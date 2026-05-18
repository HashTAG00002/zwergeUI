"""
ZwerGe-UI Retrofit Trainer

继承自 GUI-AIMA/GUI-Actor 的 AGUVISTrainer，增加：
  1. wandb 日志（grounding_loss, lm_loss, layer_weights, per-layer-acc 等）
  2. 定期 CUDA cache 清理（防止碎片化）
  3. 专为 retrofit head 设计的 optimizer group（head LR 可与 embedding LR 不同）
"""

import os
from datetime import timedelta
from functools import wraps
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin, InitProcessGroupKwargs
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer, TrainerCallback
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_accelerate_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
)
from transformers.trainer_pt_utils import LengthGroupedSampler as HFLengthGroupedSampler
from transformers.trainer_utils import seed_worker
from transformers.utils import logging

if is_datasets_available():
    import datasets


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"[Rank 0] ", *args, flush=True)
    else:
        print(*args, flush=True)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save model to disk (handles DeepSpeed and normal cases)."""
    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    if getattr(trainer, "deepspeed", None):
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class EmptyCacheCallback(TrainerCallback):
    """Periodically clear CUDA cache to reduce memory fragmentation."""
    def __init__(self, every_n_steps: int = 20):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            torch.cuda.empty_cache()


class WandbRetrofitCallback(TrainerCallback):
    """
    Custom wandb callback that logs retrofit-specific metrics:
      - grounding_loss (vs lm_loss)
      - layer_weights (omega_l per probe layer)
      - grad_norm
      - learning rate
    """
    def __init__(self, probe_layers: List[int]):
        self.probe_layers = probe_layers

    def on_log(self, args, state, control, logs: Optional[Dict] = None, **kwargs):
        if logs is None:
            return
        try:
            import wandb
            if wandb.run is not None:
                extra = {}
                # ── core losses ──────────────────────────────────────────
                if "grounding_loss" in logs:
                    extra["train/grounding_loss"] = logs["grounding_loss"]
                if "lm_loss" in logs:
                    extra["train/lm_loss"] = logs["lm_loss"]
                if "loss" in logs:
                    extra["train/total_loss"] = logs["loss"]
                # ── grad_norm ────────────────────────────────────────────
                if "grad_norm" in logs:
                    extra["train/grad_norm"] = logs["grad_norm"]
                # ── learning rate ────────────────────────────────────────
                if "learning_rate" in logs:
                    extra["train/lr"] = logs["learning_rate"]
                # ── layer weights (omega) ────────────────────────────────
                if "layer_weights" in logs:
                    omegas = logs["layer_weights"]  # list of floats, len=num_probes
                    for i, (layer_idx, w) in enumerate(zip(self.probe_layers, omegas)):
                        extra[f"layer_omega/L{layer_idx}"] = w
                if extra:
                    wandb.log(extra, step=state.global_step)
        except ImportError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────────────────────────────────────

class RetrofitTrainer(Trainer):
    """
    Trainer for ZwerGe-UI Retrofit.

    Key differences from base HF Trainer:
      1. Custom loss computation (grounding_loss + optional lm_loss)
      2. Logs grounding_loss and lm_loss separately to wandb
      3. Optimizer groups: head params get higher LR (configurable)
      4. Periodic CUDA cache flush
    """

    def __init__(self, *args, probe_layers: Optional[List[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_layers = probe_layers or []

        # Wrap _save / save_model to handle EOS token during checkpoint
        original_save = self._save
        original_save_model = self.save_model

        def wrap_save(func):
            @wraps(func)
            def wrapper(*a, **kw):
                return func(*a, **kw)
            return wrapper

        self._save = wrap_save(original_save)
        self.save_model = wrap_save(original_save_model)

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
        grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))

        dispatch_batches = getattr(self.args, "dispatch_batches", None)
        split_batches = getattr(self.args, "split_batches", None)
        self.dataloader_config = DataLoaderConfiguration(
            dispatch_batches=dispatch_batches,
            split_batches=split_batches,
        )
        self.accelerator = Accelerator(
            dataloader_config=self.dataloader_config,
            deepspeed_plugin=self.args.deepspeed_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            kwargs_handlers=[accelerator_kwargs],
        )
        self.gather_function = self.accelerator.gather_for_metrics
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                    "activation_checkpointing", fsdp_plugin.activation_checkpointing
                )
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError(
                        "activation_checkpointing in FSDP and gradient_checkpointing cannot both be True."
                    )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_length:
            lengths = getattr(self.train_dataset, "lengths", None)
            if lengths is not None:
                return HFLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                )
        return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = (
                self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None
            )
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def create_optimizer(self):
        """
        Optimizer with separate LR for grounding head vs embedding tokens.

        Parameter groups:
          Group 1: backbone + other frozen params -> NOT in optimizer (requires_grad=False)
          Group 2: layerwise_grounding_head -> args.learning_rate
          Group 3: embed_tokens (new tokens only, via hook) -> args.learning_rate_new_tokens
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [n for n in decay_parameters if "bias" not in n]

        # Identify head vs embedding params
        head_param_names = set()
        emb_param_names = set()
        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            if "layerwise_grounding_head" in n:
                head_param_names.add(n)
            elif "embed_tokens" in n:
                emb_param_names.add(n)

        lr_head = getattr(self.args, "learning_rate", 2e-4)
        lr_emb = getattr(self.args, "learning_rate_new_tokens", lr_head)

        param_groups = [
            # Head params with weight decay
            {
                "params": [p for n, p in opt_model.named_parameters()
                           if n in head_param_names and n in decay_parameters],
                "weight_decay": self.args.weight_decay,
                "lr": lr_head,
            },
            # Head params without weight decay
            {
                "params": [p for n, p in opt_model.named_parameters()
                           if n in head_param_names and n not in decay_parameters],
                "weight_decay": 0.0,
                "lr": lr_head,
            },
            # Embedding params (new tokens)
            {
                "params": [p for n, p in opt_model.named_parameters()
                           if n in emb_param_names],
                "weight_decay": 0.0,
                "lr": lr_emb,
            },
        ]
        # Filter empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        rank0_print(f"[Optimizer] Head groups: {len(head_param_names)} params, LR={lr_head}")
        rank0_print(f"[Optimizer] Emb groups: {len(emb_param_names)} params, LR={lr_emb}")

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override to accumulate grounding_loss, lm_loss, and layer_weights for logging.
        Component metrics are stashed in self._custom_metrics and flushed by
        the log() override, so they appear at the correct step in WandB.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        # Stash component losses; log() will pick them up at the right step
        if not hasattr(self, "_custom_metrics"):
            self._custom_metrics = {}
        if not hasattr(self, "_custom_counts"):
            self._custom_counts = {}

        if hasattr(outputs, "grounding_loss") and outputs.grounding_loss is not None:
            gl = outputs.grounding_loss.item()
            self._custom_metrics["grounding_loss"] = (
                self._custom_metrics.get("grounding_loss", 0.0) + gl
            )
        if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
            ll = outputs.lm_loss.item()
            self._custom_metrics["lm_loss"] = (
                self._custom_metrics.get("lm_loss", 0.0) + ll
            )

        # ── layer weights (omega) from LayerFusionScorer ──────────────────
        # outputs.layer_weights is a list of omega tensors, one per sample.
        # Average across samples in batch, accumulate across grad-accum steps.
        if hasattr(outputs, "layer_weights") and outputs.layer_weights is not None:
            # layer_weights: list[Tensor shape (num_probes,)] or None per sample
            omegas_batch = [w for w in outputs.layer_weights if w is not None]
            if omegas_batch:
                # Stack and mean over batch
                try:
                    omega_mean = torch.stack(
                        [w.float().cpu() if isinstance(w, torch.Tensor)
                         else torch.tensor(w, dtype=torch.float32)
                         for w in omegas_batch]
                    ).mean(0)  # [num_probes]
                    prev = self._custom_metrics.get("layer_weights")
                    cnt  = self._custom_counts.get("layer_weights", 0)
                    if prev is None:
                        self._custom_metrics["layer_weights"] = omega_mean
                        self._custom_counts["layer_weights"] = 1
                    else:
                        # Running mean
                        self._custom_metrics["layer_weights"] = (
                            prev * cnt + omega_mean
                        ) / (cnt + 1)
                        self._custom_counts["layer_weights"] = cnt + 1
                except Exception:
                    pass

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: dict, start_time=None):
        """
        Override to inject stashed custom metrics into the current logging event,
        avoiding the WandB 'step must be monotonically increasing' warning that
        occurs when self.log() is called independently from compute_loss().
        """
        if hasattr(self, "_custom_metrics") and self._custom_metrics:
            ga = max(1, self.args.gradient_accumulation_steps)
            for k, v in self._custom_metrics.items():
                if k == "layer_weights":
                    # Pass as list of floats for WandbRetrofitCallback to unpack
                    logs[k] = v.tolist() if isinstance(v, torch.Tensor) else list(v)
                else:
                    logs[k] = v / ga
            self._custom_metrics = {}
            self._custom_counts = {}
        # Forward to parent (handles WandB, TensorBoard, etc.)
        if start_time is not None:
            super().log(logs, start_time=start_time)
        else:
            super().log(logs)
