"""
ZwerGe-UI Retrofit Trainer

继承自 GUI-AIMA/GUI-Actor 的 AGUVISTrainer，增加：
  1. wandb 日志（grounding_loss, lm_loss, layer_weights, per-layer-acc 等）
  2. 定期 CUDA cache 清理（防止碎片化）
  3. 专为 retrofit head 设计的 optimizer group（head LR 可与 embedding LR 不同）
"""

import json
import math
import os
import pathlib
import shutil
import sys
from datetime import timedelta
from functools import wraps
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin, InitProcessGroupKwargs
from torch.utils.data import DataLoader, RandomSampler
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import has_length
from transformers.utils.import_utils import (
    is_accelerate_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
)
# ALL_LAYERNORM_LAYERS moved to pytorch_utils in transformers>=4.56; fall back gracefully
try:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
except ImportError:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import LengthGroupedSampler as HFLengthGroupedSampler
try:
    import inspect as _inspect
    from transformers.trainer_utils import seed_worker as _seed_worker
    _sw_sig = _inspect.signature(_seed_worker)
    if len(_sw_sig.parameters) > 1:
        # New transformers (>=4.57): seed_worker(worker_id, num_workers, rank)
        # Must be wrapped with partial to fill num_workers/rank at DataLoader creation time
        _SEED_WORKER_NEW_API = True
    else:
        _SEED_WORKER_NEW_API = False
except ImportError:
    _seed_worker = None
    _SEED_WORKER_NEW_API = False
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


def safe_save_model_for_hf_trainer(trainer: "Trainer", output_dir: str):
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
# Window metric plumbing (observation only; no training-behavior changes)
# ─────────────────────────────────────────────────────────────────────────────

# Fixed global mean keys (window SUM / actual COUNT). Per-layer keys are
# derived from the trainer's probe_layers so EVERY rank builds the same key
# universe even when a rank saw zero valid samples in the window.
_WINDOW_GLOBAL_MEAN_KEYS = (
    "grounding_loss",            # legacy key; now a true window mean
    "lm_loss",                   # legacy key; same fix
    "diag/loss_model_mean",      # raw microbatch model loss mean
    "diag/grounding_valid_mean", # valid-grounding-sample mean (zero-skip excluded)
    "data/uniform_ref_kl_mean",  # log(N_vis) - H(label_dist), valid-sample mean
)
_WINDOW_LAYER_METRICS = ("kl_mean", "logit_rms_centered", "entropy_norm", "gt_floor_mass")
_WINDOW_INT_KEYS = (
    "diag/nonfinite_forward_count",
    "_model_samples",
    "_valid_samples",
    "_newtoken_grad_none",
)
_WINDOW_MAX_KEYS = (
    "diag/loss_micro_max",
    "optim/grad_norm_preclip_max",
    "optim/new_tokens_grad_rms_postclip",
)
# Legacy local running-mean keys kept with unchanged semantics (NOT reduced).
_LEGACY_LOCAL_MEAN_KEYS = ("layer_weights", "omega_entropy", "p_final_entropy", "p_final_max")


def _build_window_payloads(metrics, counts, maxes, taints, probe_layers):
    """Pack window accumulators into flat payloads for cross-rank reduction.

    Returns (sum_payload, max_payload):
      sum_payload: "s:" value sums, "n:" matching counts, "t:" non-finite
                   taint counts, "c:" integer counters — all SUM-reduced.
      max_payload: "x:" window peaks (or -inf when absent) — MAX-reduced.

    The key universe is fixed globals + probe_layers so all ranks build
    identical key sets. Pure function: no I/O, no RNG, no model state.
    """
    layers = list(probe_layers or [])
    if not layers:
        layers = sorted({
            int(k.split("/")[1][1:]) for k in metrics if k.startswith("probe/L")
        })
    mean_keys = list(_WINDOW_GLOBAL_MEAN_KEYS) + [
        f"probe/L{ell}/{m}" for ell in layers for m in _WINDOW_LAYER_METRICS
    ]
    sum_payload = {}
    for k in mean_keys:
        sum_payload[f"s:{k}"] = float(metrics.get(k, 0.0))
        sum_payload[f"n:{k}"] = float(counts.get(k, 0))
        sum_payload[f"t:{k}"] = float(taints.get(k, 0))
    for k in _WINDOW_INT_KEYS:
        sum_payload[f"c:{k}"] = float(counts.get(k, 0))
    max_payload = {f"x:{k}": float(maxes.get(k, float("-inf"))) for k in _WINDOW_MAX_KEYS}
    return sum_payload, max_payload


def _window_allreduce(sum_payload, max_payload):
    """Cross-rank reduce: SUM for sum_payload, MAX for max_payload.

    Observation only — reduced values are diagnostic scalars, never model
    state. Must be entered by ALL ranks unconditionally (log() is called on
    every rank at the same step, so no conditional-collective deadlock).
    No-op when torch.distributed is not initialized.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return sum_payload, max_payload
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend() == "nccl" else torch.device("cpu")
    )
    sum_keys = sorted(sum_payload)
    t_sum = torch.tensor(
        [sum_payload[k] for k in sum_keys], dtype=torch.float64, device=device
    )
    dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
    out_sum = dict(zip(sum_keys, (float(v) for v in t_sum.tolist())))
    out_max = max_payload
    if max_payload:
        max_keys = sorted(max_payload)
        t_max = torch.tensor(
            [max_payload[k] for k in max_keys], dtype=torch.float64, device=device
        )
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
        out_max = dict(zip(max_keys, (float(v) for v in t_max.tolist())))
    return out_sum, out_max


def _emit_window_logs(sum_payload, max_payload):
    """Turn (reduced) payloads into final Trainer.log keys.

    Mean = SUM(value) / SUM(count). A mean tainted by a non-finite
    observation in this window is omitted (unavailable) rather than silently
    folding the bad value in as 0. Peaks use MAX. Pure function.
    """
    logs = {}
    mean_keys = sorted({k[2:] for k in sum_payload if k.startswith("s:")})
    for k in mean_keys:
        n = sum_payload.get(f"n:{k}", 0.0)
        t = sum_payload.get(f"t:{k}", 0.0)
        if n > 0 and t == 0:
            logs[k] = sum_payload[f"s:{k}"] / n
    n_model = sum_payload.get("c:_model_samples", 0.0)
    if n_model > 0:
        logs["data/grounding_valid_frac"] = (
            sum_payload.get("c:_valid_samples", 0.0) / n_model
        )
    logs["diag/nonfinite_forward_count"] = sum_payload.get(
        "c:diag/nonfinite_forward_count", 0.0
    )
    newtoken_grad_none = sum_payload.get("c:_newtoken_grad_none", 0.0)
    for k in _WINDOW_MAX_KEYS:
        v = max_payload.get(f"x:{k}", float("-inf"))
        if v != float("-inf") and math.isfinite(v):
            if k == "optim/new_tokens_grad_rms_postclip" and newtoken_grad_none > 0:
                continue   # grad was None this window → unavailable, not 0
            logs[k] = v
    if newtoken_grad_none > 0:
        logs["optim/new_tokens_grad_none_updates"] = newtoken_grad_none
    return logs


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class SyncNewTokenEmbCallback(TrainerCallback):
    """
    Before every checkpoint save, write the trained _new_token_emb rows back
    into embed_tokens.weight.data so that saved weights are self-contained
    (inference doesn't know about the forward hook, it reads embed_tokens directly).
    """
    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        raw = getattr(model, "module", model)
        if not (hasattr(raw, "_new_token_emb") and hasattr(raw, "_new_token_id_to_row")):
            return
        if args.local_rank not in (0, -1):
            return
        # get_input_embeddings() works for both Qwen2.5-VL and Qwen3-VL
        emb_module = raw.get_input_embeddings()
        with torch.no_grad():
            for tid, ri in raw._new_token_id_to_row.items():
                emb_module.weight.data[tid] = (
                    raw._new_token_emb.data[ri].to(emb_module.weight.dtype)
                )


class EmptyCacheCallback(TrainerCallback):
    """Periodically clear CUDA cache to reduce memory fragmentation."""
    def __init__(self, every_n_steps: int = 20):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            torch.cuda.empty_cache()


class ResumeCheckpointManagerCallback(TrainerCallback):
    """
    Manages checkpoint retention for elastic-queue (hope) training.

    Classifies saved checkpoints into two tiers:
      - Permanent: step divisible by permanent_save_steps → kept forever
      - Resume-only: all other steps → keep only the latest max_resume_ckpts

    Must be used together with save_steps_only_for_resume > 0 in TrainingArguments.
    Set save_total_limit=None before creating Trainer so HF's own rotation is disabled.

    Runs on_save (rank-0 only) after each checkpoint is written.
    """
    def __init__(self, permanent_save_steps: int, resume_save_steps: int, max_resume_ckpts: int = 2):
        self.permanent_save_steps = permanent_save_steps
        self.resume_save_steps = resume_save_steps
        self.max_resume_ckpts = max_resume_ckpts

    def on_save(self, args, state, control, **kwargs):
        if self.resume_save_steps <= 0:
            return
        if args.local_rank not in (0, -1):
            return

        output_dir = pathlib.Path(args.output_dir)
        ckpts = sorted(
            [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[1]),
        )

        resume_only = [
            ckpt for ckpt in ckpts
            if int(ckpt.name.split("-")[1]) % self.permanent_save_steps != 0
        ]

        # Write marker so eval_daemon can distinguish permanent from resume-only ckpts
        for ckpt in ckpts:
            if ckpt not in resume_only:
                (ckpt / "is_permanent_ckpt").touch(exist_ok=True)

        if len(resume_only) > self.max_resume_ckpts:
            for ckpt in resume_only[: -self.max_resume_ckpts]:
                rank0_print(f"[ResumeManager] Deleting old resume-only ckpt: {ckpt.name}")
                shutil.rmtree(ckpt, ignore_errors=True)


class SaveWandbRunIdCallback(TrainerCallback):
    """
    At the start of training, persist the wandb run ID to OUTPUT_DIR/.wandb_run_id
    so eval_daemon.py can later resume logging eval metrics to the same run.
    """
    def on_train_begin(self, args, state, control, **kwargs):
        if args.local_rank not in (0, -1):
            return
        try:
            import wandb
            if wandb.run is not None:
                run_id_file = pathlib.Path(args.output_dir) / ".wandb_run_id"
                run_id_file.write_text(wandb.run.id)
        except Exception:
            pass


class _OptimPostclipDiagCallback(TrainerCallback):
    """Reads the post-clip new-token gradient RMS (observation only).

    HF 4.51 optimizer boundary order (verified in transformers 4.51.3):
        accelerator.clip_grad_norm_  → returns the PRE-clip global norm
        on_pre_optimizer_step        ← fires here: .grad already reduced+clipped
        optimizer.step
    So .grad read here is the post-clip gradient; we never re-clip, never
    re-unscale, never touch optimizer.step. A None grad (no gradient path)
    is counted separately from a numeric zero gradient — the RMS metric is
    then reported unavailable for the window instead of a fake 0.
    """

    def __init__(self, trainer):
        self._trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        t = self._trainer
        if model is None:
            return
        raw = getattr(model, "module", model)
        p = getattr(raw, "_new_token_emb", None)
        if p is None:
            return
        g = p.grad
        counts = getattr(t, "_custom_counts", None)
        if counts is None:
            t._custom_counts = counts = {}
        if g is None:
            counts["_newtoken_grad_none"] = counts.get("_newtoken_grad_none", 0) + 1
            return
        with torch.no_grad():
            v = float(g.detach().float().pow(2).mean().sqrt())
        if math.isfinite(v):
            maxes = getattr(t, "_custom_max", None)
            if maxes is None:
                t._custom_max = maxes = {}
            prev = maxes.get("optim/new_tokens_grad_rms_postclip")
            maxes["optim/new_tokens_grad_rms_postclip"] = (
                v if prev is None else max(prev, v)
            )


# ─────────────────────────────────────────────────────────────────────────────
# In-Training Evaluation Callback
# ─────────────────────────────────────────────────────────────────────────────

class ValEvalCallback(TrainerCallback):
    """
    全量分布式验证回调（与 run_vis.sh 完全等价）。

    设计：ALL DDP ranks 同时参与评估，每个 rank 处理一个数据 shard，
    使用已加载的训练模型做推理（无需重新加载），生成 PNG 可视化 + 评测指标，
    rank-0 聚合后写入与 run_vis.sh 相同的目录结构并上报 WandB。

    触发时机：
      - on_step_end: 每 val_steps 步（val_steps>0）
      - on_train_end: 训练结束（val_steps>0）

    bench 配置：
      val_bench = "all"   → 全部 5 个 bench
      val_bench = "ss_pro" → 单 bench

    数据量：
      val_n_samples = -1  → 全量数据（默认）
      val_n_samples =  N  → 前 N 条

    输出结构（与 eval_daemon 异步评估保持一致）：
      {val_output_dir}/checkpoint-{step}/results/
        ├── {bench}_layerwise_summary.json
        ├── layerwise_all_summary.json   (bench=all 时)
        └── details/{bench}/
            ├── success/*.png
            ├── failure/*.png
            └── results.json
    val_output_dir 建议直接传入训练的 OUTPUT_DIR，使所有评估产物
    集中在对应 checkpoint 目录下。
    """

    def __init__(
        self,
        training_args,
        processor,
        probe_layers: List[int],
        model_type: str = "uitars",
        system_message: Optional[str] = None,
        ground_response: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ):
        self.val_steps       = getattr(training_args, "val_steps", -1)
        self.val_bench       = getattr(training_args, "val_bench", "all")
        self.val_n_samples   = getattr(training_args, "val_n_samples", -1)
        self.eval_dir        = getattr(training_args, "val_eval_dir", "")
        self.output_dir_root = getattr(training_args, "val_output_dir", "")
        self.decode_strategy = getattr(training_args, "val_decode_strategy", "centroid")
        self.max_pixels      = getattr(training_args, "val_max_pixels", 12_845_056)
        self.processor       = processor
        self.probe_layers    = probe_layers
        self.model_type      = model_type
        # model-specific prompt constants
        self.system_message       = system_message
        self.ground_response      = ground_response
        self.user_prompt_template = user_prompt_template

        # Cell sizes for vis (use vis_zwerge defaults)
        self.cell_w = getattr(training_args, "val_cell_w", 300)
        self.cell_h = getattr(training_args, "val_cell_h", 220)
        self.alpha  = getattr(training_args, "val_alpha",  0.55)

        _eval_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "eval")
        )
        if _eval_dir not in sys.path:
            sys.path.insert(0, _eval_dir)

    # ------------------------------------------------------------------
    def on_step_end(self, args, state, control, **kwargs):
        if self.val_steps > 0 and state.global_step % self.val_steps == 0:
            self._run_eval_distributed(args, state, kwargs.get("model"))

    def on_train_end(self, args, state, control, **kwargs):
        if self.val_steps > 0:
            self._run_eval_distributed(args, state, kwargs.get("model"))

    # ------------------------------------------------------------------
    def _run_eval_distributed(self, args, state, model):
        """
        所有 rank 同步进入评估，每个 rank 处理其 shard，rank-0 聚合。
        """
        if model is None:
            return

        # 所有 rank 同步进入
        if dist.is_initialized():
            dist.barrier()

        try:
            bench_list = self._bench_list()
            rank      = args.local_rank if args.local_rank >= 0 else 0
            n_ranks   = dist.get_world_size() if dist.is_initialized() else 1
            step      = state.global_step
            step_dir  = os.path.join(
                self.output_dir_root, f"checkpoint-{step}", "results"
            ) if self.output_dir_root else ""

            raw_model = getattr(model, "module", model)
            device    = next(raw_model.parameters()).device
            if hasattr(self.processor, "image_processor"):
                self.processor.image_processor.max_pixels = self.max_pixels

            raw_model.eval()
            all_summaries = {}

            for bench_key in bench_list:
                summary = self._eval_bench_shard(
                    bench_key=bench_key,
                    raw_model=raw_model,
                    device=device,
                    rank=rank,
                    n_ranks=n_ranks,
                    step=step,
                    step_dir=step_dir,
                )
                if summary is not None:
                    all_summaries[bench_key] = summary

            raw_model.train()

        except Exception as e:
            rank0_print(f"[ValEval] eval failed step={state.global_step}: {e}")
            import traceback; traceback.print_exc()
        finally:
            if dist.is_initialized():
                dist.barrier()

        # rank-0 聚合 + WandB
        if args.local_rank in (0, -1) and all_summaries and step_dir:
            self._report_wandb(all_summaries, step)
            if len(all_summaries) > 1:
                with open(os.path.join(step_dir, "layerwise_all_summary.json"), "w") as f:
                    json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    def _eval_bench_shard(self, bench_key, raw_model, device, rank, n_ranks, step, step_dir):
        """每个 rank 处理其 shard，rank-0 聚合后返回 summary；非 rank-0 返回 None。"""
        try:
            from inference_base import (
                BENCH_CONFIGS, scores_to_point_and_topk,
                point_in_bbox, do_boxes_overlap,
                _get_group_key, _print_layerwise_summary,
            )
            from eval_retrofit import get_inference_class, _aggregate_shards
            from vis_utils import visualize_sample
            from PIL import Image as _PIL
            import glob as _glob
        except ImportError as e:
            rank0_print(f"[ValEval] import error: {e}")
            return None

        cfg       = BENCH_CONFIGS[bench_key]
        bench_name = cfg["name"]
        eval_json  = os.path.join(self.eval_dir, cfg["eval_dir"], cfg["eval_json"])
        img_root   = os.path.join(self.eval_dir, cfg["eval_dir"])
        group_field = cfg.get("group_field")

        if not os.path.exists(eval_json):
            return None

        with open(eval_json) as f:
            all_data = json.load(f)
        if self.val_n_samples > 0:
            all_data = all_data[:self.val_n_samples]
        N = len(all_data)

        # Contiguous shard for this rank
        chunk = (N + n_ranks - 1) // n_ranks
        start = rank * chunk
        end   = min(start + chunk, N)
        shard = all_data[start:end]

        # Output dirs
        if step_dir:
            success_dir = os.path.join(step_dir, "details", bench_key, "success")
            failure_dir = os.path.join(step_dir, "details", bench_key, "failure")
            os.makedirs(success_dir, exist_ok=True)
            os.makedirs(failure_dir, exist_ok=True)
        else:
            success_dir = failure_dir = ""

        probe_layers = list(raw_model.layerwise_grounding_head.probe_layers)
        n_probes     = len(probe_layers)
        layer_stats  = [{"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
                        for _ in range(n_probes)]
        fusion_stats  = {"hit1": 0, "hitk": 0, "overlap1": 0, "overlapk": 0, "total": 0}
        fusion_groups: Dict[str, Dict] = {}
        results = []
        skip    = 0
        TOPK    = 3

        # Create inference wrapper once (wraps already-loaded training model)
        InfClass = get_inference_class(self.model_type)
        grounder = InfClass(
            model=raw_model, processor=self.processor,
            system_message=self.system_message,
            ground_response=self.ground_response,
            user_prompt_template=self.user_prompt_template,
        )

        with torch.no_grad():
            for idx, example in enumerate(shard):
                global_idx = start + idx
                img_path = os.path.join(img_root, example["image_path"])
                if not os.path.exists(img_path):
                    skip += 1; continue
                try:
                    orig_img = _PIL.open(img_path).convert("RGB")
                    W, H = float(example["image_size"][0]), float(example["image_size"][1])
                    x1, y1, x2, y2 = example["gt_bbox"]
                    gt = (x1/W, y1/H, x2/W, y2/H)

                    pred = grounder.predict_layerwise(
                        image=orig_img, instruction=example["instruction"],
                        device=device, decode_strategy=self.decode_strategy, topk=TOPK,
                    )
                    n_w, n_h = pred["n_width"], pred["n_height"]
                    phx, phy = 0.5/n_w, 0.5/n_h

                    # Per-layer metrics
                    layer_metrics = []
                    for li in range(n_probes):
                        lpt = pred["per_layer_points"][li]
                        px, py = float(lpt[0]), float(lpt[1])
                        tpts   = pred["per_layer_topk"][li]
                        hit1   = int(point_in_bbox(px, py, gt))
                        hitk   = int(any(point_in_bbox(float(p[0]),float(p[1]),gt) for p in tpts))
                        ov1    = int(do_boxes_overlap((px-phx,py-phy,px+phx,py+phy), gt))
                        ovk    = ov1
                        for pk in tpts[1:]:
                            if do_boxes_overlap((float(pk[0])-phx,float(pk[1])-phy,
                                                  float(pk[0])+phx,float(pk[1])+phy), gt):
                                ovk = 1; break
                        layer_stats[li]["hit1"]     += hit1
                        layer_stats[li]["hitk"]     += hitk
                        layer_stats[li]["overlap1"] += ov1
                        layer_stats[li]["overlapk"] += ovk
                        layer_stats[li]["total"]    += 1
                        layer_metrics.append({
                            "hit_top1": hit1, "hit_topk": hitk,
                            "overlap_top1": ov1, "overlap_topk": ovk,
                            "pred_point": list(pred["per_layer_points"][li]),
                        })

                    # Fusion metrics
                    fb, fc = scores_to_point_and_topk(
                        p=pred["p_final"], n_width=n_w, n_height=n_h,
                        activation_threshold=0.3, topk=TOPK,
                        decode_strategy=self.decode_strategy,
                    )
                    fpx, fpy = float(fb[0]), float(fb[1])
                    fhit1 = int(point_in_bbox(fpx, fpy, gt))
                    fov1  = int(do_boxes_overlap((fpx-phx,fpy-phy,fpx+phx,fpy+phy), gt))
                    fhitk = int(any(point_in_bbox(float(p[0]),float(p[1]),gt) for p in fc))
                    fovk  = fov1
                    for fk in fc[1:]:
                        if do_boxes_overlap((float(fk[0])-phx,float(fk[1])-phy,
                                              float(fk[0])+phx,float(fk[1])+phy), gt):
                            fovk = 1; break
                    fusion_stats["hit1"]     += fhit1
                    fusion_stats["overlap1"] += fov1
                    fusion_stats["hitk"]     += fhitk
                    fusion_stats["overlapk"] += fovk
                    fusion_stats["total"]    += 1
                    if group_field:
                        grp = _get_group_key(example, group_field)
                        if grp not in fusion_groups:
                            fusion_groups[grp] = {"hit1":0,"hitk":0,"overlap1":0,"overlapk":0,"total":0}
                        fusion_groups[grp]["hit1"]     += fhit1
                        fusion_groups[grp]["overlap1"] += fov1
                        fusion_groups[grp]["hitk"]     += fhitk
                        fusion_groups[grp]["overlapk"] += fovk
                        fusion_groups[grp]["total"]    += 1

                    # Visualization
                    if success_dir:
                        meta = {"bench": bench_name}
                        for k in ["ui_type","data_type","GUI_types","grounding_type",
                                  "task_type","platform","application","element_type"]:
                            if k in example: meta[k] = example[k]
                        vis_img = visualize_sample(
                            orig_img=orig_img, pred=pred, gt_bbox_norm=gt,
                            instruction=example["instruction"], meta=meta,
                            activation_threshold=0.3, decode_strategy=self.decode_strategy,
                            cell_w=self.cell_w, cell_h=self.cell_h, alpha=self.alpha,
                        )
                        save_d = success_dir if fhit1 else failure_dir
                        vis_img.save(os.path.join(save_d, f"idx{global_idx:05d}.png"))

                    # Record
                    rec = {
                        "idx": global_idx,
                        "image_path": example["image_path"],
                        "instruction": example["instruction"],
                        "gt_bbox_norm": list(gt),
                        "anchor_strategy": pred["anchor_strategy"],
                        "n_width": n_w, "n_height": n_h,
                        "omega": pred["omega"].tolist(),
                        "probe_layers": probe_layers,
                        "layer_metrics": layer_metrics,
                        "fusion_hit1": fhit1, "fusion_overlap1": fov1,
                        "fusion_hitk": fhitk, "fusion_overlapk": fovk,
                    }
                    for ext in ["id","ui_type","group","platform","application","data_type",
                                "split","grounding_type","task_type","GUI_types",
                                "category","element_type"]:
                        if ext in example: rec[ext] = example[ext]
                    results.append(rec)

                except Exception as _e:
                    import traceback as _tb
                    rank0_print(f"[ValEval] sample #{global_idx} failed: {_e}")
                    _tb.print_exc()
                    skip += 1

        # Write shard files (all ranks)
        if step_dir:
            det_dir = os.path.join(step_dir, "details", bench_key)
            os.makedirs(det_dir, exist_ok=True)
            with open(os.path.join(det_dir, f"results_{start}-{end}.json"), "w") as f:
                json.dump(results, f, ensure_ascii=False)

        # Shard summary (all ranks write, rank-0 aggregates)
        valid_cnt = len(shard) - skip
        la = []
        for li, ls in enumerate(layer_stats):
            n = ls["total"] or 1
            la.append({
                "layer_idx":    probe_layers[li], "probe_rank": li,
                "hit_top1":     round(ls["hit1"]/n*100, 4),
                "overlap_top1": round(ls["overlap1"]/n*100, 4),
                "hit_topk":     round(ls["hitk"]/n*100, 4),
                "overlap_topk": round(ls["overlapk"]/n*100, 4),
                "n": ls["total"],
            })
        fn = fusion_stats["total"] or 1
        fa = {
            "hit_top1":     round(fusion_stats["hit1"]/fn*100, 4),
            "overlap_top1": round(fusion_stats["overlap1"]/fn*100, 4),
            "hit_topk":     round(fusion_stats["hitk"]/fn*100, 4),
            "overlap_topk": round(fusion_stats["overlapk"]/fn*100, 4),
            "topk": TOPK,
        }
        fga: Dict = {}
        if group_field:
            for grp, st in sorted(fusion_groups.items()):
                gn = st["total"] or 1
                fga[grp] = {
                    "hit_top1":     round(st["hit1"]/gn*100, 2),
                    "overlap_top1": round(st["overlap1"]/gn*100, 2),
                    "hit_topk":     round(st["hitk"]/gn*100, 2),
                    "overlap_topk": round(st["overlapk"]/gn*100, 2),
                    "total": st["total"],
                }
        shard_summary = {
            "bench": bench_name, "bench_key": bench_key,
            "total": len(shard), "valid": valid_cnt, "skipped": skip,
            "slice": [start, end], "probe_layers": probe_layers,
            "layer_accs": la, "fusion_acc": fa, "fusion_group_accs": fga,
        }
        if step_dir:
            with open(os.path.join(step_dir, f"{bench_key}_layerwise_summary_{start}-{end}.json"), "w") as f:
                json.dump(shard_summary, f, ensure_ascii=False)

        # Barrier: wait for all shards to complete
        if dist.is_initialized():
            dist.barrier()

        # Rank-0 aggregates
        if rank != 0:
            return None
        if not step_dir:
            return shard_summary  # single rank, return directly

        try:
            from eval_retrofit import _aggregate_shards
            summary = _aggregate_shards(step_dir, bench_key, TOPK, skip_vis=False)
        except Exception as e:
            rank0_print(f"[ValEval] aggregation failed for {bench_key}: {e}")
            summary = shard_summary  # fallback to own shard

        _print_layerwise_summary(summary, TOPK)
        return summary

    # ------------------------------------------------------------------
    def _bench_list(self) -> List[str]:
        try:
            from inference_base import MAIN_BENCH_KEYS
        except ImportError:
            return [self.val_bench]
        if self.val_bench == "all":
            return MAIN_BENCH_KEYS
        return [self.val_bench]

    def _report_wandb(self, summaries: Dict, step: int):
        try:
            import wandb
            if wandb.run is None:
                return
            log_dict = {}
            for bench_key, sm in summaries.items():
                fa = sm.get("fusion_acc", {})
                log_dict[f"val/{bench_key}/hit_top1"]     = fa.get("hit_top1", 0)
                log_dict[f"val/{bench_key}/overlap_top1"] = fa.get("overlap_top1", 0)
                log_dict[f"val/{bench_key}/valid"]        = sm.get("valid", 0)
                # Best single layer hit@1
                best = max(sm.get("layer_accs", [{}]), key=lambda x: x.get("hit_top1", 0), default={})
                if best:
                    log_dict[f"val/{bench_key}/best_layer_hit"]     = best.get("hit_top1", 0)
                    log_dict[f"val/{bench_key}/best_layer_idx"]     = best.get("layer_idx", -1)
            wandb.log(log_dict, step=step)
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

    def __init__(self, *args, probe_layers: Optional[List[int]] = None,
                 metric_layer_labels: Optional[List[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_layers = probe_layers or []
        # Layer labels for flattening layer_weights (omega) into scalar keys.
        # For A8 this is the ACTIVE subset (same labeling the removed
        # WandbRetrofitCallback used); defaults to probe_layers.
        self.metric_layer_labels = metric_layer_labels or list(self.probe_layers)

        # ── Window diagnostics accumulators (observation only) ────────────
        self._custom_metrics: Dict[str, float] = {}
        self._custom_counts: Dict[str, float] = {}
        self._custom_max: Dict[str, float] = {}
        self._custom_taint: Dict[str, float] = {}
        self._install_optim_observers()

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

    def _install_optim_observers(self):
        """Attach pure-observation hooks to the optimizer boundary.

        1. Wrap accelerator.clip_grad_norm_ to record the PRE-clip global
           norm it actually returns (HF 4.51 calls it right before
           on_pre_optimizer_step). The original function is called through
           exactly once — no re-clip, no re-unscale, no behavior change.
        2. Register _OptimPostclipDiagCallback for post-clip new-token grad.
        """
        trainer = self
        try:
            acc = self.accelerator
            orig_clip = acc.clip_grad_norm_

            def _observing_clip_grad_norm_(parameters, max_norm, norm_type=2):
                result = orig_clip(parameters, max_norm, norm_type)
                try:
                    if result is not None:
                        v = float(result)
                        if math.isfinite(v):
                            maxes = getattr(trainer, "_custom_max", None)
                            if maxes is None:
                                trainer._custom_max = maxes = {}
                            prev = maxes.get("optim/grad_norm_preclip_max")
                            maxes["optim/grad_norm_preclip_max"] = (
                                v if prev is None else max(prev, v)
                            )
                except Exception:
                    pass
                return result

            acc.clip_grad_norm_ = _observing_clip_grad_norm_
        except Exception:
            pass
        try:
            self.add_callback(_OptimPostclipDiagCallback(self))
        except Exception:
            pass

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
        # is_tp_enabled added in transformers>=4.57 (Tensor Parallelism); we don't use TP
        self.is_tp_enabled = getattr(self.accelerator.state, "torch_tp_plugin", None) is not None

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
        # group_by_length was removed in transformers>=5.x; guard with getattr for compat
        if getattr(self.args, "group_by_length", False):
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
            if _seed_worker is not None:
                if _SEED_WORKER_NEW_API:
                    from functools import partial as _partial
                    dataloader_params["worker_init_fn"] = _partial(
                        _seed_worker,
                        num_workers=self.args.dataloader_num_workers,
                        rank=getattr(self.args, "process_index", 0),
                    )
                else:
                    dataloader_params["worker_init_fn"] = _seed_worker
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

        # Identify head vs new-token embedding params
        # _new_token_emb is a tiny [n_new, d_model] Parameter separate from the frozen
        # embed_tokens.weight, so Adam only tracks n_new*d_model states instead of 543M.
        head_param_names = set()
        emb_param_names = set()
        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            if "layerwise_grounding_head" in n:
                head_param_names.add(n)
            elif "_new_token_emb" in n or "new_token_embeddings" in n:
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
        Override to accumulate component/diagnostic metrics for logging.

        Window accounting is SUM / actual COUNT (per microbatch or per valid
        sample, depending on the metric), flushed by log(). This fixes the
        old convention that divided the window sum by
        gradient_accumulation_steps only, which inflated custom losses by
        logging_steps (constant 2.0 logged as 40.0 at logging_steps=20).
        The loss returned for backward is completely untouched; every
        diagnostic value below is detached.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        # Stash component metrics; log() will pick them up at the right step
        if not hasattr(self, "_custom_metrics"):
            self._custom_metrics = {}
        if not hasattr(self, "_custom_counts"):
            self._custom_counts = {}
        if not hasattr(self, "_custom_max"):
            self._custom_max = {}
        if not hasattr(self, "_custom_taint"):
            self._custom_taint = {}
        m, c = self._custom_metrics, self._custom_counts

        def _add_mean(key, value):
            """sum/count accumulate; non-finite → count + taint, never 0."""
            v = float(value)
            if math.isfinite(v):
                m[key] = m.get(key, 0.0) + v
                c[key] = c.get(key, 0) + 1
                return v
            c["diag/nonfinite_forward_count"] = c.get("diag/nonfinite_forward_count", 0) + 1
            self._custom_taint[key] = self._custom_taint.get(key, 0) + 1
            return None

        # ── raw microbatch model loss (HF's own 'loss' curve is untouched) ──
        _lv = _add_mean("diag/loss_model_mean", loss.detach())
        if _lv is not None:
            prev = self._custom_max.get("diag/loss_micro_max")
            self._custom_max["diag/loss_micro_max"] = _lv if prev is None else max(prev, _lv)

        # ── legacy component losses (now true window means) ──
        if getattr(outputs, "grounding_loss", None) is not None:
            _add_mean("grounding_loss", outputs.grounding_loss.detach())
        if getattr(outputs, "lm_loss", None) is not None:
            _add_mean("lm_loss", outputs.lm_loss.detach())

        # ── structured diagnostics payload from RetrofitModelMixin ──
        # When training with DDP, model is wrapped by DistributedDataParallel;
        # _grounding_diag is set on the inner (unwrapped) model.  We unwrap
        # here (and in the eval/inference paths) so diagnostics flow through
        # regardless of whether DDP/FSDP is active or not.
        raw_model = getattr(model, "module", model)
        diag = getattr(raw_model, "_grounding_diag", None)
        if diag:
            c["_model_samples"] = c.get("_model_samples", 0) + diag.get("batch_size", 0)
            c["_valid_samples"] = c.get("_valid_samples", 0) + diag.get("valid_count", 0)
            _nf = diag.get("nonfinite_forward_count", 0)
            if _nf:
                c["diag/nonfinite_forward_count"] = c.get("diag/nonfinite_forward_count", 0) + _nf
            if diag.get("valid_mean_tainted"):
                self._custom_taint["diag/grounding_valid_mean"] = (
                    self._custom_taint.get("diag/grounding_valid_mean", 0) + 1
                )
            if diag.get("valid_loss_finite_count"):
                m["diag/grounding_valid_mean"] = (
                    m.get("diag/grounding_valid_mean", 0.0) + diag["valid_loss_sum"]
                )
                c["diag/grounding_valid_mean"] = (
                    c.get("diag/grounding_valid_mean", 0) + diag["valid_loss_finite_count"]
                )
            if diag.get("uniform_ref_kl_count"):
                m["data/uniform_ref_kl_mean"] = (
                    m.get("data/uniform_ref_kl_mean", 0.0) + diag["uniform_ref_kl_sum"]
                )
                c["data/uniform_ref_kl_mean"] = (
                    c.get("data/uniform_ref_kl_mean", 0) + diag["uniform_ref_kl_count"]
                )
            for _lidx, _slot in (diag.get("layers") or {}).items():
                if _slot.get("count"):
                    for _metric, _skey in (("kl_mean", "kl_sum"),
                                           ("logit_rms_centered", "logit_rms_sum"),
                                           ("entropy_norm", "entropy_sum"),
                                           ("gt_floor_mass", "gt_floor_mass_sum")):
                        _key = f"probe/L{_lidx}/{_metric}"
                        m[_key] = m.get(_key, 0.0) + _slot[_skey]
                        c[_key] = c.get(_key, 0) + _slot["count"]
                if _slot.get("nonfinite"):
                    for _metric in ("kl_mean", "logit_rms_centered",
                                    "entropy_norm", "gt_floor_mass"):
                        _key = f"probe/L{_lidx}/{_metric}"
                        self._custom_taint[_key] = self._custom_taint.get(_key, 0) + _slot["nonfinite"]

        # ── layer weights (omega) from ContextLoRACosMetaFusion ──────────────
        if hasattr(outputs, "layer_weights") and outputs.layer_weights is not None:
            omegas_batch = [w for w in outputs.layer_weights if w is not None]
            if omegas_batch:
                try:
                    omega_stack = torch.stack(
                        [w.float().cpu() if isinstance(w, torch.Tensor)
                         else torch.tensor(w, dtype=torch.float32)
                         for w in omegas_batch]
                    )  # [B, num_probes]
                    omega_mean = omega_stack.mean(0)  # [num_probes]
                    prev = self._custom_metrics.get("layer_weights")
                    cnt  = self._custom_counts.get("layer_weights", 0)
                    if prev is None:
                        self._custom_metrics["layer_weights"] = omega_mean
                        self._custom_counts["layer_weights"] = 1
                    else:
                        self._custom_metrics["layer_weights"] = (
                            prev * cnt + omega_mean) / (cnt + 1)
                        self._custom_counts["layer_weights"] = cnt + 1
                    # omega_entropy: H(omega) per sample → mean over batch
                    eps = 1e-8
                    ent = -(omega_stack * torch.log(omega_stack.clamp(min=eps))).sum(-1).mean().item()
                    _k = "omega_entropy"
                    _c = self._custom_counts.get(_k, 0)
                    self._custom_metrics[_k] = (self._custom_metrics.get(_k, 0.0) * _c + ent) / (_c + 1)
                    self._custom_counts[_k] = _c + 1
                except Exception:
                    pass

        # ── p_final stats (sharpness / confidence) ────────────────────────
        if hasattr(outputs, "grounding_scores") and outputs.grounding_scores is not None:
            p_finals = [s for s in outputs.grounding_scores if s is not None]
            if p_finals:
                try:
                    eps = 1e-8
                    entropies, maxes = [], []
                    for pf in p_finals:
                        pf_f = pf.float()
                        entropies.append(-(pf_f * torch.log(pf_f.clamp(min=eps))).sum().item())
                        maxes.append(pf_f.max().item())
                    for _k, _v in [("p_final_entropy", sum(entropies) / len(entropies)),
                                   ("p_final_max",     sum(maxes)     / len(maxes))]:
                        _c = self._custom_counts.get(_k, 0)
                        self._custom_metrics[_k] = (self._custom_metrics.get(_k, 0.0) * _c + _v) / (_c + 1)
                        self._custom_counts[_k] = _c + 1
                except Exception:
                    pass

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: dict, start_time=None):
        """
        Flush window accumulators into this logging event, then forward to
        the HF parent. Single writer: the default HF WandbCallback adds the
        'train/' prefix itself — these keys never carry a manual prefix.

        Cross-rank: SUM for numerators/counts, MAX for peaks, entered by
        every rank at the same step (HF calls log() on all ranks).
        """
        if not hasattr(self, "_custom_metrics"):
            self._custom_metrics = {}
        if not hasattr(self, "_custom_counts"):
            self._custom_counts = {}
        if not hasattr(self, "_custom_max"):
            self._custom_max = {}
        if not hasattr(self, "_custom_taint"):
            self._custom_taint = {}
        if self._custom_metrics or self._custom_counts or self._custom_max:
            sum_payload, max_payload = _build_window_payloads(
                self._custom_metrics, self._custom_counts,
                self._custom_max, self._custom_taint,
                getattr(self, "probe_layers", None) or [],
            )
            sum_payload, max_payload = _window_allreduce(sum_payload, max_payload)
            emitted = _emit_window_logs(sum_payload, max_payload)
            # ── legacy local running means (unchanged semantics) ──────────
            legacy_omega = self._custom_metrics.get("layer_weights")
            if legacy_omega is not None:
                om = (legacy_omega.tolist() if isinstance(legacy_omega, torch.Tensor)
                      else list(legacy_omega))
                labels = list(getattr(self, "metric_layer_labels", None)
                              or getattr(self, "probe_layers", None) or [])
                if len(labels) != len(om):
                    labels = (labels[: len(om)] if len(labels) >= len(om)
                              else list(range(len(om))))
                for _l, _w in zip(labels, om):
                    logs[f"layer_omega/L{_l}"] = float(_w)
            for _k in ("omega_entropy", "p_final_entropy", "p_final_max"):
                if _k in self._custom_metrics:
                    logs[_k] = float(self._custom_metrics[_k])
            logs.update(emitted)
            self._custom_metrics = {}
            self._custom_counts = {}
            self._custom_max = {}
            self._custom_taint = {}
        # Forward to parent (handles WandB, TensorBoard, etc.)
        if start_time is not None:
            super().log(logs, start_time=start_time)
        else:
            super().log(logs)

    def _save(self, output_dir: str, state_dict=None):
        """Override to exclude _new_token_emb from saved checkpoint.
        Its values are already synced back into embed_tokens.weight by
        SyncNewTokenEmbCallback.on_save, so loading works without it.
        """
        if state_dict is None:
            state_dict = self.model.state_dict()
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("_new_token_emb")}
        super()._save(output_dir, state_dict=state_dict)
