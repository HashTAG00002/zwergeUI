"""
ZwerGe-UI Retrofit Training Script
====================================
Trains a lightweight layer-wise coordinate-free grounding head on top of
frozen UI-TARS-1.5-7B (or any Qwen2.5-VL based native GUI agent).

Core recipe (from chatgpt-export.txt Phase 4):
  - Backbone: completely frozen
  - Trainable: LayerWiseGroundingHead + new token embeddings only
  - Loss: KL divergence on patch posterior (per-layer + fused)

WandB:
  export WANDB_API_KEY=05140d124018012288eaf1d7166bef50eb16eb3b
  export WANDB_PROJECT=Look-Ahead-Agent

Usage (single-node, multi-GPU):
  torchrun --nproc_per_node=8 train_retrofit.py \
    --model_name_or_path /path/to/UI-TARS-1.5-7B \
    --data_path /path/to/data.json \
    --output_dir /path/to/output \
    --probe_layers 14,18,21,24,26,27 \
    --grounding_proj_dim 512 \
    --grounding_adapter_rank 16 \
    --grounding_lambda_layer 0.5 \
    --grounding_loss_weight 1.0 \
    --lm_loss_weight 0.0 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --learning_rate_new_tokens 2e-4 \
    --bf16 True \
    --gradient_checkpointing True \
    --report_to wandb

Data format (each .json file is a list of items):
  OS-Atlas raw format (recommended):
    [{"img_filename": "xxx.png", "elements": [{"instruction": "...", "bbox": [...]}]}]

  GUI-Actor/AIMA conversations format:
    [{"image": "xxx.png", "conversations": [...], "bbox": [...]}]

  ms-swift flat format:
    [{"image": "xxx.png", "query": "...", "response": "...", "bbox": [...]}]

See zwerge/src/zwerge_retrofit/dataset.py for details.
"""

import pathlib
import sys
import os

# Add parent directory to path so we can import zwerge_retrofit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import transformers
from PIL import ImageFile
from transformers import AutoProcessor

from zwerge_retrofit.constants import (
    ADDITIONAL_SPECIAL_TOKENS,
    DEFAULT_GROUND_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_PROBE_LAYERS,
    IGNORE_INDEX,
)
from zwerge_retrofit.dataset import RetrofitDataset, RetrofitDataCollator
from zwerge_retrofit.modeling_uitars import UITARSRetrofitModel
from zwerge_retrofit.trainer import (
    RetrofitTrainer,
    EmptyCacheCallback,
    WandbRetrofitCallback,
    rank0_print,
    safe_save_model_for_hf_trainer,
)

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")

local_rank = None


# ─────────────────────────────────────────────────────────────────────────────
# Argument classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model (UI-TARS-1.5-7B or any Qwen2.5-VL agent)"},
    )
    flash_attn_2_enabled: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2 (recommended for training)"},
    )
    # ── Grounding head architecture ──
    probe_layers: str = field(
        default="14,18,21,24,26,27",
        metadata={"help": "Comma-separated list of transformer layer indices to probe (0-indexed)"},
    )
    grounding_proj_dim: int = field(
        default=512,
        metadata={"help": "Projection dimension for query/key in grounding head"},
    )
    grounding_adapter_rank: int = field(
        default=16,
        metadata={"help": "LoRA rank for per-layer adapter in grounding head"},
    )
    grounding_lambda_layer: float = field(
        default=0.5,
        metadata={"help": "Weight for per-layer loss vs fused loss: total = loss_fuse + lambda * loss_layer"},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to training data (.json, .jsonl, .yaml, or {file1,file2}.json pattern)"},
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Root folder for images (prepended to relative image paths)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,  # 2*2*28*28 = 56*56
        metadata={"help": "Minimum number of pixels for image resizing"},
    )
    max_pixels: Optional[int] = field(
        default=5720064,  # ~3192*1792
        metadata={"help": "Maximum number of pixels for image resizing"},
    )
    max_conv_turns: Optional[int] = field(
        default=10,
        metadata={"help": "Maximum conversation turns to use"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length"},
    )
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)

    # ── Freeze/unfreeze controls ──
    unfreeze_all_parameters: bool = field(
        default=False,
        metadata={"help": "Unfreeze ALL model parameters (full fine-tuning, not recommended)"},
    )
    unfreeze_grounding_head: bool = field(
        default=True,
        metadata={"help": "Unfreeze layerwise_grounding_head parameters"},
    )
    unfreeze_new_tokens: bool = field(
        default=True,
        metadata={"help": "Unfreeze embedding for newly added special tokens (<GROUND>, etc.)"},
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={"help": "Unfreeze lm_head (not needed when lm_loss_weight=0)"},
    )
    unfreeze_last_n_layers: int = field(
        default=-1,
        metadata={"help": "Additionally unfreeze last N transformer layers (-1 = none)"},
    )
    unfreeze_visual_encoder: bool = field(
        default=False,
        metadata={"help": "Unfreeze visual encoder (not recommended for retrofit)"},
    )

    # ── Loss weights ──
    grounding_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for grounding (KL) loss"},
    )
    lm_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for LM (next-token) loss. Set to 0 to disable (frozen backbone)"},
    )

    # ── Training efficiency ──
    empty_cache_every_n_steps: int = field(
        default=20,
        metadata={"help": "Clear CUDA cache every N steps"},
    )
    learning_rate_new_tokens: float = field(
        default=2e-4,
        metadata={"help": "Learning rate for newly added token embeddings"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Setup functions
# ─────────────────────────────────────────────────────────────────────────────

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Add special tokens and resize embeddings.
    New tokens get mean embedding initialization (from GUI-Actor/GUI-AIMA recipe).
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    new_vocab_size = len(tokenizer)
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = new_vocab_size
    else:
        model.config.vocab_size = new_vocab_size
    model.vocab_size = new_vocab_size

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # Initialize new tokens with mean of existing embeddings
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    rank0_print(f"Added {num_new_tokens} new tokens. Vocab size: {new_vocab_size}")


def update_model_config_for_retrofit(
    model_config: transformers.PretrainedConfig,
    tokenizer: transformers.PreTrainedTokenizer,
    model_args: ModelArguments,
):
    """Store retrofit-specific config into model config for re-loading."""
    # Grounding head config
    probe_layers = [int(x.strip()) for x in model_args.probe_layers.split(",")]
    model_config.probe_layers = probe_layers
    model_config.grounding_proj_dim = model_args.grounding_proj_dim
    model_config.grounding_adapter_rank = model_args.grounding_adapter_rank
    model_config.grounding_lambda_layer = model_args.grounding_lambda_layer

    # Special token IDs (needed for inference)
    model_config.ground_token_id = tokenizer.encode(DEFAULT_GROUND_TOKEN)[0]
    model_config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
    model_config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
    model_config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]

    rank0_print(f"Probe layers: {probe_layers}")
    rank0_print(f"Ground token ID: {model_config.ground_token_id}")
    rank0_print(f"Pointer pad token ID: {model_config.pointer_pad_token_id}")


def setup_trainable_params(
    model: UITARSRetrofitModel,
    training_args: TrainingArguments,
):
    """
    Freeze backbone; only unfreeze grounding head and new token embeddings.

    This is the key design choice: backbone frozen, only retrofit head is trained.
    Compare with GUI-Actor which unfreezes its pointer head only.
    """
    if training_args.unfreeze_all_parameters:
        rank0_print("Unfreezing ALL parameters (full fine-tuning mode)")
        for p in model.parameters():
            p.requires_grad = True
        return

    # Default: freeze everything
    rank0_print("Freezing all backbone parameters...")
    for p in model.parameters():
        p.requires_grad = False

    if training_args.unfreeze_grounding_head:
        rank0_print("Unfreezing layerwise_grounding_head...")
        for p in model.layerwise_grounding_head.parameters():
            p.requires_grad = True

    if training_args.unfreeze_lm_head:
        rank0_print("Unfreezing lm_head...")
        for p in model.lm_head.parameters():
            p.requires_grad = True

    if training_args.unfreeze_last_n_layers > 0:
        n = training_args.unfreeze_last_n_layers
        rank0_print(f"Unfreezing last {n} transformer layers...")
        for p in model.model.layers[-n:].parameters():
            p.requires_grad = True

    if training_args.unfreeze_new_tokens:
        rank0_print("Unfreezing embed_tokens (new tokens only, via gradient hook)...")
        model.model.embed_tokens.weight.requires_grad = True

    if training_args.unfreeze_visual_encoder:
        rank0_print("Unfreezing visual encoder...")
        for p in model.visual.parameters():
            p.requires_grad = True

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )


def dump_args_to_json(model_config, processor, model_args, data_args, training_args, output_dir):
    """Save all arguments to JSON for reproducibility."""
    args_dict = {
        "model_args": {k: v for k, v in vars(model_args).items()},
        "data_args": {k: v for k, v in vars(data_args).items() if k != "processor"},
        "training_args": {k: v for k, v in training_args.to_dict().items()},
        "probe_layers": model_config.probe_layers,
        "vocab_size": model_config.vocab_size if hasattr(model_config, "vocab_size") else None,
    }
    out_path = os.path.join(output_dir, "args.json")
    with open(out_path, "w") as f:
        json.dump(args_dict, f, indent=2, default=str)
    rank0_print(f"Args saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if training_args.verbose_logging:
        rank0_print(f"model_args = {vars(model_args)}")
        rank0_print(f"data_args = {vars(data_args)}")
        rank0_print(f"training_args = {training_args.to_dict()}")

    # ── Load model ──────────────────────────────────────────────────────────
    rank0_print(f"Loading model from {model_args.model_name_or_path}...")

    # Parse probe layers from string to list
    probe_layers_list = [int(x.strip()) for x in model_args.probe_layers.split(",")]

    # Build a temporary config to pass grounding head params at init time
    # We load the base config first, then add our retrofit params
    base_config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    base_config.probe_layers = probe_layers_list
    base_config.grounding_proj_dim = model_args.grounding_proj_dim
    base_config.grounding_adapter_rank = model_args.grounding_adapter_rank
    base_config.grounding_lambda_layer = model_args.grounding_lambda_layer

    # flash_attention_2 要求 gcc >= 7（stdatomic.h），codelab 环境可能缺失
    # 关闭时退回到 sdpa（PyTorch 原生实现，无需 triton 编译，A100 也有加速）
    attn_impl = "flash_attention_2" if model_args.flash_attn_2_enabled else "sdpa"
    rank0_print(f"Using attn_implementation={attn_impl}")
    model = UITARSRetrofitModel.from_pretrained(
        model_args.model_name_or_path,
        config=base_config,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False
    model.reset_loss_weights(
        grounding_loss_weight=training_args.grounding_loss_weight,
        lm_loss_weight=training_args.lm_loss_weight,
    )

    # Gradient checkpointing hook
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # ── Tokenizer & special tokens ──────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS},
        tokenizer=tokenizer,
        model=model,
    )
    update_model_config_for_retrofit(model.config, tokenizer, model_args)

    # CRITICAL: register special token IDs for _find_ground_anchor P1/P2/P3
    # Must be called AFTER add_special_tokens so the token IDs are correct.
    _vision_end_id = getattr(model.config, "vision_end_token_id", None)
    model.setup_special_token_ids(
        ground_token_id=model.config.ground_token_id,
        pointer_start_token_id=model.config.pointer_start_token_id,
        vision_end_token_id=_vision_end_id,
    )
    rank0_print(
        f"setup_special_token_ids: ground={model.config.ground_token_id}, "
        f"pointer_start={model.config.pointer_start_token_id}, "
        f"vision_end={_vision_end_id}"
    )

    # ── Processor ────────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
    )
    processor.tokenizer = tokenizer
    data_args.processor = processor

    # ── Freeze / unfreeze params ─────────────────────────────────────────────
    setup_trainable_params(model, training_args)

    # ── Gradient hook for new token embeddings ───────────────────────────────
    if training_args.unfreeze_new_tokens and not training_args.unfreeze_all_parameters:
        emb_param = None
        for n, p in model.named_parameters():
            if n.endswith("model.embed_tokens.weight"):
                emb_param = p
                break
        if emb_param is None:
            raise ValueError("embed_tokens.weight not found in model")

        n_new_tokens = len(ADDITIONAL_SPECIAL_TOKENS)
        def mask_grad(grad):
            """Only allow gradients for the newly added tokens."""
            grad = grad.clone()
            grad[:-n_new_tokens] = 0.0
            return grad
        emb_param.register_hook(mask_grad)
        rank0_print(f"Registered gradient hook: only update last {n_new_tokens} token embeddings")

    # ── Output directory ─────────────────────────────────────────────────────
    os.makedirs(training_args.output_dir, exist_ok=True)

    if training_args.local_rank in (0, -1):
        dump_args_to_json(model.config, processor, model_args, data_args, training_args, training_args.output_dir)

    # ── Dataset & collator ───────────────────────────────────────────────────
    rank0_print(f"Loading dataset from {data_args.data_path}...")
    train_dataset = RetrofitDataset(
        tokenizer=tokenizer,
        processor=processor,
        data_path=data_args.data_path,
        data_args=data_args,
    )
    data_collator = RetrofitDataCollator(tokenizer=tokenizer)

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [
        EmptyCacheCallback(every_n_steps=training_args.empty_cache_every_n_steps),
    ]
    # WandB callback (log retrofit-specific metrics)
    if "wandb" in training_args.report_to:
        callbacks.append(WandbRetrofitCallback(probe_layers=probe_layers_list))

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = RetrofitTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=callbacks,
        probe_layers=probe_layers_list,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    rank0_print(f"Training complete. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
