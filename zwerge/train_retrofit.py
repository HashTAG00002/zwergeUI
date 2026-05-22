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
    CHAT_TEMPLATE,
    MODEL_TYPE_CONSTANTS,
)
from zwerge_retrofit.dataset import RetrofitDataset, RetrofitDataCollator
from zwerge_retrofit import get_model_class
from zwerge_retrofit.trainer import (
    RetrofitTrainer,
    EmptyCacheCallback,
    SyncNewTokenEmbCallback,
    WandbRetrofitCallback,
    ValEvalCallback,
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
    model_type: str = field(
        default="uitars",
        metadata={
            "help": (
                "Model type: 'uitars' (Qwen2.5-VL, default), "
                "'guiowl' (GUI-Owl-1.5, Qwen3-VL), "
                "'uivenus' (UI-Venus-1.5, Qwen3-VL)"
            )
        },
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
    grounding_fusion_type: str = field(
        default="cos_meta",
        metadata={"help": "Fusion scorer: 'cos_meta' (q_meta·q_l, default) or 'readiness' (original 5-feature MLP)"},
    )
    grounding_use_shared_mlp: bool = field(
        default=True,
        metadata={"help": "If False, skip shared q/k MLP projectors and use LoRA-adapted states directly for dot product (pure LoRA mode)"},
    )
    grounding_independent_layers: bool = field(
        default=False,
        metadata={"help": (
            "Independent per-layer mode: no fusion scorer, no shared MLP (forced). "
            "Each probe layer is supervised only by its own grounding accuracy. "
            "Loss = mean_l KL(y || p_l). "
            "Eval: p_final = uniform mean of per-layer probs (for display only)."
        )},
    )
    grounding_adapter_type: str = field(
        default="lora",
        metadata={"help": (
            "Probe adapter type: 'lora' (default, rank-16 LoRA + dot-product) | "
            "'attn' (cross-attention probe, full-rank W_q/W_k, multi-head scoring, ~0.52% of 8B). "
            "When 'attn', shared MLP is automatically disabled."
        )},
    )
    grounding_attn_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in cross-attention probe (adapter_type='attn', default 8)"},
    )
    grounding_attn_head_dim: int = field(
        default=64,
        metadata={"help": "Per-head dimension in cross-attention probe (adapter_type='attn', default 64 → d_attn=512)"},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": (
            "Training data path. Supported formats:\n"
            "  Single file:       /path/a.json\n"
            "  Comma-separated:   /path/a.json,/path/b.json\n"
            "  Newline-separated: $'path/a.json\\npath/b.json' (shell multiline var)\n"
            "  Brace expansion:   /path/{a,b,c}.json (same dir/suffix)\n"
            "  YAML config:       /path/config.yaml\n"
            "Multiple files are merged then globally shuffled."
        )},
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
        default=12_845_056,
        # uitars (Qwen2.5-VL, patch_size=14): 16384 × 14² × 2² = 16384 × 784 = 12,845,056
        # guiowl/uivenus (Qwen3-VL, patch_size=16): 12544 × 16² × 2² = 12544 × 1024 = 12,845,056
        #   OR use 16,777,216 (= 16384 × 1024) to keep same token budget of 16384 for Qwen3-VL.
        # Pass via --max_pixels in training scripts (see train_ablation_A3_gaussian_cos_meta.sh).
        metadata={"help": "Maximum number of pixels for image resizing. "
                          "uitars: 12845056 (16384 tokens @ 14×14×4). "
                          "guiowl/uivenus: 16777216 (16384 tokens @ 16×16×4)."},
    )
    max_conv_turns: Optional[int] = field(
        default=10,
        metadata={"help": "Maximum conversation turns to use"},
    )
    gt_label_type: str = field(
        default="binary",
        metadata={"help": "GT label type: 'binary' (bbox overlap) or 'gaussian' (anisotropic Gaussian centered at bbox center)"},
    )
    gaussian_sigma_factor: float = field(
        default=0.5,
        metadata={"help": "For gt_label_type=gaussian: σ_x = bbox_width * factor, σ_y = bbox_height * factor (default 0.5 → σ = half bbox size)"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=18432,
        # uitars (Qwen2.5-VL): 12845056 / (14*14*4) = 16384 tokens + ~2048 text budget
        # guiowl (Qwen3-VL): 16777216 / (16*16*4) = 16384 tokens + ~2048 text budget
        # Both result in ~16384 visual tokens → same model_max_length.
        metadata={"help": "Maximum sequence length (visual_tokens + text_budget)"},
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

    # ── In-training evaluation ──
    val_steps: int = field(
        default=-1,
        metadata={"help": "Run distributed vis+eval every N steps (-1 = disabled)"},
    )
    val_bench: str = field(
        default="all",
        metadata={"help": "Benches for in-training eval: 'all' or one of ss_pro/ss_v2/osworld_g/mmbench/ui_vision"},
    )
    val_n_samples: int = field(
        default=-1,
        metadata={"help": "Samples per bench for eval (-1 = all data)"},
    )
    val_eval_dir: str = field(
        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation",
        metadata={"help": "Root directory of eval datasets"},
    )
    val_output_dir: str = field(
        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results/zwerge_layerwise",
        metadata={"help": "Root output dir — {root}/{decode_strategy}/{run_name}/checkpoint-{step}/"},
    )
    val_decode_strategy: str = field(
        default="centroid",
        metadata={"help": "Decode strategy for val eval"},
    )
    val_max_pixels: int = field(
        default=12_845_056,
        # Default matches uitars (Qwen2.5-VL, patch_size=14).
        # For guiowl/uivenus (Qwen3-VL, patch_size=16) override with --val_max_pixels 16777216.
        # Training scripts (train_ablation_A3_gaussian_cos_meta.sh) pass --val_max_pixels ${MAX_PIXELS}
        # which is already set correctly per MODEL_TYPE.
        metadata={"help": "max_pixels for val eval (must match training max_pixels). "
                          "uitars: 12845056, guiowl/uivenus: 16777216."},
    )
    val_cell_w: int = field(default=300, metadata={"help": "Vis PNG cell width"})
    val_cell_h: int = field(default=220, metadata={"help": "Vis PNG cell height"})
    val_alpha:  float = field(default=0.55, metadata={"help": "Vis heatmap alpha"})


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
    model_config.grounding_fusion_type        = model_args.grounding_fusion_type
    model_config.grounding_use_shared_mlp     = model_args.grounding_use_shared_mlp
    model_config.grounding_independent_layers = model_args.grounding_independent_layers
    model_config.grounding_adapter_type       = model_args.grounding_adapter_type
    model_config.grounding_attn_heads         = model_args.grounding_attn_heads
    model_config.grounding_attn_head_dim      = model_args.grounding_attn_head_dim

    # Special token IDs (needed for inference)
    # convert_tokens_to_ids is safer than encode()[0] — avoids BOS/extra-token prepending
    model_config.ground_token_id        = tokenizer.convert_tokens_to_ids(DEFAULT_GROUND_TOKEN)
    model_config.pointer_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_START_TOKEN)
    model_config.pointer_end_token_id   = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_END_TOKEN)
    model_config.pointer_pad_token_id   = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)

    rank0_print(f"Probe layers: {probe_layers}")
    rank0_print(f"Ground token ID: {model_config.ground_token_id}")
    rank0_print(f"Pointer pad token ID: {model_config.pointer_pad_token_id}")


def setup_trainable_params(
    model,   # any RetrofitModelMixin subclass
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

    # NOTE: embed_tokens is kept frozen here; the 4 new-token rows are handled via a
    # separate _new_token_emb Parameter + forward hook (registered after this function).

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

    # ── Qwen3-VL retrofit: DDP / gradient-checkpointing hardening ──────────────
    # Why the slowness:
    #   use_reentrant=False (saved_tensors_hooks) intercepts EVERY tensor op in all
    #   36 backbone layers on FORWARD — huge overhead even with no backward.
    # Why we can use use_reentrant=True safely:
    #   - use_reentrant=True only saves layer INPUT tensors (near-zero overhead)
    #   - probe layer hooks do .detach() → backward stops there → GC recompute
    #     never actually runs (no gradient reaches backbone)
    #   - Memory: GC saves layer inputs (~8 GB) not all activations (~50 GB)
    #   - _new_token_emb stays in graph (consistent with UI-TARS: it CAN be trained
    #     if probe detach is ever removed; currently has no gradient path anyway)
    # Correct fix: keep gc=True + use_reentrant=True (default) + find_unused=True.
    if model_args.model_type in ("guiowl", "uivenus"):
        # Keep gradient_checkpointing=True — needed for memory with 8B model.
        # Do NOT set use_reentrant=False (that caused 600s/step overhead).
        # use_reentrant=True (default) is free when backward doesn't run through backbone.
        if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is not None:
            # If someone explicitly passed use_reentrant=False, override it.
            if training_args.gradient_checkpointing_kwargs.get("use_reentrant") is False:
                training_args.gradient_checkpointing_kwargs["use_reentrant"] = True
                rank0_print("[INFO] Overriding use_reentrant=False → True for Qwen3-VL retrofit.")
        if training_args.ddp_find_unused_parameters is None:
            training_args.ddp_find_unused_parameters = True
            rank0_print(
                "[INFO] ddp_find_unused_parameters=True for Qwen3-VL retrofit "
                "(explicit; Trainer default find_unused = not gc = False → broken)."
            )
    # ─────────────────────────────────────────────────────────────────────────────

    if training_args.verbose_logging:
        rank0_print(f"model_args = {vars(model_args)}")
        rank0_print(f"data_args = {vars(data_args)}")
        rank0_print(f"training_args = {training_args.to_dict()}")

    # ── Load model ──────────────────────────────────────────────────────────
    rank0_print(f"Loading model from {model_args.model_name_or_path}...")
    rank0_print(f"model_type = {model_args.model_type}")

    # Resolve model class and model-specific constants
    ModelClass = get_model_class(model_args.model_type)
    model_constants = MODEL_TYPE_CONSTANTS[model_args.model_type]

    # Parse probe layers from string to list
    probe_layers_list = [int(x.strip()) for x in model_args.probe_layers.split(",")]

    # Build a temporary config to pass grounding head params at init time
    base_config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    base_config.probe_layers = probe_layers_list
    base_config.grounding_proj_dim = model_args.grounding_proj_dim
    base_config.grounding_adapter_rank = model_args.grounding_adapter_rank
    base_config.grounding_lambda_layer = model_args.grounding_lambda_layer
    base_config.grounding_fusion_type        = model_args.grounding_fusion_type
    base_config.grounding_use_shared_mlp     = model_args.grounding_use_shared_mlp
    base_config.grounding_independent_layers = model_args.grounding_independent_layers
    base_config.grounding_adapter_type       = model_args.grounding_adapter_type
    base_config.grounding_attn_heads         = model_args.grounding_attn_heads
    base_config.grounding_attn_head_dim      = model_args.grounding_attn_head_dim

    attn_impl = "flash_attention_2" if model_args.flash_attn_2_enabled else "sdpa"
    rank0_print(f"Using attn_implementation={attn_impl}")
    model = ModelClass.from_pretrained(
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

    # Save model_type into config for auto-detection at inference time
    model.config.model_type_retrofit = model_args.model_type

    # CRITICAL: register special token IDs for _find_ground_anchor P0-P3.
    # reinit_grounding_head=True only when starting fresh from a base model;
    # False when resuming from a retrofit checkpoint (trained weights must not be reset).
    _is_resuming = len(list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))) > 0
    _vision_end_id = getattr(model.config, "vision_end_token_id", None)
    model.setup_special_token_ids(
        ground_token_id=model.config.ground_token_id,
        pointer_start_token_id=model.config.pointer_start_token_id,
        vision_end_token_id=_vision_end_id,
        reinit_grounding_head=not _is_resuming,
    )
    rank0_print(f"setup_special_token_ids: reinit_head={'yes' if not _is_resuming else 'NO (resuming)'}")
    rank0_print(
        f"setup_special_token_ids: ground={model.config.ground_token_id}, "
        f"pointer_start={model.config.pointer_start_token_id}, "
        f"vision_end={_vision_end_id}"
    )

    # ── Inject model-specific constants into data_args for RetrofitDataset ──
    data_args.system_message       = model_constants["system_message"]
    data_args.ground_response      = model_constants["ground_response"]
    data_args.user_prompt_template = model_constants.get("user_prompt_template")

    # ── Processor ────────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
    )
    processor.tokenizer = tokenizer
    # 对 Qwen2.5-VL (uitars)，强制使用项目内定义的 CHAT_TEMPLATE，
    # 确保 <|ground|> 等新 token 在 apply_chat_template 时被正确处理。
    # 对 Qwen3-VL (guiowl/uivenus)，保留模型自带的 chat_template（更完整，含 Qwen3 特殊标签）。
    if model_args.model_type == "uitars":
        processor.tokenizer.chat_template = CHAT_TEMPLATE
    data_args.processor = processor

    # ── Freeze / unfreeze params ─────────────────────────────────────────────
    setup_trainable_params(model, training_args)

    # ── New-token embeddings: separate nn.Parameter (avoids 543M Adam state) ──
    # Instead of gradient-hook on the full embed_tokens.weight (which forces Adam to
    # allocate momentum buffers for all 543M params even though only 4 rows are trained),
    # we create a small standalone Parameter [n_new, d_model] and inject its values
    # back into the embedding output via a forward hook.
    if training_args.unfreeze_new_tokens and not training_args.unfreeze_all_parameters:
        new_token_ids = [
            tokenizer.convert_tokens_to_ids(t)
            for t in ADDITIONAL_SPECIAL_TOKENS
            if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id
        ]
        if not new_token_ids:
            rank0_print("[WARN] No new token IDs found; skipping new-token embedding setup")
        else:
            # Use get_input_embeddings() — works for both Qwen2.5-VL (model.model.embed_tokens)
            # and Qwen3-VL (model.model.language_model.embed_tokens)
            embed_module = model.get_input_embeddings()
            embed_weight  = embed_module.weight
            # embed_tokens stays FROZEN — no requires_grad, no Adam state for 543M rows
            embed_weight.requires_grad_(False)
            # Standalone trainable parameter for just the new token rows
            with torch.no_grad():
                init_rows = embed_weight.data[new_token_ids].clone()
            model._new_token_emb = torch.nn.Parameter(init_rows)  # [n_new, d_model]
            model._new_token_id_to_row = {tid: i for i, tid in enumerate(new_token_ids)}
            rank0_print(
                f"[NewTokenEmb] Created _new_token_emb {list(init_rows.shape)} "
                f"for token IDs {new_token_ids} — Adam state: {init_rows.numel()} params"
            )

            # Forward hook: substitute embedding outputs for new token positions
            _id_to_row = model._new_token_id_to_row
            def _patch_new_token_outputs(module, inputs, output):
                token_ids = inputs[0]
                hits = [(tid, ri) for tid, ri in _id_to_row.items()
                        if (token_ids == tid).any()]
                if not hits:
                    return output
                out = output.clone()
                for tid, ri in hits:
                    mask = (token_ids == tid)
                    out[mask] = model._new_token_emb[ri].to(out.dtype)
                return out
            embed_module.register_forward_hook(_patch_new_token_outputs)

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
        SyncNewTokenEmbCallback(),  # write _new_token_emb back to embed_tokens before each save
    ]
    # WandB callback (log retrofit-specific metrics)
    if "wandb" in training_args.report_to:
        callbacks.append(WandbRetrofitCallback(probe_layers=probe_layers_list))
    # In-training eval callback
    if training_args.val_steps > 0:
        callbacks.append(ValEvalCallback(
            training_args=training_args,
            processor=processor,
            probe_layers=probe_layers_list,
            model_type=model_args.model_type,
            system_message=model_constants["system_message"],
            ground_response=model_constants["ground_response"],
            user_prompt_template=model_constants.get("user_prompt_template"),
        ))

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
