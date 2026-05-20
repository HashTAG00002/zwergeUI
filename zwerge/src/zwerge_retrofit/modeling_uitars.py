"""
ZwerGe-UI Retrofit: UITARSRetrofitModel
=========================================
将 UI-TARS-1.5-7B（或任何 Qwen2.5-VL 系坐标生成型 GUI agent）
retrofit 为具有 layer-wise coordinate-free grounding 能力的模型。

核心设计（来自 chatgpt-export.txt Phase 4 / §4.2）：

  问题：UI-TARS 等坐标生成模型没有 <ACTOR>/<anchor> 这类 token，
       无法直接用 GUI-Actor 的方式取 query hidden state。

  主方案：Pre-coordinate action-prefix token
    构造：click(start_box='<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>')
    对应 UI-TARS-1.5 真实固化输出格式（issue #183/#138 确认），坐标部分替换为 pointer tokens
    <|ground|> 的 hidden state 特性：
      - 已见过 image + instruction + action type (click(start_box='
      - 还没见到任何坐标数字 → 无 label leakage
      - 天然与 UI-TARS action interface 对齐
      - 类比 GUI-Actor 的 <ACTOR> token，但更贴近坐标生成模型范式

  动态 anchor 查找 (_find_ground_anchor)，优先级从高到低：
    P1. 序列中的 <|ground|> token（主方案，最干净，无 label leakage）
    P2. <|pointer_start|> 之前的 token（pre-coordinate action-prefix token）
    P3. <|vision_end|> 之后第一个 token（post-visual-context token）
    P4. 外部 hint（dataset 预计算的位置）
    P5. 最后一个 non-padding token（带 UserWarning，可能有 label leakage 风险）

  对 GUI-Actor/GUI-AIMA 格式（已有 pointer tokens）：
    click(<|pointer_start|><|ground|><|pointer_pad|><|pointer_end|>)
    <|ground|> 在 pointer_start 之后，同样是 pre-coordinate position

Architecture:
  backbone (frozen) → all_hidden_states[0..L]
  LayerWiseGroundingHead:
    for l in probe_layers:
      h_q = h[l][ground_anchor_pos]        # pre-coordinate action-prefix hidden state
      h_v = h[l][visual_token_positions]   # visual token hidden states
      h_q_adapted = LayerLoRAAdapter_q(h_q)
      h_v_adapted = LayerLoRAAdapter_v(h_v)
      q = shared_q_proj(LN(h_q_adapted))
      k = shared_k_proj(LN(h_v_adapted))
      logits_l = q @ k^T / sqrt(d_proj)
      p_l = softmax(logits_l)   # patch posterior at layer l
      feat_l = readiness_features(p_l)   # [entropy, margin, top3mass, top5mass, active_area]
    omega = softmax(fusion_mlp(concat(feat_l, layer_emb_l)))   # learned layer weights
    p_final = sum_l(omega_l * p_l)   # fused posterior

Loss:
  L_fuse  = KL(y || p_final)
  L_layer = mean_l KL(y || p_l)
  L_total = L_fuse + lambda_layer * L_layer

  where y is the patch soft label (Gaussian for point GT, binary/uniform for bbox GT)

FA2 compatible: output_hidden_states=True, output_attentions=False (no full attn matrix)
"""

import dataclasses
import math
import warnings
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
)

from .constants import (
    IGNORE_INDEX,
    DEFAULT_GROUND_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
)


# =============================================================================
# AnchorStrategy Enum
# =============================================================================

class AnchorStrategy(str, Enum):
    """
    Enum recording how the grounding anchor token was selected.

    Priority ordering (P1 → P5 fallback):
      P1: EXPLICIT_GROUND_TOKEN   — <|ground|> explicitly present in sequence
          Model has seen image + instruction + "click(" but NOT coordinates → no leakage
      P2: BEFORE_POINTER_START    — token immediately before <|pointer_start|>
          Pre-coordinate action-prefix position; safe from leakage
      P3: AFTER_VISION_END        — first token after <|vision_end|>
          Has seen full image; may miss instruction tail → weaker but safe
      P4: EXTERNAL_HINT           — position pre-computed by RetrofitDataset
          Used when P1-P3 not found in sequence but dataset has pre-computed index
      P5: LAST_NON_PAD            — last non-padding token (WARNING: label leakage risk)
          For UI-TARS native format, last token may follow coordinates → leakage risk
    """
    EXPLICIT_GROUND_TOKEN = "P1:explicit_ground_token"
    BEFORE_POINTER_START  = "P2:before_pointer_start"
    AFTER_VISION_END      = "P3:after_vision_end"
    EXTERNAL_HINT         = "P4:external_hint"
    LAST_NON_PAD          = "P5:last_non_pad_WARNING"


# =============================================================================
# Output dataclass
# =============================================================================

@dataclasses.dataclass
class RetrofitOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Extended output class carrying retrofit-specific losses and scores.

    NOTE: Must be a @dataclass (not __init__ override) so that ModelOutput's
    __init_subclass__ machinery works correctly and doesn't confuse field names.
    Using @dataclass lets us add fields safely on top of the parent dataclass fields.
    """
    # NOTE: rename to avoid collision with any parent field names.
    # Parent Qwen2_5_VLCausalLMOutputWithPast fields:
    #   loss, logits, past_key_values, hidden_states, attentions, rope_deltas
    grounding_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    per_layer_losses: Optional[torch.FloatTensor] = None
    grounding_scores: Optional[object] = None   # list of p_final tensors per sample (detached CPU)
    layer_weights: Optional[object] = None      # list of omega_l tensors per sample (detached CPU)
    anchor_positions: Optional[object] = None   # list of (anchor_idx, AnchorStrategy) or None


# =============================================================================
# Lightweight 2-Layer MLP
# =============================================================================

class MLP2(nn.Module):
    """2-layer MLP with GELU activation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Per-layer LoRA Adapter
# =============================================================================

class LayerLoRAAdapter(nn.Module):
    """
    Per-layer lightweight LoRA adapter with numerical stability.

    Formula:
      h_norm = LN(h)                          # normalize input
      a_out  = A(h_norm) * (1 / sqrt(d_model)) # scaled projection to rank space
      delta_h = B(a_out)                       # B initialized to zero → identity at start
      output = h_norm + delta_h               # residual connection

    Key design choices for stability:
      1. A's output is scaled by 1/sqrt(d_model) to prevent explosion on large-norm inputs.
         This is critical because deep-layer hidden states in 7B+ models can have norm > 10000.
      2. B is initialized to zero so adapter starts as identity (no impact on forward pass).
      3. LayerNorm before A ensures input has unit variance regardless of layer depth.

    Why this matters:
      In Qwen2.5-VL-7B, Layer 26/27 hidden states have norms of 8000-14000.
      Without scaling, A(LN(h)) can produce values that overflow float32 when passed
      through the subsequent MLP projector (q_proj / k_proj).
    """

    def __init__(self, d_model: int, rank: int = 16):
        super().__init__()
        self.d_model = d_model
        self.ln = nn.LayerNorm(d_model)
        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.B.weight)  # zero init → identity at start
        # Initialize A with small values for initial stability
        nn.init.xavier_uniform_(self.A.weight, gain=0.1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.ln(h)                                    # [d_model] or [N, d_model]
        a_out = self.A(h_norm) / math.sqrt(self.d_model)       # scaled down
        return h_norm + self.B(a_out)                           # residual


# =============================================================================
# Single-layer Grounding Probe Head
# =============================================================================

class LayerGroundingProbe(nn.Module):
    """
    Query-conditioned dot-product grounding probe for a single layer.

    Architecture (per-layer adapters + shared projectors):
      h_q_adapted = q_adapter(h_q)         # per-layer LoRA on query
      h_v_adapted = k_adapter(h_vis)       # per-layer LoRA on visual tokens
      q = q_proj(LN(h_q_adapted))          # shared q projector  [d_model → d_proj]
      k = k_proj(LN(h_v_adapted))          # shared k projector  [d_model → d_proj]
      logits = k @ q / sqrt(d_proj)        # [N_vis]
      p = softmax(logits)

    q_proj / k_proj are shared across all layers (passed in from LayerWiseGroundingHead).
    q_adapter / k_adapter are per-layer (instantiated here).
    """

    def __init__(self, d_model: int, adapter_rank: int = 16):
        super().__init__()
        self.q_adapter = LayerLoRAAdapter(d_model, rank=adapter_rank)
        self.k_adapter = LayerLoRAAdapter(d_model, rank=adapter_rank)
        self.q_ln = nn.LayerNorm(d_model)
        self.k_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        h_query: torch.Tensor,                  # [d_model]  anchor hidden state
        h_vis: torch.Tensor,                    # [N_vis, d_model]  visual token hidden states
        q_proj: Optional[nn.Module],            # shared q projector (d_model → d_proj); None = no MLP
        k_proj: Optional[nn.Module],            # shared k projector (d_model → d_proj); None = no MLP
        d_eff: int,                             # d_proj when MLP present, d_model otherwise
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (p, logits, q): p and logits [N_vis], q [d_eff]."""
        d_model = h_query.shape[-1]
        target_norm = math.sqrt(d_model)

        # ── Stage 1: RMS-normalize BEFORE adapter (bfloat16 safety) ────────
        # Deep-layer hidden states can have ||h||_2 >> sqrt(d_model), which
        # overflows bfloat16 range (max ≈ 3.4e4) inside the LoRA adapter.
        # Pre-normalize to ||h||_2 ≈ sqrt(d_model) before any computation.
        rms_q_pre = (h_query.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        rms_v_pre = (h_vis.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        h_q_safe = h_query / rms_q_pre     # [d_model], norm ≈ sqrt(d_model)
        h_v_safe = h_vis / rms_v_pre       # [N_vis, d_model]

        # ── Stage 2: per-layer LoRA adaptation (residual) ────────────────────
        h_q_adapted = self.q_adapter(h_q_safe.unsqueeze(0)).squeeze(0)   # [d_model]
        h_v_adapted = self.k_adapter(h_v_safe)                            # [N_vis, d_model]

        # ── Stage 3: LN + RMS-normalize BEFORE projector (double safety) ────
        # LayerNorm does NOT constrain L2 norm. Apply a second RMS scaling
        # so the shared MLP projector always receives controlled-magnitude input.
        h_q_ln = self.q_ln(h_q_adapted)
        h_v_ln = self.k_ln(h_v_adapted)
        rms_scale_q = (h_q_ln.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        rms_scale_v = (h_v_ln.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        if q_proj is not None:
            q = q_proj(h_q_ln / rms_scale_q)   # [d_proj]
            k = k_proj(h_v_ln / rms_scale_v)   # [N_vis, d_proj]
        else:
            # No shared MLP: use LoRA-adapted hidden states directly
            q = h_q_ln / rms_scale_q           # [d_model]
            k = h_v_ln / rms_scale_v           # [N_vis, d_model]

        logits = torch.matmul(k, q) / math.sqrt(d_eff)   # [N_vis]
        p = torch.softmax(logits, dim=-1)
        return p, logits, q


# =============================================================================
# Readiness Features
# =============================================================================

def compute_readiness_features(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute scalar readiness features from patch posterior p: [N_vis]
    Returns feature vector of shape [5].

    Features:
      [0] entropy:     H(p) = -sum p*log(p)  — lower = more peaked = more ready
      [1] margin:      top1 - top2 prob      — higher = clearer winner
      [2] top3_mass:   sum of top-3 probs    — higher = more focused
      [3] top5_mass:   sum of top-5 probs
      [4] active_area: fraction of patches with p > 0.1 * max(p)
    """
    entropy = -(p * torch.log(p.clamp(min=eps))).sum()

    k = min(5, p.shape[0])
    topk_vals, _ = torch.topk(p, k=k)
    margin     = topk_vals[0] - topk_vals[1] if k > 1 else topk_vals[0]
    top3_mass  = topk_vals[:3].sum() if k >= 3 else topk_vals.sum()
    top5_mass  = topk_vals.sum()
    active_area = (p > p.max() * 0.1).float().mean()

    return torch.stack([entropy, margin, top3_mass, top5_mass, active_area])


# =============================================================================
# Layer Fusion Scorer (Readiness Selector)
# =============================================================================

class LayerFusionScorer(nn.Module):
    """
    Learned layer fusion scorer. Two modes, selected at construction time:

    cos_meta (default, recommended):
      score_l = alpha_l + cos(q_meta, q_l)
      q_meta  [d_proj]  — single global trainable query (D+L new params)
      alpha   [L]       — per-layer learnable prior (init 0 → uniform start)
      q_l is the per-layer anchor query vector already computed by the probe.

    readiness (original, kept for backward compat):
      s_l = concat(readiness_features(p_l), layer_emb_l)
      score_l = fusion_mlp(s_l)
    """

    def __init__(
        self,
        num_layers: int,
        feature_dim: int = 5,
        layer_emb_dim: int = 8,
        fusion_type: str = "cos_meta",
        d_proj: int = 512,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "cos_meta":
            self.q_meta = nn.Parameter(torch.empty(d_proj))
            self.alpha  = nn.Parameter(torch.zeros(num_layers))
            nn.init.normal_(self.q_meta, std=0.01)
        else:
            self.layer_embeddings = nn.Embedding(num_layers, layer_emb_dim)
            in_dim = feature_dim + layer_emb_dim
            self.scorer = MLP2(in_dim, in_dim * 2, 1)

    def forward(
        self,
        readiness_features: List[torch.Tensor],   # list of [5] per probe layer
        probe_positions: List[int],               # 0-indexed in probe set
        per_layer_queries: Optional[List[torch.Tensor]] = None,  # list of [d_proj]
    ) -> torch.Tensor:
        """Returns omega: [num_probes] softmax weights."""
        if self.fusion_type == "cos_meta":
            param_dtype = self.q_meta.dtype
            q_norm = F.normalize(self.q_meta.to(param_dtype), dim=-1)
            scores = []
            for i, q_l in enumerate(per_layer_queries):
                ql_norm = F.normalize(q_l.to(param_dtype), dim=-1)
                cos_sim = (q_norm * ql_norm).sum()
                scores.append(self.alpha[i] + cos_sim)
            return torch.softmax(torch.stack(scores), dim=-1)
        else:
            device = readiness_features[0].device
            param_dtype = next(self.scorer.parameters()).dtype
            scores = []
            for feat, pos in zip(readiness_features, probe_positions):
                l_emb = self.layer_embeddings(torch.tensor(pos, device=device))
                combined = torch.cat([feat.to(param_dtype), l_emb], dim=-1)
                scores.append(self.scorer(combined))
            return torch.softmax(torch.cat(scores, dim=0), dim=-1)


# =============================================================================
# Full Layer-Wise Grounding Head
# =============================================================================

class LayerWiseGroundingHead(nn.Module):
    """
    Complete layer-wise coordinate-free grounding head.

    Architecture:
      Shared:    q_proj (MLP d_model → d_proj), k_proj (MLP d_model → d_proj)
      Per-layer: LayerGroundingProbe (q_adapter + k_adapter + q_ln + k_ln)
      Fusion:    LayerFusionScorer (layer_embeddings + fusion_mlp)

    Forward:
      For each probe layer l:
        (p_l, logits_l) = probe_l(h_query_l, h_vis_l)
        feat_l = readiness_features(p_l.detach())
      omega = fusion_scorer(feats)
      p_final = sum_l(omega_l * p_l)

    Loss (when labels provided):
      L_fuse = KL(y || p_final)
      L_layer = mean_l KL(y || p_l)
      L_total = L_fuse + lambda_layer * L_layer
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int,
        probe_layers: List[int],
        adapter_rank: int = 16,
        layer_emb_dim: int = 8,
        lambda_layer: float = 0.5,
        fusion_type: str = "cos_meta",
        use_shared_mlp: bool = True,
    ):
        super().__init__()
        self.probe_layers    = sorted(probe_layers)
        self.num_probes      = len(self.probe_layers)
        self.d_model         = d_model
        self.d_proj          = d_proj
        self.lambda_layer    = lambda_layer
        self.use_shared_mlp  = use_shared_mlp

        # Shared MLP projectors (optional; skip when use_shared_mlp=False)
        if use_shared_mlp:
            self.q_proj = MLP2(d_model, d_proj, d_proj)
            self.k_proj = MLP2(d_model, d_proj, d_proj)
        else:
            self.q_proj = None   # type: ignore[assignment]
            self.k_proj = None   # type: ignore[assignment]

        # Per-layer probes (LoRA adapters are always present)
        self.probes = nn.ModuleList([
            LayerGroundingProbe(d_model, adapter_rank)
            for _ in range(self.num_probes)
        ])

        # Layer fusion scorer — q_meta dimension = d_proj if MLP, else d_model
        d_for_fusion = d_proj if use_shared_mlp else d_model
        self.fusion = LayerFusionScorer(
            num_layers=self.num_probes,
            feature_dim=5,
            layer_emb_dim=layer_emb_dim,
            fusion_type=fusion_type,
            d_proj=d_for_fusion,
        )

    def forward(
        self,
        all_hidden_states: Tuple[torch.Tensor, ...],
        # Tuple of (num_layers+1) tensors, each [seq_len, d_model]
        # Index 0 = embedding output; index l+1 = output of transformer layer l
        ground_token_idx: int,
        visual_indices: torch.Tensor,   # [N_vis]
        labels: Optional[torch.Tensor] = None,   # [N_vis] soft label
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys: p_final, omega, per_layer_probs
        (+ loss_fuse, loss_layer, total_grounding_loss if labels given)
        """
        per_layer_probs   = []
        readiness_feats   = []
        per_layer_queries = []

        for probe_i, layer_idx in enumerate(self.probe_layers):
            hs = all_hidden_states[layer_idx + 1]   # [seq_len, d_model]
            h_query = hs[ground_token_idx]           # [d_model]
            h_vis   = hs[visual_indices]             # [N_vis, d_model]

            p_l, _, q_l = self.probes[probe_i](
                h_query, h_vis,
                self.q_proj, self.k_proj,
                self.d_proj if self.use_shared_mlp else self.d_model,
            )
            per_layer_probs.append(p_l)
            readiness_feats.append(compute_readiness_features(p_l.detach()))
            per_layer_queries.append(q_l)

        omega = self.fusion(
            readiness_feats, list(range(self.num_probes)), per_layer_queries
        )  # [num_probes]
        p_final = sum(omega[i] * per_layer_probs[i] for i in range(self.num_probes))

        result = {
            "p_final": p_final,
            "omega": omega,
            "per_layer_probs": per_layer_probs,
        }

        if labels is not None:
            eps = 1e-8
            labels_f = labels.float()
            label_dist = labels_f / (labels_f.sum() + eps)

            loss_fuse = F.kl_div(
                torch.log(p_final.clamp(min=eps)),
                label_dist,
                reduction="sum",
            )

            loss_layer = torch.zeros((), device=p_final.device)
            for p_l in per_layer_probs:
                loss_layer = loss_layer + F.kl_div(
                    torch.log(p_l.clamp(min=eps)),
                    label_dist,
                    reduction="sum",
                )
            loss_layer = loss_layer / self.num_probes

            result["loss_fuse"] = loss_fuse
            result["loss_layer"] = loss_layer
            result["total_grounding_loss"] = loss_fuse + self.lambda_layer * loss_layer

        return result


# =============================================================================
# Main Model: UITARSRetrofitModel
# =============================================================================

class UITARSRetrofitModel(Qwen2_5_VLForConditionalGeneration):
    """
    UI-TARS-1.5-7B (or any Qwen2.5-VL-based GUI agent) retrofitted with
    a lightweight layer-wise coordinate-free grounding head.

    Key design decisions:
      1. BACKBONE FROZEN: no gradient through backbone params
         (original grounding / action formatting preserved; non-invasive retrofit)
      2. QUERY ANCHOR: <|ground|> at pre-coordinate action-prefix position
         click(start_box='<|ground|><|pointer_start|><|pointer_pad|><|pointer_end|>')
         Has seen image + instruction + action-type but NOT coordinates → no leakage
      3. FA2 COMPATIBLE: output_hidden_states=True, output_attentions=False
         (no full B×L×H×S×S attention matrix stored)
      4. LAYER-WISE PROBING: per-layer probe reads hidden states at probe_layers
         (per-layer q/k adapters + shared q_proj + k_proj)
      5. LEARNED FUSION: readiness scorer learns which layer is most "ready"

    Usage:
      model = UITARSRetrofitModel.from_pretrained(model_path, config=config)
      model.setup_special_token_ids(ground_token_id, pointer_start_token_id)
      model.reset_loss_weights(grounding_loss_weight=1.0, lm_loss_weight=0.0)
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        probe_layers    = getattr(config, "probe_layers",              [14, 18, 21, 24, 26, 27])
        d_proj          = getattr(config, "grounding_proj_dim",         512)
        adapter_rank    = getattr(config, "grounding_adapter_rank",     16)
        lambda_layer    = getattr(config, "grounding_lambda_layer",     0.5)
        fusion_type     = getattr(config, "grounding_fusion_type",     "readiness")
        use_shared_mlp  = getattr(config, "grounding_use_shared_mlp",  True)

        self.layerwise_grounding_head = LayerWiseGroundingHead(
            d_model=config.hidden_size,
            d_proj=d_proj,
            probe_layers=probe_layers,
            adapter_rank=adapter_rank,
            lambda_layer=lambda_layer,
            fusion_type=fusion_type,
            use_shared_mlp=use_shared_mlp,
        )

        # Loss weights
        self.grounding_loss_weight: float = 1.0
        self.lm_loss_weight: float = 0.0   # disabled by default (backbone frozen)

        # Special token IDs for _find_ground_anchor()
        # Must be set via setup_special_token_ids() after add_special_tokens().
        self._ground_token_id: Optional[int] = getattr(config, "ground_token_id", None)
        self._pointer_start_token_id: Optional[int] = getattr(
            config, "pointer_start_token_id", None
        )
        # <|vision_end|> token: auto-detect from config (naming varies by model version)
        _vid = getattr(config, "vision_end_token_id", None)
        if _vid is None:
            _vid = getattr(config, "vision_token_id", None)
        self._vision_end_token_id: Optional[int] = _vid

        # Anchor source stats (for debugging / wandb logging)
        self._anchor_source_counts: Dict[str, int] = {}

        self.post_init()

    # ─────────────────────────────────────────────────────────────────────────
    # Public setup helpers
    # ─────────────────────────────────────────────────────────────────────────

    def reinit_grounding_head(self) -> None:
        """
        Re-initialize all LayerNorm weights/biases in the grounding head to
        canonical values (weight=1, bias=0).  Also re-applies zero-init to
        LoRA B matrices and xavier-uniform to LoRA A matrices.

        MUST be called after from_pretrained() because HuggingFace may leave
        newly-added parameters in uninitialized bfloat16 memory (which can be
        NaN), even though post_init() is invoked. This is a known issue when
        using torch_dtype=bfloat16 with models that have new (non-checkpoint)
        parameters.

        Called automatically at the end of setup_special_token_ids().
        """
        import math as _math
        for module in self.layerwise_grounding_head.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Detect LoRA B matrices (named 'B') by checking parent module
                pass  # handled per-class below
        # Re-init LoRA adapters explicitly
        for probe in self.layerwise_grounding_head.probes:
            for adapter in [probe.q_adapter, probe.k_adapter]:
                nn.init.ones_(adapter.ln.weight)
                nn.init.zeros_(adapter.ln.bias)
                nn.init.xavier_uniform_(adapter.A.weight, gain=0.1)
                nn.init.zeros_(adapter.B.weight)
            nn.init.ones_(probe.q_ln.weight)
            nn.init.zeros_(probe.q_ln.bias)
            nn.init.ones_(probe.k_ln.weight)
            nn.init.zeros_(probe.k_ln.bias)
        # cos_meta fusion params
        fusion = self.layerwise_grounding_head.fusion
        if hasattr(fusion, "q_meta"):
            nn.init.normal_(fusion.q_meta, std=0.01)
        if hasattr(fusion, "alpha"):
            nn.init.zeros_(fusion.alpha)

    def setup_special_token_ids(
        self,
        ground_token_id: int,
        pointer_start_token_id: int,
        vision_end_token_id: Optional[int] = None,
        reinit_grounding_head: bool = True,
    ) -> None:
        """
        Register special token IDs needed by _find_ground_anchor().

        MUST be called after:
          tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
          model.resize_token_embeddings(len(tokenizer))

        Args:
            ground_token_id:        token id of <|ground|>
            pointer_start_token_id: token id of <|pointer_start|>
            vision_end_token_id:    token id of <|vision_end|>
                                    (optional; auto-detected from config if not provided)
            reinit_grounding_head:  call reinit_grounding_head() to fix NaN weights from
                                    bfloat16 from_pretrained() of newly-added parameters.
                                    Set True when initializing from a base (non-retrofit)
                                    checkpoint; set False when resuming from a trained
                                    retrofit checkpoint to preserve learned weights.
        """
        self._ground_token_id = ground_token_id
        self._pointer_start_token_id = pointer_start_token_id
        if vision_end_token_id is not None:
            self._vision_end_token_id = vision_end_token_id
        if reinit_grounding_head:
            self.reinit_grounding_head()

    def reset_loss_weights(
        self, grounding_loss_weight: float, lm_loss_weight: float
    ) -> None:
        self.grounding_loss_weight = grounding_loss_weight
        self.lm_loss_weight = lm_loss_weight

    # ─────────────────────────────────────────────────────────────────────────
    # Anchor token finder (instance method, accesses self._*_token_id)
    # ─────────────────────────────────────────────────────────────────────────

    def _find_ground_anchor(
        self,
        token_ids: torch.Tensor,            # [seq_len] (single sample, no batch dim)
        external_hint: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[int, AnchorStrategy]:
        """
        Dynamically find the best anchor token for grounding query.

        Priority:
          P0. external_hint from dataset     → EXTERNAL_HINT  (pre-computed, most reliable)
          P1. <|ground|> AFTER vision_end    → EXPLICIT_GROUND_TOKEN
          P2. last <|pointer_start|> AFTER vision_end → BEFORE_POINTER_START
          P3. first token after <|vision_end|> → AFTER_VISION_END
          P4. Last non-padding token         → LAST_NON_PAD (WARNING: leakage risk)

        P1 and P2 are restricted to positions after the last <|vision_end|> to prevent
        accidentally selecting <|ground|> / <|pointer_start|> in the system-message
        action-space example, which appears before the image tokens.

        Returns (anchor_position_index, AnchorStrategy).
        """
        seq_len = token_ids.shape[0]

        # Determine the cut point: last <|vision_end|> position.
        # P1 and P2 only consider tokens after this point.
        vision_cut = -1
        if self._vision_end_token_id is not None:
            vis_ends = (token_ids == self._vision_end_token_id).nonzero(as_tuple=False)
            if vis_ends.numel() > 0:
                vision_cut = int(vis_ends[-1].item())

        # P0: external hint from dataset (pre-computed by get_ground_token_idx_in_sequence)
        if external_hint is not None and 0 <= external_hint < seq_len:
            return external_hint, AnchorStrategy.EXTERNAL_HINT

        # P1: last <|ground|> AFTER vision_end
        if self._ground_token_id is not None:
            positions = (token_ids == self._ground_token_id).nonzero(as_tuple=False).squeeze(-1)
            candidates = positions[positions > vision_cut]
            if candidates.numel() > 0:
                return int(candidates[-1].item()), AnchorStrategy.EXPLICIT_GROUND_TOKEN

        # P2: token before last <|pointer_start|> AFTER vision_end
        if self._pointer_start_token_id is not None:
            positions = (token_ids == self._pointer_start_token_id).nonzero(as_tuple=False).squeeze(-1)
            candidates = positions[positions > vision_cut]
            if candidates.numel() > 0:
                ptr_pos = int(candidates[-1].item())
                if ptr_pos > 0:
                    if verbose:
                        warnings.warn(
                            f"Anchor P2: before pointer_start at pos {ptr_pos}. "
                            f"<|ground|> not found after vision_end. "
                            f"Ensure training data injects <|ground|> for best quality.",
                            UserWarning, stacklevel=3,
                        )
                    return ptr_pos - 1, AnchorStrategy.BEFORE_POINTER_START

        # P3: first token after <|vision_end|>
        if vision_cut >= 0 and vision_cut + 1 < seq_len:
            if verbose:
                warnings.warn(
                    f"Anchor P3: first token after vision_end at pos {vision_cut}. "
                    f"No <|ground|> or <|pointer_start|> found after vision_end.",
                    UserWarning, stacklevel=3,
                )
            return vision_cut + 1, AnchorStrategy.AFTER_VISION_END

        # P4: last non-padding token (WARNING)
        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is not None:
            non_pad = (token_ids != pad_id).nonzero(as_tuple=False)
        else:
            non_pad = torch.arange(seq_len, device=token_ids.device).unsqueeze(1)

        if non_pad.numel() > 0:
            last_np = int(non_pad[-1].item())
            warnings.warn(
                f"Anchor P4: last non-pad token at pos {last_np}. "
                f"For UI-TARS native format this may follow coordinates → LABEL LEAKAGE RISK! "
                f"Inject <|ground|> token into training responses to avoid this.",
                UserWarning, stacklevel=3,
            )
            return last_np, AnchorStrategy.LAST_NON_PAD

        return seq_len - 1, AnchorStrategy.LAST_NON_PAD

    # ─────────────────────────────────────────────────────────────────────────
    # Visual token index finder
    # ─────────────────────────────────────────────────────────────────────────

    def _get_visual_indices(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get indices of visual (image patch) tokens in the flat token sequence.
        In Qwen2.5-VL, image tokens are identified by config.image_token_id.
        """
        vis_mask = (token_ids == self.config.image_token_id)
        return vis_mask.nonzero(as_tuple=False).squeeze(-1)   # [N_vis]

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # ── Retrofit grounding supervision ──────────────────────────────────
        # ground_token_indices: pre-computed anchor positions from RetrofitDataset.
        #   Each element may be None (fallback to _find_ground_anchor).
        ground_token_indices: Optional[List[Optional[int]]] = None,
        multi_patch_labels: Optional[List[Optional[torch.Tensor]]] = None,
        # Legacy compat (unused but kept for API stability)
        visual_token_indices_of_coordinates: Optional[List[torch.Tensor]] = None,
        coordinates: Optional[List[Tuple[float, float]]] = None,
        verbose: bool = False,
    ) -> Union[Tuple, RetrofitOutputWithPast]:
        """
        Full forward pass.

        When multi_patch_labels is provided:
          - Computes layerwise grounding loss per sample
          - Logs anchor strategy used per sample
        When labels is provided and lm_loss_weight > 0:
          - Also computes standard LM cross-entropy loss on assistant tokens

        Returns RetrofitOutputWithPast with .loss, .grounding_loss, .anchor_positions.
        """

        # Always enable hidden states; always disable full attention matrix (FA2 compat)
        output_hidden_states = True
        output_attentions = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ── Embed inputs ─────────────────────────────────────────────────────
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features ({n_image_features}) != image tokens "
                        f"({n_image_tokens}) in batch"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # ── RoPE position ids ─────────────────────────────────────────────────
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # ── Transformer forward ───────────────────────────────────────────────
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,    # FA2 compatible
            output_hidden_states=True,  # needed for layer probe
            return_dict=True,
            cache_position=cache_position,
        )

        all_hidden_states = outputs.hidden_states   # tuple of (num_layers+1) tensors
        hidden_states = outputs.last_hidden_state   # [batch, seq_len, d_model]
        logits = self.lm_head(hidden_states)

        # ── Optional LM loss ─────────────────────────────────────────────────
        lm_loss = None
        if labels is not None and self.lm_loss_weight > 0:
            logits_f = logits.float()
            shift_logits = logits_f[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
            )

        # ── Grounding loss ────────────────────────────────────────────────────
        grounding_loss = None
        all_grounding_scores: List = []
        all_layer_weights: List = []
        all_anchor_positions: List = []

        if multi_patch_labels is not None:
            batch_size = input_ids.shape[0]
            grounding_losses = []

            for i in range(batch_size):
                token_ids_i = input_ids[i]   # [seq_len]

                # Step 1: visual token indices
                visual_indices = self._get_visual_indices(token_ids_i)   # [N_vis]

                if visual_indices.numel() == 0:
                    grounding_losses.append(logits[i].sum() * 0.0)
                    all_grounding_scores.append(None)
                    all_layer_weights.append(None)
                    all_anchor_positions.append(None)
                    continue

                # Step 2: anchor token (P1→P5 priority)
                hint = None
                if ground_token_indices is not None:
                    hint = ground_token_indices[i]

                anchor_idx, anchor_strategy = self._find_ground_anchor(
                    token_ids=token_ids_i,
                    external_hint=hint,
                    verbose=verbose,
                )
                self._anchor_source_counts[anchor_strategy.value] = (
                    self._anchor_source_counts.get(anchor_strategy.value, 0) + 1
                )
                all_anchor_positions.append((anchor_idx, anchor_strategy))

                # Step 3: patch label
                sample_label = multi_patch_labels[i]
                if sample_label is None:
                    grounding_losses.append(logits[i].sum() * 0.0)
                    all_grounding_scores.append(None)
                    all_layer_weights.append(None)
                    continue

                n_vis = visual_indices.numel()
                sample_label = sample_label.to(input_ids.device)

                # Skip placeholder labels
                if sample_label.shape[0] == 1 and sample_label.sum() == 0:
                    grounding_losses.append(logits[i].sum() * 0.0)
                    all_grounding_scores.append(None)
                    all_layer_weights.append(None)
                    continue

                # Handle label size mismatch (processor resize may differ)
                if sample_label.shape[0] != n_vis:
                    if abs(sample_label.shape[0] - n_vis) <= 10:
                        sample_label = F.interpolate(
                            sample_label.unsqueeze(0).unsqueeze(0).float(),
                            size=n_vis, mode="linear", align_corners=False,
                        ).squeeze()
                        sample_label = sample_label / (sample_label.sum() + 1e-8)
                    else:
                        if verbose:
                            print(
                                f"[WARN] Sample {i}: label={sample_label.shape[0]} "
                                f"!= N_vis={n_vis}, skipping"
                            )
                        grounding_losses.append(logits[i].sum() * 0.0)
                        all_grounding_scores.append(None)
                        all_layer_weights.append(None)
                        continue

                # Step 4: per-sample hidden states (slice from batch)
                sample_hidden_states = tuple(hs[i] for hs in all_hidden_states)

                # Step 5: run grounding head
                head_out = self.layerwise_grounding_head(
                    all_hidden_states=sample_hidden_states,
                    ground_token_idx=anchor_idx,
                    visual_indices=visual_indices,
                    labels=sample_label,
                )
                grounding_losses.append(head_out["total_grounding_loss"])
                all_grounding_scores.append(head_out["p_final"].detach().cpu())
                all_layer_weights.append(head_out["omega"].detach().cpu())

            grounding_loss = torch.stack(grounding_losses).mean()

        # ── Combine losses ────────────────────────────────────────────────────
        total_loss = None
        if lm_loss is not None and grounding_loss is not None:
            total_loss = (
                self.lm_loss_weight * lm_loss
                + self.grounding_loss_weight * grounding_loss
            )
        elif grounding_loss is not None:
            total_loss = self.grounding_loss_weight * grounding_loss
        elif lm_loss is not None:
            total_loss = lm_loss

        # ── Build output ─────────────────────────────────────────────────────
        if return_dict:
            return RetrofitOutputWithPast(
                lm_loss=lm_loss,
                grounding_loss=grounding_loss,
                grounding_scores=all_grounding_scores,
                layer_weights=all_layer_weights,
                anchor_positions=(
                    all_anchor_positions if multi_patch_labels is not None else []
                ),
                loss=total_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=None,   # do NOT return full hidden states (memory)
                attentions=None,
                rope_deltas=self.rope_deltas,
            )
        else:
            if total_loss is not None:
                return (total_loss, logits) + outputs[1:]
            return (logits,) + outputs[1:]
