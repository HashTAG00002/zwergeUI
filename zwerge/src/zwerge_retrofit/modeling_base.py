"""
ZwerGe-UI Retrofit: Base Components (Model-Agnostic)
=====================================================
所有与具体 backbone 无关的 grounding 组件：

  - AnchorStrategy    — anchor 查找策略枚举
  - BaseRetrofitOutput — 通用 retrofit 输出 dataclass（继承 ModelOutput）
  - MLP2              — 轻量 2-layer MLP
  - LayerLoRAAdapter  — 单层 LoRA 适配器
  - LayerGroundingProbe — 单层 grounding probe
  - ContextLoRACosMetaFusion — context-aware cos-meta fusion head (~200K params)
  - LayerWiseGroundingHead — 完整的 layer-wise grounding head
  - RetrofitModelMixin — 共享 mixin，供各模型继承

具体模型文件（modeling_uitars.py / modeling_guiowl.py / modeling_uivenus.py）
继承 RetrofitModelMixin 并各自的 backbone 基类，只需实现：
  - forward() — 处理 backbone 特有参数，调用 self._compute_grounding_loss()
  - 可选：_get_visual_indices() 如需修改 image_token_id 查找方式
"""

import dataclasses
import math
import warnings
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import ModelOutput


# =============================================================================
# AnchorStrategy Enum
# =============================================================================

class AnchorStrategy(str, Enum):
    """
    Enum recording how the grounding anchor token was selected.

    Priority ordering (P1 → P5 fallback):
      P1: EXPLICIT_GROUND_TOKEN   — <|ground|> explicitly present in sequence
      P2: BEFORE_POINTER_START    — token immediately before <|pointer_start|>
      P3: AFTER_VISION_END        — first token after <|vision_end|>
      P4: EXTERNAL_HINT           — position pre-computed by RetrofitDataset
      P5: LAST_NON_PAD            — last non-padding token (WARNING: label leakage risk)
    """
    EXPLICIT_GROUND_TOKEN = "P1:explicit_ground_token"
    BEFORE_POINTER_START  = "P2:before_pointer_start"
    AFTER_VISION_END      = "P3:after_vision_end"
    EXTERNAL_HINT         = "P4:external_hint"
    LAST_NON_PAD          = "P5:last_non_pad_WARNING"


# =============================================================================
# Output dataclass (generic, works with any backbone)
# =============================================================================

@dataclasses.dataclass
class BaseRetrofitOutput(ModelOutput):
    """
    Generic retrofit output class.  Extends ModelOutput rather than a
    backbone-specific class so it can be reused across Qwen2.5-VL, Qwen3-VL, etc.

    The Trainer only requires `.loss`; all other fields are optional.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[object] = None
    hidden_states: Optional[object] = None
    attentions: Optional[object] = None
    rope_deltas: Optional[torch.LongTensor] = None
    # ── Retrofit-specific ──────────────────────────────────────────────────
    grounding_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    per_layer_losses: Optional[torch.FloatTensor] = None
    grounding_scores: Optional[object] = None   # list[Tensor | None]
    layer_weights: Optional[object] = None      # list[Tensor | None]
    anchor_positions: Optional[object] = None   # list[(int, AnchorStrategy) | None]


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
      h_norm = LN(h)
      a_out  = A(h_norm) * (1 / sqrt(d_model))
      delta_h = B(a_out)
      output = h_norm + delta_h

    Key design: B zero-initialized → identity at start; input pre-normalized
    to prevent bfloat16 overflow on deep-layer hidden states (norm > 10000).
    """

    def __init__(self, d_model: int, rank: int = 16):
        super().__init__()
        self.d_model = d_model
        self.ln = nn.LayerNorm(d_model)
        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.B.weight)
        nn.init.xavier_uniform_(self.A.weight, gain=0.1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = self.ln(h)
        a_out = self.A(h_norm) / math.sqrt(self.d_model)
        return h_norm + self.B(a_out)


# =============================================================================
# Single-layer Grounding Probe Head
# =============================================================================

class LayerGroundingProbe(nn.Module):
    """
    Query-conditioned dot-product grounding probe for a single layer.

    Architecture (per-layer adapters + shared projectors):
      h_q_adapted = q_adapter(h_q)
      h_v_adapted = k_adapter(h_vis)
      q = q_proj(LN(h_q_adapted))    [d_model → d_proj]
      k = k_proj(LN(h_v_adapted))    [d_model → d_proj]
      logits = k @ q / sqrt(d_proj)  [N_vis]
      p = softmax(logits)
    """

    def __init__(self, d_model: int, adapter_rank: int = 16):
        super().__init__()
        self.q_adapter = LayerLoRAAdapter(d_model, rank=adapter_rank)
        self.k_adapter = LayerLoRAAdapter(d_model, rank=adapter_rank)
        self.q_ln = nn.LayerNorm(d_model)
        self.k_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        h_query: torch.Tensor,
        h_vis: torch.Tensor,
        q_proj: Optional[nn.Module],
        k_proj: Optional[nn.Module],
        d_eff: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (p, logits, q): p and logits [N_vis], q [d_eff]."""
        d_model = h_query.shape[-1]
        target_norm = math.sqrt(d_model)

        # Stage 1: RMS-normalize BEFORE adapter (bfloat16 safety)
        rms_q_pre = (h_query.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        rms_v_pre = (h_vis.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        h_q_safe = h_query / rms_q_pre
        h_v_safe = h_vis / rms_v_pre

        # Stage 2: per-layer LoRA
        h_q_adapted = self.q_adapter(h_q_safe.unsqueeze(0)).squeeze(0)
        h_v_adapted = self.k_adapter(h_v_safe)

        # Stage 3: LN + second RMS-normalize BEFORE shared projector
        h_q_ln = self.q_ln(h_q_adapted)
        h_v_ln = self.k_ln(h_v_adapted)
        rms_scale_q = (h_q_ln.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        rms_scale_v = (h_v_ln.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        if q_proj is not None:
            q = q_proj(h_q_ln / rms_scale_q)
            k = k_proj(h_v_ln / rms_scale_v)
        else:
            q = h_q_ln / rms_scale_q
            k = h_v_ln / rms_scale_v

        logits = torch.matmul(k, q) / math.sqrt(d_eff)
        p = torch.softmax(logits, dim=-1)
        return p, logits, q


class CrossAttnGroundingProbe(nn.Module):
    """
    Cross-attention grounding probe for a single layer.

    Replaces LayerGroundingProbe when adapter_type="attn".
    Key difference: removes the rank-16 LoRA bottleneck and shared MLP,
    uses full-rank per-probe projections W_q/W_k with multi-head attention scoring.

    Architecture:
      1. RMS pre-normalize (bfloat16 safety, same as LayerGroundingProbe)
      2. LayerNorm + second RMS (stability)
      3. W_q [d_model → n_heads*d_head]: full-rank query projection
         W_k [d_model → n_heads*d_head]: full-rank key projection
      4. Per-head dot-product scores: [N_vis, n_heads]
      5. Learnable head_gate (softmax) → weighted sum → logits [N_vis]
      6. p = softmax(logits)

    Parameter budget (8B model, d_model=4096, n_heads=8, d_head=64):
      W_q + W_k: 2 × 4096×512 ≈ 4.19M per probe
      10 probes: ~42M  (0.52% of 8B)  ← within 0.5–0.75% target
    """

    def __init__(self, d_model: int, n_heads: int = 8, d_head: int = 64):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_head
        d_attn = n_heads * d_head
        self.q_ln     = nn.LayerNorm(d_model)
        self.k_ln     = nn.LayerNorm(d_model)
        self.W_q      = nn.Linear(d_model, d_attn, bias=False)
        self.W_k      = nn.Linear(d_model, d_attn, bias=False)
        # head_gate: learnable head-combination weights, init zeros → uniform softmax at start
        self.head_gate = nn.Parameter(torch.zeros(n_heads))
        nn.init.xavier_uniform_(self.W_q.weight, gain=0.02)
        nn.init.xavier_uniform_(self.W_k.weight, gain=0.02)

    def forward(
        self,
        h_query: torch.Tensor,
        h_vis: torch.Tensor,
        q_proj: Optional[nn.Module],   # ignored (probe has its own W_q/W_k)
        k_proj: Optional[nn.Module],   # ignored
        d_eff: int,                    # ignored
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (p, logits, q_l): p and logits [N_vis], q_l [n_heads*d_head]."""
        d_model     = h_query.shape[-1]
        target_norm = math.sqrt(d_model)

        # Stage 1: RMS pre-normalize (same bfloat16 safety as LayerGroundingProbe)
        rms_q = (h_query.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        rms_v = (h_vis.norm(dim=-1, keepdim=True)   / target_norm).clamp(min=1e-6)
        h_q_safe = h_query / rms_q
        h_v_safe = h_vis   / rms_v

        # Stage 2: LayerNorm
        h_q_ln = self.q_ln(h_q_safe)
        h_v_ln = self.k_ln(h_v_safe)

        # Stage 3: second RMS before projection
        rms_q2 = (h_q_ln.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        rms_v2 = (h_v_ln.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)

        # Stage 4: full-rank multi-head projections
        Q = self.W_q(h_q_ln / rms_q2).view(self.n_heads, self.d_head)     # [n_heads, d_head]
        K = self.W_k(h_v_ln / rms_v2).view(-1, self.n_heads, self.d_head) # [N_vis, n_heads, d_head]

        # Stage 5: per-head attention scores → [N_vis, n_heads]
        scores_h = torch.einsum("hd,nhd->nh", Q, K) / math.sqrt(self.d_head)

        # Stage 6: learnable head combination
        omega  = torch.softmax(self.head_gate.to(scores_h.dtype), dim=-1)  # [n_heads]
        logits = scores_h @ omega                                            # [N_vis]

        p   = torch.softmax(logits, dim=-1)
        q_l = Q.view(-1)   # [n_heads*d_head], for fusion use (detach applied in LayerWiseGroundingHead)
        return p, logits, q_l


# =============================================================================
# Context-Aware Cos-Meta Fusion Head
# =============================================================================

class ContextLoRACosMetaFusion(nn.Module):
    """
    ~200K-param context-aware cos-meta fusion head for A8 Stage-2.

    Architecture per sample (M active layers):
      1. Per-layer LoRA residual: z_l = LN_f(q_l) + B_f(A_f(LN_f(q_l)))
      2. Cross-layer mean context: z_bar = mean_l(z_l)
      3. Context LoRA: c = B_c(A_c(LN_c(z_bar)))
      4. Context-aware representation: z_tilde_l = LN_o(z_l + c)
      5. Cos-meta score: a_l = alpha_l + tau * cos(q_meta, z_tilde_l)
      6. Layer weights: omega = softmax(a)

    Parameter count (d_attn=512, M=10, lora_rank=128, context_rank=64):
      A_f/B_f: 131,072 + A_c/B_c: 65,536 + 3 LNs: 3,072 + q_meta: 512 + alpha: 10 + rho: 1
      Total: 200,203
    """

    def __init__(
        self,
        num_layers: int,
        d_attn: int,
        lora_rank: int = 128,
        context_rank: int = 64,
        learn_temperature: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.A_f = nn.Linear(d_attn, lora_rank, bias=False)
        self.B_f = nn.Linear(lora_rank, d_attn, bias=False)
        self.A_c = nn.Linear(d_attn, context_rank, bias=False)
        self.B_c = nn.Linear(context_rank, d_attn, bias=False)
        self.ln_f = nn.LayerNorm(d_attn)
        self.ln_c = nn.LayerNorm(d_attn)
        self.ln_o = nn.LayerNorm(d_attn)
        self.q_meta = nn.Parameter(torch.empty(d_attn))
        self.alpha  = nn.Parameter(torch.zeros(num_layers))
        if learn_temperature:
            self.rho = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413) ≈ 1.0
        else:
            self.register_buffer("rho", torch.tensor(0.5413))
        nn.init.xavier_uniform_(self.A_f.weight, gain=0.02)
        nn.init.zeros_(self.B_f.weight)
        nn.init.xavier_uniform_(self.A_c.weight, gain=0.02)
        nn.init.zeros_(self.B_c.weight)
        nn.init.normal_(self.q_meta, std=0.01)

    def forward(self, per_layer_queries: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            per_layer_queries: list of M tensors, each [d_attn]
        Returns:
            omega: [M] softmax layer weights
        """
        q_stack = torch.stack(per_layer_queries, dim=0)   # [M, d_attn]
        z  = self.ln_f(q_stack)
        z  = z + self.B_f(self.A_f(z))                   # [M, d_attn]
        z_bar   = z.mean(dim=0)
        c  = self.B_c(self.A_c(self.ln_c(z_bar)))        # [d_attn]
        z_tilde = self.ln_o(z + c.unsqueeze(0))           # [M, d_attn]
        q_meta  = F.normalize(self.q_meta.to(z_tilde.dtype), dim=-1)
        z_norm  = F.normalize(z_tilde.to(q_meta.dtype), dim=-1)
        cos     = (z_norm * q_meta.unsqueeze(0)).sum(dim=-1)   # [M]
        tau     = F.softplus(self.rho) + 1e-4
        scores  = self.alpha.to(cos.dtype) + tau.to(cos.dtype) * cos
        return torch.softmax(scores, dim=-1)


# =============================================================================
# Full Layer-Wise Grounding Head
# =============================================================================

class LayerWiseGroundingHead(nn.Module):
    """
    Complete layer-wise coordinate-free grounding head.

    Two modes controlled by `independent_layers`:

    Normal mode (independent_layers=False):
      Fusion:    ContextLoRACosMetaFusion (fusion_type="cos_meta_context_lora") → omega → p_final
      Loss:      L_fuse + lambda_layer * L_layer (only active probes)

    Independent mode (independent_layers=True):
      No ContextLoRACosMetaFusion (no omega parameters)
      Each probe supervised only by its own per-layer KL
      Loss:      mean_l KL(y || p_l)   [no fusion term]
      p_final:   uniform mean of per-layer probs (for eval only, no gradient)
      omega:     uniform [1/L, ..., 1/L]  (for eval display only)

    Active-subset design (for stage-2 / A8):
      probe_layers: all layers in the model (must match checkpoint)
      active_probe_layers: subset used in forward/fusion/loss
      Inactive probes are retained in the model (for clean checkpoint loading)
      but frozen by setup_trainable_params in train_retrofit.py.
    """

    def __init__(
        self,
        d_model: int,
        d_proj: int,
        probe_layers: List[int],
        active_probe_layers: Optional[List[int]] = None,
        adapter_rank: int = 16,
        lambda_layer: float = 0.5,
        fusion_type: str = "cos_meta_context_lora",
        use_shared_mlp: bool = True,
        independent_layers: bool = False,
        adapter_type: str = "lora",
        attn_n_heads: int = 8,
        attn_d_head: int = 64,
        fusion_lora_rank: int = 128,
        fusion_context_rank: int = 64,
        fusion_learn_temperature: bool = True,
        fusion_detach_queries: bool = True,
    ):
        super().__init__()
        self.probe_layers      = sorted(probe_layers)
        self.num_probes        = len(self.probe_layers)
        self.d_model           = d_model
        self.d_proj            = d_proj
        self.lambda_layer      = lambda_layer
        self.independent_layers = independent_layers
        self.adapter_type      = adapter_type
        self.fusion_detach_queries = fusion_detach_queries

        # Active-subset: which probe indices participate in forward/fusion/loss
        if active_probe_layers is not None:
            _active_set = set(active_probe_layers)
            for l in _active_set:
                if l not in set(self.probe_layers):
                    raise ValueError(
                        f"active_probe_layers contains layer {l} not in probe_layers {self.probe_layers}"
                    )
            self.active_probe_indices = [
                i for i, l in enumerate(self.probe_layers) if l in _active_set
            ]
            self.active_probe_layers = sorted(active_probe_layers)
        else:
            self.active_probe_indices = list(range(self.num_probes))
            self.active_probe_layers  = list(self.probe_layers)
        self.num_active_probes = len(self.active_probe_indices)

        # "attn" probe has its own W_q/W_k — shared MLP not needed
        if independent_layers or adapter_type == "attn":
            use_shared_mlp = False
        self.use_shared_mlp = use_shared_mlp

        if use_shared_mlp:
            self.q_proj = MLP2(d_model, d_proj, d_proj)
            self.k_proj = MLP2(d_model, d_proj, d_proj)
        else:
            self.q_proj = None   # type: ignore[assignment]
            self.k_proj = None   # type: ignore[assignment]

        if adapter_type == "attn":
            self.probes = nn.ModuleList([
                CrossAttnGroundingProbe(d_model, n_heads=attn_n_heads, d_head=attn_d_head)
                for _ in range(self.num_probes)
            ])
        else:
            self.probes = nn.ModuleList([
                LayerGroundingProbe(d_model, adapter_rank)
                for _ in range(self.num_probes)
            ])

        # Fusion head only in non-independent mode
        if not independent_layers:
            if fusion_type == "cos_meta_context_lora":
                if adapter_type != "attn":
                    raise ValueError("cos_meta_context_lora requires adapter_type='attn'")
                d_attn = attn_n_heads * attn_d_head
                self.fusion = ContextLoRACosMetaFusion(
                    num_layers=self.num_active_probes,
                    d_attn=d_attn,
                    lora_rank=fusion_lora_rank,
                    context_rank=fusion_context_rank,
                    learn_temperature=fusion_learn_temperature,
                )
            else:
                raise ValueError(
                    f"Unknown fusion_type: {fusion_type!r}. Use 'cos_meta_context_lora'."
                )

    def forward(
        self,
        all_hidden_states: Tuple[torch.Tensor, ...],
        ground_token_idx: int,
        visual_indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys: p_final, omega, per_layer_probs
        (+ loss_fuse, loss_layer, total_grounding_loss if labels given)

        per_layer_probs has len == num_probes (all probes, for eval inspection).
        omega and p_final are computed over active probes only.
        """
        all_p: List[torch.Tensor] = []
        all_q: List[torch.Tensor] = []

        for probe_i, layer_idx in enumerate(self.probe_layers):
            hs = all_hidden_states[layer_idx + 1]   # [seq_len, d_model]
            h_query = hs[ground_token_idx]           # [d_model]
            h_vis   = hs[visual_indices]             # [N_vis, d_model]

            p_l, _, q_l = self.probes[probe_i](
                h_query, h_vis,
                self.q_proj, self.k_proj,
                self.d_proj if self.use_shared_mlp else self.d_model,
            )
            all_p.append(p_l)
            all_q.append(q_l)

        # Slice to active subset
        active_p = [all_p[i] for i in self.active_probe_indices]

        if self.independent_layers:
            # Uniform mean over active probes for eval; no fusion weights
            p_final = sum(active_p) / self.num_active_probes
            omega   = torch.full(
                (self.num_active_probes,), 1.0 / self.num_active_probes,
                device=p_final.device, dtype=p_final.dtype,
            )
        else:
            active_q = [
                all_q[i].detach() if self.fusion_detach_queries else all_q[i]
                for i in self.active_probe_indices
            ]
            omega   = self.fusion(active_q)   # [num_active_probes]
            p_final = sum(omega[j] * active_p[j] for j in range(self.num_active_probes))

        result = {
            "p_final": p_final,
            "omega": omega,
            "per_layer_probs": all_p,   # all probes retained for eval inspection
        }

        if labels is not None:
            eps = 1e-8
            labels_f   = labels.float()
            label_dist = labels_f / (labels_f.sum() + eps)

            if self.independent_layers:
                # Loss = mean per-layer KL over active probes only; no fusion term
                loss_layer = torch.zeros((), device=label_dist.device)
                for p_l in active_p:
                    loss_layer = loss_layer + F.kl_div(
                        torch.log(p_l.clamp(min=eps)), label_dist, reduction="sum",
                    )
                loss_layer = loss_layer / self.num_active_probes
                result["loss_fuse"]           = torch.zeros_like(loss_layer)
                result["loss_layer"]          = loss_layer
                result["total_grounding_loss"] = loss_layer
            else:
                loss_fuse = F.kl_div(
                    torch.log(p_final.clamp(min=eps)), label_dist, reduction="sum",
                )
                loss_layer = torch.zeros((), device=p_final.device)
                for p_l in active_p:
                    loss_layer = loss_layer + F.kl_div(
                        torch.log(p_l.clamp(min=eps)), label_dist, reduction="sum",
                    )
                loss_layer = loss_layer / self.num_active_probes
                result["loss_fuse"]           = loss_fuse
                result["loss_layer"]          = loss_layer
                result["total_grounding_loss"] = loss_fuse + self.lambda_layer * loss_layer

        return result


# =============================================================================
# RetrofitModelMixin — shared logic for all retrofit model classes
# =============================================================================

class RetrofitModelMixin:
    """
    Mixin providing layer-wise grounding head capabilities to any VLM.

    Usage:
      class MyRetrofitModel(RetrofitModelMixin, SomeVLMClass):
          def __init__(self, config, *args, **kwargs):
              super().__init__(config, *args, **kwargs)
              self._init_retrofit_from_config(config)
              self.post_init()

          def forward(self, ..., ground_token_indices=None, multi_patch_labels=None, ...):
              # 1. call backbone forward to get all_hidden_states + logits
              # 2. call self._compute_grounding_loss(...)
              # 3. return output

    Concrete models must also implement:
      - forward()  with backbone-specific parameter signature
    """

    def _init_retrofit_from_config(self, config) -> None:
        """Initialize the grounding head and retrofit state from model config."""
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        probe_layers          = getattr(config, "probe_layers",                      [14, 18, 21, 24, 26, 27])
        active_probe_layers   = getattr(config, "grounding_active_probe_layers",     None)
        d_proj                = getattr(config, "grounding_proj_dim",                512)
        adapter_rank          = getattr(config, "grounding_adapter_rank",            16)
        lambda_layer          = getattr(config, "grounding_lambda_layer",            0.5)
        fusion_type           = getattr(config, "grounding_fusion_type",             "cos_meta_context_lora")
        use_shared_mlp        = getattr(config, "grounding_use_shared_mlp",          True)
        independent_layers    = getattr(config, "grounding_independent_layers",      False)
        adapter_type          = getattr(config, "grounding_adapter_type",            "lora")
        attn_n_heads          = getattr(config, "grounding_attn_heads",              8)
        attn_d_head           = getattr(config, "grounding_attn_head_dim",           64)
        fusion_lora_rank      = getattr(config, "grounding_fusion_lora_rank",        128)
        fusion_context_rank   = getattr(config, "grounding_fusion_context_rank",     64)
        fusion_learn_temp     = getattr(config, "grounding_fusion_learn_temperature", True)
        fusion_detach_q       = getattr(config, "grounding_fusion_detach_queries",   True)

        self.layerwise_grounding_head = LayerWiseGroundingHead(
            d_model=config.hidden_size,
            d_proj=d_proj,
            probe_layers=probe_layers,
            active_probe_layers=active_probe_layers,
            adapter_rank=adapter_rank,
            lambda_layer=lambda_layer,
            fusion_type=fusion_type,
            use_shared_mlp=use_shared_mlp,
            independent_layers=independent_layers,
            adapter_type=adapter_type,
            attn_n_heads=attn_n_heads,
            attn_d_head=attn_d_head,
            fusion_lora_rank=fusion_lora_rank,
            fusion_context_rank=fusion_context_rank,
            fusion_learn_temperature=fusion_learn_temp,
            fusion_detach_queries=fusion_detach_q,
        )
        head = self.layerwise_grounding_head
        _logger.info(
            f"[RetrofitHead] probe_layers={head.probe_layers} "
            f"active={head.active_probe_layers} "
            f"active_indices={head.active_probe_indices}"
        )

        self.grounding_loss_weight: float = 1.0
        self.lm_loss_weight: float = 0.0

        self._ground_token_id: Optional[int] = getattr(config, "ground_token_id", None)
        self._pointer_start_token_id: Optional[int] = getattr(
            config, "pointer_start_token_id", None
        )
        _vid = getattr(config, "vision_end_token_id", None)
        if _vid is None:
            _vid = getattr(config, "vision_token_id", None)
        self._vision_end_token_id: Optional[int] = _vid

        self._anchor_source_counts: Dict[str, int] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Public setup helpers
    # ─────────────────────────────────────────────────────────────────────────

    def reinit_grounding_head(self) -> None:
        """
        Re-initialize all LayerNorm / LoRA weights in the grounding head.
        Must be called after from_pretrained() to fix NaN bfloat16 parameters.
        Called automatically at the end of setup_special_token_ids() when
        reinit_grounding_head=True.
        """
        for probe in self.layerwise_grounding_head.probes:
            if isinstance(probe, CrossAttnGroundingProbe):
                nn.init.xavier_uniform_(probe.W_q.weight, gain=0.02)
                nn.init.xavier_uniform_(probe.W_k.weight, gain=0.02)
                nn.init.zeros_(probe.head_gate)
                nn.init.ones_(probe.q_ln.weight)
                nn.init.zeros_(probe.q_ln.bias)
                nn.init.ones_(probe.k_ln.weight)
                nn.init.zeros_(probe.k_ln.bias)
            else:   # LayerGroundingProbe (LoRA)
                for adapter in [probe.q_adapter, probe.k_adapter]:
                    nn.init.ones_(adapter.ln.weight)
                    nn.init.zeros_(adapter.ln.bias)
                    nn.init.xavier_uniform_(adapter.A.weight, gain=0.1)
                    nn.init.zeros_(adapter.B.weight)
                nn.init.ones_(probe.q_ln.weight)
                nn.init.zeros_(probe.q_ln.bias)
                nn.init.ones_(probe.k_ln.weight)
                nn.init.zeros_(probe.k_ln.bias)
        fusion = getattr(self.layerwise_grounding_head, "fusion", None)
        if fusion is not None and isinstance(fusion, ContextLoRACosMetaFusion):
            nn.init.xavier_uniform_(fusion.A_f.weight, gain=0.02)
            nn.init.zeros_(fusion.B_f.weight)
            nn.init.xavier_uniform_(fusion.A_c.weight, gain=0.02)
            nn.init.zeros_(fusion.B_c.weight)
            for ln in [fusion.ln_f, fusion.ln_c, fusion.ln_o]:
                nn.init.ones_(ln.weight)
                nn.init.zeros_(ln.bias)
            nn.init.normal_(fusion.q_meta, std=0.01)
            nn.init.zeros_(fusion.alpha)
            nn.init.constant_(fusion.rho, 0.5413)

    def reinit_fusion_only(self) -> None:
        """
        Re-initialize ONLY the fusion head parameters (ContextLoRACosMetaFusion).
        Does NOT touch probe weights (W_q, W_k, head_gate, LNs).

        Must be called for A8 stage-2 runs where:
          - reinit_grounding_head=False  (preserve A7 probe weights)
          - fusion_type="cos_meta_context_lora"  (newly created fusion params need init)

        Without this, HF from_pretrained(torch_dtype=bfloat16) leaves new fusion
        parameters as uninitialized bfloat16 memory (~3e38 max value), causing NaN.
        """
        fusion = getattr(self.layerwise_grounding_head, "fusion", None)
        if fusion is None:
            return
        if isinstance(fusion, ContextLoRACosMetaFusion):
            nn.init.xavier_uniform_(fusion.A_f.weight, gain=0.02)
            nn.init.zeros_(fusion.B_f.weight)
            nn.init.xavier_uniform_(fusion.A_c.weight, gain=0.02)
            nn.init.zeros_(fusion.B_c.weight)
            for ln in [fusion.ln_f, fusion.ln_c, fusion.ln_o]:
                nn.init.ones_(ln.weight)
                nn.init.zeros_(ln.bias)
            nn.init.normal_(fusion.q_meta, std=0.01)
            nn.init.zeros_(fusion.alpha)
            nn.init.constant_(fusion.rho, 0.5413)

    def setup_special_token_ids(
        self,
        ground_token_id: int,
        pointer_start_token_id: int,
        vision_end_token_id: Optional[int] = None,
        reinit_grounding_head: bool = True,
    ) -> None:
        """
        Register special token IDs needed by _find_ground_anchor().
        Must be called after add_special_tokens() and resize_token_embeddings().
        """
        self._ground_token_id = ground_token_id
        self._pointer_start_token_id = pointer_start_token_id
        if vision_end_token_id is not None:
            self._vision_end_token_id = vision_end_token_id
        if reinit_grounding_head:
            self.reinit_grounding_head()
        else:
            # Even when not reinitializing probes, always init newly created
            # fusion parameters to prevent bfloat16 uninitialized memory NaN.
            self.reinit_fusion_only()

    def reset_loss_weights(
        self, grounding_loss_weight: float, lm_loss_weight: float
    ) -> None:
        self.grounding_loss_weight = grounding_loss_weight
        self.lm_loss_weight = lm_loss_weight

    def _zero_grounding_loss(self, device=None) -> torch.Tensor:
        """
        Returns 0.0 connected to all trainable parameters in the grounding head
        (and _new_token_emb if present).

        Used in skip branches of _compute_grounding_loss so that:
          (a) all-skip batches can still call backward() without graph errors;
          (b) DDP always sees every trainable parameter participating in the graph,
              preventing _rebuild_buckets failures with find_unused_parameters=False.

        This is essential for Qwen3-VL (guiowl/uivenus) where the hook-based
        grounding path and frozen-backbone setup create a data-dependent graph.
        It is also a defensive practice for UI-TARS.
        """
        loss = None
        for p in self.layerwise_grounding_head.parameters():
            if p.requires_grad:
                z = p.sum() * 0.0
                if device is not None:
                    z = z.to(device)
                loss = z if loss is None else loss + z
        # Include _new_token_emb if the model uses the standalone-param embedding approach
        new_token_emb = getattr(self, "_new_token_emb", None)
        if new_token_emb is not None and new_token_emb.requires_grad:
            z = new_token_emb.sum() * 0.0
            if device is not None:
                z = z.to(device)
            loss = z if loss is None else loss + z
        if loss is None:
            # Absolute fallback: create a dummy zero on the requested device
            dev = device or (next(self.parameters()).device if list(self.parameters()) else "cpu")
            loss = torch.zeros((), device=dev, requires_grad=True)
        return loss

    # ─────────────────────────────────────────────────────────────────────────
    # Anchor token finder
    # ─────────────────────────────────────────────────────────────────────────

    def _find_ground_anchor(
        self,
        token_ids: torch.Tensor,
        external_hint: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[int, AnchorStrategy]:
        """
        Dynamically find the best anchor token for grounding query.

        Priority:
          P0. external_hint         → EXTERNAL_HINT
          P1. <|ground|> AFTER vision_end → EXPLICIT_GROUND_TOKEN
          P2. last <|pointer_start|> AFTER vision_end → BEFORE_POINTER_START
          P3. first token after <|vision_end|> → AFTER_VISION_END
          P4. Last non-padding token → LAST_NON_PAD (WARNING: leakage risk)
        """
        seq_len = token_ids.shape[0]

        vision_cut = -1
        if self._vision_end_token_id is not None:
            vis_ends = (token_ids == self._vision_end_token_id).nonzero(as_tuple=False)
            if vis_ends.numel() > 0:
                vision_cut = int(vis_ends[-1].item())

        if external_hint is not None and 0 <= external_hint < seq_len:
            return external_hint, AnchorStrategy.EXTERNAL_HINT

        if self._ground_token_id is not None:
            positions = (token_ids == self._ground_token_id).nonzero(as_tuple=False).squeeze(-1)
            candidates = positions[positions > vision_cut]
            if candidates.numel() > 0:
                return int(candidates[-1].item()), AnchorStrategy.EXPLICIT_GROUND_TOKEN

        if self._pointer_start_token_id is not None:
            positions = (token_ids == self._pointer_start_token_id).nonzero(as_tuple=False).squeeze(-1)
            candidates = positions[positions > vision_cut]
            if candidates.numel() > 0:
                ptr_pos = int(candidates[-1].item())
                if ptr_pos > 0:
                    if verbose:
                        warnings.warn(
                            f"Anchor P2: before pointer_start at pos {ptr_pos}.",
                            UserWarning, stacklevel=3,
                        )
                    return ptr_pos - 1, AnchorStrategy.BEFORE_POINTER_START

        if vision_cut >= 0 and vision_cut + 1 < seq_len:
            if verbose:
                warnings.warn(
                    f"Anchor P3: first token after vision_end at pos {vision_cut}.",
                    UserWarning, stacklevel=3,
                )
            return vision_cut + 1, AnchorStrategy.AFTER_VISION_END

        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is not None:
            non_pad = (token_ids != pad_id).nonzero(as_tuple=False)
        else:
            non_pad = torch.arange(seq_len, device=token_ids.device).unsqueeze(1)

        if non_pad.numel() > 0:
            last_np = int(non_pad[-1].item())
            warnings.warn(
                f"Anchor P4: last non-pad token at pos {last_np}. LABEL LEAKAGE RISK!",
                UserWarning, stacklevel=3,
            )
            return last_np, AnchorStrategy.LAST_NON_PAD

        return seq_len - 1, AnchorStrategy.LAST_NON_PAD

    # ─────────────────────────────────────────────────────────────────────────
    # Visual token index finder
    # ─────────────────────────────────────────────────────────────────────────

    def _get_visual_indices(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get indices of visual (image patch) tokens.
        Uses config.image_token_id — works for both Qwen2.5-VL and Qwen3-VL.
        """
        vis_mask = (token_ids == self.config.image_token_id)
        return vis_mask.nonzero(as_tuple=False).squeeze(-1)

    # ─────────────────────────────────────────────────────────────────────────
    # Core grounding computation (shared across all model classes)
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # Grounding inference: get all layer hidden states
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _forward_hidden_states_for_grounding(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        device: torch.device,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Default implementation for Qwen2.5-VL: manually embeds tokens and visual,
        then runs the transformer with output_hidden_states=True.

        GUIOwlRetrofitModel / UIVenusRetrofitModel override this to call
        super().forward(output_hidden_states=True) directly (official Qwen3-VL handles
        DeepStack injection automatically).
        Note: deepstack_visual_indexes=[8, 16, 24] are ViT layer indices, NOT LLM layer
        indices. The actual LLM injection points are decoder layers 0, 1, 2.

        Returns:
            all_hidden_states: tuple of (num_layers+1) tensors [seq_len, d_model]
        """
        # ── Embed tokens ─────────────────────────────────────────────────────
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pv = pixel_values.to(self.dtype)
            image_embeds = self.visual(pv, grid_thw=image_grid_thw)
            n_img_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_img_feats  = image_embeds.shape[0]
            if n_img_tokens != n_img_feats:
                warnings.warn(
                    f"Image token mismatch: seq={n_img_tokens}, visual={n_img_feats}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # ── RoPE position ids ─────────────────────────────────────────────────
        position_ids = None
        if hasattr(self, "get_rope_index"):
            try:
                position_ids, _ = self.get_rope_index(
                    input_ids, image_grid_thw, None, attention_mask
                )
            except Exception:
                position_ids = None

        # ── Transformer forward ───────────────────────────────────────────────
        transformer_out = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return transformer_out.hidden_states

    def _compute_grounding_loss(
        self,
        all_hidden_states: Tuple[torch.Tensor, ...],
        input_ids: torch.LongTensor,                          # [batch, seq_len]
        logits: torch.FloatTensor,                            # [batch, seq_len, vocab]
        ground_token_indices: Optional[List[Optional[int]]],
        multi_patch_labels: Optional[List[Optional[torch.Tensor]]],
        verbose: bool = False,
    ) -> Tuple[
        Optional[torch.FloatTensor],
        List, List, List
    ]:
        """
        Run the layer-wise grounding head over a batch.

        Returns:
          grounding_loss          — scalar tensor or None
          all_grounding_scores    — list[Tensor|None]
          all_layer_weights       — list[Tensor|None]
          all_anchor_positions    — list[(int, AnchorStrategy)|None]
        """
        if multi_patch_labels is None:
            return None, [], [], []

        batch_size = input_ids.shape[0]
        grounding_losses = []
        all_grounding_scores: List = []
        all_layer_weights: List = []
        all_anchor_positions: List = []

        for i in range(batch_size):
            token_ids_i = input_ids[i]

            visual_indices = self._get_visual_indices(token_ids_i)

            if visual_indices.numel() == 0:
                grounding_losses.append(self._zero_grounding_loss(device=logits.device))
                all_grounding_scores.append(None)
                all_layer_weights.append(None)
                all_anchor_positions.append(None)
                continue

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

            sample_label = multi_patch_labels[i]
            if sample_label is None:
                grounding_losses.append(self._zero_grounding_loss(device=logits.device))
                all_grounding_scores.append(None)
                all_layer_weights.append(None)
                continue

            n_vis = visual_indices.numel()
            sample_label = sample_label.to(input_ids.device)

            if sample_label.shape[0] == 1 and sample_label.sum() == 0:
                grounding_losses.append(self._zero_grounding_loss(device=logits.device))
                all_grounding_scores.append(None)
                all_layer_weights.append(None)
                continue

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
                    grounding_losses.append(self._zero_grounding_loss(device=logits.device))
                    all_grounding_scores.append(None)
                    all_layer_weights.append(None)
                    continue

            # Support sparse tuples (GUI-Owl hook path has None at non-probe positions)
            sample_hidden_states = tuple(
                hs[i] if hs is not None else None for hs in all_hidden_states
            )

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
        return grounding_loss, all_grounding_scores, all_layer_weights, all_anchor_positions
