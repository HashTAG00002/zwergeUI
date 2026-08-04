"""
ZwerGe-UI — Frozen-Backbone Retrofit (minimal, self-contained)
================================================================
This file contains the *model-side* core of ZwerGe-UI as described in the
paper "Middle Layers Know Where to Click: Coordinate Serialization
Bottlenecks in GUI Agents":

  * Per-layer cross-attention grounding probes (Eq. 1, "Per-Layer Spatial
    Probes" section).  Each probe reads the anchor query h_t* and the
    visual keys H_v at ONE intermediate layer of a frozen backbone and
    outputs a patch-level posterior p_l via multi-head cross-attention
    with a learnable head gate.

  * RMSNorm pre-scale (used twice, before every LayerNorm) to keep hidden
    state L2 norms bounded to ~sqrt(d) regardless of depth, which is what
    makes the probes numerically stable in bfloat16 across very different
    layer-norm scales (paper: "RMSNorm pre-scale" paragraph).

  * Cross-layer fusion (Eq. 3, "Cross-Layer Fusion" section): a cross-layer
    LoRA refines each active layer's query, and a cosine-meta scorer
    assigns per-layer weights omega_l = softmax(alpha_l + tau^-1 *
    cos(q_tilde_l, q_meta)); p_final = sum_l omega_l * p_l.

  * RetrofitModelMixin — the model-agnostic glue that any frozen VLM
    backbone (Qwen2.5-VL, Qwen3-VL, ...) can inherit from to gain a
    grounding head.  It implements the anchor-token lookup used to read
    the query representation with ZERO label leakage (the anchor token
    is emitted BEFORE any coordinate value, see the paper's "Prefill-forced
    inference" paragraph), and the visual-token index lookup.

The backbone itself (Qwen2.5-VL / Qwen3-VL) is NOT included here — it must
be downloaded separately from HuggingFace (see README.md).  This file only
contains the retrofit components that sit ON TOP of a frozen backbone; the
backbone contributes hidden states, nothing else, and receives zero
gradient (see RetrofitModelMixin._compute_grounding_loss).

Concrete per-backbone subclasses (illustrative only — trimmed to the
minimum needed to explain how a backbone is wired in; see the full
repository for GUI-Owl-1.5 / UI-Venus-1.5 / Qwen3.5 variants) are provided
at the bottom of this file behind a lazy import so this module has zero
hard dependency on any specific backbone package.
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
# AnchorStrategy — records how the grounding anchor token was located
# =============================================================================

class AnchorStrategy(str, Enum):
    """
    How the <|ground|> anchor token position was resolved for a sample.

    Priority ordering (P1 is preferred; fallbacks exist for malformed or
    truncated sequences and should essentially never fire in practice):
      P1: EXPLICIT_GROUND_TOKEN — <|ground|> explicitly present in sequence
                                   (the paper's primary, zero-label-leakage
                                   design: the token appears in
                                   `click(<|ground|><|pointer_start|>...)`
                                   AFTER the action prefix but BEFORE any
                                   coordinate value).
      P2: BEFORE_POINTER_START  — token immediately before <|pointer_start|>
      P3: AFTER_VISION_END      — first token after <|vision_end|>
      P4: EXTERNAL_HINT         — position pre-computed by the data loader
      P5: LAST_NON_PAD          — last non-padding token (WARNING: label
                                   leakage risk; should never be reached
                                   when the prompt template is well-formed)
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
    Generic retrofit output. Extends ModelOutput rather than a
    backbone-specific class so it can be reused across Qwen2.5-VL,
    Qwen3-VL, etc.  Only `.loss` is required by a HuggingFace Trainer;
    everything else is diagnostic / optional.
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
    grounding_scores: Optional[object] = None   # list[Tensor | None]  (p_final per sample)
    layer_weights: Optional[object] = None      # list[Tensor | None]  (omega per sample)
    anchor_positions: Optional[object] = None   # list[(int, AnchorStrategy) | None]


# =============================================================================
# Lightweight 2-layer MLP (shared q/k projector, optional)
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
# Cross-Attention Grounding Probe (the "≈4.2M parameters" probe in the paper)
# =============================================================================

class CrossAttnGroundingProbe(nn.Module):
    """
    Cross-attention grounding probe for ONE frozen backbone layer.

    Implements Eq. 1 of the paper:
        Q_l = W_q^(l) Norm(h_t*^(l))
        K_l = W_k^(l) Norm(H_v^(l))
        p_l = softmax( sum_h softmax(alpha)_h * Q_l^T K_{l,h} / sqrt(d_h) )

    where Norm(.) is the RMS pre-scale followed by LayerNorm (see
    `_rms_prescale` below), and W_q^(l), W_k^(l) are FULL-RANK per-layer
    projections (no LoRA bottleneck) so each probe can learn a geometry
    suited to its own layer.

    Parameter budget (paper): d_model=4096, n_heads=8, d_head=64
      W_q + W_k: 2 x 4096 x 512 ~= 4.19M parameters per probe.
      A set of ~10 probes totals ~42M, i.e. ~0.5% of an 8B backbone.
    """

    def __init__(self, d_model: int, n_heads: int = 8, d_head: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_head
        d_attn = n_heads * d_head
        self.q_ln = nn.LayerNorm(d_model)
        self.k_ln = nn.LayerNorm(d_model)
        self.W_q  = nn.Linear(d_model, d_attn, bias=False)
        self.W_k  = nn.Linear(d_model, d_attn, bias=False)
        # head_gate: learnable per-head combination weights (softmax(alpha) in
        # Eq. 1). Zero-init -> uniform head weighting at the start of training.
        self.head_gate = nn.Parameter(torch.zeros(n_heads))
        nn.init.xavier_uniform_(self.W_q.weight, gain=0.02)
        nn.init.xavier_uniform_(self.W_k.weight, gain=0.02)

    @staticmethod
    def _rms_prescale(h: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm pre-scale (paper: "Hidden-state l2 norms grow with depth,
        geometrically misaligning layers"). Rescales h so that
        ||h||_2 ~= sqrt(d) regardless of the raw magnitude at this depth,
        which prevents bfloat16 overflow in the LayerNorm/Linear that
        follow, and puts every probed layer on a comparable numeric scale.
        """
        d = h.shape[-1]
        target_norm = math.sqrt(d)
        rms = (h.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1e-6)
        return h / rms

    def forward(
        self,
        h_query: torch.Tensor,   # [d_model]           anchor hidden state at this layer
        h_vis: torch.Tensor,     # [N_vis, d_model]     visual-patch hidden states at this layer
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (p, logits, q_l): p and logits [N_vis], q_l [n_heads*d_head]."""
        # Stage 1: RMS pre-scale (bfloat16 numerical safety).
        h_q_safe = self._rms_prescale(h_query)
        h_v_safe = self._rms_prescale(h_vis)

        # Stage 2: LayerNorm.
        h_q_ln = self.q_ln(h_q_safe)
        h_v_ln = self.k_ln(h_v_safe)

        # Stage 3: second RMS pre-scale immediately before the projection
        # (this is the "before every LayerNorm" application referenced in
        # the class docstring: once on the raw hidden state, once again on
        # the LayerNorm output, both feeding into numerically-unstable
        # matrix multiplies).
        h_q_ln = self._rms_prescale(h_q_ln)
        h_v_ln = self._rms_prescale(h_v_ln)

        # Stage 4: full-rank multi-head projections (Eq. 1's W_q^(l), W_k^(l)).
        Q = self.W_q(h_q_ln).view(self.n_heads, self.d_head)          # [n_heads, d_head]
        K = self.W_k(h_v_ln).view(-1, self.n_heads, self.d_head)      # [N_vis, n_heads, d_head]

        # Stage 5: per-head dot-product scores -> [N_vis, n_heads].
        scores_h = torch.einsum("hd,nhd->nh", Q, K) / math.sqrt(self.d_head)

        # Stage 6: learnable head gate sigma(alpha) in Eq. 1.
        omega  = torch.softmax(self.head_gate.to(scores_h.dtype), dim=-1)   # [n_heads]
        logits = scores_h @ omega                                          # [N_vis]

        p   = torch.softmax(logits, dim=-1)
        q_l = Q.reshape(-1)   # [n_heads*d_head], consumed by the cross-layer fusion head
        return p, logits, q_l


# =============================================================================
# Cross-Layer Fusion Head (Eq. 3, "Cross-Layer Fusion" section; ~200K params)
# =============================================================================

class CrossLayerFusion(nn.Module):
    """
    Cross-layer LoRA + cosine-meta scorer that fuses the active layers'
    posteriors into p_final (Eq. 3 of the paper):

      z_l     = LN_f(q_l) + B_f(A_f(LN_f(q_l)))            (per-layer LoRA)
      z_bar   = mean_l(z_l)                                 (cross-layer context)
      c       = B_c(A_c(LN_c(z_bar)))                       (context LoRA)
      z~_l    = LN_o(z_l + c)
      omega_l = softmax_l( alpha_l + tau^-1 * cos(z~_l, q_meta) )
      p_final = sum_l omega_l * p_l

    alpha_l is a sample-independent trainable prior on layer l's average
    quality, q_meta is a global trainable "prototype of the ideal
    grounding representation", and tau is a learnable temperature
    (parameterised as softplus(rho) for positivity).

    Parameter count (paper, d_attn=512, M=10 active layers, lora_rank=128,
    context_rank=64):
      A_f/B_f: 131,072 + A_c/B_c: 65,536 + 3 LayerNorms: 3,072
      + q_meta: 512 + alpha: 10 + rho: 1  =  200,203 total.
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
            self.rho = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413) ~= 1.0
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
            per_layer_queries: list of M tensors, each [d_attn] (the q_l
                returned by CrossAttnGroundingProbe.forward for each of the
                M active layers).
        Returns:
            omega: [M] softmax layer weights (Eq. 3).
        """
        q_stack = torch.stack(per_layer_queries, dim=0)    # [M, d_attn]
        z  = self.ln_f(q_stack)
        z  = z + self.B_f(self.A_f(z))                     # per-layer LoRA residual
        z_bar   = z.mean(dim=0)                             # cross-layer mean context
        c  = self.B_c(self.A_c(self.ln_c(z_bar)))          # context LoRA
        z_tilde = self.ln_o(z + c.unsqueeze(0))
        q_meta  = F.normalize(self.q_meta.to(z_tilde.dtype), dim=-1)
        z_norm  = F.normalize(z_tilde.to(q_meta.dtype), dim=-1)
        cos     = (z_norm * q_meta.unsqueeze(0)).sum(dim=-1)   # [M]
        tau     = F.softplus(self.rho) + 1e-4
        scores  = self.alpha.to(cos.dtype) + tau.to(cos.dtype) * cos
        return torch.softmax(scores, dim=-1)


# =============================================================================
# Full Layer-Wise Grounding Head (Stage 1 probes + Stage 2 fusion)
# =============================================================================

class LayerWiseGroundingHead(nn.Module):
    """
    The complete grounding head: a bank of per-layer CrossAttnGroundingProbe
    modules plus (optionally) a CrossLayerFusion head over an "active"
    subset of them.

    Training happens in two stages (paper, "Cross-Layer Fusion" section):

      Stage 1 (independent_layers=True): every probe is supervised only by
        its own KL loss against the anisotropic-Gaussian label (Eq. 2);
        there is no fusion head yet.  L_S1 = mean_l KL(y || p_l).

      Stage 2 (independent_layers=False): a CrossLayerFusion head is
        introduced over `active_probe_layers` (a learned subset of
        `probe_layers`, drawn from the spatial plateau identified by the
        Stage-1 analysis); the backbone and any INACTIVE probes stay
        frozen while the fusion head and the active probes remain
        trainable.  L_S2 = KL(y || p_final) + lambda * mean_l KL(y || p_l),
        lambda=0.2 (Eq. 4).
    """

    def __init__(
        self,
        d_model: int,
        probe_layers: List[int],
        active_probe_layers: Optional[List[int]] = None,
        lambda_layer: float = 0.2,
        independent_layers: bool = False,
        attn_n_heads: int = 8,
        attn_d_head: int = 64,
        fusion_lora_rank: int = 128,
        fusion_context_rank: int = 64,
        fusion_learn_temperature: bool = True,
        fusion_detach_queries: bool = True,
    ):
        super().__init__()
        self.probe_layers       = sorted(probe_layers)
        self.num_probes         = len(self.probe_layers)
        self.d_model             = d_model
        self.lambda_layer        = lambda_layer
        self.independent_layers  = independent_layers
        self.fusion_detach_queries = fusion_detach_queries

        # Active-subset design: which probe indices participate in
        # fusion/loss during Stage 2 (paper: "we draw active layers
        # L_active subset L_probe from the spatial plateau").
        if active_probe_layers is not None:
            active_set = set(active_probe_layers)
            for l in active_set:
                if l not in set(self.probe_layers):
                    raise ValueError(
                        f"active_probe_layers contains layer {l} not in probe_layers {self.probe_layers}"
                    )
            self.active_probe_indices = [
                i for i, l in enumerate(self.probe_layers) if l in active_set
            ]
            self.active_probe_layers = sorted(active_probe_layers)
        else:
            self.active_probe_indices = list(range(self.num_probes))
            self.active_probe_layers  = list(self.probe_layers)
        self.num_active_probes = len(self.active_probe_indices)

        self.probes = nn.ModuleList([
            CrossAttnGroundingProbe(d_model, n_heads=attn_n_heads, d_head=attn_d_head)
            for _ in range(self.num_probes)
        ])

        self.fusion: Optional[CrossLayerFusion] = None
        if not independent_layers:
            d_attn = attn_n_heads * attn_d_head
            self.fusion = CrossLayerFusion(
                num_layers=self.num_active_probes,
                d_attn=d_attn,
                lora_rank=fusion_lora_rank,
                context_rank=fusion_context_rank,
                learn_temperature=fusion_learn_temperature,
            )

    def forward(
        self,
        all_hidden_states: Tuple[torch.Tensor, ...],
        ground_token_idx: int,
        visual_indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          all_hidden_states: tuple of (num_layers+1) tensors [seq_len, d_model]
              (index 0 = embedding output, index l+1 = output of decoder
              layer l), as produced by a HuggingFace VLM forward pass with
              output_hidden_states=True.
          ground_token_idx: sequence position of the <|ground|> anchor token.
          visual_indices: LongTensor of sequence positions holding visual
              patch tokens.
          labels: optional [N_vis] soft/binary target distribution (Eq. 2's
              anisotropic Gaussian, or a binary bbox mask).

        Returns a dict with keys p_final, omega, per_layer_probs (+
        loss_fuse, loss_layer, total_grounding_loss if labels is given).
        """
        all_p: List[torch.Tensor] = []
        all_q: List[torch.Tensor] = []

        for probe_i, layer_idx in enumerate(self.probe_layers):
            hs = all_hidden_states[layer_idx + 1]    # [seq_len, d_model]
            h_query = hs[ground_token_idx]            # [d_model]
            h_vis   = hs[visual_indices]               # [N_vis, d_model]

            p_l, _, q_l = self.probes[probe_i](h_query, h_vis)
            all_p.append(p_l)
            all_q.append(q_l)

        active_p = [all_p[i] for i in self.active_probe_indices]

        if self.independent_layers or self.fusion is None:
            # Stage 1: uniform mean over active probes for inspection; no
            # fusion weights are learned yet.
            p_final = sum(active_p) / self.num_active_probes
            omega   = torch.full(
                (self.num_active_probes,), 1.0 / self.num_active_probes,
                device=p_final.device, dtype=p_final.dtype,
            )
        else:
            # Stage 2: learned cross-layer fusion (Eq. 3).
            active_q = [
                all_q[i].detach() if self.fusion_detach_queries else all_q[i]
                for i in self.active_probe_indices
            ]
            omega   = self.fusion(active_q)                              # [num_active_probes]
            p_final = sum(omega[j] * active_p[j] for j in range(self.num_active_probes))

        result = {
            "p_final": p_final,
            "omega": omega,
            "per_layer_probs": all_p,   # all probes retained for inspection
        }

        if labels is not None:
            eps = 1e-8
            labels_f   = labels.float()
            label_dist = labels_f / (labels_f.sum() + eps)

            if self.independent_layers or self.fusion is None:
                # Stage 1 loss: L_S1 = mean_l KL(y || p_l), no fusion term.
                loss_layer = torch.zeros((), device=label_dist.device)
                for p_l in active_p:
                    loss_layer = loss_layer + F.kl_div(
                        torch.log(p_l.clamp(min=eps)), label_dist, reduction="sum",
                    )
                loss_layer = loss_layer / self.num_active_probes
                result["loss_fuse"]            = torch.zeros_like(loss_layer)
                result["loss_layer"]           = loss_layer
                result["total_grounding_loss"] = loss_layer
            else:
                # Stage 2 loss: L_S2 = KL(y || p_final) + lambda * mean_l KL(y || p_l) (Eq. 4).
                loss_fuse = F.kl_div(
                    torch.log(p_final.clamp(min=eps)), label_dist, reduction="sum",
                )
                loss_layer = torch.zeros((), device=p_final.device)
                for p_l in active_p:
                    loss_layer = loss_layer + F.kl_div(
                        torch.log(p_l.clamp(min=eps)), label_dist, reduction="sum",
                    )
                loss_layer = loss_layer / self.num_active_probes
                result["loss_fuse"]            = loss_fuse
                result["loss_layer"]           = loss_layer
                result["total_grounding_loss"] = loss_fuse + self.lambda_layer * loss_layer

        return result


def gaussian_bbox_label(
    n_width: int,
    n_height: int,
    bbox_norm: Tuple[float, float, float, float],
    sigma_factor: float = 0.5,
) -> torch.Tensor:
    """
    Build the anisotropic-Gaussian supervision target of Eq. 2:

        y_i ~ exp( -(x_i - c_x)^2 / (2 sigma_x^2) - (y_i - c_y)^2 / (2 sigma_y^2) )
        sigma_x = eta * w_b,  sigma_y = eta * h_b

    centered at the bbox centroid (c_x, c_y), with sigma proportional to the
    bbox width/height (w_b, h_b) via a factor eta (sigma_factor). This
    reduces the over-penalization from patch-grid quantization on small
    targets relative to a hard binary bbox mask.

    Args:
      n_width, n_height: patch grid dimensions (N_vis = n_width * n_height).
      bbox_norm: (x1, y1, x2, y2) normalized to [0, 1].
      sigma_factor: eta in the formula above.

    Returns:
      [N_vis] tensor, normalized to sum to 1 (falls back to a one-hot label
      at the bbox center if sigma degenerates to ~0).
    """
    x1, y1, x2, y2 = bbox_norm
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w_b, h_b = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    sigma_x = max(sigma_factor * w_b, 1e-6)
    sigma_y = max(sigma_factor * h_b, 1e-6)

    xs = (torch.arange(n_width, dtype=torch.float32) + 0.5) / n_width    # [n_width]
    ys = (torch.arange(n_height, dtype=torch.float32) + 0.5) / n_height  # [n_height]
    gx = torch.exp(-((xs - cx) ** 2) / (2 * sigma_x ** 2))               # [n_width]
    gy = torch.exp(-((ys - cy) ** 2) / (2 * sigma_y ** 2))               # [n_height]
    grid = gy.unsqueeze(1) * gx.unsqueeze(0)                             # [n_height, n_width]
    label = grid.reshape(-1)                                             # [N_vis]

    total = label.sum()
    if not torch.isfinite(total) or total <= 1e-12:
        # Degenerate sigma (e.g. bbox smaller than a patch): fall back to a
        # one-hot label at the nearest patch to the bbox center.
        col = min(n_width - 1, max(0, int(cx * n_width)))
        row = min(n_height - 1, max(0, int(cy * n_height)))
        label = torch.zeros(n_width * n_height)
        label[row * n_width + col] = 1.0
        return label
    return label / total


# =============================================================================
# RetrofitModelMixin — shared model-agnostic logic
# =============================================================================

class RetrofitModelMixin:
    """
    Mixin providing layer-wise grounding-head capabilities to any frozen
    VLM backbone.

    Usage (schematically — see the full repository for concrete
    Qwen2.5-VL / Qwen3-VL subclasses):

      class MyRetrofitModel(RetrofitModelMixin, SomeBackboneForConditionalGeneration):
          def __init__(self, config, *args, **kwargs):
              super().__init__(config, *args, **kwargs)
              self._init_retrofit_from_config(config)
              self.post_init()

          def forward(self, ..., ground_token_indices=None, multi_patch_labels=None, ...):
              outputs = super().forward(..., output_hidden_states=True, return_dict=True)
              grounding_loss, scores, weights, anchors = self._compute_grounding_loss(
                  all_hidden_states=outputs.hidden_states,
                  input_ids=input_ids, logits=outputs.logits,
                  ground_token_indices=ground_token_indices,
                  multi_patch_labels=multi_patch_labels,
              )
              ...

    The backbone (`self.model`, `self.lm_head`, `self.visual`, etc. as
    defined by the concrete HuggingFace class you inherit from) is expected
    to be FROZEN (all `requires_grad_(False)`); only `self.layerwise_grounding_head`
    parameters receive gradient from `_compute_grounding_loss`.
    """

    def _init_retrofit_from_config(self, config) -> None:
        """Initialize the grounding head and retrofit state from model config."""
        probe_layers        = getattr(config, "probe_layers", [14, 18, 21, 24, 26, 27])
        active_probe_layers = getattr(config, "grounding_active_probe_layers", None)
        lambda_layer        = getattr(config, "grounding_lambda_layer", 0.2)
        independent_layers  = getattr(config, "grounding_independent_layers", False)
        attn_n_heads        = getattr(config, "grounding_attn_heads", 8)
        attn_d_head         = getattr(config, "grounding_attn_head_dim", 64)
        fusion_lora_rank    = getattr(config, "grounding_fusion_lora_rank", 128)
        fusion_context_rank = getattr(config, "grounding_fusion_context_rank", 64)
        fusion_learn_temp   = getattr(config, "grounding_fusion_learn_temperature", True)
        fusion_detach_q     = getattr(config, "grounding_fusion_detach_queries", True)

        self.layerwise_grounding_head = LayerWiseGroundingHead(
            d_model=config.hidden_size,
            probe_layers=probe_layers,
            active_probe_layers=active_probe_layers,
            lambda_layer=lambda_layer,
            independent_layers=independent_layers,
            attn_n_heads=attn_n_heads,
            attn_d_head=attn_d_head,
            fusion_lora_rank=fusion_lora_rank,
            fusion_context_rank=fusion_context_rank,
            fusion_learn_temperature=fusion_learn_temp,
            fusion_detach_queries=fusion_detach_q,
        )

        self.grounding_loss_weight: float = 1.0
        self.lm_loss_weight: float = 0.0

        self._ground_token_id: Optional[int] = getattr(config, "ground_token_id", None)
        self._pointer_start_token_id: Optional[int] = getattr(config, "pointer_start_token_id", None)
        vid = getattr(config, "vision_end_token_id", None)
        if vid is None:
            vid = getattr(config, "vision_token_id", None)
        self._vision_end_token_id: Optional[int] = vid
        self._anchor_source_counts: Dict[str, int] = {}

    def setup_special_token_ids(
        self,
        ground_token_id: int,
        pointer_start_token_id: int,
        vision_end_token_id: Optional[int] = None,
    ) -> None:
        """Register special token IDs needed by _find_ground_anchor()."""
        self._ground_token_id = ground_token_id
        self._pointer_start_token_id = pointer_start_token_id
        if vision_end_token_id is not None:
            self._vision_end_token_id = vision_end_token_id

    def reset_loss_weights(self, grounding_loss_weight: float, lm_loss_weight: float) -> None:
        self.grounding_loss_weight = grounding_loss_weight
        self.lm_loss_weight = lm_loss_weight

    def _zero_grounding_loss(self, device=None) -> torch.Tensor:
        """
        Returns 0.0 connected to every trainable grounding-head parameter,
        so that "skip" branches (e.g. no visual tokens found, malformed
        label) can still call backward() without breaking the autograd
        graph or upsetting DDP's parameter-participation bookkeeping.
        """
        loss = None
        for p in self.layerwise_grounding_head.parameters():
            if p.requires_grad:
                z = p.sum() * 0.0
                if device is not None:
                    z = z.to(device)
                loss = z if loss is None else loss + z
        if loss is None:
            dev = device or (next(self.parameters()).device if list(self.parameters()) else "cpu")
            loss = torch.zeros((), device=dev, requires_grad=True)
        return loss

    # ─────────────────────────────────────────────────────────────────────
    # Anchor token finder — this is what gives ZwerGe-UI its "prefill-forced,
    # zero label leakage" property described in the paper.
    # ─────────────────────────────────────────────────────────────────────

    def _find_ground_anchor(
        self,
        token_ids: torch.Tensor,
        external_hint: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[int, AnchorStrategy]:
        """
        Locate the anchor token whose hidden state is read by every probe.

        Priority:
          P0. external_hint                            -> EXTERNAL_HINT
          P1. last <|ground|> occurring AFTER <|vision_end|>  -> EXPLICIT_GROUND_TOKEN
          P2. last <|pointer_start|> AFTER <|vision_end|>, minus 1 -> BEFORE_POINTER_START
          P3. first token after <|vision_end|>          -> AFTER_VISION_END
          P4. last non-padding token                    -> LAST_NON_PAD (WARNING)

        At the P1 anchor, the model has processed the full screenshot and
        instruction but has NOT yet produced any coordinate token — this is
        the "prefill-forced inference" property from the paper that gives
        zero label leakage.
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
                        warnings.warn(f"Anchor P2: before pointer_start at pos {ptr_pos}.",
                                      UserWarning, stacklevel=3)
                    return ptr_pos - 1, AnchorStrategy.BEFORE_POINTER_START

        if vision_cut >= 0 and vision_cut + 1 < seq_len:
            if verbose:
                warnings.warn(f"Anchor P3: first token after vision_end at pos {vision_cut}.",
                              UserWarning, stacklevel=3)
            return vision_cut + 1, AnchorStrategy.AFTER_VISION_END

        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is not None:
            non_pad = (token_ids != pad_id).nonzero(as_tuple=False)
        else:
            non_pad = torch.arange(seq_len, device=token_ids.device).unsqueeze(1)

        if non_pad.numel() > 0:
            last_np = int(non_pad[-1].item())
            warnings.warn(f"Anchor P4: last non-pad token at pos {last_np}. LABEL LEAKAGE RISK!",
                          UserWarning, stacklevel=3)
            return last_np, AnchorStrategy.LAST_NON_PAD

        return seq_len - 1, AnchorStrategy.LAST_NON_PAD

    def _get_visual_indices(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Sequence positions of visual (image-patch) tokens (config.image_token_id)."""
        vis_mask = (token_ids == self.config.image_token_id)
        return vis_mask.nonzero(as_tuple=False).squeeze(-1)

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
        Single prefill-only forward pass with output_hidden_states=True.

        This is the "single forward pass" referenced throughout the paper:
        one call gives every layer's hidden states, from which every probe
        (Stage 1) and the fusion head (Stage 2) read their inputs — no
        extra backbone forward is needed per probe or per layer.

        Default implementation assumes a Qwen2.5-VL-style API
        (`self.model.embed_tokens`, `self.visual`, `self.model(...)`).
        Backbones with a different internal module layout (e.g. Qwen3-VL)
        should override this method to call their own
        `super().forward(output_hidden_states=True, ...)` instead — see the
        full repository's modeling_guiowl.py / modeling_uivenus.py for a
        worked example. The paper's numeric results were obtained with a
        single forward pass either way; the override is purely an
        engineering necessity for backbones that route image embedding
        differently.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pv = pixel_values.to(self.dtype)
            image_embeds = self.visual(pv, grid_thw=image_grid_thw)
            n_img_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_img_feats  = image_embeds.shape[0]
            if n_img_tokens != n_img_feats:
                warnings.warn(f"Image token mismatch: seq={n_img_tokens}, visual={n_img_feats}")
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = None
        if hasattr(self, "get_rope_index"):
            try:
                position_ids, _ = self.get_rope_index(input_ids, image_grid_thw, None, attention_mask)
            except Exception:
                position_ids = None

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
        logits: torch.FloatTensor,                             # [batch, seq_len, vocab]
        ground_token_indices: Optional[List[Optional[int]]],
        multi_patch_labels: Optional[List[Optional[torch.Tensor]]],
        verbose: bool = False,
    ) -> Tuple[Optional[torch.FloatTensor], List, List, List]:
        """
        Run the layer-wise grounding head over a batch. The backbone
        receives zero gradient from this call (only layerwise_grounding_head
        parameters are trainable, per RetrofitModelMixin's design).

        Returns:
          grounding_loss       — scalar tensor or None
          all_grounding_scores — list[Tensor|None]  (p_final per sample)
          all_layer_weights    — list[Tensor|None]  (omega per sample)
          all_anchor_positions — list[(int, AnchorStrategy)|None]
        """
        if multi_patch_labels is None:
            return None, [], [], []

        batch_size = input_ids.shape[0]
        grounding_losses: List[torch.Tensor] = []
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

            hint = ground_token_indices[i] if ground_token_indices is not None else None
            anchor_idx, anchor_strategy = self._find_ground_anchor(
                token_ids=token_ids_i, external_hint=hint, verbose=verbose,
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
                        print(f"[WARN] Sample {i}: label={sample_label.shape[0]} != N_vis={n_vis}, skipping")
                    grounding_losses.append(self._zero_grounding_loss(device=logits.device))
                    all_grounding_scores.append(None)
                    all_layer_weights.append(None)
                    continue

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


# =============================================================================
# Illustrative concrete subclass — Qwen2.5-VL backbone (e.g. UI-TARS-1.5-7B,
# GUI-Owl-7B). Lazily imports transformers so this module has no hard
# dependency on any specific backbone / transformers version.
# =============================================================================

def build_qwen25vl_retrofit_model_class():
    """
    Returns a RetrofitModelMixin + Qwen2_5_VLForConditionalGeneration class.

    This mirrors the paper's UI-TARS-1.5-7B / GUI-Owl-7B retrofit (both are
    built on the 28-layer Qwen2.5-VL). Call this AFTER installing
    `transformers` with Qwen2.5-VL support; see requirements.txt.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    class Qwen25VLRetrofitModel(RetrofitModelMixin, Qwen2_5_VLForConditionalGeneration):
        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            self._init_retrofit_from_config(config)
            self.post_init()

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            ground_token_indices: Optional[List[Optional[int]]] = None,
            multi_patch_labels: Optional[List[Optional[torch.Tensor]]] = None,
            verbose: bool = False,
            **_unused,
        ) -> Union[Tuple, BaseRetrofitOutput]:
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.visual.dtype)
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                    image_mask = (
                        (input_ids == self.config.image_token_id)
                        .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                    )
                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if position_ids is None:
                position_ids, _ = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
            )
            all_hidden_states = outputs.hidden_states
            logits = self.lm_head(outputs.last_hidden_state)

            lm_loss = None
            if labels is not None and self.lm_loss_weight > 0:
                shift_logits = logits[..., :-1, :].float().contiguous()
                shift_labels = labels[..., 1:].contiguous()
                lm_loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1).to(shift_logits.device),
                )

            grounding_loss, scores, weights, anchors = self._compute_grounding_loss(
                all_hidden_states=all_hidden_states, input_ids=input_ids, logits=logits,
                ground_token_indices=ground_token_indices,
                multi_patch_labels=multi_patch_labels, verbose=verbose,
            )

            total_loss = None
            if lm_loss is not None and grounding_loss is not None:
                total_loss = self.lm_loss_weight * lm_loss + self.grounding_loss_weight * grounding_loss
            elif grounding_loss is not None:
                total_loss = self.grounding_loss_weight * grounding_loss
            elif lm_loss is not None:
                total_loss = lm_loss

            if not return_dict:
                return (total_loss, logits) + outputs[1:] if total_loss is not None else (logits,) + outputs[1:]
            return BaseRetrofitOutput(
                lm_loss=lm_loss, grounding_loss=grounding_loss,
                grounding_scores=scores, layer_weights=weights,
                anchor_positions=anchors if multi_patch_labels is not None else [],
                loss=total_loss, logits=logits, past_key_values=outputs.past_key_values,
            )

    return Qwen25VLRetrofitModel


def build_qwen3vl_retrofit_model_class():
    """
    Returns a RetrofitModelMixin + Qwen3VLForConditionalGeneration class.

    This mirrors the paper's GUI-Owl-1.5-8B / UI-Venus-1.5-8B retrofit
    (both are built on the 36-layer Qwen3-VL). Qwen3-VL's official forward
    already handles DeepStack visual-feature injection internally, so the
    retrofit subclass simply calls `super().forward(output_hidden_states=True)`
    and reuses the exact same _compute_grounding_loss / grounding-head
    machinery as the Qwen2.5-VL subclass above.
    """
    from transformers import Qwen3VLForConditionalGeneration

    class Qwen3VLRetrofitModel(RetrofitModelMixin, Qwen3VLForConditionalGeneration):
        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            # Qwen3-VL keeps hidden_size / num_hidden_layers in config.text_config.
            text_cfg = getattr(config, "text_config", config)
            if not hasattr(config, "hidden_size"):
                config.hidden_size = getattr(text_cfg, "hidden_size", 4096)
            if not hasattr(config, "num_hidden_layers"):
                config.num_hidden_layers = getattr(text_cfg, "num_hidden_layers", 36)
            self._init_retrofit_from_config(config)
            if self._vision_end_token_id is None:
                self._vision_end_token_id = getattr(config, "vision_end_token_id", None)
            self.post_init()

        @torch.no_grad()
        def _forward_hidden_states_for_grounding(
            self, input_ids, attention_mask, pixel_values, image_grid_thw,
            device, mm_token_type_ids=None,
        ):
            outputs = super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                output_hidden_states=True, output_attentions=False,
                return_dict=True, use_cache=False,
            )
            return outputs.hidden_states

        def forward(
            self,
            input_ids=None, attention_mask=None, position_ids=None,
            past_key_values=None, inputs_embeds=None, labels=None,
            use_cache=None, return_dict=None, pixel_values=None,
            image_grid_thw=None, cache_position=None,
            ground_token_indices=None, multi_patch_labels=None,
            verbose=False, **extra_kwargs,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=None,
                use_cache=use_cache, output_attentions=False, output_hidden_states=True,
                return_dict=True, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                cache_position=cache_position, **extra_kwargs,
            )
            all_hidden_states = outputs.hidden_states
            logits = outputs.logits

            lm_loss = None
            if labels is not None and self.lm_loss_weight > 0:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                lm_loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1).to(shift_logits.device),
                )

            grounding_loss, scores, weights, anchors = self._compute_grounding_loss(
                all_hidden_states=all_hidden_states, input_ids=input_ids, logits=logits,
                ground_token_indices=ground_token_indices,
                multi_patch_labels=multi_patch_labels, verbose=verbose,
            )

            total_loss = None
            if lm_loss is not None and grounding_loss is not None:
                total_loss = self.lm_loss_weight * lm_loss + self.grounding_loss_weight * grounding_loss
            elif grounding_loss is not None:
                total_loss = self.grounding_loss_weight * grounding_loss
            elif lm_loss is not None:
                total_loss = lm_loss

            if not return_dict:
                return (total_loss, logits) if total_loss is not None else (logits,)
            return BaseRetrofitOutput(
                lm_loss=lm_loss, grounding_loss=grounding_loss,
                grounding_scores=scores, layer_weights=weights,
                anchor_positions=anchors if multi_patch_labels is not None else [],
                loss=total_loss, logits=logits, past_key_values=outputs.past_key_values,
                rope_deltas=getattr(outputs, "rope_deltas", None),
            )

    return Qwen3VLRetrofitModel


def get_retrofit_model_class(backbone_family: str):
    """
    Factory: backbone_family in {"qwen2_5_vl", "qwen3_vl"} ->
    a ready-to-use `RetrofitModelMixin`-based model class.

    Both factories are lazy (transformers is only imported when the
    requested class is actually built), so importing this module does not
    require any specific backbone to be installed.
    """
    if backbone_family == "qwen2_5_vl":
        return build_qwen25vl_retrofit_model_class()
    if backbone_family == "qwen3_vl":
        return build_qwen3vl_retrofit_model_class()
    raise ValueError(
        f"Unknown backbone_family {backbone_family!r}. "
        f"Choose from: ['qwen2_5_vl', 'qwen3_vl']."
    )
