"""Experimental initialization for ZwerGe A7 CrossAttnGroundingProbe.

Audited source revision: 9329b271709aabceb54ca16224f75733116a9ee0.
This is NOT an implementation of TMEM. It compresses the *mean pre-RoPE,
weight-only QK content kernel*. It deliberately leaves the ZwerGe probe's
normalization and trainable parameterization unchanged. It does not reproduce
native attention (biases, RMSNorm, QK normalization, RoPE, and per-head softmax
are not included).

No downloads or checkpoint writes are performed by this module.
Compute once outside the training loop; save initialized weights and reuse
that artifact across ranks/runs. Do not initialize trained/resumed A7/A8 heads.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn


def decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """Resolve text decoder layers, never the vision encoder's blocks."""
    for path in ("model.language_model.layers", "model.layers", "language_model.layers"):
        cur: Any = model
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if isinstance(cur, nn.ModuleList):
            return path, cur
    raise ValueError("Cannot find supported Qwen text decoder layers.")


def _working_copy(weight: Tensor, device: str | torch.device) -> Tensor:
    if weight.is_meta or weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("Expected a materialized, dense floating-point weight matrix.")
    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    result = weight.detach().to(device=device, dtype=dtype)
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite donor weights.")
    return result


@torch.no_grad()
def mean_qk_factors(
    q_weight: Tensor, k_weight: Tensor, native_head_dim: int,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor, float]:
    """Return A, B, c such that M = c A.T B is mean native content logits.

    M = sum_h Q_h.T K_g(h) / (Hq * sqrt(native_head_dim)).
    GQA head assignment is the standard repeat_kv contiguous grouping.
    A averages query weights within each KV group instead of expanding K.
    Both A and B have shape [Hkv * native_head_dim, hidden_size].
    """
    if native_head_dim <= 0:
        raise ValueError("native_head_dim must be positive.")
    q = _working_copy(q_weight, device)
    k = _working_copy(k_weight, device).to(q.dtype)
    if q.shape[1] != k.shape[1]:
        raise ValueError("Q and K must consume the same hidden dimension.")
    if q.shape[0] % native_head_dim or k.shape[0] % native_head_dim:
        raise ValueError("Projection rows must be divisible by native_head_dim.")
    hq, hkv = q.shape[0] // native_head_dim, k.shape[0] // native_head_dim
    if hq < hkv or hq % hkv:
        raise ValueError("Unsupported GQA head grouping.")
    group = hq // hkv
    a = q.reshape(hkv, group, native_head_dim, q.shape[1]).mean(dim=1)
    return a.reshape(-1, q.shape[1]), k, 1.0 / (hkv * math.sqrt(native_head_dim))


@torch.no_grad()
def paired_qk_weights(
    q_weight: Tensor, k_weight: Tensor, native_head_dim: int,
    probe_heads: int = 8, probe_head_dim: int = 64,
    *, device: str | torch.device = "cpu", logit_gain: float = 1.0,
    random_orientation_seed: int | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Build [probe_heads * probe_head_dim, d] paired initial projections.

    Exact QR + economy SVD computes the singular system of M without making
    its d-by-d matrix. For GQA, the core is often only 512 or 1024 wide.
    At uniform head_gate, the resulting probe kernel is logit_gain * M_r.

    random_orientation_seed preserves the kernel singular values while
    replacing its left/right singular vectors with random orthonormal bases.
    This is a control, not a trained method; use the same logit scale controls.
    """
    if probe_heads <= 0 or probe_head_dim <= 0:
        raise ValueError("Probe head dimensions must be positive.")
    if not math.isfinite(logit_gain) or logit_gain <= 0:
        raise ValueError("logit_gain must be finite and positive.")
    a, b, native_scale = mean_qk_factors(q_weight, k_weight, native_head_dim, device)
    d = a.shape[1]
    r = probe_heads * probe_head_dim
    if r > d:
        raise ValueError("This implementation requires probe projection width <= hidden size.")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    qb, rb = torch.linalg.qr(b.T, mode="reduced")
    core = (ra @ rb.T) * native_scale
    uc, s, vhc = torch.linalg.svd(core, full_matrices=False)
    if r > s.numel():
        raise ValueError("Probe width exceeds donor factor dimension; zero-padding BOTH factors would create dead rows.")
    kept = r
    if float(s.square().sum()) == 0.0:
        raise ValueError("Donor kernel is zero; two zero projections would have zero gradients.")
    u = qa @ uc[:, :kept]
    v = qb @ vhc[:kept].T
    # Canonicalize paired signs (scores are sign-invariant, fusion queries are not).
    pivot = u.abs().argmax(dim=0)
    sign = torch.sign(u[pivot, torch.arange(kept, device=u.device)])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    u, v = u * sign, v * sign
    if random_orientation_seed is not None:
        # CPU generator avoids advancing the caller's model/training RNG state.
        gen = torch.Generator(device="cpu").manual_seed(random_orientation_seed)
        ru = torch.randn(d, kept, generator=gen, dtype=a.dtype).to(a.device)
        rv = torch.randn(d, kept, generator=gen, dtype=a.dtype).to(a.device)
        u = torch.linalg.qr(ru, mode="reduced")[0]
        v = torch.linalg.qr(rv, mode="reduced")[0]
    # Probe logits are sum_i <Q_i,K_i> / (probe_heads * sqrt(probe_head_dim)).
    compensation = probe_heads * math.sqrt(probe_head_dim) * logit_gain
    root = (s[:kept] * compensation).clamp_min(0).sqrt()
    wq = torch.zeros(r, d, dtype=a.dtype, device=a.device)
    wk = torch.zeros_like(wq)
    wq[:kept] = root[:, None] * u.T
    wk[:kept] = root[:, None] * v.T
    # Distribute large singular components across probe heads. This changes
    # neither Wq.T @ Wk nor the initial score with uniform gates.
    order = torch.arange(r, device=a.device).reshape(probe_head_dim, probe_heads).T.flatten()
    wq, wk = wq[order].contiguous(), wk[order].contiguous()
    energy = s.square().sum()
    info = {
        "source": "mean_pre_rope_weight_only_qk_kernel",
        "native_head_dim": native_head_dim,
        "native_query_heads": q_weight.shape[0] // native_head_dim,
        "native_kv_heads": k_weight.shape[0] // native_head_dim,
        "probe_projection_width": r,
        "core_shape": list(core.shape),
        "retained_energy_fraction": float(s[:kept].square().sum() / energy),
        "relative_frobenius_tail": float((s[kept:].square().sum() / energy).sqrt()),
        "logit_gain": logit_gain,
        "random_orientation_seed": random_orientation_seed,
        "omitted_operations": ["bias", "input_norm", "qk_norm", "rope", "native_head_softmax"],
    }
    return wq, wk, info


@torch.no_grad()
def initialize_a7_probes(
    model: nn.Module, *, donor_mode: str = "next", terminal_policy: str = "same",
    device: str | torch.device = "cpu", logit_gain: float = 1.0,
    random_orientation_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Copy an initialization into a freshly reset A7 model, in place.

    Call AFTER setup_special_token_ids(... reinit_grounding_head=True),
    BEFORE optimizer creation. Existing LayerNorm and trainable flags are kept.
    A8/non-independent heads are rejected to avoid replacing trained probes.
    No base-model weights are changed. Last-layer reuse is explicitly reported.
    """
    head = getattr(model, "layerwise_grounding_head", None)
    if head is None or getattr(head, "adapter_type", None) != "attn":
        raise ValueError("Requires the A7/A8 attn probe architecture.")
    if not getattr(head, "independent_layers", False):
        raise ValueError("Initialize fresh independent A7 probes only, not A8 fusion runs.")
    if donor_mode not in {"same", "next"}:
        raise ValueError("donor_mode must be 'same' or 'next'.")
    if terminal_policy not in {"same", "error"}:
        raise ValueError("terminal_policy must be 'same' or 'error'.")
    prefix, layers = decoder_layers(model)
    if donor_mode == "next" and terminal_policy == "error" and len(layers) - 1 in head.probe_layers:
        raise ValueError("Final probe has no next attention layer; no weights have been changed.")
    report: list[dict[str, Any]] = []
    for probe_i, ell in enumerate(head.probe_layers):
        donor = ell if donor_mode == "same" else ell + 1
        fallback = donor >= len(layers)
        if fallback:
            if terminal_policy == "error":
                raise ValueError(f"Layer {ell} has no next attention layer. Treat endpoint separately.")
            donor = ell
        if not 0 <= donor < len(layers):
            raise ValueError(f"Invalid donor layer: {donor}")
        attn = getattr(layers[donor], "self_attn", None)
        if attn is None or not hasattr(attn, "q_proj") or not hasattr(attn, "k_proj"):
            raise ValueError(f"{prefix}.{donor} is not a supported dense attention block.")
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            cfg = getattr(attn, "config", None)
            head_dim = getattr(cfg, "head_dim", None)
            if head_dim is None and cfg is not None:
                head_dim = cfg.hidden_size // cfg.num_attention_heads
        if head_dim is None:
            raise ValueError("Cannot determine the native attention head dimension.")
        probe = head.probes[probe_i]
        seed = None if random_orientation_seed is None else random_orientation_seed + ell
        wq, wk, info = paired_qk_weights(
            attn.q_proj.weight, attn.k_proj.weight, int(head_dim),
            probe.n_heads, probe.d_head, device=device, logit_gain=logit_gain,
            random_orientation_seed=seed,
        )
        if wq.shape != probe.W_q.weight.shape or wk.shape != probe.W_k.weight.shape:
            raise ValueError("Source factorization does not match the probe shapes.")
        probe.W_q.weight.copy_(wq.to(probe.W_q.weight))
        probe.W_k.weight.copy_(wk.to(probe.W_k.weight))
        probe.head_gate.zero_()
        info.update({
            "probe_index": probe_i, "hidden_state_index": ell + 1,
            "probe_layer": ell, "donor_layer": donor,
            "terminal_same_layer_fallback": fallback,
            "q_source": f"{prefix}.{donor}.self_attn.q_proj.weight",
            "k_source": f"{prefix}.{donor}.self_attn.k_proj.weight",
            "source_has_qk_norm": hasattr(attn, "q_norm") or hasattr(attn, "k_norm"),
            "source_has_bias": attn.q_proj.bias is not None or attn.k_proj.bias is not None,
        })
        report.append(info)
    return report


@torch.no_grad()
def match_probe_logit_rms(
    probe: nn.Module, examples: Sequence[tuple[Tensor, Tensor]], target_rms: float,
) -> dict[str, float]:
    """Match centered score RMS using TRAIN-split states only; no bbox labels.

    Examples are (anchor_hidden[d], visual_hidden[n,d]) in the actual probe
    input convention. Recompute states if trainable anchor embeddings change.
    A common target is an experimental hyperparameter, not a calibration claim.
    """
    if not examples or not math.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("Need examples and a positive target_rms.")
    squares = []
    for query, visual in examples:
        query = query.to(probe.W_q.weight)
        visual = visual.to(probe.W_k.weight)
        if visual.shape[0] < 2:
            raise ValueError("Need at least two visual tokens to measure centered logits.")
        _, logits, _ = probe(query, visual, None, None, query.shape[-1])
        centered = logits.double() - logits.double().mean()
        squares.append(centered.square().mean().cpu())
    current = float(torch.stack(squares).mean().sqrt())
    if not math.isfinite(current) or current <= 1e-12:
        raise ValueError("Logits are effectively constant; scaling cannot supply missing signal.")
    factor = math.sqrt(target_rms / current)
    probe.W_q.weight.mul_(factor)
    probe.W_k.weight.mul_(factor)
    return {"before_centered_logit_rms": current, "target_rms": target_rms,
            "projection_scale_factor": factor}


@torch.no_grad()
def audit_hidden_state_contract(
    model: nn.Module, forward_fn: Callable[[], Sequence[Tensor]],
    token_positions: Sequence[int], layer_ids: Sequence[int],
) -> dict[str, Any]:
    """Compare hs[ell+1] to actual decoder outputs, inputs, and final norm.

    forward_fn must return the hidden-state tuple (e.g. call the model's
    _forward_hidden_states_for_grounding with one prepared sample).
    This audit runs a complete forward, never aborts it with a hook exception.
    Only selected positions in batch item zero are copied to CPU.
    """
    if not token_positions or not layer_ids:
        raise ValueError("Specify sampled token positions and decoder layer IDs.")
    prefix, layers = decoder_layers(model)
    norm_path = prefix.rsplit(".", 1)[0] + ".norm"
    norm = model.get_submodule(norm_path)
    saved: dict[str, Tensor] = {}
    handles = []

    def sample(value: Any) -> Tensor:
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not isinstance(value, Tensor):
            raise TypeError("Expected a decoder hidden-state tensor.")
        if value.ndim == 3:
            value = value[0]
        if value.ndim != 2:
            raise ValueError("Expected [batch,seq,d] or [seq,d].")
        return value[list(token_positions)].detach().float().cpu().clone()

    def out_hook(key: str):
        def hook(module, args, output):
            saved[key] = sample(output)
        return hook

    def in_hook(key: str):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            saved[key] = sample(hidden)
        return hook

    try:
        for ell in layer_ids:
            if not 0 <= ell < len(layers):
                raise ValueError(f"Invalid layer {ell}")
            handles.append(layers[ell].register_forward_hook(out_hook(f"output_{ell}")))
            if ell + 1 < len(layers):
                handles.append(layers[ell + 1].register_forward_pre_hook(
                    in_hook(f"next_input_{ell}"), with_kwargs=True))
        handles.append(norm.register_forward_hook(out_hook("final_norm_output")))
        hidden = forward_fn()
        if hidden is None or len(hidden) != len(layers) + 1:
            raise ValueError("Hidden states missing or unexpected length: check runtime Transformers version.")
        result: dict[str, Any] = {"decoder_path": prefix, "tuple_length": len(hidden), "layers": {}}
        for ell in layer_ids:
            h = sample(hidden[ell + 1])
            keys = [f"output_{ell}"]
            keys += [f"next_input_{ell}"] if ell + 1 < len(layers) else ["final_norm_output"]
            result["layers"][str(ell)] = {
                key: {"max_abs_error": float((h - saved[key]).abs().max()),
                      "relative_l2_error": float((h - saved[key]).norm() / saved[key].norm().clamp_min(1e-12))}
                for key in keys
            }
        return result
    finally:
        for handle in handles:
            handle.remove()
