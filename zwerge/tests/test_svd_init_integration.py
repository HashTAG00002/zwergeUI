"""Integration tests for paired QK-SVD Stage-1 (A7) probe initialization.

- AST-extracts the real CrossAttnGroundingProbe from zwerge_retrofit/modeling_base.py
  (no Transformers import, no checkpoint download).
- Verifies initialize_a7_probes against the real probe class:
  donor mapping, kernel match ((1/(8*sqrt(d_head))) * W_q^T W_k vs donor kernel
  rank-r truncation), head_gate=0, q_ln/k_ln preserved at reinit values.
- Also includes the full original CPU algebra test suite from
  docs/oracle_new/svd/test_paired_qk_svd.py, adapted to import from the
  in-repo copy zwerge_retrofit/paired_qk_svd.py.

Run:  python zwerge/tests/test_svd_init_integration.py
"""
from __future__ import annotations

import ast
import math
import sys
import unittest
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "zwerge" / "src"
sys.path.insert(0, str(SRC_DIR))

from zwerge_retrofit.paired_qk_svd import (  # noqa: E402
    mean_qk_factors, paired_qk_weights, initialize_a7_probes,
    match_probe_logit_rms, audit_hidden_state_contract,
)

# ── AST-extract the REAL CrossAttnGroundingProbe (same technique as the
#    audited docs/oracle_new/svd/test_paired_qk_svd.py) ─────────────────────
_SOURCE = SRC_DIR / "zwerge_retrofit" / "modeling_base.py"
_TREE = ast.parse(_SOURCE.read_text())
_NODE = next(
    n for n in _TREE.body
    if isinstance(n, ast.ClassDef) and n.name == "CrossAttnGroundingProbe"
)
_NS = {"torch": torch, "nn": nn, "math": math, "Optional": Optional, "Tuple": Tuple}
exec(compile(ast.Module(body=[_NODE], type_ignores=[]), str(_SOURCE), "exec"), _NS)
Probe = _NS["CrossAttnGroundingProbe"]
torch.set_num_threads(2)


def weights():
    gen = torch.Generator().manual_seed(13)
    return (torch.randn(32, 32, dtype=torch.float64, generator=gen),
            torch.randn(8, 32, dtype=torch.float64, generator=gen))


def direct_kernel(q, k, hd=4):
    hq, hkv = q.shape[0] // hd, k.shape[0] // hd
    qh = q.reshape(hq, hd, -1)
    kh = k.reshape(hkv, hd, -1)
    return sum(qh[h].T @ kh[h // (hq // hkv)] for h in range(hq)) / (hq * math.sqrt(hd))


def probe_kernel(wq, wk, heads=2, hd=4):
    return wq.T @ wk / (heads * math.sqrt(hd))


class Block(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.head_dim = 4
        self.self_attn.q_proj = nn.Linear(d, 32, bias=False)
        self.self_attn.k_proj = nn.Linear(d, 8, bias=False)

    def forward(self, hidden_states):
        return hidden_states * 1.3 + 0.7


class ToyModel(nn.Module):
    def __init__(self, nested=False):
        super().__init__()
        decoder = nn.Module()
        decoder.layers = nn.ModuleList([Block() for _ in range(3)])
        decoder.norm = nn.LayerNorm(32)
        self.model = nn.Module() if nested else decoder
        if nested:
            self.model.language_model = decoder
        head = nn.Module()
        head.independent_layers = True
        head.adapter_type = "attn"
        head.probe_layers = [0, 2]
        head.probes = nn.ModuleList([Probe(32, 2, 4) for _ in range(2)])
        self.layerwise_grounding_head = head

    def states(self, x):
        dec = getattr(self.model, "language_model", self.model)
        hs = []
        for block in dec.layers:
            hs.append(x)
            x = block(x)
        hs.append(dec.norm(x))
        return tuple(hs)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: real probe class + Qwen2.5-VL-like donor geometry
# ─────────────────────────────────────────────────────────────────────────────

class Qwen25LikeBlock(nn.Module):
    """Qwen2.5-VL-like attention block: d_model=64, 8 q heads x dim 8,
    hkv=8 so the donor core is 64 wide = probe width (8 heads x 8 dim),
    i.e. zero truncation, mirroring Qwen2.5-VL's hkv*d0 = 4*128 = 512."""

    def __init__(self, d=64):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.head_dim = 8
        self.self_attn.q_proj = nn.Linear(d, 64, bias=False)
        self.self_attn.k_proj = nn.Linear(d, 64, bias=False)


class Qwen25LikeToyModel(nn.Module):
    """28-layer toy backbone with A7-style head probing layers [14, 20, 27]."""

    PROBE_LAYERS = [14, 20, 27]

    def __init__(self):
        super().__init__()
        decoder = nn.Module()
        decoder.layers = nn.ModuleList([Qwen25LikeBlock() for _ in range(28)])
        decoder.norm = nn.LayerNorm(64)
        self.model = decoder
        head = nn.Module()
        head.independent_layers = True
        head.adapter_type = "attn"
        head.probe_layers = list(self.PROBE_LAYERS)
        head.probes = nn.ModuleList(
            [Probe(64, 8, 8).double() for _ in self.PROBE_LAYERS]
        )
        self.layerwise_grounding_head = head


class SVDInitIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = Qwen25LikeToyModel().double()
        # Simulate reinit_grounding_head() values on the real probe class
        for probe in self.model.layerwise_grounding_head.probes:
            nn.init.xavier_uniform_(probe.W_q.weight, gain=0.02)
            nn.init.xavier_uniform_(probe.W_k.weight, gain=0.02)
            nn.init.zeros_(probe.head_gate)
            nn.init.ones_(probe.q_ln.weight)
            nn.init.zeros_(probe.q_ln.bias)
            nn.init.ones_(probe.k_ln.weight)
            nn.init.zeros_(probe.k_ln.bias)
        self.report = initialize_a7_probes(self.model, donor_mode="next",
                                           terminal_policy="same", device="cpu")

    def test_donor_mapping_next_with_terminal_fallback(self):
        donors = [info["donor_layer"] for info in self.report]
        self.assertEqual(donors, [15, 21, 27])
        self.assertEqual([info["probe_layer"] for info in self.report], [14, 20, 27])
        self.assertFalse(self.report[0]["terminal_same_layer_fallback"])
        self.assertFalse(self.report[1]["terminal_same_layer_fallback"])
        self.assertTrue(self.report[2]["terminal_same_layer_fallback"])

    def test_kernel_matches_donor_truncation(self):
        head = self.model.layerwise_grounding_head
        layers = self.model.model.layers
        for info, probe in zip(self.report, head.probes):
            donor = layers[info["donor_layer"]].self_attn
            m = direct_kernel(donor.q_proj.weight, donor.k_proj.weight, hd=8)
            # rank-512 truncation analog: here core width == probe width == 64,
            # so the rank-r truncation is exact (Qwen2.5 zero-truncation case)
            self.assertAlmostEqual(info["retained_energy_fraction"], 1.0)
            self.assertAlmostEqual(info["relative_frobenius_tail"], 0.0)
            kernel = probe_kernel(probe.W_q.weight, probe.W_k.weight, heads=8, hd=8)
            torch.testing.assert_close(kernel, m)

    def test_gate_and_layernorms_preserved(self):
        for probe in self.model.layerwise_grounding_head.probes:
            torch.testing.assert_close(probe.head_gate,
                                       torch.zeros_like(probe.head_gate))
            torch.testing.assert_close(probe.q_ln.weight,
                                       torch.ones_like(probe.q_ln.weight))
            torch.testing.assert_close(probe.q_ln.bias,
                                       torch.zeros_like(probe.q_ln.bias))
            torch.testing.assert_close(probe.k_ln.weight,
                                       torch.ones_like(probe.k_ln.weight))
            torch.testing.assert_close(probe.k_ln.bias,
                                       torch.zeros_like(probe.k_ln.bias))

    def test_report_sources_point_to_donors(self):
        for info, expect_donor in zip(self.report, [15, 21, 27]):
            self.assertEqual(
                info["q_source"],
                f"model.layers.{expect_donor}.self_attn.q_proj.weight")
            self.assertEqual(
                info["k_source"],
                f"model.layers.{expect_donor}.self_attn.k_proj.weight")


# ─────────────────────────────────────────────────────────────────────────────
# Original audited algebra/integration suite (docs/oracle_new/svd/
# test_paired_qk_svd.py), imported here so one file runs everything.
# ─────────────────────────────────────────────────────────────────────────────

class SVDTests(unittest.TestCase):
    def test_gqa_factorization(self):
        q, k = weights()
        a, b, c = mean_qk_factors(q, k, 4)
        torch.testing.assert_close(c * a.T @ b, direct_kernel(q, k))

    def test_full_reconstruction(self):
        q, k = weights()
        wq, wk, info = paired_qk_weights(q, k, 4, 2, 4)
        torch.testing.assert_close(probe_kernel(wq, wk), direct_kernel(q, k))
        self.assertAlmostEqual(info["retained_energy_fraction"], 1.0)

    def test_truncated_optimum(self):
        q, k = weights()
        wq, wk, info = paired_qk_weights(q, k, 4, 2, 2)
        m = direct_kernel(q, k)
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        mr = (u[:, :4] * s[:4]) @ vh[:4]
        torch.testing.assert_close(probe_kernel(wq, wk, 2, 2), mr)
        measured = torch.linalg.norm(m - mr) / torch.linalg.norm(m)
        self.assertAlmostEqual(float(measured), info["relative_frobenius_tail"])

    def test_actual_probe_forward_and_gradients(self):
        q, k = weights()
        wq, wk, _ = paired_qk_weights(q, k, 4, 2, 4)
        p = Probe(32, 2, 4).double()
        with torch.no_grad():
            p.W_q.weight.copy_(wq)
            p.W_k.weight.copy_(wk)
        x, y = torch.randn(32, dtype=torch.float64), torch.randn(9, 32, dtype=torch.float64)
        prob, logits, ql = p(x, y, None, None, 32)
        xn = p.q_ln(x / (x.norm() / math.sqrt(32)))
        yn = p.k_ln(y / (y.norm(dim=-1, keepdim=True) / math.sqrt(32)))
        xn = xn / (xn.norm() / math.sqrt(32))
        yn = yn / (yn.norm(dim=-1, keepdim=True) / math.sqrt(32))
        expected = xn @ direct_kernel(q, k) @ yn.T
        torch.testing.assert_close(logits, expected)
        self.assertEqual(ql.numel(), 8)
        self.assertAlmostEqual(float(prob.sum().detach()), 1.0)
        (logits.square().mean()).backward()
        self.assertGreater(float(p.W_q.weight.grad.norm()), 0)
        self.assertGreater(float(p.W_k.weight.grad.norm()), 0)

    def test_orientation_control(self):
        q, k = weights()
        a, b, _ = paired_qk_weights(q, k, 4, 2, 4)
        c, d, _ = paired_qk_weights(q, k, 4, 2, 4, random_orientation_seed=5)
        torch.testing.assert_close(torch.linalg.svdvals(probe_kernel(a, b)),
                                   torch.linalg.svdvals(probe_kernel(c, d)))
        self.assertGreater(float((probe_kernel(a, b) - probe_kernel(c, d)).norm()), 1.0)

    def test_logit_gain(self):
        q, k = weights()
        a, b, _ = paired_qk_weights(q, k, 4, 2, 4, logit_gain=0.3)
        torch.testing.assert_close(probe_kernel(a, b), 0.3 * direct_kernel(q, k))

    def test_scale_matching(self):
        p = Probe(32, 2, 4).double()
        samples = [(torch.randn(32).double(), torch.randn(10, 32).double()) for _ in range(3)]
        match_probe_logit_rms(p, samples, 0.5)
        vals = []
        for x, y in samples:
            logits = p(x, y, None, None, 32)[1]
            vals.append((logits - logits.mean()).square().mean())
        self.assertAlmostEqual(float(torch.stack(vals).mean().sqrt().detach()), 0.5, places=7)

    def test_model_paths_and_frozen_donors(self):
        for nested in (False, True):
            model = ToyModel(nested)
            before = {n: p.clone() for n, p in model.named_parameters() if not n.startswith("layerwise")}
            report = initialize_a7_probes(model)
            self.assertEqual(report[0]["donor_layer"], 1)
            self.assertTrue(report[1]["terminal_same_layer_fallback"])
            for n, p in model.named_parameters():
                if n in before:
                    torch.testing.assert_close(p, before[n], rtol=0, atol=0)

    def test_refuse_a8(self):
        model = ToyModel()
        model.layerwise_grounding_head.independent_layers = False
        with self.assertRaises(ValueError):
            initialize_a7_probes(model)

    def test_endpoint_error_no_changes(self):
        model = ToyModel()
        before = {n: p.clone() for n, p in model.named_parameters()}
        with self.assertRaises(ValueError):
            initialize_a7_probes(model, terminal_policy="error")
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)

    def test_hidden_state_endpoint_audit(self):
        model = ToyModel(True)
        x = torch.randn(1, 5, 32)
        result = audit_hidden_state_contract(model, lambda: model.states(x), [0, 2], [0, 2])
        self.assertEqual(result["layers"]["0"]["next_input_0"]["max_abs_error"], 0)
        self.assertEqual(result["layers"]["2"]["final_norm_output"]["max_abs_error"], 0)
        self.assertGreater(result["layers"]["2"]["output_2"]["max_abs_error"], 0)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_invalid_or_zero_donors(self):
        with self.assertRaises(ValueError):
            paired_qk_weights(torch.zeros(32, 32), torch.zeros(8, 32), 4, 2, 4)
        with self.assertRaises(ValueError):
            paired_qk_weights(torch.randn(7, 32), torch.randn(8, 32), 4, 2, 4)
        with self.assertRaises(ValueError):
            paired_qk_weights(torch.randn(32, 32), torch.randn(8, 32), 4, 4, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
