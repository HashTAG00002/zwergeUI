"""CPU-only algebra/integration tests. No pretrained-model accuracy claim.
Usage: python test_paired_qk_svd.py --repo /path/to/zwergeUI
The exact CrossAttnGroundingProbe class is AST-extracted from audited source;
no Transformers import or model checkpoint is needed for these tests.
"""
from __future__ import annotations
import argparse
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional, Tuple
import unittest

import torch
from torch import nn

from paired_qk_svd import (
    mean_qk_factors, paired_qk_weights, initialize_a7_probes,
    match_probe_logit_rms, audit_hidden_state_contract,
)

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
args, rest = parser.parse_known_args()
source = Path(args.repo) / "zwerge/src/zwerge_retrofit/modeling_base.py"
tree = ast.parse(source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CrossAttnGroundingProbe")
namespace = {"torch": torch, "nn": nn, "math": math, "Optional": Optional, "Tuple": Tuple}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
Probe = namespace["CrossAttnGroundingProbe"]
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
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
