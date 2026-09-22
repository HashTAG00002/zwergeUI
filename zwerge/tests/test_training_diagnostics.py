"""Regression/convention tests for training-time observability (P0/P1 subset).

What is covered (all CPU, no transformers import, no model weights, no W&B):
  1. Log convention fix: custom metrics are window SUM / actual COUNT
     (constant loss 2.0, GA=4, logging windows 20 / 5 / trailing short → 2.0;
     the pre-fix formula window_sum/GA → 40/10 is asserted as the documented
     counterexample).
  2. Model loss (zero-skip diluted, unchanged) vs diag/grounding_valid_mean
     (valid samples only) + data/grounding_valid_frac.
  3. Non-finite handling: counted, never folded into means as 0, the window
     mean is omitted (unavailable).
  4. Per-layer probe metrics keyed by REAL layer number + taint semantics.
  5. DDP aggregation correctness with 2 REAL gloo processes: SUM for
     numerators/counts, MAX for peaks, taints propagate across ranks.
  6. Non-intrusion: real LayerWiseGroundingHead loss+gradients are
     bit-identical with collect_diagnostics on/off; the refactored KL
     bookkeeping matches the old arithmetic bit-for-bit.
  7. Ports of the applicable items from
     docs/oracle_new/w&b/audit_repro_tests.py (uniform-ref KL, entropy
     normalization, impulse max, floor toy, zero-skip dilution contract).

Run:  python zwerge/tests/test_training_diagnostics.py
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_SRC = REPO_ROOT / "zwerge" / "src" / "zwerge_retrofit" / "trainer.py"
BASE_SRC = REPO_ROOT / "zwerge" / "src" / "zwerge_retrofit" / "modeling_base.py"
torch.set_num_threads(2)


# ─────────────────────────────────────────────────────────────────────────────
# AST extraction (same technique as docs/oracle_new/w&b/audit_repro_tests.py):
# run the REAL source without importing transformers.
# ─────────────────────────────────────────────────────────────────────────────

def _extract(path: Path, class_specs=None, func_names=None, const_names=None):
    tree = ast.parse(path.read_text())
    nodes = []
    for n in tree.body:
        if class_specs and isinstance(n, ast.ClassDef) and n.name in class_specs:
            orig, methods, new_name, bases = class_specs[n.name]
            node = copy.deepcopy(n)
            if methods is not None:
                node.body = [
                    b for b in node.body
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and b.name in methods
                ]
            node.name = new_name or orig
            node.bases = [
                copy.deepcopy(ast.parse(b, mode="eval").body)
                for b in (bases or [])
            ]
            nodes.append(node)
        elif func_names and isinstance(n, ast.FunctionDef) and n.name in func_names:
            nodes.append(copy.deepcopy(n))
        elif const_names and isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in const_names:
                    nodes.append(copy.deepcopy(n))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    return ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))


class LogSink:
    def log(self, logs, start_time=None):
        self.last_logs = dict(logs)


_NS = {
    "torch": torch, "nn": nn, "F": F, "math": math, "dist": dist,
    "Optional": Optional, "Dict": Dict, "List": List, "Tuple": Tuple,
    "LogSink": LogSink,
}
exec(compile(_extract(
    TRAINER_SRC,
    class_specs={"RetrofitTrainer": ("RetrofitTrainer", {"compute_loss", "log"}, "AuditTrainer", ["LogSink"])},
    func_names={"_build_window_payloads", "_window_allreduce", "_emit_window_logs"},
    const_names={"_WINDOW_GLOBAL_MEAN_KEYS", "_WINDOW_LAYER_METRICS",
                 "_WINDOW_INT_KEYS", "_WINDOW_MAX_KEYS"},
), str(TRAINER_SRC), "exec"), _NS)
AuditTrainer = _NS["AuditTrainer"]
_build_window_payloads = _NS["_build_window_payloads"]
_window_allreduce = _NS["_window_allreduce"]
_emit_window_logs = _NS["_emit_window_logs"]

exec(compile(_extract(
    BASE_SRC,
    class_specs={
        "MLP2": ("MLP2", None, "MLP2", ["nn.Module"]),
        "LayerLoRAAdapter": ("LayerLoRAAdapter", None, "LayerLoRAAdapter", ["nn.Module"]),
        "LayerGroundingProbe": ("LayerGroundingProbe", None, "LayerGroundingProbe", ["nn.Module"]),
        "CrossAttnGroundingProbe": ("CrossAttnGroundingProbe", None, "CrossAttnGroundingProbe", ["nn.Module"]),
        "ContextLoRACosMetaFusion": ("ContextLoRACosMetaFusion", None, "ContextLoRACosMetaFusion", ["nn.Module"]),
        "LayerWiseGroundingHead": ("LayerWiseGroundingHead", None, "LayerWiseGroundingHead", ["nn.Module"]),
        "RetrofitModelMixin": ("RetrofitModelMixin", {"_compute_grounding_loss"}, "AuditMixin", []),
    },
    func_names={"probe_output_stats"},
), str(BASE_SRC), "exec"), _NS)
Probe = _NS["CrossAttnGroundingProbe"]
Head = _NS["LayerWiseGroundingHead"]
AuditMixin = _NS["AuditMixin"]
probe_output_stats = _NS["probe_output_stats"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_trainer(ga=4):
    t = AuditTrainer()
    t.args = SimpleNamespace(gradient_accumulation_steps=ga)
    return t


def _const_out(value):
    return SimpleNamespace(
        loss=torch.tensor(value), grounding_loss=torch.tensor(value),
        lm_loss=None, grounding_scores=None, layer_weights=None,
    )


class _FakeDiagModel:
    """Callable returning fixed outputs + a structured _grounding_diag."""

    def __init__(self, out, diag):
        self._out = out
        self._grounding_diag = diag

    def __call__(self, **kwargs):
        return self._out


def _diag(batch_size=2, valid_count=1, valid_loss_sum=10.0, urk_sum=3.0, layers=None):
    return {
        "batch_size": batch_size,
        "valid_count": valid_count,
        "valid_loss_sum": valid_loss_sum,
        "valid_loss_finite_count": valid_count,
        "valid_mean_tainted": 0,
        "uniform_ref_kl_sum": urk_sum,
        "uniform_ref_kl_count": valid_count,
        "nonfinite_forward_count": 0,
        "layers": layers or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Log convention: window SUM/COUNT
# ─────────────────────────────────────────────────────────────────────────────

class LogConventionTests(unittest.TestCase):
    def test_constant_loss_window_invariant(self):
        # constant loss 2.0, GA=4: full window 20, short window 5, trailing 3
        for window_updates in (20, 5, 3):
            trainer = _make_trainer(ga=4)
            out = _const_out(2.0)
            model = lambda **kw: out
            for _ in range(window_updates * 4):
                trainer.compute_loss(model, {})
            trainer.log({})
            self.assertAlmostEqual(trainer.last_logs["grounding_loss"], 2.0)
            self.assertAlmostEqual(trainer.last_logs["diag/loss_model_mean"], 2.0)
            self.assertAlmostEqual(trainer.last_logs["diag/loss_micro_max"], 2.0)

    def test_old_formula_counterexample(self):
        # Pre-fix convention: window_sum / gradient_accumulation_steps.
        # With constant loss c, GA=A, K updates per window: K*A*c/A = K*c.
        c, a = 2.0, 4
        self.assertEqual((c * 20 * a) / a, 40.0)   # old logged value at logging_steps=20
        self.assertEqual((c * 5 * a) / a, 10.0)    # old logged value at logging_steps=5
        self.assertNotAlmostEqual(40.0, c)

    def test_trailing_short_window_after_full_window(self):
        trainer = _make_trainer(ga=4)
        out = _const_out(2.0)
        model = lambda **kw: out
        for _ in range(20 * 4):
            trainer.compute_loss(model, {})
        trainer.log({})
        self.assertAlmostEqual(trainer.last_logs["grounding_loss"], 2.0)
        # trailing short window: 3 updates = 12 microbatches
        for _ in range(3 * 4):
            trainer.compute_loss(model, {})
        trainer.log({})
        self.assertAlmostEqual(trainer.last_logs["grounding_loss"], 2.0)
        self.assertAlmostEqual(trainer.last_logs["diag/loss_model_mean"], 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2/3/4. valid vs model loss, non-finite, per-layer metrics
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticsPayloadTests(unittest.TestCase):
    def test_valid_mean_excludes_zero_skip(self):
        # batch of 2: one valid sample (loss 10), one zero-skip placeholder.
        # Model loss (unchanged convention) = 5; valid-only mean = 10.
        trainer = _make_trainer()
        out = _const_out(5.0)
        model = _FakeDiagModel(out, _diag())
        trainer.compute_loss(model, {})
        trainer.log({})
        logs = trainer.last_logs
        self.assertAlmostEqual(logs["grounding_loss"], 5.0)          # model convention kept
        self.assertAlmostEqual(logs["diag/grounding_valid_mean"], 10.0)
        self.assertAlmostEqual(logs["data/grounding_valid_frac"], 0.5)
        self.assertAlmostEqual(logs["data/uniform_ref_kl_mean"], 3.0)

    def test_layer_metrics_use_real_layer_numbers(self):
        trainer = _make_trainer()
        layers = {18: {"kl_sum": 6.0, "logit_rms_sum": 4.0, "entropy_sum": 1.5,
                       "gt_floor_mass_sum": 0.5, "count": 2, "nonfinite": 0}}
        model = _FakeDiagModel(_const_out(5.0), _diag(layers=layers))
        trainer.compute_loss(model, {})
        trainer.log({})
        logs = trainer.last_logs
        self.assertAlmostEqual(logs["probe/L18/kl_mean"], 3.0)
        self.assertAlmostEqual(logs["probe/L18/logit_rms_centered"], 2.0)
        self.assertAlmostEqual(logs["probe/L18/entropy_norm"], 0.75)
        self.assertAlmostEqual(logs["probe/L18/gt_floor_mass"], 0.25)

    def test_nonfinite_model_loss_marks_window_unavailable(self):
        trainer = _make_trainer()
        out = _const_out(5.0)
        out.loss = torch.tensor(float("nan"))
        model = lambda **kw: out
        trainer.compute_loss(model, {})
        trainer.log({})
        logs = trainer.last_logs
        self.assertNotIn("diag/loss_model_mean", logs)   # unavailable, not 0
        self.assertNotIn("diag/loss_micro_max", logs)
        self.assertEqual(logs["diag/nonfinite_forward_count"], 1.0)
        self.assertAlmostEqual(logs["grounding_loss"], 5.0)  # finite key unaffected

    def test_layer_taint_omits_only_that_layer(self):
        trainer = _make_trainer()
        layers = {
            18: {"kl_sum": 0.0, "logit_rms_sum": 0.0, "entropy_sum": 0.0,
                 "gt_floor_mass_sum": 0.0, "count": 0, "nonfinite": 1},
            20: {"kl_sum": 4.0, "logit_rms_sum": 2.0, "entropy_sum": 1.0,
                 "gt_floor_mass_sum": 0.0, "count": 2, "nonfinite": 0},
        }
        d = _diag()
        d["nonfinite_forward_count"] = 1
        d["layers"] = layers
        model = _FakeDiagModel(_const_out(5.0), d)
        trainer.compute_loss(model, {})
        trainer.log({})
        logs = trainer.last_logs
        self.assertNotIn("probe/L18/kl_mean", logs)      # tainted → unavailable
        self.assertAlmostEqual(logs["probe/L20/kl_mean"], 2.0)
        self.assertEqual(logs["diag/nonfinite_forward_count"], 1.0)

    def test_payload_universe_stable_with_empty_accumulators(self):
        # a rank that saw zero valid samples must still build the full key set
        sum_p, max_p = _build_window_payloads({}, {}, {}, {}, [14, 18])
        self.assertIn("s:probe/L14/kl_mean", sum_p)
        self.assertIn("s:probe/L18/gt_floor_mass", sum_p)
        self.assertIn("c:diag/nonfinite_forward_count", sum_p)
        self.assertEqual(sum_p["n:probe/L14/kl_mean"], 0.0)
        logs = _emit_window_logs(sum_p, max_p)
        self.assertNotIn("probe/L14/kl_mean", logs)
        self.assertNotIn("data/grounding_valid_frac", logs)
        self.assertEqual(logs["diag/nonfinite_forward_count"], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DDP aggregation with 2 real gloo processes
# ─────────────────────────────────────────────────────────────────────────────

def _gloo_worker(rank, world_size, port, out_dir):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size,
    )
    try:
        if rank == 0:
            metrics = {"diag/grounding_valid_mean": 20.0, "probe/L18/kl_mean": 1.0}
            counts = {"diag/grounding_valid_mean": 2, "probe/L18/kl_mean": 1,
                      "_model_samples": 2, "_valid_samples": 2}
            maxes = {"diag/loss_micro_max": 3.0}
            taints = {}
        else:
            # mean 1.0 over 8 samples; a spike; non-finite on this rank; L18 tainted
            metrics = {"diag/grounding_valid_mean": 8.0, "probe/L18/kl_mean": 5.0}
            counts = {"diag/grounding_valid_mean": 8, "probe/L18/kl_mean": 1,
                      "_model_samples": 8, "_valid_samples": 8,
                      "diag/nonfinite_forward_count": 2}
            maxes = {"diag/loss_micro_max": 41.0}
            taints = {"probe/L18/kl_mean": 1}
        sum_p, max_p = _build_window_payloads(metrics, counts, maxes, taints, [18])
        sum_p, max_p = _window_allreduce(sum_p, max_p)
        logs = _emit_window_logs(sum_p, max_p)
        with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
            json.dump(logs, f, sort_keys=True)
    finally:
        dist.destroy_process_group()


class DdpAggregationTests(unittest.TestCase):
    def test_gloo_two_rank_sum_count_max(self):
        if not dist.is_available():
            self.skipTest("torch.distributed unavailable")
        import socket
        import torch.multiprocessing as mp

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        with tempfile.TemporaryDirectory(dir=".") as out_dir:
            mp.spawn(_gloo_worker, args=(2, port, out_dir), nprocs=2, join=True)
            results = {}
            for rank in (0, 1):
                with open(os.path.join(out_dir, f"rank{rank}.json")) as f:
                    results[rank] = json.load(f)
        # both ranks must compute identical global statistics
        self.assertEqual(results[0], results[1])
        logs = results[0]
        # ratio of SUMS, not mean of rank means ((10+1)/2=5.5 would be wrong)
        self.assertAlmostEqual(logs["diag/grounding_valid_mean"], 2.8)
        # impulse preserved by MAX across ranks
        self.assertAlmostEqual(logs["diag/loss_micro_max"], 41.0)
        # integer counters SUM across ranks
        self.assertEqual(logs["diag/nonfinite_forward_count"], 2.0)
        self.assertAlmostEqual(logs["data/grounding_valid_frac"], 1.0)
        # taint on rank 1 suppresses the layer metric on ALL ranks
        self.assertNotIn("probe/L18/kl_mean", logs)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Non-intrusion: real head loss/grad bit-identity, refactored KL arithmetic
# ─────────────────────────────────────────────────────────────────────────────

class NonIntrusionTests(unittest.TestCase):
    def _head_and_inputs(self, seed=7):
        torch.manual_seed(seed)
        head = Head(
            d_model=32, d_proj=16, probe_layers=[0, 1],
            independent_layers=True, adapter_type="attn",
            attn_n_heads=2, attn_d_head=4,
        )
        gen = torch.Generator().manual_seed(11)
        hidden = tuple(torch.randn(7, 32, generator=gen) for _ in range(3))
        visual = torch.tensor([1, 2, 3, 4])
        labels = torch.tensor([0.1, 0.6, 0.2, 0.1])
        return head, hidden, visual, labels

    def test_monitor_on_off_bit_identical(self):
        head_a, hidden, visual, labels = self._head_and_inputs()
        head_b = copy.deepcopy(head_a)
        head_a.collect_diagnostics = True
        head_b.collect_diagnostics = False
        outs = {}
        for tag, head in (("on", head_a), ("off", head_b)):
            result = head(
                all_hidden_states=hidden, ground_token_idx=5,
                visual_indices=visual, labels=labels,
            )
            result["total_grounding_loss"].backward()
            outs[tag] = result
        torch.testing.assert_close(
            outs["on"]["total_grounding_loss"], outs["off"]["total_grounding_loss"],
            rtol=0, atol=0,
        )
        for pa, pb in zip(head_a.parameters(), head_b.parameters()):
            if pa.grad is None or pb.grad is None:
                self.assertIs(pa.grad, pb.grad)
            else:
                torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
        # diag present only when on; values consistent with the loss itself
        self.assertIn("diag", outs["on"])
        self.assertNotIn("diag", outs["off"])
        kl_sum = sum(l["kl"] for l in outs["on"]["diag"]["layers"])
        self.assertAlmostEqual(
            kl_sum / 2, float(outs["on"]["total_grounding_loss"]), places=6,
        )
        self.assertEqual([l["layer_idx"] for l in outs["on"]["diag"]["layers"]], [0, 1])

    def test_refactored_kl_bookkeeping_matches_old_arithmetic(self):
        head, hidden, visual, labels = self._head_and_inputs()
        head.collect_diagnostics = False
        labeled = head(all_hidden_states=hidden, ground_token_idx=5,
                       visual_indices=visual, labels=labels)
        # old arithmetic: sequential += of per-layer KL from zeros, / num_active
        unlabeled = head(all_hidden_states=hidden, ground_token_idx=5,
                         visual_indices=visual, labels=None)
        y = labels / (labels.sum() + 1e-8)
        expected = torch.zeros(())
        for p_l in unlabeled["per_layer_probs"]:
            expected = expected + F.kl_div(
                torch.log(p_l.clamp(min=1e-8)), y, reduction="sum",
            )
        expected = expected / 2
        torch.testing.assert_close(labeled["loss_layer"], expected, rtol=0, atol=0)
        torch.testing.assert_close(labeled["total_grounding_loss"], expected, rtol=0, atol=0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. probe_output_stats math + audit ports
# ─────────────────────────────────────────────────────────────────────────────

class ProbeOutputStatsTests(unittest.TestCase):
    def test_entropy_normalization_and_uniform_ref(self):
        for n in (4, 16, 128):
            p = torch.full((n,), 1.0 / n, dtype=torch.float64)
            stats = probe_output_stats(torch.zeros(n), p)
            self.assertAlmostEqual(stats["entropy_norm"], 1.0, places=5)
            self.assertAlmostEqual(stats["posterior_max"], 1.0 / n)
            # uniform reference: KL(y || u) = log(N) - H(y)
            y = torch.arange(1, n + 1, dtype=torch.float64)
            y /= y.sum()
            hy = -(y * y.log()).sum()
            kl = F.kl_div(p.log(), y, reduction="sum")
            self.assertAlmostEqual(kl.item(), math.log(n) - hy.item())

    def test_n_vis_one_entropy_defined_zero(self):
        stats = probe_output_stats(torch.tensor([3.0]), torch.tensor([1.0]))
        self.assertEqual(stats["entropy_norm"], 0.0)
        self.assertEqual(stats["logit_rms_centered"], 0.0)
        self.assertEqual(stats["posterior_max"], 1.0)

    def test_centered_rms_shift_invariant(self):
        z = torch.tensor([1.0, 2.0, 4.0, 8.0])
        p = torch.softmax(z, dim=-1)
        a = probe_output_stats(z, p)["logit_rms_centered"]
        b = probe_output_stats(z + 100.0, p)["logit_rms_centered"]
        self.assertAlmostEqual(a, b, places=5)
        self.assertAlmostEqual(a, float((z - z.mean()).square().mean().sqrt()), places=6)

    def test_mean_does_not_preserve_impulse_but_max_does(self):
        values = [1.0] * 19 + [41.0]
        self.assertEqual(sum(values) / len(values), 3.0)
        self.assertEqual(max(values), 41.0)

    def test_target_floor_can_hide_corrective_gradient_in_toy_case(self):
        z = torch.tensor([0.0, -30.0], requires_grad=True)
        y = torch.tensor([0.0, 1.0])
        p = z.softmax(-1)
        floored = F.kl_div(p.clamp(min=1e-8).log(), y, reduction="sum")
        floored.backward()
        self.assertTrue(torch.equal(z.grad, torch.zeros_like(z)))
        floor_mass = (y * (p.detach() < 1e-8)).sum()
        self.assertEqual(floor_mass.item(), 1.0)
        z2 = z.detach().clone().requires_grad_()
        F.kl_div(z2.log_softmax(-1), y, reduction="sum").backward()
        self.assertGreater(z2.grad.norm().item(), 1.0)

    def test_detached_stats_do_not_change_gradients(self):
        torch.manual_seed(7)
        a = Probe(d_model=16, n_heads=2, d_head=4)
        b = copy.deepcopy(a)
        hq, hv = torch.randn(16), torch.randn(11, 16)
        ya = torch.arange(1, 12, dtype=torch.float32)
        ya /= ya.sum()
        for model, monitor in ((a, False), (b, True)):
            p, logits, q = model(hq, hv, None, None, 0)
            if monitor:
                stats = probe_output_stats(logits, p)
                self.assertTrue(all(math.isfinite(v) for v in stats.values()))
            F.kl_div(p.clamp_min(1e-8).log(), ya, reduction="sum").backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            if pa.grad is None or pb.grad is None:
                self.assertIs(pa.grad, pb.grad)
            else:
                torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)

    def test_skip_branch_dilution_contract_preserved(self):
        # _compute_grounding_loss model-loss behavior is intentionally
        # UNCHANGED: one valid sample (loss 10) + one placeholder → mean 5.
        model = AuditMixin()
        model._anchor_source_counts = {}
        model._grounding_diag = None
        model._get_visual_indices = lambda tokens: torch.tensor([1, 2])
        model._find_ground_anchor = lambda **kwargs: (0, SimpleNamespace(value="P1"))
        model._zero_grounding_loss = lambda device=None: torch.tensor(0.0, device=device)
        model.layerwise_grounding_head = lambda **kwargs: {
            "total_grounding_loss": torch.tensor(10.0),
            "p_final": torch.tensor([0.5, 0.5]), "omega": torch.tensor([1.0]),
        }
        output = model._compute_grounding_loss(
            all_hidden_states=(torch.zeros(2, 3, 4),),
            input_ids=torch.tensor([[0, 1, 1], [0, 1, 1]]),
            logits=torch.zeros(2, 3, 1), ground_token_indices=[0, 0],
            multi_patch_labels=[torch.tensor([1.0, 0.0]), torch.zeros(1)],
        )
        self.assertEqual(output[0].item(), 5.0)
        self.assertEqual(sum(p is not None for p in output[1]), 1)
        # and the diagnostics payload reflects the same accounting
        diag = model._grounding_diag
        self.assertEqual(diag["valid_count"], 1)
        self.assertEqual(diag["batch_size"], 2)
        self.assertAlmostEqual(diag["valid_loss_sum"], 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
