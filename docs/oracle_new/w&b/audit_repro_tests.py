"""CPU source reproductions and mathematical contracts; NOT a Trainer patch.

Run:
  python audit_repro_tests.py --repo-root /path/to/zwergeUI
Requires Python 3.9+ and torch. Does not contact W&B or load model weights.
"""
from __future__ import annotations
import argparse
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--repo-root', type=Path, required=True)
args, remaining = parser.parse_known_args()
ROOT = args.repo_root.resolve()
BASE = ROOT / 'zwerge/src/zwerge_retrofit/modeling_base.py'
TRAINER = ROOT / 'zwerge/src/zwerge_retrofit/trainer.py'
for path in (BASE, TRAINER):
    if not path.is_file():
        parser.error(f'Missing source file: {path}')


def extract_class(path: Path, original: str, methods=None, new_name=None, bases=None):
    tree = ast.parse(path.read_text())
    source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == original)
    result = copy.deepcopy(source)
    if methods is not None:
        result.body = [n for n in result.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods]
    if new_name:
        result.name = new_name
    if bases is not None:
        result.bases = [ast.Name(id=b, ctx=ast.Load()) for b in bases]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    return ast.fix_missing_locations(ast.Module(body=[future, result], type_ignores=[]))


class LogSink:
    def log(self, logs, start_time=None):
        self.last_logs = dict(logs)


ns = dict(torch=torch, nn=nn, F=F, math=math, LogSink=LogSink)
exec(compile(extract_class(TRAINER, 'RetrofitTrainer', {'compute_loss', 'log'}, 'AuditTrainer', ['LogSink']), str(TRAINER), 'exec'), ns)
exec(compile(extract_class(BASE, 'RetrofitModelMixin', {'_compute_grounding_loss'}, 'AuditMixin', []), str(BASE), 'exec'), ns)
exec(compile(extract_class(BASE, 'CrossAttnGroundingProbe'), str(BASE), 'exec'), ns)
AuditTrainer = ns['AuditTrainer']
AuditMixin = ns['AuditMixin']
Probe = ns['CrossAttnGroundingProbe']


def old_logged_loss(window_steps=20, accumulation=4, loss_value=2.0):
    trainer = AuditTrainer()
    trainer.args = SimpleNamespace(gradient_accumulation_steps=accumulation)
    out = SimpleNamespace(loss=torch.tensor(loss_value), grounding_loss=torch.tensor(loss_value), lm_loss=None,
                          grounding_scores=None, layer_weights=None)
    model = lambda **kwargs: out
    for _ in range(window_steps * accumulation):
        trainer.compute_loss(model, {})
    trainer.log({'loss': loss_value})
    return trainer.last_logs['grounding_loss']


def ratio_of_sums(sums_counts):
    total = sum(s for s, n in sums_counts)
    count = sum(n for s, n in sums_counts)
    return total / count if count else None


class ObservabilityAudit(unittest.TestCase):
    def test_original_log_full_window_is_scaled_by_logging_interval(self):
        self.assertAlmostEqual(old_logged_loss(), 40.0)
        self.assertNotAlmostEqual(old_logged_loss(), 2.0)

    def test_original_log_scale_changes_with_partial_window(self):
        self.assertAlmostEqual(old_logged_loss(window_steps=5), 10.0)
        self.assertAlmostEqual(old_logged_loss(window_steps=20), 40.0)

    def test_count_based_mean_is_window_invariant(self):
        for steps in (1, 5, 20):
            count = 4 * steps
            self.assertEqual(ratio_of_sums([(2.0 * count, count)]), 2.0)

    def test_ddp_valid_mean_requires_sum_and_count_not_mean_of_means(self):
        self.assertAlmostEqual(ratio_of_sums([(20.0, 2), (8.0, 8)]), 2.8)
        self.assertNotAlmostEqual((10.0 + 1.0) / 2, 2.8)

    def test_original_skip_branch_dilutes_batch_loss(self):
        model = AuditMixin()
        model._anchor_source_counts = {}
        model._get_visual_indices = lambda tokens: torch.tensor([1, 2])
        model._find_ground_anchor = lambda **kwargs: (0, SimpleNamespace(value='P1'))
        model._zero_grounding_loss = lambda device=None: torch.tensor(0.0, device=device)
        model.layerwise_grounding_head = lambda **kwargs: {
            'total_grounding_loss': torch.tensor(10.0),
            'p_final': torch.tensor([0.5, 0.5]), 'omega': torch.tensor([1.0]),
        }
        output = model._compute_grounding_loss(
            all_hidden_states=(torch.zeros(2, 3, 4),),
            input_ids=torch.tensor([[0, 1, 1], [0, 1, 1]]),
            logits=torch.zeros(2, 3, 1), ground_token_indices=[0, 0],
            multi_patch_labels=[torch.tensor([1.0, 0.0]), torch.zeros(1)],
        )
        self.assertEqual(output[0].item(), 5.0)
        self.assertEqual(sum(p is not None for p in output[1]), 1)

    def test_target_floor_can_hide_corrective_gradient_in_toy_case(self):
        z = torch.tensor([0.0, -30.0], requires_grad=True)
        y = torch.tensor([0.0, 1.0])
        p = z.softmax(-1)
        floored = F.kl_div(p.clamp(min=1e-8).log(), y, reduction='sum')
        floored.backward()
        self.assertTrue(torch.equal(z.grad, torch.zeros_like(z)))
        floor_mass = (y * (p.detach() < 1e-8)).sum()
        self.assertEqual(floor_mass.item(), 1.0)
        z2 = z.detach().clone().requires_grad_()
        F.kl_div(z2.log_softmax(-1), y, reduction='sum').backward()
        self.assertGreater(z2.grad.norm().item(), 1.0)

    def test_uniform_reference_and_entropy_normalization(self):
        for n in (4, 16, 128):
            p = torch.full((n,), 1.0 / n, dtype=torch.float64)
            entropy = -(p * p.log()).sum()
            self.assertAlmostEqual((entropy / math.log(n)).item(), 1.0)
            y = torch.arange(1, n + 1, dtype=torch.float64)
            y /= y.sum()
            hy = -(y * y.log()).sum()
            kl = F.kl_div(p.log(), y, reduction='sum')
            self.assertAlmostEqual(kl.item(), math.log(n) - hy.item())

    def test_mean_does_not_preserve_impulse_but_max_does(self):
        values = [1.0] * 19 + [41.0]
        self.assertEqual(sum(values) / len(values), 3.0)
        self.assertEqual(max(values), 41.0)

    def test_detached_probe_scalar_collection_does_not_change_gradients(self):
        torch.manual_seed(7)
        a = Probe(d_model=16, n_heads=2, d_head=4)
        b = copy.deepcopy(a)
        hq, hv = torch.randn(16), torch.randn(11, 16)
        ya = torch.arange(1, 12, dtype=torch.float32)
        ya /= ya.sum()
        for model, monitor in ((a, False), (b, True)):
            p, logits, q = model(hq, hv, None, None, 0)
            if monitor:
                with torch.no_grad():
                    z = logits.detach().float()
                    rms = ((z - z.mean()).square().mean()).sqrt()
                    entropy = -(p.detach().float() * p.detach().float().clamp_min(1e-8).log()).sum() / math.log(len(p))
                    self.assertTrue(torch.isfinite(rms) and torch.isfinite(entropy))
            F.kl_div(p.clamp_min(1e-8).log(), ya, reduction='sum').backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            if pa.grad is None or pb.grad is None:
                self.assertIs(pa.grad, pb.grad)
            else:
                torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)


if __name__ == '__main__':
    print('Scope: CPU source snippets and mathematical contracts only.')
    print('No real checkpoint, GPU training, W&B integration, or multi-process DDP test.')
    print(f'Original full-window reported grounding_loss: {old_logged_loss()} (true constant = 2.0)')
    unittest.main(argv=['audit_repro_tests.py'] + remaining, verbosity=2)
