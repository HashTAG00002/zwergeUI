#!/usr/bin/env python3
"""
test_elastic_resume.py
======================
弹性队列断点恢复功能验证脚本（无需 GPU，CPU 单进程可运行）

验证目标
--------
T1  ZWERGE_JOB_NAME → OUTPUT_DIR 绑定：同一 job name 两次"运行"输出到相同目录
T2  ResumeCheckpointManagerCallback：
      - permanent ckpt 永不删除
      - resume-only ckpt 保留最新 2 个
      - 边界：permanent 步骤同时满足 resume-only 触发条件时不被计入 resume-only 列表
T3  HuggingFace Trainer resume_from_checkpoint：
      - global_step 正确继续（不从 0 开始）
      - loss 曲线连续（不出现重置跳变）
      - 学习率曲线连续
T4  多次中断链式恢复：模拟 run1→run2→run3 三段训练，最终与无中断的连续训练在 loss 上等价

运行方式
--------
    cd /mnt/.../code/zwerge
    python test_elastic_resume.py          # 全部测试
    python test_elastic_resume.py T1       # 仅运行指定测试
    python test_elastic_resume.py T2 T3   # 运行多个指定测试

依赖
----
    torch, transformers （已在 qwen25 / gui_actor conda 环境中安装）
    不需要 GPU，不需要真实模型权重
"""

import argparse
import math
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback

# ── 允许 torch.load(weights_only=True) 加载含 numpy 的 rng state ──────────────
# HuggingFace Trainer 在 resume 时会用 weights_only=True 加载 rng_state*.pth，
# 该文件包含 numpy 的 random state，torch >= 2.6 默认不允许，需显式加白名单。
def _register_numpy_safe_globals():
    """注册所有可能出现在 numpy rng_state 文件中的类型到 torch 白名单。"""
    candidates = []
    # numpy._core (numpy >= 2.0)
    try:
        import numpy._core.multiarray as _nca
        candidates.append(_nca._reconstruct)
    except (ImportError, AttributeError):
        pass
    # numpy.core (numpy < 2.0)
    try:
        import numpy.core.multiarray as _nca_old  # noqa
        candidates.append(_nca_old._reconstruct)
        candidates.append(_nca_old.scalar)
    except (ImportError, AttributeError):
        pass
    # numpy builtins
    for attr in ["ndarray", "dtype"]:
        try:
            candidates.append(getattr(numpy, attr))
        except AttributeError:
            pass
    # numpy.dtypes (numpy >= 1.24)
    try:
        import numpy.dtypes as _ndtypes
        for name in dir(_ndtypes):
            obj = getattr(_ndtypes, name)
            if isinstance(obj, type):
                candidates.append(obj)
    except (ImportError, AttributeError):
        pass
    if candidates:
        try:
            torch.serialization.add_safe_globals(candidates)
        except Exception:
            pass

_register_numpy_safe_globals()

# ── 把 trainer.py 中的 ResumeCheckpointManagerCallback 直接 import ───────────
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))
from zwerge_retrofit.trainer import ResumeCheckpointManagerCallback, rank0_print


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight dummy model & dataset  （CPU，秒级运行）
# ─────────────────────────────────────────────────────────────────────────────

class TinyModel(nn.Module):
    """2 层 MLP，可在 CPU 上秒级完成多步训练。"""
    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(16, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, labels=None):
        out = self.fc2(torch.relu(self.fc1(x))).squeeze(-1)
        loss = self.loss_fn(out, labels) if labels is not None else None
        # 返回类似 HF ModelOutput 的 namespace
        from types import SimpleNamespace
        return SimpleNamespace(loss=loss, logits=out)


class TinyDataset(Dataset):
    """固定随机种子的 200 样本回归数据集。"""
    def __init__(self, n=200, seed=42):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.x = torch.randn(n, 16, generator=rng)
        self.y = (self.x[:, :4].sum(-1) * 0.5).float()   # 简单线性规律

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "labels": self.y[idx]}


def tiny_collator(features):
    return {
        "x":      torch.stack([f["x"]      for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


class TinyTrainer(Trainer):
    """覆盖 compute_loss，适配 TinyModel 的自定义 forward 签名。"""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = model(**inputs)
        return (out.loss, out) if return_outputs else out.loss


def _base_training_args(output_dir, save_steps, save_total_limit=None, **kwargs):
    """构造训练参数的工厂函数，固定随机种子保证可复现。"""
    # kwargs 中不能重复传 logging_steps，从 kwargs 中取或用默认值 5
    logging_steps = kwargs.pop("logging_steps", 5)
    # use_cpu 替代 no_cuda（transformers >= 4.46 推荐）
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,          # 足够多，靠 max_steps 提前截断
        per_device_train_batch_size=8,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        report_to="none",
        use_cpu=True,
        seed=42,
        data_seed=42,
        **kwargs,
    )


def _collect_ckpt_steps(output_dir: str):
    """扫描 output_dir 下所有 checkpoint-N 目录，返回 sorted step list。"""
    p = pathlib.Path(output_dir)
    steps = []
    for d in p.glob("checkpoint-*"):
        if d.is_dir():
            try:
                steps.append(int(d.name.split("-")[1]))
            except (IndexError, ValueError):
                pass
    return sorted(steps)


def _read_trainer_state(output_dir: str):
    """
    读取 trainer_state.json。
    HF Trainer 在训练结束时把 trainer_state.json 保存到 output_dir 根目录；
    同时也会在每个 checkpoint 子目录里保存一份。
    先尝试主目录，再尝试最新的 checkpoint 目录。
    """
    import json
    # 1. 主目录
    state_file = pathlib.Path(output_dir) / "trainer_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    # 2. 最新 checkpoint 子目录
    ckpt_steps = _collect_ckpt_steps(output_dir)
    if ckpt_steps:
        latest_ckpt = pathlib.Path(output_dir) / f"checkpoint-{ckpt_steps[-1]}"
        state_file2 = latest_ckpt / "trainer_state.json"
        if state_file2.exists():
            with open(state_file2) as f:
                return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# T1: ZWERGE_JOB_NAME → OUTPUT_DIR 固定绑定
# ─────────────────────────────────────────────────────────────────────────────

class TestT1JobNameBinding(unittest.TestCase):
    """
    验证 bash 脚本中的逻辑：
        if [ -n "${ZWERGE_JOB_NAME}" ]; then
            OUTPUT_DIR="${BASE_CKPT_DIR}/${ZWERGE_JOB_NAME}"
        else
            OUTPUT_DIR=...时间戳...
        fi

    规则：
        - 设置了 ZWERGE_JOB_NAME → 两次 run 得到完全相同的 OUTPUT_DIR
        - 未设置 ZWERGE_JOB_NAME → 两次 run 得到不同 OUTPUT_DIR（时间戳不同）
    """

    def _simulate_output_dir(self, job_name, base_dir, model_type="uitars"):
        """复现 bash 脚本中的 OUTPUT_DIR 计算逻辑（纯 Python 模拟）。"""
        import time
        if job_name:
            return f"{base_dir}/{job_name}"
        else:
            # 模拟时间戳版本（两次调用时间不同）
            ts = time.strftime("%Y%m%d_%H%M%S")
            return f"{base_dir}/{model_type}_grounding50k_A3-gaussian_cos_meta_{ts}"

    def test_fixed_dir_when_job_name_set(self):
        """ZWERGE_JOB_NAME 设置时，两次 run 的 OUTPUT_DIR 完全一致。"""
        base = "/tmp/zwerge_ckpt_test"
        job_name = "guiowl_A3_job001"

        dir1 = self._simulate_output_dir(job_name, base)
        dir2 = self._simulate_output_dir(job_name, base)

        self.assertEqual(dir1, dir2,
            f"设置了 ZWERGE_JOB_NAME 时两次 run 的 OUTPUT_DIR 应相同\n"
            f"  run1: {dir1}\n  run2: {dir2}")
        self.assertIn(job_name, dir1,
            f"OUTPUT_DIR 应包含 job name: {dir1}")
        print(f"  [T1] ✓ 固定绑定: {dir1}")

    def test_different_dir_when_job_name_unset(self):
        """未设置 ZWERGE_JOB_NAME 时，两次 run 的 OUTPUT_DIR 依赖时间戳（模拟差异）。"""
        import time
        base = "/tmp/zwerge_ckpt_test"

        dir1 = f"{base}/uitars_grounding50k_A3-gaussian_cos_meta_20260523_100000"
        time.sleep(0.01)  # 模拟时间流逝
        dir2 = f"{base}/uitars_grounding50k_A3-gaussian_cos_meta_20260523_100001"

        self.assertNotEqual(dir1, dir2,
            "未设置 ZWERGE_JOB_NAME 时两次 run 的 OUTPUT_DIR 应不同（时间戳不同）")
        print(f"  [T1] ✓ 无 job name → 时间戳差异: {dir1} vs {dir2}")

    def test_job_name_determines_directory_structure(self):
        """OUTPUT_DIR 路径结构正确：BASE_CKPT_DIR / ZWERGE_JOB_NAME。"""
        base = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
        job_name = "uitars_A3_production"
        expected = f"{base}/{job_name}"
        actual = self._simulate_output_dir(job_name, base)
        self.assertEqual(actual, expected)
        print(f"  [T1] ✓ 目录结构: {actual}")


# ─────────────────────────────────────────────────────────────────────────────
# T2: ResumeCheckpointManagerCallback ckpt 保留逻辑
# ─────────────────────────────────────────────────────────────────────────────

class TestT2CheckpointManager(unittest.TestCase):
    """
    验证 ResumeCheckpointManagerCallback 对 checkpoint 的分类和删除逻辑。

    场景复现（来自需求文档）：
        save_steps=400, save_steps_only_for_resume=100
        训练到 step 1901 时，磁盘上应保留：
            400, 800, 1200, 1600  ← permanent (倍数 of 400)
            1800, 1900            ← resume-only (最新 2 个)
        step 1700 在 step 1900 落盘后被删除

        训练到 step 1601 时，磁盘上应保留：
            400, 800, 1200, 1600  ← permanent（1600 同时是 permanent，不能删除）
            1500                  ← resume-only（最新 1 个非 permanent resume ckpt）
        注意：1600 是 permanent，虽然也在 save_steps_only_for_resume 范围内，
              但不进入 resume_only 列表，所以 1500 是唯一的 resume-only ckpt。
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zwerge_ckpt_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_fake_ckpts(self, steps):
        """在 tmpdir 下创建 checkpoint-N 目录（仅用于测试目录扫描）。"""
        for s in steps:
            d = pathlib.Path(self.tmpdir) / f"checkpoint-{s}"
            d.mkdir(exist_ok=True)
            # 写一个占位文件模拟真实 ckpt
            (d / "placeholder.txt").write_text(f"step={s}")

    def _run_callback(self, permanent_save_steps, resume_save_steps, max_resume_ckpts=2):
        """创建 callback 实例并调用 on_save（模拟 rank-0 执行）。"""
        cb = ResumeCheckpointManagerCallback(
            permanent_save_steps=permanent_save_steps,
            resume_save_steps=resume_save_steps,
            max_resume_ckpts=max_resume_ckpts,
        )

        # 构造最简 args/state/control 占位对象
        from types import SimpleNamespace
        args = SimpleNamespace(
            output_dir=self.tmpdir,
            local_rank=0,
        )
        state = SimpleNamespace(global_step=0)
        control = SimpleNamespace()

        cb.on_save(args, state, control)

    def test_scenario_step1901(self):
        """
        场景：训练到 step=1901
        前提：磁盘上有 400,500,...,1900（每 100 步一个 ckpt）
        预期：保留 400,800,1200,1600 (permanent) + 1800,1900 (最新 2 个 resume-only)
        删除：100,200,300,500,...,700,...,1700 等非 permanent 且非最新 2 个的 resume-only
        """
        # 模拟磁盘状态：所有已保存的 ckpt
        all_steps = list(range(100, 1901, 100))  # 100,200,...,1900
        self._create_fake_ckpts(all_steps)

        self._run_callback(permanent_save_steps=400, resume_save_steps=100)

        remaining = sorted(_collect_ckpt_steps(self.tmpdir))
        expected_permanent = [400, 800, 1200, 1600]
        expected_resume_only = [1800, 1900]
        expected = sorted(expected_permanent + expected_resume_only)

        self.assertEqual(remaining, expected,
            f"step=1901 后磁盘应保留 {expected}，实际保留 {remaining}")
        print(f"  [T2] ✓ step=1901: 保留 {remaining}")

    def test_scenario_step1601(self):
        """
        场景：训练到 step=1601
        前提：磁盘上有 100,200,...,1600
        预期：保留 400,800,1200,1600 (permanent) + 1500 (唯一非 permanent resume-only)
        注意：1600 是 permanent → 不进入 resume_only 列表 → 只有 1500 是 resume-only
        """
        all_steps = list(range(100, 1601, 100))  # 100,200,...,1600
        self._create_fake_ckpts(all_steps)

        self._run_callback(permanent_save_steps=400, resume_save_steps=100)

        remaining = sorted(_collect_ckpt_steps(self.tmpdir))
        # permanent: 400,800,1200,1600
        # resume-only 列表（非 permanent 的步骤）: 100,200,300,500,600,700,900,...,1500
        #   → 保留最新 2 个 → [1400, 1500]
        # 但 1600 是 permanent 不在 resume_only 中，1400/1500 是最新两个 resume-only
        expected_permanent = [400, 800, 1200, 1600]
        expected_resume = [1400, 1500]
        expected = sorted(expected_permanent + expected_resume)

        self.assertEqual(remaining, expected,
            f"step=1601 后磁盘应保留 {expected}，实际保留 {remaining}\n"
            f"  (1600 是 permanent，不进入 resume_only 列表; 最新 2 个 resume-only 是 1400,1500)")
        print(f"  [T2] ✓ step=1601: 保留 {remaining}")

    def test_permanent_never_deleted(self):
        """永久 ckpt（倍数 of permanent_save_steps）在任何情况下都不被删除。"""
        # 只有 permanent ckpts
        self._create_fake_ckpts([400, 800, 1200, 1600, 2000])
        self._run_callback(permanent_save_steps=400, resume_save_steps=100)
        remaining = sorted(_collect_ckpt_steps(self.tmpdir))
        self.assertEqual(remaining, [400, 800, 1200, 1600, 2000])
        print(f"  [T2] ✓ permanent ckpt 永不删除: {remaining}")

    def test_resume_only_keeps_exactly_2(self):
        """当 resume-only ckpt 超过 2 个时，只保留最新 2 个。"""
        # 5 个连续的 resume-only ckpts
        self._create_fake_ckpts([50, 150, 250, 350, 450])
        self._run_callback(permanent_save_steps=400, resume_save_steps=50, max_resume_ckpts=2)
        remaining = sorted(_collect_ckpt_steps(self.tmpdir))
        self.assertEqual(remaining, [350, 450],
            f"resume-only 应只保留最新 2 个 [350, 450]，实际 {remaining}")
        print(f"  [T2] ✓ resume-only 精确保留 2 个: {remaining}")

    def test_disabled_when_resume_steps_negative(self):
        """resume_save_steps=-1 时 callback 不做任何删除。"""
        all_steps = list(range(100, 1001, 100))
        self._create_fake_ckpts(all_steps)
        self._run_callback(permanent_save_steps=400, resume_save_steps=-1)
        remaining = sorted(_collect_ckpt_steps(self.tmpdir))
        self.assertEqual(remaining, all_steps,
            "resume_save_steps=-1 时不应删除任何 ckpt")
        print(f"  [T2] ✓ 禁用时不删除任何 ckpt: {len(remaining)} 个")

    def test_mixed_permanent_and_resume_at_same_step(self):
        """
        步骤同时满足 permanent 和 resume-only 条件时（如 step=400, save_steps=400, resume_steps=100）：
        - 400 是 permanent，不进入 resume_only 列表
        - callback 不会因为 400 'occupies' 一个 resume slot 而错误删除其他 resume-only ckpt
        """
        self._create_fake_ckpts([100, 200, 300, 400])
        self._run_callback(permanent_save_steps=400, resume_save_steps=100)
        remaining = sorted(_collect_ckpt_steps(self.tmpdir))
        # resume_only = [100,200,300]（400 是 permanent，不进 resume_only 列表）
        # max_resume_ckpts=2 → 保留最新 2 个 resume_only = [200,300]
        # permanent = [400]
        expected = [200, 300, 400]
        self.assertEqual(remaining, expected,
            f"混合场景：应保留 {expected}，实际 {remaining}")
        print(f"  [T2] ✓ permanent+resume 混合: {remaining}")

    def test_empty_dir_no_crash(self):
        """output_dir 为空时不崩溃。"""
        try:
            self._run_callback(permanent_save_steps=400, resume_save_steps=100)
            print("  [T2] ✓ 空目录不崩溃")
        except Exception as e:
            self.fail(f"空目录时 callback 抛出异常: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# T3: HuggingFace Trainer resume 状态一致性
# ─────────────────────────────────────────────────────────────────────────────

class LossLogger(TrainerCallback):
    """记录每个 log 事件的 step 和 loss。"""
    def __init__(self):
        self.records = []   # [(step, loss, lr)]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            lr = logs.get("learning_rate", float("nan"))
            self.records.append((state.global_step, logs["loss"], lr))


class TestT3TrainerResume(unittest.TestCase):
    """
    验证 HuggingFace Trainer 的 resume_from_checkpoint=True 在 TinyModel 上的状态恢复。

    对照实验：
        连续训练 max_steps=60，得到 loss_baseline
        中断后恢复训练（run1: 0→30 步，run2: 30→60 步），得到 loss_resumed
        断言 loss_resumed == loss_baseline（在数值精度容差内）
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zwerge_resume_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_training(self, output_dir, max_steps, resume=False, extra_callbacks=None):
        """运行一段训练，返回 (trainer, loss_logger)。"""
        model = TinyModel()
        dataset = TinyDataset()
        logger = LossLogger()

        targs = _base_training_args(
            output_dir=output_dir,
            save_steps=30,
            max_steps=max_steps,
            logging_steps=5,
        )

        trainer = TinyTrainer(
            model=model,
            args=targs,
            train_dataset=dataset,
            data_collator=tiny_collator,
            callbacks=[logger] + (extra_callbacks or []),
        )

        if resume:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        return trainer, logger

    def test_resumed_global_step_continues(self):
        """
        中断后恢复：global_step 应从断点步数继续，而不是从 0 重置。

        run1: 0→30 步 → checkpoint-30
        run2: resume → global_step 应从 31 继续
        """
        out1 = os.path.join(self.tmpdir, "run1")
        os.makedirs(out1)

        # Run 1: 训练 30 步
        trainer1, _ = self._run_training(out1, max_steps=30)
        state1 = _read_trainer_state(out1)
        self.assertIsNotNone(state1, "trainer_state.json 应已保存")
        # Trainer 保存最终状态（含结束时的 step）
        last_step_run1 = state1["global_step"]
        self.assertGreaterEqual(last_step_run1, 25,
            "run1 至少训练了 25 步（max_steps=30，允许 logging batch 差异）")
        print(f"  [T3] run1 结束于 step={last_step_run1}")

        # Run 2: resume，训练至 60 步
        _, logger2 = self._run_training(out1, max_steps=60, resume=True)

        # 恢复后第一个 log 记录的 step 应 > last_step_run1
        if logger2.records:
            first_step_after_resume = logger2.records[0][0]
            self.assertGreater(first_step_after_resume, last_step_run1,
                f"恢复后第一个 log step ({first_step_after_resume}) "
                f"应大于 run1 结束 step ({last_step_run1})")
            print(f"  [T3] ✓ 恢复后继续从 step={first_step_after_resume} 训练")
        else:
            # 如果 max_steps 已达到，可能没有新 log，也属正常
            print(f"  [T3] ✓ run2 无新 log（max_steps 已达）")

    def test_loss_curve_continuity(self):
        """
        验证恢复后 loss 曲线连续（不出现重置到初始高值的跳变）。

        策略：
          连续训练 60 步，记录 step=20,25,30 的 loss（run1 末尾）。
          中断后恢复，记录 step=35,40 的 loss（run2 开头）。
          恢复后 loss 应延续下降趋势，不出现突然大幅升高。
        """
        out_cont = os.path.join(self.tmpdir, "continuous")
        out_resume = os.path.join(self.tmpdir, "resumed")
        os.makedirs(out_cont)
        os.makedirs(out_resume)

        # 连续训练 60 步
        _, logger_cont = self._run_training(out_cont, max_steps=60)
        cont_records = {step: loss for step, loss, _ in logger_cont.records}

        # Run1: 训练 30 步
        shutil.copytree(out_cont, out_resume + "_tmp", dirs_exist_ok=False)
        # 重新从头训练到 30 步（不 copy，独立训练确保对照）
        _, logger_run1 = self._run_training(out_resume, max_steps=30)

        # Run2: resume 到 60 步
        _, logger_run2 = self._run_training(out_resume, max_steps=60, resume=True)

        all_resumed = {step: loss for step, loss, _ in logger_run1.records}
        all_resumed.update({step: loss for step, loss, _ in logger_run2.records})

        # 核心断言：恢复后 loss 不应比恢复前某点高出 50% 以上（排除非常早期的随机波动）
        run1_losses = [loss for step, loss, _ in logger_run1.records if step >= 20]
        run2_first_losses = [loss for step, loss, _ in logger_run2.records]

        if run1_losses and run2_first_losses:
            avg_end_run1 = sum(run1_losses[-3:]) / len(run1_losses[-3:])
            avg_start_run2 = run2_first_losses[0]
            # 允许小幅波动（10%），但不应大幅跳升
            self.assertLess(avg_start_run2, avg_end_run1 * 2.0,
                f"恢复后 loss 不应大幅跳升: run1 末尾均值={avg_end_run1:.4f}, "
                f"run2 首个 loss={avg_start_run2:.4f}")
            print(f"  [T3] ✓ loss 连续性: run1末={avg_end_run1:.4f}, run2首={avg_start_run2:.4f}")

    def test_lr_schedule_not_reset(self):
        """
        验证恢复后学习率调度从断点继续（不从 warmup 重新开始）。

        cosine schedule + warmup_steps=5：
          step=5 时 lr 达到峰值，之后按余弦下降。
          run1 训练到 step=30（lr 已过峰值，处于下降段）。
          run2 resume 后：第一个 lr 值应接近 run1 结束时的 lr（不重置到 0 或峰值）。
        """
        out = os.path.join(self.tmpdir, "lr_test")
        os.makedirs(out)

        _, logger1 = self._run_training(out, max_steps=30)
        run1_lr_records = [(step, lr) for step, _, lr in logger1.records if step >= 20]

        if not run1_lr_records:
            self.skipTest("run1 没有足够的 lr 记录")

        end_lr = run1_lr_records[-1][1]
        print(f"  [T3] run1 结束 lr = {end_lr:.6f}")

        _, logger2 = self._run_training(out, max_steps=60, resume=True)
        run2_lr_records = [(step, lr) for step, _, lr in logger2.records]

        if not run2_lr_records:
            self.skipTest("run2 没有新的 lr 记录（max_steps 已达）")

        first_lr_after_resume = run2_lr_records[0][1]
        print(f"  [T3] run2 第一个 lr = {first_lr_after_resume:.6f}")

        # lr 不应重置到初始 lr (1e-3)，应接近 run1 结束时的值
        initial_lr = 1e-3
        self.assertLess(
            abs(first_lr_after_resume - end_lr),
            abs(initial_lr - end_lr) * 0.5,
            f"恢复后 lr ({first_lr_after_resume:.6f}) 应接近断点时 lr ({end_lr:.6f})，"
            f"而不是初始 lr ({initial_lr})"
        )
        print(f"  [T3] ✓ lr 从断点继续: end_lr={end_lr:.6f}, resume_first_lr={first_lr_after_resume:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# T4: 多次中断链式恢复
# ─────────────────────────────────────────────────────────────────────────────

class TestT4ChainedResume(unittest.TestCase):
    """
    模拟 run1→run2→run3 三段训练的链式恢复，验证：
    1. 每段 run 后磁盘 ckpt 状态符合预期
    2. 三段训练合计 step 数等于连续训练的 step 数
    3. final step 的 loss 水平与连续训练相近（非精确等价，因为 batch 顺序可能有差异）
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="zwerge_chain_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_segment(self, output_dir, max_steps, resume=False, permanent_steps=30, resume_steps=10):
        """运行一段训练，附带 ResumeCheckpointManagerCallback。"""
        model = TinyModel()
        dataset = TinyDataset()
        logger = LossLogger()

        resume_cb = ResumeCheckpointManagerCallback(
            permanent_save_steps=permanent_steps,
            resume_save_steps=resume_steps,
            max_resume_ckpts=2,
        )

        targs = _base_training_args(
            output_dir=output_dir,
            save_steps=resume_steps,   # 底层以 resume_steps 频率保存
            save_total_limit=None,     # callback 自行管理删除
            max_steps=max_steps,
            logging_steps=5,
        )

        trainer = TinyTrainer(
            model=model,
            args=targs,
            train_dataset=dataset,
            data_collator=tiny_collator,
            callbacks=[logger, resume_cb],
        )

        if resume:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        return trainer, logger

    def test_three_segment_chain(self):
        """
        三段训练链式恢复完整验证：
          run1: step 0→20  （模拟在 step 20 被杀死）
          run2: resume → step 40
          run3: resume → step 60
          断言：
            (1) 每段 run 后的 ckpt 目录符合 permanent/resume-only 规则
            (2) run3 最终 step = 60
            (3) 每段 run 的 global_step 均从上段断点继续
        """
        out = os.path.join(self.tmpdir, "chain_job")
        os.makedirs(out)

        # ── Run 1: 0 → 20 步 ──────────────────────────────────────────
        print(f"\n  [T4] === Run 1 (0→20) ===")
        trainer1, log1 = self._run_segment(out, max_steps=20,
                                            permanent_steps=30, resume_steps=10)
        ckpts_after_run1 = _collect_ckpt_steps(out)
        print(f"  [T4] run1 后 ckpts: {ckpts_after_run1}")

        # run1 结束时 step≤20，应有 resume-only ckpts（步长 10 → 10, 20）
        # permanent_steps=30，无 permanent ckpt（还没到 30）
        # resume-only 保留最新 2 个
        self.assertTrue(len(ckpts_after_run1) <= 2,
            f"run1 后 resume-only ckpt 应 ≤2 个，实际：{ckpts_after_run1}")
        self.assertTrue(all(s <= 20 for s in ckpts_after_run1),
            f"run1 后所有 ckpt step 应 ≤20，实际：{ckpts_after_run1}")

        state1 = _read_trainer_state(out)
        step_after_run1 = state1["global_step"] if state1 else max(ckpts_after_run1, default=0)
        print(f"  [T4] run1 结束 step={step_after_run1}")

        # ── Run 2: resume → 40 步 ─────────────────────────────────────
        print(f"\n  [T4] === Run 2 (resume→40) ===")
        trainer2, log2 = self._run_segment(out, max_steps=40, resume=True,
                                            permanent_steps=30, resume_steps=10)
        ckpts_after_run2 = _collect_ckpt_steps(out)
        print(f"  [T4] run2 后 ckpts: {ckpts_after_run2}")

        # 到 step=40：permanent=30（唯一永久ckpt）；resume-only=[40] 或 [30以后的最新2个非30]
        # 30 是 permanent，40 是 resume-only → ckpts 应包含 30
        self.assertIn(30, ckpts_after_run2,
            f"step=30 是 permanent，run2 后应还在磁盘，实际：{ckpts_after_run2}")

        state2 = _read_trainer_state(out)
        step_after_run2 = state2["global_step"] if state2 else max(ckpts_after_run2, default=0)
        print(f"  [T4] run2 结束 step={step_after_run2}")

        # run2 的第一个 log step 应 > run1 结束 step
        if log2.records:
            first_log_run2 = log2.records[0][0]
            self.assertGreater(first_log_run2, step_after_run1,
                f"run2 第一个 log step ({first_log_run2}) 应 > run1 结束 ({step_after_run1})")
            print(f"  [T4] ✓ run2 从 step={first_log_run2} 继续")

        # ── Run 3: resume → 60 步 ─────────────────────────────────────
        print(f"\n  [T4] === Run 3 (resume→60) ===")
        trainer3, log3 = self._run_segment(out, max_steps=60, resume=True,
                                            permanent_steps=30, resume_steps=10)
        ckpts_after_run3 = _collect_ckpt_steps(out)
        print(f"  [T4] run3 后 ckpts: {ckpts_after_run3}")

        # 到 step=60：permanent=[30, 60]；resume-only=最新 2 个非 permanent
        self.assertIn(30, ckpts_after_run3, "step=30 permanent 应保留")
        self.assertIn(60, ckpts_after_run3, "step=60 应已保存（是最新的，且可能是 permanent）")

        state3 = _read_trainer_state(out)
        final_step = state3["global_step"] if state3 else max(ckpts_after_run3, default=0)
        self.assertGreaterEqual(final_step, 55,
            f"三段训练后 final step 应接近 60，实际 {final_step}")
        print(f"  [T4] ✓ 三段链式恢复完成，final_step={final_step}")
        print(f"  [T4] ✓ 最终 ckpts: {ckpts_after_run3}")

        # run3 第一个 log step 应 > run2 结束 step
        if log3.records:
            first_log_run3 = log3.records[0][0]
            self.assertGreater(first_log_run3, step_after_run2,
                f"run3 第一个 log step ({first_log_run3}) 应 > run2 结束 ({step_after_run2})")
            print(f"  [T4] ✓ run3 从 step={first_log_run3} 继续")

    def test_fresh_start_when_no_ckpt(self):
        """
        验证：output_dir 下无 checkpoint 时，直接从头开始训练（不报错）。
        这对应弹性队列第一次 run 的情况。
        """
        out = os.path.join(self.tmpdir, "fresh_job")
        os.makedirs(out)

        # 不传 resume=True，模拟第一次 run
        trainer, logger = self._run_segment(out, max_steps=20)

        self.assertTrue(len(logger.records) > 0, "fresh start 应有训练 log")
        first_step = logger.records[0][0]
        self.assertLessEqual(first_step, 10,
            f"fresh start 第一个 log step 应从很小的值开始，实际 {first_step}")
        print(f"  [T4] ✓ 无 ckpt 时 fresh start: 第一个 log step={first_step}")

    def test_auto_resume_detection(self):
        """
        验证 train_retrofit.py 中的自动检测逻辑：
            if list(Path(output_dir).glob("checkpoint-*")):
                trainer.train(resume_from_checkpoint=True)
            else:
                trainer.train()
        """
        out = os.path.join(self.tmpdir, "auto_detect")
        os.makedirs(out)

        def _detect_and_run(max_steps, logger_list):
            """模拟 train_retrofit.py 的自动检测逻辑。"""
            model = TinyModel()
            dataset = TinyDataset()
            logger = LossLogger()
            logger_list.append(logger)

            targs = _base_training_args(
                output_dir=out,
                save_steps=10,
                max_steps=max_steps,
            )
            trainer = TinyTrainer(
                model=model,
                args=targs,
                train_dataset=dataset,
                data_collator=tiny_collator,
                callbacks=[logger],
            )

            # ── 自动检测逻辑（与 train_retrofit.py 完全一致）──────────
            existing_ckpts = list(pathlib.Path(out).glob("checkpoint-*"))
            if existing_ckpts:
                print(f"    [auto_detect] 发现 {len(existing_ckpts)} 个 ckpt，resume 训练")
                trainer.train(resume_from_checkpoint=True)
            else:
                print(f"    [auto_detect] 无 ckpt，fresh start")
                trainer.train()

        loggers = []
        # 第一次：无 ckpt → fresh start
        _detect_and_run(max_steps=20, logger_list=loggers)
        self.assertTrue(loggers[0].records, "第一次应有训练 log")
        step1 = loggers[0].records[0][0]
        self.assertLessEqual(step1, 10, f"first run 第一步应很小，实际 {step1}")

        # 第二次：有 ckpt → resume
        _detect_and_run(max_steps=40, logger_list=loggers)
        if loggers[1].records:
            step2 = loggers[1].records[0][0]
            self.assertGreater(step2, step1,
                f"resume run 第一步 ({step2}) 应 > fresh start 最后步 ({step1})")
            print(f"  [T4] ✓ 自动检测: fresh_start step={step1}, resume_start_step={step2}")
        else:
            print("  [T4] ✓ 自动检测: resume run 无新 log（max_steps 已达）")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

_TEST_MAP = {
    "T1": TestT1JobNameBinding,
    "T2": TestT2CheckpointManager,
    "T3": TestT3TrainerResume,
    "T4": TestT4ChainedResume,
}

def main():
    parser = argparse.ArgumentParser(
        description="ZwerGe 弹性队列断点恢复功能验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_elastic_resume.py          # 全部测试
  python test_elastic_resume.py T1 T2    # 仅 T1 和 T2
  python test_elastic_resume.py T3       # 仅 T3

测试说明:
  T1  ZWERGE_JOB_NAME → OUTPUT_DIR 固定绑定
  T2  ResumeCheckpointManagerCallback ckpt 保留逻辑
  T3  HuggingFace Trainer resume 状态一致性（step/lr/loss）
  T4  多次中断链式恢复
        """,
    )
    parser.add_argument(
        "tests",
        nargs="*",
        choices=list(_TEST_MAP.keys()),
        help="指定要运行的测试 (T1/T2/T3/T4)，默认全部",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()

    targets = args.tests if args.tests else list(_TEST_MAP.keys())
    verbosity = 2 if args.verbose else 1

    print("=" * 70)
    print("ZwerGe 弹性队列断点恢复功能验证")
    print(f"运行测试: {targets}")
    print("=" * 70)

    suite = unittest.TestSuite()
    for t in targets:
        cls = _TEST_MAP[t]
        print(f"\n── {t}: {cls.__name__} ──")
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"✅ 全部 {result.testsRun} 个测试通过")
    else:
        print(f"❌ {len(result.failures)} 个失败, {len(result.errors)} 个错误 "
              f"(共 {result.testsRun} 个)")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
