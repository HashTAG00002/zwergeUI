#!/usr/bin/env python3
"""
eval_daemon.py — Train + Async Eval Orchestrator for ZwerGe

Submits a training hope job, then watches the output directory for new
permanent checkpoints (marked by `is_permanent_ckpt` file) and submits
eval hope jobs asynchronously via `hope run`.

Usage (inside a tmux session on CodeLab):

    # Submit training + start watching for eval:
    python eval_daemon.py --config experiments/A3_guiowl.yaml

    # Watch only (training already submitted):
    python eval_daemon.py --config experiments/A3_guiowl.yaml --skip_train

    # Dry-run: print generated hope files without submitting:
    python eval_daemon.py --config experiments/A3_guiowl.yaml --dry_run

How it works:
  1. Reads experiment YAML config.
  2. Generates a temp train .hope file (injecting ZWERGE_JOB_NAME + env),
     submits via `hope run`.
  3. Every poll_interval_sec:
       - Scans OUTPUT_DIR/checkpoint-*/is_permanent_ckpt for new ckpts.
       - For each unseen ckpt: generates temp eval .hope with CKPT and
         OUTPUT_DIR_FINAL injected, submits via `hope run`.
       - Records submission in OUTPUT_DIR/.eval_submitted.json (idempotent).
  4. Exits when training is complete AND all permanent ckpts are submitted.
"""

import argparse
import datetime
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    print("PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


ZWERGE_ROOT = pathlib.Path(__file__).parent.resolve()
_BASE_CKPT_DIR = (
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
)

# ── 网络代理 ─────────────────────────────────────────────────────────────────
# CodeLab 容器需要通过内网代理访问外网（WandB、hope CLI 等）。
# 如果环境变量已设，不覆盖；否则注入默认值。
_PROXY = "http://10.70.11.143:8412"
os.environ.setdefault("http_proxy",  _PROXY)
os.environ.setdefault("https_proxy", _PROXY)
# WandB 在 Python 中读取 WANDB_API_KEY 和 WANDB_PROJECT 环境变量
os.environ.setdefault("WANDB_API_KEY",
    "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")
os.environ.setdefault("WANDB_PROJECT", "zwerge")
os.environ.setdefault("WANDB_ENTITY",  "yangwenkui-cas")


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    exp_name = cfg["experiment_name"]
    # Expand ${experiment_name} placeholder anywhere in config values
    def _expand(obj):
        if isinstance(obj, str):
            return obj.replace("${experiment_name}", exp_name)
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        return obj
    return _expand(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Hope file manipulation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_bash_script(hope_path: str) -> str:
    """Return the absolute bash script path from the worker.script line."""
    with open(hope_path) as f:
        for line in f:
            if line.strip().startswith("worker.script"):
                m = re.search(r"bash\s+(/[^\s]+\.sh)", line)
                if m:
                    return m.group(1)
    raise ValueError(f"Cannot find 'bash /path/to/script.sh' in {hope_path}")


def generate_hope_file(
    template_path: str,
    output_path: str,
    env_vars: Dict[str, str],
    positional_args: str = "",
    bash_script: Optional[str] = None,
    queue: Optional[str] = None,
) -> str:
    """
    Copy a .hope template, replacing worker.script and optionally the queue.

    bash_script: override the bash script path from the template.
    queue:       override the active (uncommented) queue = ... line.
                 Useful when different experiments need different resource pools.
    Returns the worker.script value written (for logging).
    """
    if bash_script is None:
        bash_script = _extract_bash_script(template_path)
    env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items())
    new_worker_script = f"{env_prefix} bash {bash_script}"
    if positional_args:
        new_worker_script += f" {positional_args}"

    with open(template_path) as f:
        content = f.read()

    new_content = re.sub(
        r"^(worker\.script\s*=\s*).*$",
        f"worker.script = {new_worker_script}",
        content,
        flags=re.MULTILINE,
    )

    if queue:
        # Replace the active (uncommented) queue line; leave #queue comment lines intact
        new_content = re.sub(
            r"^queue\s*=\s*.+$",
            f"queue = {queue}",
            new_content,
            flags=re.MULTILINE,
        )

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(new_content)
    return new_worker_script


# ─────────────────────────────────────────────────────────────────────────────
# hope run
# ─────────────────────────────────────────────────────────────────────────────

def hope_run(hope_file: pathlib.Path, logger: logging.Logger, dry_run: bool = False) -> bool:
    """Run `hope run <filename>` from the hope file's parent directory."""
    if dry_run:
        logger.info(f"  [dry_run] would run: hope run {hope_file.name}  (cwd={hope_file.parent})")
        return True

    logger.info(f"  hope run {hope_file.name}  (cwd={hope_file.parent})")
    result = subprocess.run(
        ["hope", "run", hope_file.name],
        cwd=str(hope_file.parent),
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        logger.error(f"  hope run FAILED (rc={result.returncode}): {output}")
        return False  # keep the file for debugging
    if output:
        logger.info(f"  {output}")
    # Clean up temp hope file on success to prevent accumulation
    try:
        hope_file.unlink(missing_ok=True)
    except Exception:
        pass
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint detection
# ─────────────────────────────────────────────────────────────────────────────

def find_permanent_ckpts(output_dir: pathlib.Path) -> List[pathlib.Path]:
    """Return sorted list of checkpoint dirs that have `is_permanent_ckpt` marker."""
    result = []
    for d in sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    ):
        if d.is_dir() and (d / "is_permanent_ckpt").exists():
            result.append(d)
    return result


def load_submitted(output_dir: pathlib.Path) -> dict:
    path = output_dir / ".eval_submitted.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_submitted(output_dir: pathlib.Path, submitted: dict):
    (output_dir / ".eval_submitted.json").write_text(
        json.dumps(submitted, indent=2, ensure_ascii=False)
    )


# ─────────────────────────────────────────────────────────────────────────────
# WandB eval logging
# ─────────────────────────────────────────────────────────────────────────────

def _read_wandb_run_id(output_dir: pathlib.Path) -> Optional[str]:
    """Read the wandb run ID saved by SaveWandbRunIdCallback during training."""
    p = output_dir / ".wandb_run_id"
    if p.exists():
        run_id = p.read_text().strip()
        return run_id if run_id else None
    return None


def _extract_wandb_metrics(summary: dict) -> dict:
    """
    Extract key metrics from layerwise_all_summary.json for WandB.

    Per bench: fusion_hit1, fusion_overlap1, best_layer_hit1, best_layer_overlap1.
    Skips group-level breakdowns (too verbose) and per-layer arrays.
    """
    metrics: Dict[str, float] = {}
    for bench_key, bench_data in summary.items():
        if not isinstance(bench_data, dict):
            continue
        pfx = f"eval/{bench_key}"

        fusion = bench_data.get("fusion_acc", {})
        if isinstance(fusion, dict):
            for src, dst in [
                ("hit_top1",      "fusion_hit1"),
                ("overlap_top1",  "fusion_overlap1"),
                ("hit_topk",      "fusion_hit_topk"),
                ("overlap_topk",  "fusion_overlap_topk"),
            ]:
                if src in fusion and isinstance(fusion[src], (int, float)):
                    metrics[f"{pfx}/{dst}"] = float(fusion[src])

        sorted_layers = bench_data.get("layer_accs_sorted", [])
        if sorted_layers and isinstance(sorted_layers[0], dict):
            best = sorted_layers[0]
            for src, dst in [
                ("hit_top1",     "best_layer_hit1"),
                ("overlap_top1", "best_layer_overlap1"),
                ("layer_idx",    "best_layer_idx"),
            ]:
                if src in best and isinstance(best[src], (int, float)):
                    metrics[f"{pfx}/{dst}"] = float(best[src])

    return metrics


def _wandb_log_eval(
    metrics: dict,
    step: int,
    exp_name: str,
    wandb_run_id: Optional[str],
    cfg: dict,
    logger: logging.Logger,
) -> bool:
    """
    Log eval metrics to WandB, resuming the training run if possible.

    Uses wandb_run_id (from .wandb_run_id) to attach to the exact training run.
    Falls back to name-based resume if run_id is unavailable.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("[wandb] wandb not installed; skipping eval metric upload")
        return False

    project = cfg.get("eval", {}).get("wandb_project") or os.environ.get("WANDB_PROJECT", "zwerge")

    try:
        init_kwargs: dict = {"project": project, "reinit": True}
        if wandb_run_id:
            # Attach to the exact training run (written by SaveWandbRunIdCallback)
            init_kwargs["id"] = wandb_run_id
            init_kwargs["resume"] = "must"
        else:
            # Fallback: resume by name (creates a new run if not found)
            init_kwargs["name"] = exp_name
            init_kwargs["resume"] = "allow"

        run = wandb.init(**init_kwargs)
        wandb.log(metrics, step=step)
        wandb.finish()
        logger.info(
            f"[wandb] Logged {len(metrics)} eval metrics at step={step} "
            f"(run_id={run.id}, project={project})"
        )
        return True
    except Exception as e:
        logger.warning(f"[wandb] Failed to log eval metrics: {e}")
        return False


def _result_json_path(ckpt_path: pathlib.Path, cfg: dict) -> pathlib.Path:
    """Return the expected layerwise summary JSON path for a checkpoint.

    Results live at {ckpt_path}/results/ so everything for a checkpoint
    is co-located in one directory.
    """
    bench = cfg.get("eval", {}).get("bench", "all")
    base = ckpt_path / "results"
    if bench == "all":
        return base / "layerwise_all_summary.json"
    return base / f"{bench}_layerwise_summary.json"


# ─────────────────────────────────────────────────────────────────────────────
# Training completion detection
# ─────────────────────────────────────────────────────────────────────────────

def is_training_complete(output_dir: pathlib.Path, cfg: dict) -> bool:
    """
    Check if training is done by reading trainer_state.json.

    HF Trainer writes trainer_state.json to output_dir root via save_state()
    at the very end of training.  During training it also exists inside each
    checkpoint dir.  We prefer the root-level file as the definitive signal.
    """
    root_state = output_dir / "trainer_state.json"
    ckpt_states = sorted(
        output_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )

    state_path = root_state if root_state.exists() else (ckpt_states[-1] if ckpt_states else None)
    if state_path is None:
        return False

    # Track whether this is the root-level state (written only by save_state() at end of training)
    # vs. a checkpoint-level state (written at every save_steps, even mid-training).
    is_root_state = (state_path == root_state)

    try:
        state = json.loads(state_path.read_text())
    except Exception:
        return False

    train_cfg = cfg.get("train", {})
    max_steps = train_cfg.get("max_steps", -1)
    if max_steps and max_steps > 0:
        if state.get("global_step", 0) >= max_steps:
            return True

    num_epochs = train_cfg.get("num_train_epochs")
    if num_epochs and is_root_state:
        # Only trust epoch-based completion from the root-level trainer_state.json;
        # checkpoint-level states have epoch values that look "complete" before the final step.
        if state.get("epoch", 0.0) >= float(num_epochs) - 0.01:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent eval job count (best-effort, non-blocking)
# ─────────────────────────────────────────────────────────────────────────────

def count_running_eval_jobs(exp_name: str, logger: logging.Logger) -> int:
    """
    Count RUNNING hope jobs whose output contains exp_name.
    Returns 0 on any error so concurrent limit degrades gracefully (no submission blocking).

    NOTE: The exact format of `hope ls --status RUNNING` output is cluster-specific.
    If this always returns 0 (no concurrent limiting), run `hope ls` manually and check
    whether job names / descriptions include the experiment_name string, then adjust the
    matching logic (e.g. use a different field or grep pattern).
    """
    try:
        result = subprocess.run(
            ["hope", "ls", "--status", "RUNNING"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return 0
        if result.stdout.strip():
            logger.debug(f"[count_jobs] hope ls output (first 5 lines):\n"
                         + "\n".join(result.stdout.splitlines()[:5]))
        return sum(1 for line in result.stdout.splitlines() if exp_name in line)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Job submission helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tmp_dir(cfg: dict, output_dir: pathlib.Path) -> pathlib.Path:
    val = cfg.get("daemon", {}).get("tmp_dir")
    return pathlib.Path(val) if val else output_dir / ".eval_daemon_tmp"


def submit_train_job(cfg: dict, logger: logging.Logger, dry_run: bool = False) -> bool:
    train_cfg = cfg.get("train", {})
    if train_cfg.get("skip_submit", False):
        logger.info("[train] skip_submit=true, skipping training submission")
        return True

    template = str(ZWERGE_ROOT / train_cfg["hope_file"])
    if not pathlib.Path(template).exists():
        logger.error(f"[train] hope template not found: {template}")
        return False

    env_vars = {"ZWERGE_JOB_NAME": cfg["experiment_name"]}
    env_vars.update(train_cfg.get("env", {}))

    # Optional: override the bash script path (e.g. point to A7 script using A3 hope template)
    bash_script = train_cfg.get("bash_script")
    if bash_script and not pathlib.Path(bash_script).is_absolute():
        bash_script = str(ZWERGE_ROOT / bash_script)

    output_dir = pathlib.Path(
        cfg.get("watch", {}).get("output_dir", f"{_BASE_CKPT_DIR}/{cfg['experiment_name']}")
    )
    out_hope = _tmp_dir(cfg, output_dir) / f"train_{cfg['experiment_name']}.hope"
    queue = train_cfg.get("queue")
    worker_script = generate_hope_file(
        template, str(out_hope), env_vars, bash_script=bash_script, queue=queue
    )

    logger.info(f"[train] Generated hope: {out_hope}")
    if queue:
        logger.info(f"[train]   queue         = {queue}")
    logger.info(f"[train]   worker.script = {worker_script}")
    return hope_run(out_hope, logger, dry_run=dry_run)


def submit_eval_job(
    ckpt_path: pathlib.Path,
    cfg: dict,
    logger: logging.Logger,
    dry_run: bool = False,
) -> bool:
    eval_cfg = cfg.get("eval", {})
    template = str(ZWERGE_ROOT / eval_cfg["hope_template"])
    if not pathlib.Path(template).exists():
        logger.error(f"[eval] hope template not found: {template}")
        return False

    ckpt_name = ckpt_path.name  # e.g. checkpoint-400
    # Results live inside the checkpoint directory: {ckpt_path}/results/
    output_dir_final = ckpt_path / "results"

    env_vars: Dict[str, str] = {
        "CKPT": str(ckpt_path),
        "OUTPUT_DIR_FINAL": str(output_dir_final),
    }
    env_vars.update(eval_cfg.get("env", {}))

    bench = eval_cfg.get("bench", "all")
    step = ckpt_name.split("-")[1]

    output_dir = pathlib.Path(
        cfg.get("watch", {}).get("output_dir", f"{_BASE_CKPT_DIR}/{cfg['experiment_name']}")
    )
    out_hope = _tmp_dir(cfg, output_dir) / f"eval_{cfg['experiment_name']}_ckpt{step}.hope"
    queue = eval_cfg.get("queue")
    worker_script = generate_hope_file(
        template, str(out_hope), env_vars, positional_args=bench, queue=queue
    )

    logger.info(f"[eval] Generated hope for {ckpt_name}: {out_hope}")
    logger.info(f"[eval]   CKPT            = {ckpt_path}")
    logger.info(f"[eval]   OUTPUT_DIR_FINAL= {output_dir_final}")
    logger.info(f"[eval]   worker.script   = {worker_script}")
    return hope_run(out_hope, logger, dry_run=dry_run)


# ─────────────────────────────────────────────────────────────────────────────
# Main daemon loop
# ─────────────────────────────────────────────────────────────────────────────

def run_daemon(cfg: dict, logger: logging.Logger, dry_run: bool = False):
    exp_name = cfg["experiment_name"]
    eval_cfg = cfg.get("eval", {})
    watch_cfg = cfg.get("watch", {})

    output_dir = pathlib.Path(watch_cfg.get("output_dir", f"{_BASE_CKPT_DIR}/{exp_name}"))
    poll_interval = int(watch_cfg.get("poll_interval_sec", 60))
    max_concurrent = int(eval_cfg.get("max_concurrent_jobs", 2))

    logger.info("=" * 60)
    logger.info(f"experiment      : {exp_name}")
    logger.info(f"output_dir      : {output_dir}")
    logger.info(f"poll_interval   : {poll_interval}s")
    logger.info(f"max_concurrent  : {max_concurrent} eval jobs")
    if dry_run:
        logger.info("DRY RUN MODE — hope run commands will be printed but not executed")
    logger.info("=" * 60)

    # Phase 1: submit training
    if not cfg.get("train", {}).get("skip_submit", False):
        ok = submit_train_job(cfg, logger, dry_run=dry_run)
        if ok:
            logger.info("[train] Submitted OK — waiting 10s before watching...")
            if not dry_run:
                time.sleep(10)
        else:
            logger.warning("[train] Submission failed — proceeding to watch anyway")

    # Cached wandb run ID (written by SaveWandbRunIdCallback once training starts)
    _wandb_run_id: Optional[str] = None

    # Phase 2 + 3: watch loop
    done_rounds = 0
    while True:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Refresh wandb run ID if not yet found (training may not have started yet)
        if _wandb_run_id is None:
            _wandb_run_id = _read_wandb_run_id(output_dir)
            if _wandb_run_id:
                logger.info(f"[wandb] Found training run ID: {_wandb_run_id}")

        permanent_ckpts = find_permanent_ckpts(output_dir)
        submitted = load_submitted(output_dir)
        pending = [c for c in permanent_ckpts if c.name not in submitted]

        logger.info(
            f"[watch] {len(permanent_ckpts)} permanent ckpt(s) found, "
            f"{len(pending)} pending eval submission"
        )

        # ── Phase 2: submit eval jobs for new permanent ckpts ─────────────────
        for ckpt in pending:
            running = count_running_eval_jobs(exp_name, logger)
            if running >= max_concurrent:
                logger.info(
                    f"[eval] {running}/{max_concurrent} eval jobs already running, "
                    f"will retry next cycle"
                )
                break

            ok = submit_eval_job(ckpt, cfg, logger, dry_run=dry_run)
            if ok:
                submitted[ckpt.name] = {
                    "submitted_at": datetime.datetime.now().isoformat(),
                    "status": "submitted",
                    "ckpt_path": str(ckpt),
                    "wandb_logged": False,
                }
                save_submitted(output_dir, submitted)
                logger.info(f"[eval] Recorded submission of {ckpt.name}")
                if not dry_run:
                    time.sleep(3)
            else:
                logger.warning(f"[eval] Submit failed for {ckpt.name}, will retry next cycle")

        # ── Phase 3: upload completed eval results to WandB ───────────────────
        wandb_dirty = False
        for ckpt_name, record in submitted.items():
            if record.get("wandb_logged"):
                continue
            ckpt_path_str = record.get("ckpt_path")
            if not ckpt_path_str:
                continue  # old record without path info
            result_json = _result_json_path(pathlib.Path(ckpt_path_str), cfg)
            if not result_json.exists():
                continue  # eval job not done yet

            try:
                summary = json.loads(result_json.read_text())
            except Exception as e:
                logger.warning(f"[wandb] Cannot read {result_json}: {e}")
                continue

            metrics = _extract_wandb_metrics(summary)
            if not metrics:
                logger.warning(f"[wandb] No metrics extracted from {result_json}")
                record["wandb_logged"] = True  # avoid retry loop on empty file
                wandb_dirty = True
                continue

            step = int(ckpt_name.split("-")[1])
            ok = _wandb_log_eval(metrics, step, exp_name, _wandb_run_id, cfg, logger)
            if ok or dry_run:
                record["wandb_logged"] = True
                wandb_dirty = True

        if wandb_dirty:
            save_submitted(output_dir, submitted)

        # ── Exit condition ─────────────────────────────────────────────────────
        training_done = is_training_complete(output_dir, cfg)
        all_submitted = bool(permanent_ckpts) and all(
            c.name in submitted for c in permanent_ckpts
        )
        all_logged = all(
            submitted.get(c.name, {}).get("wandb_logged", False)
            for c in permanent_ckpts
        )

        if training_done and all_submitted and all_logged:
            done_rounds += 1
            logger.info(
                f"[daemon] Training complete, all ckpts submitted and logged "
                f"({done_rounds}/2 confirmation rounds)"
            )
            if done_rounds >= 2:
                logger.info("[daemon] All done. Exiting.")
                break
        else:
            done_rounds = 0
            if training_done:
                pending_log = sum(
                    1 for c in permanent_ckpts
                    if not submitted.get(c.name, {}).get("wandb_logged", False)
                )
                logger.info(
                    f"[daemon] Training complete — "
                    f"{sum(1 for c in permanent_ckpts if c.name not in submitted)} ckpt(s) "
                    f"pending submission, {pending_log} pending wandb log"
                )

        if dry_run:
            logger.info("[dry_run] Single cycle complete, exiting.")
            break

        time.sleep(poll_interval)


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("eval_daemon")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ZwerGe eval daemon: submit training + async eval via hope run"
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--skip_train", action="store_true",
        help="Skip training submission (watch mode only, training already running)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print generated hope files without calling hope run",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.skip_train:
        cfg.setdefault("train", {})["skip_submit"] = True

    # Auto-set log_file to OUTPUT_DIR/eval_daemon.log if not specified
    daemon_cfg = cfg.setdefault("daemon", {})
    if not daemon_cfg.get("log_file"):
        exp_name = cfg["experiment_name"]
        out_dir = cfg.get("watch", {}).get("output_dir", f"{_BASE_CKPT_DIR}/{exp_name}")
        daemon_cfg["log_file"] = str(pathlib.Path(out_dir) / "eval_daemon.log")

    logger = setup_logger(daemon_cfg["log_file"])
    logger.info(f"Config: {args.config}")
    logger.info(f"Log:    {daemon_cfg['log_file']}")

    try:
        run_daemon(cfg, logger, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
