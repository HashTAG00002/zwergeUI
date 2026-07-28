#!/usr/bin/env python3
"""
submit_p2p_jobs.py — Launch ZWERGE-P2P eval/cache jobs via `hope run`.

Reuses eval_daemon.generate_hope_file / hope_run so the generated .hope files
are byte-identical to what the daemon produces.

Job matrix (canonical A8 checkpoint-2800 for every backbone):
  cache jobs    : 4 models × {ss_pro, ss_v2},  DECODE_STRATEGY=centroid, CACHE_POSTERIORS=1
                  → posteriors .pt for offline p2p_sweep.py (no backbone generate; fast)
  native-gate   : 2 Qwen3 models × {ss_pro, ss_v2}, DECODE_STRATEGY=p2p_native, CACHE_POSTERIORS=1
                  → caches native_point for the Posterior-Constrained Native gate

Usage:
  python submit_p2p_jobs.py --dry_run          # print generated hope files, do NOT submit
  python submit_p2p_jobs.py --only cache       # submit cache jobs
  python submit_p2p_jobs.py --only native      # submit native-gate jobs
  python submit_p2p_jobs.py                    # submit both
"""

import argparse
import glob
import os
import pathlib
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))   # for eval_daemon import

import eval_daemon as ed   # noqa: E402

ZWERGE_ROOT = pathlib.Path(_HERE).parent.resolve()
_CKPT_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
_OUT_BASE  = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/data/results"

# Canonical A8 retrofit checkpoints (checkpoint-2800, the last permanent one).
CKPTS = {
    "guiowl7b": f"{_CKPT_BASE}/guiowl7b_A8_cosmeta_ctx_exp001/checkpoint-2800",
    "uitars":   f"{_CKPT_BASE}/uitars_A8_cosmeta_ctx_exp001/checkpoint-2800",
    "guiowl":   f"{_CKPT_BASE}/guiowl_A8_cosmeta_ctx_exp001/checkpoint-2800",
    "uivenus":  f"{_CKPT_BASE}/uivenus_A8_cosmeta_ctx_exp003/checkpoint-2800",
}
# Qwen2.5 → eval_zwerge.hope ; Qwen3 → eval_zwerge_qwen3.hope
TEMPLATE = {
    "guiowl7b": "scripts/eval/eval_zwerge.hope",
    "uitars":   "scripts/eval/eval_zwerge.hope",
    "guiowl":   "scripts/eval/eval_zwerge_qwen3.hope",
    "uivenus":  "scripts/eval/eval_zwerge_qwen3.hope",
}
BENCHES = ["ss_pro", "ss_v2"]
ZOOM_BENCHES = ["ss_pro", "osworld_g", "ui_vision"]
# Two elastic queues available — distribute jobs across both to avoid blocking.
QUEUES = [
    "root.zw05_training_cluster.hadoop-vision.elastic_job",
    "root.zw05_training_cluster.hadoop-aipnlp.elastic",
]


def _tmp_dir() -> pathlib.Path:
    d = pathlib.Path(_OUT_BASE) / ".p2p_job_tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _done_already(out_final, bench):
    """Skip if a job already created this output dir (finished, running, or cached)."""
    if not os.path.isdir(out_final):
        return False
    if os.path.exists(os.path.join(out_final, f"{bench}_layerwise_summary.json")):
        return True
    det = os.path.join(out_final, "details", bench)
    if os.path.isdir(det):
        return True   # details/ exists → job is running or finished
    # output dir exists but no details yet (job just started) → still skip to avoid duplicates
    return True


def _make_job(model, bench, decode, cache, logger, queue=None):
    import glob
    env = {
        "MODEL_TYPE": model,
        "CKPT": CKPTS[model],
        "DECODE_STRATEGY": decode,
        "SKIP_VIS": "1",
    }
    if cache:
        env["CACHE_POSTERIORS"] = "1"
    tag = "cache" if decode == "centroid" else decode
    out_final = f"{_OUT_BASE}/p2p_{tag}/{model}_{bench}"
    env["OUTPUT_DIR_FINAL"] = out_final
    template = str(ZWERGE_ROOT / TEMPLATE[model])
    out_hope = _tmp_dir() / f"p2p_{tag}_{model}_{bench}.hope"
    worker_script = ed.generate_hope_file(
        template_path=template, output_path=str(out_hope),
        env_vars=env, positional_args=bench, queue=queue,
    )
    return out_hope, env, worker_script, out_final, bench


def main():
    import glob
    p = argparse.ArgumentParser(description="Submit ZWERGE-P2P eval/cache hope jobs")
    p.add_argument("--dry_run", action="store_true", help="print generated hope files, do not submit")
    p.add_argument("--only", choices=["cache", "native", "zoom", "both"], default="both",
                   help="cache = centroid+cache_posteriors (all 4 models); native = p2p_native (Qwen3 only); "
                        "zoom = p2p_zoom (guiowl on SS-Pro/OSWorld-G/UI-Vision, beat-base sweep)")
    p.add_argument("--zoom_compare", action="store_true",
                   help="with --only zoom, also submit legacy zoom_backbone (max-region) for comparison")
    p.add_argument("--zoom_gated", action="store_true",
                   help="with --only zoom, also submit p2p_zoom_gated (native-fallback, guaranteed >= base)")
    p.add_argument("--zoom_model", default="guiowl", choices=list(CKPTS.keys()),
                   help="backbone for the zoom sweep (default guiowl = GUI-Owl-1.5-8B)")
    p.add_argument("--skip_existing", action="store_true", default=True,
                   help="skip jobs whose output dir already has results/posteriors (default on)")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--queue", default="auto",
                   help="queue override: 'auto' (default) alternates across the two elastic queues, "
                        "or a full queue name, or 'none' to use the template default")
    args = p.parse_args()

    logger = ed.setup_logger()

    def _q(i):
        if args.queue == "none":
            return None
        if args.queue == "auto":
            return QUEUES[i % len(QUEUES)]
        return args.queue

    jobs = []
    _qi = 0
    if args.only in ("cache", "both"):
        for m in ["guiowl7b", "uitars", "guiowl", "uivenus"]:
            for b in BENCHES:
                jobs.append(_make_job(m, b, "centroid", cache=True, logger=logger, queue=_q(_qi))); _qi += 1
    if args.only in ("native", "both"):
        for m in ["guiowl", "uivenus"]:           # native gate only for Qwen3
            for b in BENCHES:
                jobs.append(_make_job(m, b, "p2p_native", cache=True, logger=logger, queue=_q(_qi))); _qi += 1
    if args.only in ("zoom", "both"):
        for b in ZOOM_BENCHES:
            jobs.append(_make_job(args.zoom_model, b, "p2p_zoom", cache=False, logger=logger, queue=_q(_qi))); _qi += 1
            if args.zoom_compare:
                jobs.append(_make_job(args.zoom_model, b, "zoom_backbone", cache=False, logger=logger, queue=_q(_qi))); _qi += 1
            if args.zoom_gated:
                jobs.append(_make_job(args.zoom_model, b, "p2p_zoom_gated", cache=False, logger=logger, queue=_q(_qi))); _qi += 1

    # Submit jobs that aren't already done/running.
    to_submit = []
    skipped = 0
    for out_hope, env, ws, out_final, bench in jobs:
        if args.skip_existing and _done_already(out_final, bench):
            logger.info(f"[skip] {out_hope.name} already has results")
            skipped += 1
            continue
        to_submit.append((out_hope, env, ws))

    logger.info(f"Prepared {len(to_submit)} job(s) to submit, {skipped} skipped ({'DRY RUN' if args.dry_run else 'SUBMIT'})")
    for out_hope, env, ws in to_submit:
        logger.info("-" * 70)
        logger.info(f"hope file : {out_hope}")
        logger.info(f"worker    : {ws}")
        logger.info(f"output    : {env['OUTPUT_DIR_FINAL']}")
        if args.dry_run:
            continue
        ed.hope_run(out_hope, logger, dry_run=False)


if __name__ == "__main__":
    main()
