"""
analysis_random_instruction.py — P0-3 (inference-time): random-instruction control.

Reviewer concern addressed:
  eKpD-W2 / eKpD-Comments "random instruction" control; reinforces Finding 3
  (the mid-layer posterior is instruction-conditioned, not a learned position prior).

TRAINING-FREE: uses the existing A8 CrossAttn probe (no retraining). For each
sample we run the probe twice on the SAME image:
  (a) real instruction       → hit@1_real, GT-mass_real
  (b) a RANDOM instruction   → hit@1_rand, GT-mass_rand   (instruction from an
                               unrelated sample, shuffled across the batch)
If the probe were merely a learned position prior (e.g. "always click center" /
"click the most-common icon location"), the random-instruction posterior would
NOT collapse---hit@1_rand would stay high. If the probe reads instruction-
conditioned spatial info, randomizing the instruction must collapse hit@1 toward
chance and drop GT-mass. This is the cheap, full-benchmark version of the
instruction-switch counterfactual (Finding 3, n=150 pairs).
"""
import argparse, gc, json, os, sys, warnings
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PROBE_DIR, "../.."))
for _d in [_PROBE_DIR, os.path.join(_REPO_ROOT, "zwerge", "eval"),
           os.path.join(_REPO_ROOT, "zwerge", "src")]:
    if _d not in sys.path: sys.path.insert(0, _d)
from probe_utils import (get_inference_class, load_eval_records, sample_records,
                         compute_target_mass)

OUT_DIR = Path(_PROBE_DIR) / "outputs"
TBL_DIR = OUT_DIR / "tables"; TBL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation"
CKPT_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
MODEL_CFG = {
    "guiowl7b": ("GUI-Owl-7B", "guiowl7b_A8_cosmeta_ctx_exp001", "qwen25"),
    "uitars":   ("UI-TARS-1.5-7B", "uitars_A8_cosmeta_ctx_exp001", "qwen25"),
}


def _img_path(rec, root):
    for k in ("image_path", "img_path", "image_filename", "image"):
        v = rec.get(k)
        if v and os.path.exists(os.path.join(root, v)): return os.path.join(root, v)
    return None


def _bbox_norm(rec):
    v = rec.get("gt_bbox_norm")
    if v and len(v) == 4:
        vals = [float(x) for x in v]
        if max(vals) > 1.5: vals = [x/1000.0 for x in vals]
        return tuple(vals)
    b = rec.get("gt_bbox") or rec.get("bbox"); sz = rec.get("image_size")
    if b and sz and len(b)==4 and len(sz)==2:
        return (b[0]/sz[0], b[1]/sz[1], b[2]/sz[0], b[3]/sz[1])
    return None


def run(model_type, n_samples=200, device_str="cuda:0", seed=42):
    disp, mdir, _ = MODEL_CFG[model_type]
    ckpt = os.path.join(CKPT_BASE, mdir, "checkpoint-2800")
    eval_json = os.path.join(DATA_DIR, "ScreenSpot-Pro", "eval.json")
    image_root = os.path.join(DATA_DIR, "ScreenSpot-Pro")
    device = torch.device(device_str)
    print(f"[rand-instr] loading {disp}")
    InfClass = get_inference_class(model_type)
    grounder = InfClass.from_checkpoint(ckpt, device=device_str)
    grounder.model.eval(); grounder.model.to(device)

    records = sample_records(load_eval_records(eval_json), n_samples, seed=seed)
    # shuffle instructions across samples (rotation by half the list, seed-shuffled)
    rng = np.random.default_rng(seed)
    instrs = [r.get("instruction") or r.get("query") or "" for r in records]
    perm = rng.permutation(len(instrs))
    # ensure no sample gets its own instruction
    rand_instrs = [instrs[perm[(i + len(instrs)//2) % len(instrs)]] for i in range(len(instrs))]

    real_hit = []; rand_hit = []; real_mass = []; rand_mass = []; n_ok = 0
    for i, rec in enumerate(tqdm(records, desc=f"rand/{model_type}")):
        ip = _img_path(rec, image_root)
        if ip is None: continue
        try: img = Image.open(ip).convert("RGB")
        except Exception: continue
        instr_real = instrs[i]; instr_rand = rand_instrs[i]
        bn = _bbox_norm(rec)
        if bn is None: continue
        try:
            with torch.no_grad():
                pred_real = grounder.predict_layerwise(img, instr_real, device=device)
                pred_rand = grounder.predict_layerwise(img, instr_rand, device=device)
        except Exception as e:
            warnings.warn(f"fwd fail: {e}"); torch.cuda.empty_cache(); continue
        # use the FUSION posterior (p_final) — the deployable output
        n_w = pred_real["n_width"]; n_h = pred_real["n_height"]
        p_real = pred_real["p_final"]; p_rand = pred_rand["p_final"]
        # fusion hit@1 / mass (point-in-box)
        # find the argmax patch -> point for each
        def _hit(p):
            idx = int(p.argmax())
            y, x = idx // n_w, idx % n_w
            px, py = (x+0.5)/n_w, (y+0.5)/n_h
            x1,y1,x2,y2 = bn
            return 1.0 if (x1<=px<=x2 and y1<=py<=y2) else 0.0
        real_hit.append(_hit(p_real)); rand_hit.append(_hit(p_rand))
        real_mass.append(compute_target_mass(p_real, bn, n_w, n_h))
        rand_mass.append(compute_target_mass(p_rand, bn, n_w, n_h))
        del pred_real, pred_rand; gc.collect(); torch.cuda.empty_cache()
        n_ok += 1
    real_hit = 100*np.mean(real_hit); rand_hit = 100*np.mean(rand_hit)
    real_mass = float(np.mean(real_mass)); rand_mass = float(np.mean(rand_mass))
    print(f"[rand-instr] {disp} n={n_ok}: hit@1 real={real_hit:.1f} rand={rand_hit:.1f}  "
          f"GT-mass real={real_mass:.3f} rand={rand_mass:.3f}")
    out = {"model": disp, "model_type": model_type, "n": n_ok,
           "hit1_real": real_hit, "hit1_rand": rand_hit,
           "mass_real": real_mass, "mass_rand": rand_mass}
    json.dump(out, (OUT_DIR / f"random_instruction_{model_type}.json").open("w"), indent=2)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=list(MODEL_CFG))
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    run(a.model_type, a.n_samples, a.device)
