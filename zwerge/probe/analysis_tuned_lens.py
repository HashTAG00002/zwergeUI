"""
analysis_tuned_lens.py — P0-2b: tuned-lens control for the serialization lens.

Reviewer concern addressed:
  eKpD-W3 "applying a final LM head to intermediate states is a logit-lens-style
  analysis that can be heavily confounded by layerwise alignment to the output
  embedding space... controls such as... calibrated tuned lenses."

The raw serialization lens applies norm+lm_head to each layer's hidden state.
Late layers are more aligned with the output space, so their NLL is lower for
reasons unrelated to coordinate serialization. A tuned lens learns, per layer,
a lightweight affine A_ℓ·h_ℓ + b_ℓ → h_final (the final-layer representation),
fit by ridge regression on a train split, then applies the FROZEN norm+lm_head.
If the tuned-lens plateau L†_tuned matches the raw-lens L†_raw, the plateau
position is robust to alignment calibration (not just an alignment artifact).

Standalone: own forward via _forward_hidden_states_for_grounding; reuses
probe_utils pure helpers (find_coord_token_positions, build_native_gt_response,
_get_lm_norm/_get_lm_head). No edits to inference_base / probe_utils.
"""
import argparse, gc, json, os, sys, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PROBE_DIR, "../.."))
for _d in [_PROBE_DIR, os.path.join(_REPO_ROOT, "zwerge", "eval"),
           os.path.join(_REPO_ROOT, "zwerge", "src")]:
    if _d not in sys.path: sys.path.insert(0, _d)

from probe_utils import (get_inference_class, load_eval_records, sample_records,
                         find_coord_token_positions, build_native_gt_response,
                         _get_lm_norm, _get_lm_head, make_next_token_targets)
from inference_base import build_zwerge_inputs, _ZOOM_NOT_SET

OUT_DIR = Path(_PROBE_DIR) / "outputs"
TBL_DIR = OUT_DIR / "tables"; TBL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/datasets/evaluation"
CKPT_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/ckpt/zwerge"
MODEL_CFG = {
    "guiowl7b": ("GUI-Owl-7B", "guiowl7b_A8_cosmeta_ctx_exp001", "qwen25"),
    "uitars":   ("UI-TARS-1.5-7B", "uitars_A8_cosmeta_ctx_exp001", "qwen25"),
}


def _build_native_inputs(grounder, image, instruction, native_response, max_pixels, device):
    sys_msg = grounder._zoom_native_system_message
    if sys_msg is _ZOOM_NOT_SET: sys_msg = grounder.system_message
    user_tmpl = grounder._zoom_native_user_template
    if user_tmpl is _ZOOM_NOT_SET: user_tmpl = grounder.user_prompt_template
    return build_zwerge_inputs(image=image, instruction=instruction,
        processor=grounder.processor, system_message=sys_msg,
        ground_response=native_response, max_pixels=max_pixels,
        user_prompt_template=user_tmpl)


def _img_path(rec, root):
    for k in ("image_path", "img_path", "image_filename", "image"):
        v = rec.get(k)
        if v and os.path.exists(os.path.join(root, v)): return os.path.join(root, v)
    return None


def collect(model_type, n_samples=200, max_pixels=6_400_000, device_str="cuda:0", seed=42):
    disp, mdir, _ = MODEL_CFG[model_type]
    ckpt = os.path.join(CKPT_BASE, mdir, "checkpoint-2800")
    eval_json = os.path.join(DATA_DIR, "ScreenSpot-Pro", "eval.json")
    image_root = os.path.join(DATA_DIR, "ScreenSpot-Pro")
    device = torch.device(device_str)
    print(f"[tuned] loading {disp} from {ckpt}")
    InfClass = get_inference_class(model_type)
    grounder = InfClass.from_checkpoint(ckpt, device=device_str, max_pixels=max_pixels)
    grounder.model.eval(); grounder.model.to(device)
    probe_layers = list(grounder.model.layerwise_grounding_head.probe_layers)
    # final layer = last probe layer (output of all_hs[final+1]); for tuned-lens target
    final_layer = probe_layers[-1]
    print(f"[tuned] probe_layers={probe_layers} final={final_layer}")

    norm = _get_lm_norm(grounder.model); lm_head = _get_lm_head(grounder.model)
    _dtype = next(norm.parameters()).dtype

    records = sample_records(load_eval_records(eval_json), n_samples, seed=seed)
    # storage
    h_per_layer = {ℓ: [] for ℓ in probe_layers}   # list of [n_coord, d] float16 cpu
    h_final_list = []; coord_ids_all = []; n_ok = 0
    for rec in tqdm(records, desc=f"tuned/{model_type}"):
        ip = _img_path(rec, image_root)
        if ip is None: continue
        try: img = Image.open(ip).convert("RGB")
        except Exception: continue
        instruction = rec.get("instruction") or rec.get("query") or ""
        gt_bbox = rec.get("gt_bbox") or rec.get("bbox"); img_size = rec.get("image_size")
        if gt_bbox is None or img_size is None: continue
        try: native_resp = build_native_gt_response(gt_bbox, img_size, model_type)
        except ValueError: continue
        try: ni = _build_native_inputs(grounder, img, instruction, native_resp, max_pixels, device)
        except Exception as e: warnings.warn(f"inputs: {e}"); continue
        input_ids_1d = ni["input_ids"][0]
        coord_pos, coord_ids, _proto_pos, _proto_ids = find_coord_token_positions(
            input_ids_1d, grounder.processor.tokenizer, model_type, native_response=native_resp)
        if len(coord_ids) == 0: continue
        pred_pos, shifted_ids = make_next_token_targets(coord_pos, coord_ids)
        if not pred_pos: continue
        # move inputs to device
        kw = {k: (v.to(device) if k in ("input_ids","attention_mask","image_grid_thw","mm_token_type_ids")
                   else (v.to(device, dtype=grounder.model.dtype) if k=="pixel_values" else v))
              for k,v in ni.items() if k in ("input_ids","attention_mask","pixel_values","image_grid_thw","mm_token_type_ids")}
        try:
            with torch.no_grad():
                all_hs = grounder.model._forward_hidden_states_for_grounding(
                    input_ids=kw.get("input_ids"), attention_mask=kw.get("attention_mask"),
                    pixel_values=kw.get("pixel_values"), image_grid_thw=kw.get("image_grid_thw"),
                    device=device, mm_token_type_ids=kw.get("mm_token_type_ids"))
        except Exception as e:
            warnings.warn(f"fwd: {e}"); torch.cuda.empty_cache(); del ni; continue
        tgt = torch.tensor(shifted_ids, dtype=torch.long, device=device)
        hf = None
        try:
            hf = all_hs[final_layer+1][0] if all_hs[final_layer+1] is not None else None
            if hf is not None:
                h_final_list.append(hf[pred_pos].to(torch.float16).cpu().numpy())
                coord_ids_all.append(shifted_ids)
            for ℓ in probe_layers:
                raw = all_hs[ℓ+1]
                if raw is None: continue
                hs = raw[0] if raw.dim()==3 else raw
                h_per_layer[ℓ].append(hs[pred_pos].to(torch.float16).cpu().numpy())
            n_ok += 1
        finally:
            del all_hs, ni; gc.collect(); torch.cuda.empty_cache()
    print(f"[tuned] collected {n_ok} samples")
    # stack: each layer -> [n_total, n_coord_max, d]; but n_coord varies. Flatten across samples.
    def _flatten(layer_lists):
        return np.concatenate([np.concatenate(samps,0) for samps in layer_lists if samps],0) if any(layer_lists) else np.array([])
    # Build flat arrays matched by sample order. Simpler: collect per-sample lists already aligned.
    # h_per_layer[ℓ] is list of [n_coord_i, d] per sample; flatten -> [N_total, d]
    H = {ℓ: np.concatenate(h_per_layer[ℓ], 0).astype(np.float32) for ℓ in probe_layers if h_per_layer[ℓ]}
    Hf = np.concatenate(h_final_list, 0).astype(np.float32) if h_final_list else np.array([])
    Y_ids = np.concatenate(coord_ids_all, 0)  # [N_total]
    # filter to layers present
    layers = [ℓ for ℓ in probe_layers if ℓ in H]
    np.savez(OUT_DIR / f"tuned_lens_{model_type}.npz",
             **{f"H_{ℓ}": H[ℓ] for ℓ in layers}, Hf=Hf, Y_ids=Y_ids,
             layers=np.array(layers), n_ok=n_ok)
    print(f"[tuned] saved {OUT_DIR}/tuned_lens_{model_type}.npz  (N={len(Y_ids)})")
    return model_type, layers, H, Hf, Y_ids, grounder, norm, lm_head, _dtype, device


def fit_eval(model_type, layers, H, Hf, Y_ids, grounder, norm, lm_head, _dtype, device,
             lam=10.0, train_frac=0.8, seed=42):
    rng = np.random.default_rng(seed)
    N = len(Y_ids)
    idx = rng.permutation(N)
    n_tr = int(N*train_frac)
    tr, te = idx[:n_tr], idx[n_tr:]
    # targets on GPU
    yte = torch.tensor(Y_ids[te], dtype=torch.long, device=device)
    Hf_t = torch.tensor(Hf, dtype=torch.float32, device=device)
    results = {}
    for ℓ in layers:
        Hl = torch.tensor(H[ℓ], dtype=torch.float32, device=device)
        Htr = Hl[tr]; Hte = Hl[te]
        # ridge: W = (XᵀX + λI)⁻¹ Xᵀ Y, X=[h,1], Y=Hf
        Xtr = torch.cat([Htr, torch.ones(len(tr),1,device=device)],1)  # [n,d+1]
        d1 = Xtr.shape[1]
        XtX = Xtr.T @ Xtr
        XtX += lam * torch.eye(d1, device=device)
        XtY = Xtr.T @ Hf_t[tr]   # [d+1, d]
        W = torch.linalg.solve(XtX, XtY)   # [d+1, d]
        # eval tuned NLL on held-out
        Xte = torch.cat([Hte, torch.ones(len(te),1,device=device)],1)
        h_tuned = Xte @ W   # [n_te, d]
        with torch.no_grad():
            h_n = norm(h_tuned.to(_dtype))
            logits = lm_head(h_n).float()  # [n_te, vocab]
            tuned_nll = float(F.cross_entropy(logits, yte[:logits.shape[0]]).item())
        # raw NLL on held-out (norm+lm_head on raw h_ℓ)
        with torch.no_grad():
            h_raw_n = norm(Hte.to(_dtype))
            raw_logits = lm_head(h_raw_n).float()
            raw_nll = float(F.cross_entropy(raw_logits, yte[:raw_logits.shape[0]]).item())
        results[ℓ] = {"raw_nll": raw_nll, "tuned_nll": tuned_nll}
        print(f"  L{ℓ}: raw_nll={raw_nll:.3f} tuned_nll={tuned_nll:.3f}")
        del Hl, Xtr, Hte, Xte, h_tuned, h_n, logits, W, XtX, XtY
        torch.cuda.empty_cache()
    # L† = argmin NLL
    L_raw = min(results, key=lambda ℓ: results[ℓ]["raw_nll"])
    L_tuned = min(results, key=lambda ℓ: results[ℓ]["tuned_nll"])
    print(f"\n[tuned] L†_raw = L{L_raw}  |  L†_tuned = L{L_tuned}  (match={L_raw==L_tuned})")
    json.dump({"model": model_type, "lam": lam, "N": int(N), "n_train": n_tr,
               "layers": list(results.keys()),
               "per_layer": {str(k): v for k,v in results.items()},
               "L_dagger_raw": int(L_raw), "L_dagger_tuned": int(L_tuned)},
              (OUT_DIR / f"tuned_lens_{model_type}.json").open("w"), indent=2)
    return results, L_raw, L_tuned


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=list(MODEL_CFG))
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_pixels", type=int, default=6_400_000)
    args = p.parse_args()
    out = collect(args.model_type, args.n_samples, args.max_pixels, args.device)
    fit_eval(*out, lam=args.lam)
