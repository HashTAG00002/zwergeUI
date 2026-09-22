"""Gate-C runtime audit for the SVD Stage-1 experiment.

Runs on ONE real backbone per invocation:
  1) hidden-state contract audit (what does hidden_states[l+1] actually equal?)
  2) module-level SVD init verification on the real checkpoint
     (donor mapping / backbone untouched / energy retention / norm scale vs Xavier)

Writes JSON evidence next to this script. Read-only w.r.t. the repo and checkpoints.

Usage (GUI-Owl-7B, gui_actor env):
  python run_gatec_audit.py --model guiowl7b
Usage (GUI-Owl-1.5-8B, qwen3 env):
  python run_gatec_audit.py --model guiowl15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

CODE_ROOT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/code"
sys.path.insert(0, os.path.join(CODE_ROOT, "zwerge/src"))
sys.path.insert(0, os.path.join(CODE_ROOT, "docs/oracle_new/svd"))

import torch  # noqa: E402
import transformers  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

from paired_qk_svd import audit_hidden_state_contract, initialize_a7_probes  # noqa: E402
from zwerge_retrofit.modeling_base import LayerWiseGroundingHead  # noqa: E402

CONFIGS = {
    "guiowl7b": {
        "path": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/.hdd/models/huggingface.co/GUI_Agents/GUI-Owl-7B",
        "d_model": 3584,
        "probe_layers": list(range(14, 28)),
        "audit_layers": [14, 20, 27],
        "expected_energy": 1.0,
        "expected_qk_norm": False,
        "expected_bias": True,
        "out": "hidden_contract_guiowl7b.json",
        "out_init": "init_verify_guiowl7b.json",
    },
    "guiowl15": {
        "path": "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/models/huggingface.co/GUI_Agents/GUI-Owl-1.5-8B-Instruct",
        "d_model": 4096,
        "probe_layers": list(range(18, 36)),
        "audit_layers": [18, 26, 35],
        "expected_energy": None,  # truncated; record actual
        "expected_qk_norm": True,
        "expected_bias": False,
        "out": "hidden_contract_guiowl15.json",
        "out_init": "init_verify_guiowl15.json",
    },
}

IMG_DIR = ("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/"
           "datasets/evaluation/ScreenSpot-Pro/images")
HERE = os.path.dirname(os.path.abspath(__file__))


def find_sample_image() -> str:
    for sub in sorted(os.listdir(IMG_DIR)):
        d = os.path.join(IMG_DIR, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                return os.path.join(d, f)
    raise FileNotFoundError("no sample image under ScreenSpot-Pro/images")


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--device", default="cuda:0",
                    help="cuda:0 when GPU has room; cpu fallback (contract is device-independent)")
    args = ap.parse_args()
    cfg = CONFIGS[args.model]

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        cfg["path"], min_pixels=3136, max_pixels=256 * 28 * 28,
    )

    img_path = find_sample_image()
    img = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Locate the confirmation button and describe it."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(args.device)
    seq_len = int(inputs["input_ids"].shape[1])
    token_positions = sorted({0, seq_len // 2, seq_len - 2})

    def fwd():
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                use_cache=False, output_hidden_states=True, return_dict=True,
            )
        return out.hidden_states

    # ── Part 1: hidden-state contract ────────────────────────────────────────
    contract = audit_hidden_state_contract(
        model, fwd, token_positions, cfg["audit_layers"],
    )
    contract.update({
        "model_key": args.model,
        "model_path": cfg["path"],
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa (V100: no FA2)",
        "sample_image": img_path,
        "seq_len": seq_len,
        "token_positions": token_positions,
        "audit_layers": cfg["audit_layers"],
        "elapsed_s": round(time.time() - t0, 1),
    })
    p1 = os.path.join(HERE, cfg["out"])
    with open(p1, "w") as f:
        json.dump(contract, f, indent=2)
    print("WROTE", p1)
    for ell, entry in contract["layers"].items():
        print(f"  layer {ell}: " + "; ".join(
            f"{k}: max_abs={v['max_abs_error']:.3e} rel_l2={v['relative_l2_error']:.3e}"
            for k, v in entry.items()))

    # ── Part 2: module-level SVD init verification ───────────────────────────
    head = LayerWiseGroundingHead(
        d_model=cfg["d_model"], d_proj=1024, probe_layers=cfg["probe_layers"],
        adapter_type="attn", independent_layers=True, attn_n_heads=8, attn_d_head=64,
    ).to(device=args.device, dtype=torch.bfloat16)
    # replicate reinit_grounding_head() (bf16 NaN fix) exactly as modeling_base does
    import torch.nn as nn
    for probe in head.probes:
        nn.init.xavier_uniform_(probe.W_q.weight, gain=0.02)
        nn.init.xavier_uniform_(probe.W_k.weight, gain=0.02)
        nn.init.zeros_(probe.head_gate)
        nn.init.ones_(probe.q_ln.weight)
        nn.init.zeros_(probe.q_ln.bias)
        nn.init.ones_(probe.k_ln.weight)
        nn.init.zeros_(probe.k_ln.bias)
    model.layerwise_grounding_head = head

    xavier_norms = {
        "W_q": [float(p.W_q.weight.float().norm()) for p in head.probes],
        "W_k": [float(p.W_k.weight.float().norm()) for p in head.probes],
    }

    from paired_qk_svd import decoder_layers
    prefix, layers = decoder_layers(model)
    donor_watch = {}
    for ell in sorted({cfg["probe_layers"][0] + 1, cfg["probe_layers"][5] + 1,
                       cfg["probe_layers"][-1]}):
        donor_watch[f"layers.{ell}.q_proj"] = layers[ell].self_attn.q_proj.weight
        donor_watch[f"layers.{ell}.k_proj"] = layers[ell].self_attn.k_proj.weight
    hash_before = {k: tensor_hash(v) for k, v in donor_watch.items()}

    reports = {}
    for mode in ("next", "same"):
        # reset to Xavier before each mode so both start from the identical state
        for probe in head.probes:
            nn.init.xavier_uniform_(probe.W_q.weight, gain=0.02)
            nn.init.xavier_uniform_(probe.W_k.weight, gain=0.02)
            nn.init.zeros_(probe.head_gate)
        rep = initialize_a7_probes(
            model, donor_mode=mode, terminal_policy="same", device="cpu",
        )
        svd_norms = {
            "W_q": [float(p.W_q.weight.float().norm()) for p in head.probes],
            "W_k": [float(p.W_k.weight.float().norm()) for p in head.probes],
        }
        reports[mode] = {
            "donor_map": [
                {"probe_index": r["probe_index"], "probe_layer": r["probe_layer"],
                 "donor_layer": r["donor_layer"],
                 "terminal_same_layer_fallback": r["terminal_same_layer_fallback"],
                 "retained_energy_fraction": r["retained_energy_fraction"],
                 "relative_frobenius_tail": r["relative_frobenius_tail"],
                 "source_has_qk_norm": r["source_has_qk_norm"],
                 "source_has_bias": r["source_has_bias"]}
                for r in rep
            ],
            "svd_norms": svd_norms,
        }
        print(f"  init[{mode}] donors:",
              [(r["probe_layer"], r["donor_layer"], r["terminal_same_layer_fallback"])
               for r in rep])
        print(f"  init[{mode}] energy:",
              [round(r["retained_energy_fraction"], 6) for r in rep])

    hash_after = {k: tensor_hash(v) for k, v in donor_watch.items()}
    backbone_untouched = all(hash_before[k] == hash_after[k] for k in hash_before)

    out2 = {
        "model_key": args.model,
        "model_path": cfg["path"],
        "transformers_version": transformers.__version__,
        "decoder_path": prefix,
        "probe_layers": cfg["probe_layers"],
        "xavier_norms": xavier_norms,
        "reports": reports,
        "backbone_hash_before": hash_before,
        "backbone_hash_after": hash_after,
        "backbone_untouched": backbone_untouched,
        "expected": {"energy": cfg["expected_energy"],
                     "qk_norm": cfg["expected_qk_norm"],
                     "bias": cfg["expected_bias"]},
        "note": ("module-level verification: head constructed from repo modeling_base, "
                 "initialize_a7_probes called directly (Gate-A CLI integration pending)."),
    }
    p2 = os.path.join(HERE, cfg["out_init"])
    with open(p2, "w") as f:
        json.dump(out2, f, indent=2)
    print("WROTE", p2)
    print("  backbone_untouched:", backbone_untouched)
    print("  xavier W_q norm[0]:", round(xavier_norms["W_q"][0], 4),
          "-> svd(next) W_q norm[0]:", round(reports["next"]["svd_norms"]["W_q"][0], 4))


if __name__ == "__main__":
    main()
