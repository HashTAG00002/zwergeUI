# ZwerGe-UI — Minimal Supplementary Code

This directory is a **minimal, self-contained** extraction of the ZwerGe-UI
retrofit described in *"Middle Layers Know Where to Click: Coordinate
Serialization Bottlenecks in GUI Agents"*. It contains only the core
algorithmic logic needed to understand and reproduce the method; it does
**not** contain training data, model weights, or full evaluation
infrastructure (see "What is intentionally NOT included" below).

## Files

| File               | What it is |
|--------------------|------------|
| `modeling.py`       | Frozen-backbone retrofit model components: `CrossAttnGroundingProbe` (Eq. 1, the ≈4.2M-parameter per-layer cross-attention probe with RMSNorm pre-scale), `CrossLayerFusion` (Eq. 3, the ≈200K-parameter cross-layer LoRA + cosine-meta scorer), `LayerWiseGroundingHead` (combines both, Stage-1/Stage-2 training losses, Eq. 4), `gaussian_bbox_label` (Eq. 2's anisotropic-Gaussian supervision target), and `RetrofitModelMixin` (the model-agnostic glue: anchor-token lookup with zero label leakage, visual-token lookup, single-prefill hidden-state extraction). Two illustrative concrete subclasses (Qwen2.5-VL and Qwen3-VL backbones) are included behind lazy imports. |
| `inference.py`      | `RetrofitInference.predict_layerwise()` (the single prefill-forced forward pass that returns every probed layer's posterior + the fused posterior) and `.predict_zoom_backbone()` (Stage-1 ZwerGe ROI + Stage-2 backbone `generate()`, which with `full_image=True` reproduces the "Native" baseline). Also contains the legacy region-growing posterior→point decoder and shared prompt-building / bbox utilities. |
| `ensemble.py`        | **The deployed decoder**: `run_p2p_ensemble()` — the margin-gated fusion↔native ensemble that produces Table 2's `+ZwerGe` column — plus the training-free posterior-to-point decoder `decode_p2p` (Level 1 region re-ranking, Level 2 cross-layer consensus, Level 3 local Gaussian inversion for sub-patch precision). |
| `probe_lens.py`      | The two analytical lenses behind Figure 1 / §"The Coordinate Serialization Bottleneck": the **spatial lens** (`compute_spatial_metrics_from_pred`, reusing the Stage-1 probes) and the **serialization lens** (`logit_lens_nll_hooks` + `build_native_gt_response` + `find_coord_token_positions`, a parameter-free logit-lens NLL over teacher-forced coordinate tokens). Also includes the instruction-switch counterfactual pair sampler behind Figure 2 / Finding 3. |
| `requirements.txt`   | Minimal Python dependencies. |
| `example_run.sh`     | One annotated, end-to-end CLI sketch for reproducing a single Table-2 cell. |

## The core algorithm, in one paragraph

A frozen VLM backbone is prefilled ONCE with a fixed template ending in a
`<|ground|>` anchor token placed *before* any coordinate value is emitted
(zero label leakage). At each of several intermediate layers, a
`CrossAttnGroundingProbe` reads the anchor's hidden state as a query and
the visual-patch hidden states as keys/values, producing a patch-level
posterior via multi-head cross-attention (`modeling.py`, Eq. 1). A
`CrossLayerFusion` head (`modeling.py`, Eq. 3) combines the posteriors of a
learned "active" subset of layers into `p_final`, weighted per-sample by a
cosine-similarity-to-meta-query score. At inference time, a **training-free
confidence gate** (`ensemble.py::run_p2p_ensemble`) computes both this
fused point `f` (via the parameter-free `decode_p2p` posterior→point
decoder) and the backbone's own native autoregressive coordinate `n`
(one ordinary `generate()` call — the same compute as the baseline), then
emits `f` when the fused posterior is *sharp* (`margin = top1_patch_prob -
top2_patch_prob > τ`, default `τ=0.20` for the main table) and `n`
otherwise. No labels are used at inference time; the entire retrofit adds
at most one extra forward pass over the native baseline.

## Reproducing the main-table numbers

The exact command-line switches used for the paper's main results
(Table 2, `+ZwerGe` columns) are:

```bash
--decode_strategy p2p_ensemble \
--p2p_ensemble_gate margin \
--p2p_ensemble_margin_thr 0.20
```

which map directly onto `ensemble.py::run_p2p_ensemble`'s
`p2p_cfg["ensemble_gate"] = "margin"` and
`p2p_cfg["ensemble_margin_thr"] = 0.20` — see `default_p2p_cfg()` in
`ensemble.py`, whose defaults already match this configuration.

**Gate formula (must not be silently changed):**

```
margin = top1_patch_prob(p_final) - top2_patch_prob(p_final)
use_fusion = margin > ensemble_margin_thr        # 0.20 for the main table
emit fusion_point  if use_fusion  else  emit native_point
```

See `example_run.sh` for a full annotated command sketch (one cell, e.g.
UI-TARS-1.5-7B @ ScreenSpot-Pro).

## Data / checkpoints (not included in this package)

* **Backbones** (frozen, must be downloaded separately, each under its own
  license): UI-TARS-1.5-7B and GUI-Owl-7B (both Qwen2.5-VL, 28 layers,
  hidden_size=3584), GUI-Owl-1.5-8B-Instruct and UI-Venus-1.5-8B (both
  Qwen3-VL, 36 layers, hidden_size=4096). See each backbone's own
  HuggingFace model card for its license and download instructions.
* **Retrofit checkpoints** (the ≈42M-parameter probe bank + ≈200K-parameter
  fusion head trained on top of each frozen backbone) are the paper's own
  artifacts and are not redistributed in this minimal code package.
* **Benchmarks**: ScreenSpot-Pro, ScreenSpot-v2, OSWorld-G (original +
  refusal-removed splits), MMBench-GUI, UI-Vision — each has its own
  distribution terms; see the respective benchmark papers cited in the
  main paper's references.
* **Training data** (the 200k grounding samples used for Stage 1 / Stage 2)
  is not included, per the "no data / no model weights / no large JSON"
  packaging constraint for this supplementary code drop.

## What is intentionally NOT included

* Full multi-GPU evaluation harness, checkpoint I/O, argument parsing for
  every benchmark/ablation combination, visualization utilities.
* The `zwerge_retrofit` package's per-backbone prompt-constant tables
  (`MODEL_TYPE_CONSTANTS`) and the `get_model_class` / `from_checkpoint`
  registry that wires a `--model_type` string to a concrete model class +
  its exact system/user/assistant prompt strings. `inference.py`'s
  `RetrofitInference.from_checkpoint()` raises `NotImplementedError` with a
  pointer to instantiate the class directly instead — this is a deliberate
  boundary of this minimal package, not a bug.
* GUI-Owl-1.5 / UI-Venus-1.5-specific DeepStack visual-feature-injection
  code beyond the illustrative `build_qwen3vl_retrofit_model_class()` in
  `modeling.py` (the official `transformers` `Qwen3VLForConditionalGeneration`
  handles DeepStack automatically; the retrofit only needs
  `output_hidden_states=True`).
* Data preprocessing / label construction pipeline beyond the single
  `gaussian_bbox_label()` helper (Eq. 2).
* Additional analyses referenced in the paper (bootstrap confidence
  intervals, tuned-lens control, non-coordinate protocol-token control,
  target-size regression, OOD generalization) — these are standalone
  scripts layered on top of the same `predict_layerwise` /
  `logit_lens_nll_hooks` primitives provided here.

## Correctness

Every `.py` file in this package passes `python -m py_compile`. No dangling
references to removed helpers remain: every function called from
`ensemble.py` / `probe_lens.py` is either defined locally, imported from
`modeling.py` / `inference.py` within this package, or is a call into
`torch` / `transformers` / `PIL` (all declared in `requirements.txt`).
