# ZwerGe paired-QK SVD initialization: implementation notes

Audited revision: `9329b271709aabceb54ca16224f75733116a9ee0`.

## Status and provenance

The specified GitHub commit and its pinned core source files were readable via the web tool. The execution container could not resolve github.com, so local source inspection and tests used the user-provided `ZwerGeUI_main_9329b27.zip`. The complete archive was not independently byte-verified against a Git clone. The source manifest records the local file hashes.

These are experimental helper functions, not changes pushed to the repository. No pretrained checkpoint, training images, full-model runtime, GPU run, or grounding accuracy evaluation was available. Twelve CPU algebra/integration tests passed, using the actual `CrossAttnGroundingProbe` class AST-extracted from the supplied source. This is NOT TMEM's algorithm and NOT evidence of an accuracy gain.

## What the current implementation actually does

Source locations below are physical lines in the supplied archive (web renderers collapse blank lines).

- `zwerge/scripts/train_ablation_A7_crossattn_probe.sh:161-176`: `attn`, independent probes, 8 heads x 64 dimensions. `grounding_proj_dim=1024` and `grounding_adapter_rank=16` do NOT control this probe width.
- `zwerge/src/zwerge_retrofit/modeling_base.py:199-275`: `CrossAttnGroundingProbe`: RMS safety normalization, LayerNorm, second RMS, separate dense W_q/W_k, per-head dot products, weighted-logit aggregation, one spatial softmax. No V/O projection. W_q/W_k each have shape [512, hidden_size].
- Same file, lines 481-510: reads `all_hidden_states[layer_idx + 1]`, extracts the anchor and visual tokens, and optionally fuses the layer distributions.
- Same file, lines 282-346: the A8 low-rank matrices operate on 512-dimensional projected queries, not the backbone residual space. Do not initialize these from a 3584/4096-dimensional FFN matrix.
- Same file, lines 639-725: a post-loading reset overwrites W_q/W_k with Xavier gain 0.02. Apply any SVD initialization AFTER this call.
- `zwerge/train_retrofit.py:700-705`: the reset-triggering `setup_special_token_ids` call.
- Same file, lines 732-796: trainable parameter selection, inactive-probe freezing, and optional learned new-token embeddings. Freeze new-token embeddings in an initial fixed-representation diagnostic, or share exactly the same embedding checkpoint across methods.
- `zwerge/probe/probe_utils.py:420-429`: the serialization lens unconditionally applies final norm to the selected hidden states. With stock Qwen2.5-VL 4.51.3, its final tuple entry is ALREADY final-normalized. Audit and correct this endpoint convention before interpreting the final-layer lens curve. Qwen3-VL must be checked in the actual runtime; its output capture path differs by version.

## Proposed operator

For a donor attention layer with H query heads, head dimension d0, and GQA key assignment g(h), define the weight-only content kernel:

    M = sum_h Wq_h.T @ Wk_g(h) / (H * sqrt(d0))

Compute the rank-r SVD `M_r = U_r diag(s_r) V_r.T`, where r = probe_heads * probe_head_dim. The current probe uses an initial uniform head gate, so set:

    compensation = probe_heads * sqrt(probe_head_dim)
    Wq_init = sqrt(compensation) * diag(sqrt(s_r)) @ U_r.T
    Wk_init = sqrt(compensation) * diag(sqrt(s_r)) @ V_r.T

Then the initialized probe's bilinear score kernel equals M_r exactly, up to numerical precision. Both dense probe matrices remain fully trainable. The implementation handles GQA using group-averaged queries and a QR-reduced SVD, without constructing a full hidden_size-squared matrix.

This preserves a surrogate matching operator, NOT native attention. It omits native biases, input RMSNorm, Q/K RMSNorm, RoPE, and the native per-head softmax. Average logits are not average attention probabilities. Qwen3-VL's Q/K normalization makes the exact attention function input-dependent, so raw QK weights alone are a weaker approximation there.

## Donor mapping

Use `donor_mode="next"` as the input-aligned hypothesis for a post-block state. Compare with `donor_mode="same"`. The final layer has no next attention reader: `terminal_policy="same"` explicitly reports a same-layer fallback, while `"error"` rejects the configuration BEFORE touching any parameters. Do not silently replace the final donor with lm_head; it changes the operator family from token-token matching to vocabulary prediction.

The audit helper compares the runtime tuple entries to actual raw block outputs, next-block inputs, and final-norm output, using selected token positions only and removing hooks afterward. It does not interrupt a forward with a hook exception.

## Integration

Copy `paired_qk_svd.py` into `zwerge/src/zwerge_retrofit/`. The following is an integration example, NOT an already-supported CLI flag:

```python
from zwerge_retrofit.paired_qk_svd import initialize_a7_probes

# AFTER model.setup_special_token_ids(...), and only for a fresh A7 run:
if _is_resuming or model_args.stage2_from_retrofit_checkpoint or not _reinit_head:
    raise RuntimeError("SVD initialization must not overwrite trained/resumed probes")
report = initialize_a7_probes(
    model,
    donor_mode="next",          # paired ablation: "same"
    terminal_policy="same",    # endpoint is logged, not treated as a next-layer match
    device="cpu",              # change to available GPU for one-time preprocessing
)
```

A8 should load the new A7 checkpoint and retain its existing `reinit_grounding_head=False` logic. Do not initialize A8 fusion LoRAs in this experiment.

For distributed training, prefer computing an initialization artifact once in a single process and loading the exact same head state on all ranks after the default reset. Save only head tensors plus metadata; there is no need to duplicate the base checkpoint. Record backbone/checkpoint identity, source matrix names, donor mapping, head dimensions, decomposition dtype, energy retention, initial logit scale, and seed. Use a new optimizer rather than resuming old momentum against a replaced head.

## Scale control

The current gain-0.02 random initialization may start with very small logits. A raw pretrained factorization can have a very different scale. Measure centered logit RMS and entropy on a fixed TRAIN-split state sample. `match_probe_logit_rms` can match both random and spectral initializations to the same preselected RMS without bbox labels. The common RMS is a hyperparameter; choose it on development data, not a final benchmark.

`random_orientation_seed` preserves the singular values of the surrogate kernel but changes its singular vectors to random orthonormal bases. It helps distinguish useful pretrained directions from spectral scale/conditioning. With a uniform gate, the helper interleaves singular components across probe heads without changing the initial aggregate kernel.

## TMEM-style FFN comparator (not the preferred attention initializer)

TMEM initializes a LoRA A from `diag(s_r) @ V_r.T`, sets B=0, freezes A, and trains B for selected FFN matrices. Its zero update preserves a separate pretrained W0 path. ZwerGe's fresh standalone Q/K projections have no such path, so do not zero both projections.

An optional FFN-row-space comparison can use the donor `mlp.gate_proj.weight` (shape [d_ff, d_hidden]) and initialize BOTH probe projections from the SAME right-singular basis. This creates a positive-semidefinite similarity metric, not an inherited asymmetric QK matching operator. Right singular vectors of `down_proj` live in d_ff, not d_hidden, so they cannot be directly copied into the probe. SVD is an initializer here: keep the dense W_q/W_k trainable for a clean comparison with the current architecture.

## Minimum experiment

1. Audit tuple indexing and final normalization; keep the endpoint visible separately.
2. Train scale-controlled random, same-layer QK-SVD, next-layer QK-SVD on a representative backbone and several depths. Add the spectrum-matched random-orientation control where useful.
3. Compare step-zero behavior, learning curves, held-out grounding, and per-layer convergence, not only training loss or the A8 gate.
4. Test whether the layer-order pattern survives BOTH random and spectral initializations, with equal model capacity, input features, data order, and tuning budget.
5. Only then expand all layers / another backbone and run A8.

A better initializer reduces one optimization confound; it does not prove intermediate states are intrinsically superior or guarantee generalization beyond 200K examples. Layer-specific donors also add layer-specific initialization biases.

## Run the numerical tests

```bash
python test_paired_qk_svd.py --repo /path/to/zwergeUI
```

Only PyTorch is needed. The tests extract the relevant class definition without importing the repository's training entrypoint, downloading weights, or requiring Transformers.

## Public-source hygiene

The supplied training entrypoint contains an API-key-looking literal in its introductory docstring. It is intentionally not reproduced in this package. Review public credentials and rotate any still-valid exposed key.
