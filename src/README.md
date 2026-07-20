Two invariants worth preserving:
- `rdm.py`, `pca.py`, `metrics.py` never touch the filesystem — that's what makes them testable.
- `analyze.py` never imports `collect.py`. Analysis is re-run constantly; collection costs money.

## Modules

| File | Contains |
|---|---|
| `config.py` | All constants: paths, `ATTRIBUTES`, `MODELS`, seeds, parameters. Values only — no logic, no imports of pandas. |
| `prompts.py` | `PROMPTS` dict, keyed by condition (`direct`, `predict_human`, `predict_human_biased`). |
| `context.py` | Builds the 25-face priming grid (`make_context_grid`, `context_ids`). |
| `data_io.py` | `load_ratings` (enforces `ATTRIBUTES` order, rejects duplicates), `load_model`, `load_human_means`, `load_human_raw`, `common_stimuli`. |
| `collect.py` | OpenRouter calls. Resume-safe: skips stimuli already in the output CSV. Logs tokens + failures. |
| `rdm.py` | `build_rdm` (1 − Pearson r between attributes), `get_unique_correlations`, `compare_rdms` (Spearman on upper triangle). |
| `pca.py` | `fit_pca` (on standardized ratings), `match_components` (correlation-matching, sign-invariant). |
| `metrics.py` | `pearsonr_correlation`, `calculate_human_reliability` (split-half), `compute_self_reliability`, `build_evaluation_dataset`. |
| `plots.py` | `plot_model_comparison`, `plot_rdm_comparison`, `plot_dissimilarities`, `plot_pca_loadings`. All return `fig`; none call `show()`. |

## Running collection

```python
from collect import main
main(["direct"], pilot=True)                 # 100 stimuli
main(["direct"], pilot=True, primed=True)    # 100 stimuli + context grid
main(["direct"], pilot=False)                # all 1,004 — costs real money
```

Requires `OPENROUTER_API_KEY` in the environment and images at `IMAGE_DIR`
(see top-level README — images are not in the repo).

Output: `data/model_ratings/<model>/<label>_<suffix>.csv`, where suffix is
`pilot` / `primed_pilot` / `main` / `primed_main`.

## Worth Noting / Potential Pitfalls:

- **Component order is arbitrary.** PCA sign and component ordering are not
  meaningful when eigenvalues are close (human PC1/PC2 are 7.60 vs 7.50 at
  n=1004 and swap at n=100). Use `match_components`, never index-wise comparison.
- **Never compare across sample sizes.** Smaller n attenuates every correlation.
  Always run `common_stimuli` before `build_rdm` or `fit_pca`.
- **`SUBSAMPLE_SEED = 42` is historical.** It identifies the 100 stimuli used by
  reps 2–3 and all pilots. Changing it silently breaks comparability with
  already-collected data.
- **Model aliases don't resolve to snapshots.** `resolved_model` returns the
  alias you requested, so version drift can't be detected retroactively.
- The context grid is fixed and never rated. 25 faces drawn once via CONTEXT_SEED, excluding the pilot 100, so pilot and main runs share an identical grid. Primed runs therefore have 25 fewer targets than unprimed — always common_stimuli before comparing.