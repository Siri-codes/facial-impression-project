# Methods

## Data
OMI dataset: 1,004 synthetic (StyleGAN2) face images with existing human ratings across 34 attributes (Peterson et al., 2022). All stimuli are synthetic; the real/artificial distinction in OMI applies only to its separate validation set, which is not used here.

## Models
Three multimodal LLMs, chosen for provider diversity and size contrast, queried via OpenRouter:
- Gemini 2.5 Flash-Lite
- GPT-5.4 Mini
- Claude Sonnet 5

Temperature = 0.1 (low enough to suppress sampling noise without forcing purely deterministic output). Each model rated all 34 attributes for each image in a single call, returning JSON. Note: model aliases were requested via OpenRouter, which returns the alias rather than a dated snapshot, so exact model versions cannot be recovered retroactively.

## Prompting

The **primary condition is `direct`**: the model is asked to rate the face, with no reference to humans. Two alternate conditions were also collected and are used only in the prompt-comparison control (below).

**`direct` (primary):**
```
Rate this face on each attribute using a scale from 0 to 100.

For most attributes:
0 = the face absolutely does not display this trait
50 = neutral or average with respect to this trait
100 = the face strongly displays this trait

For these attributes, the scale runs between two opposite descriptors:
- age: 0 = young, 100 = old
- gender: 0 = feminine, 100 = masculine
- weight: 0 = skinny, 100 = fat
- skin-color: 0 = light, 100 = dark
- hair-color: 0 = light, 100 = dark

Provide a rating for every attribute, including ones that feel subjective.
```

**`predict_human` (control condition):** identical to `direct`, but the opening line asks the model to *predict how an average person would rate this face* rather than to rate it itself. This isolates the effect of framing the task as human-prediction.

**`predict_human_biased` (control condition):** the initial prompt tested. It instructs the model to predict human consensus and to reflect human "cognitive biases, halo effects, and rapid social stereotyping." It was replaced as the primary condition because it confounds the model's own representation with instruction-following. (It also predates the refined scale definitions, so its anchors are less complete — a further reason it is used only for comparison, not as a main result.)

**Format instruction** (appended per call): requires a raw JSON object with exactly the 34 attribute keys, each an integer 0–100, no prose or markdown. Model outputs were normalized on parse (lowercased, underscores/spaces to hyphens) to absorb minor key-formatting variation (e.g. `middle_eastern` -> `middle-eastern`).

## Collection pipeline
A resumable pipeline rates each stimulus and appends results to a per-condition CSV. Runs skip already-processed stimuli, so interrupted collections resume without duplication. Failed calls (API or parse errors) are skipped and logged to a separate failures file; they can be retried by re-running, which only re-attempts the missing stimuli. Per-call token counts and the resolved model string are logged alongside each result.

All analyses align judges to a common stimulus set before computing anything, so every comparison is on identical faces. Results below are computed on the `direct` condition unless stated otherwise.

---

## Analysis 1 — Per-attribute accuracy (Peterson-style)
For each attribute, the Pearson correlation between model ratings and OMI human mean ratings, benchmarked against the human split-half reliability ceiling (following Peterson et al., 2022). Split-half reliability is estimated by repeatedly partitioning each image's human ratings into halves, averaging, and correlating, over 100 random splits; this matches the published benchmark and reproduces its reported values. The `rating_type` column in the raw human data distinguishes normal from repeat trials (a re-showing of a face to the same participant); repeats were retained to match Peterson et al., and excluding them changes the ceiling negligibly.

## Analysis 2 — Representational Similarity Analysis (RSA)
For each judge (humans and each model), a 34×34 representational dissimilarity matrix (RDM) was constructed, where dissimilarity between two attributes is 1 − Pearson r across stimuli. Model and human RDMs were then compared by Spearman rank correlation over the upper triangle (excluding the diagonal), so the result depends on the rank ordering of attribute-pair dissimilarities rather than their absolute values.

On the `direct` condition, RSA values were 0.62 (Gemini), 0.72 (GPT), and 0.77 (Claude). (Values differ slightly from an earlier collection on the biased prompt; the model ordering is unchanged.)

## Analysis 3 — Model self-reliability
Three repetitions on a fixed 100-image random subsample (seed 42), summarized as the mean pairwise correlation across repetitions. This is a model-side reliability estimate: how much of a model's shortfall against human ratings is attributable to its own trial-to-trial inconsistency versus genuine misalignment. Note this is a *within-model* consistency measure and is not directly analogous to the *between-rater* human split-half reliability in Analysis 1.

## Analysis 4 — Representational structure (PCA)
The RSA score summarizes overall similarity in a single value but does not reveal *which* dimensions of human judgment a model reproduces or distorts. To characterize this, PCA was applied to each judge's standardized ratings (humans and each model), and the resulting components were compared across judges.

Because component sign is arbitrary and component order is unstable when eigenvalues are close (the top two human components explain 22.3% and 22.0% of variance and swap order between the full set and the 100-image subsample), components were matched by maximum absolute correlation between loading vectors, never by index. For each matched pair we report the absolute correlation (does the dimension exist in the model?) and the variance each explains (how salient is it?).

**Result.** All three models reproduced the leading valence/dominance dimension (human PC1, 22.3% of variance) well (|r| = 0.81–0.89). The racial dimension (human PC2, 22.0%) was systematically compressed: it explained 11–16% of variance in the models versus 22% in humans, and the fidelity of its reproduction varied by model (|r| = 0.62 Gemini, 0.67 GPT, 0.89 Claude). In the human structure, this dimension places `white` opposite the other race categories and carries `privileged`, `typical`, `familiar`, and `looks-like-you` toward the `white` pole — an association consistent with OMI's predominantly White (~71%) rater pool. Claude reproduces this dimension, including that association, most faithfully. See `results/main_pca_comparison.csv` and `results/figures/`.

---

## Controls

### Prompt condition
On the 100-stimulus subsample, the biased prompt produced a *worse* match to human representational structure than `direct` (e.g. Claude's correlation with human PC1/PC2 was 0.70/0.67 under the biased prompt vs 0.85/0.84 under `direct`; the pattern holds for all three models). The human-like structure reported above is therefore not an artifact of instructing the model to imitate human bias. See `results/prompt_comparison.csv`.

### Context priming
Peterson et al.'s participants viewed 25 example faces before rating, to convey the range of the stimulus set. To test whether this task difference affected model ratings, the `direct` condition was replicated with a 25-face context grid prepended to each call (n=100). The grid was selected by minimizing mean absolute deviation from the full set's mean human ratings across all 34 attributes (seed 55; mean deviation 1.10 on a 0–100 scale), a criterion fixed before any analysis of the primed condition, and the 25 context faces were excluded from the rated set.

Priming changed mean per-attribute R² by less than 0.03 for all three models. Bootstrap 95% confidence intervals over stimuli were [0.003, 0.045] (Gemini), [−0.025, 0.018] (GPT), and [−0.004, 0.022] (Claude): the effect is bounded below ~0.045 R² and indistinguishable from zero for two of three models. Priming was therefore omitted from the main collection. See `results/priming_comparison.csv`.

### Replication
To confirm the PCA structure is stable across repeated collections rather than an artifact of a single run, three independent collections of the biased prompt (the only condition collected in triplicate) were compared on the 100-image subsample. Pairwise component correlations were high across models (mean |r| = 0.90–0.98), supporting the reliability of single-collection structure. The one lower value (Claude's PC1, |r| = 0.69 between the two smallest-sample repetitions) is consistent with sample-size noise in the 100-image files. See `results/reliability_replication.csv`.

---

## Reproducibility
Every result above is regenerated by a single function in `src/analyze.py`, which writes to `results/`. Collection is separated from analysis so that re-running analyses never triggers paid API calls. Fixed seeds: `SUBSAMPLE_SEED = 42` (the 100-image subsample), `CONTEXT_SEED = 55` (the priming grid). These identify already-collected data and must not be changed without re-collecting.