"""
Consolidate the scattered per-prompt analysis outputs into paper-ready tables.

Produces:
  master_accuracy.csv     - one row per model, one column per prompt (filtered acc)
  master_dissociation.csv - competence vs each decision prompt, per model, + summary
  paper_summary.txt       - a human-readable digest of the headline numbers

Run after the per-prompt analyses have written their CSVs.
"""
import matplotlib
matplotlib.use("Agg")           # file output, no display needed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

from scipy.stats import binomtest

from config import ELECTION_MODELS, RESULTS, FIGURES, MMMU_PRO
from analyze import _save

# the prompts to include, in the order they should appear in the paper table.
# 'competence' is the impression; the rest are decision framings.
IMPRESSION = 'competence'
DECISIONS = ['electability', 'advisor_direct', 'prediction']


def _load_acc(prompt_key):
    """Load a per-prompt both-ways accuracy table (written by accuracy_both_ways)
    from the results directory."""
    path = RESULTS / f"multitrial_accuracy_both_{prompt_key}_multitrial.csv"
    if not path.exists():
        print(f"  (missing: {path.name} — run accuracy_both_ways('{prompt_key}_multitrial') first)")
        return None
    return pd.read_csv(path)


def master_accuracy_table(acc_col='filtered_acc', save=True):
    """One row per model; accuracy for each prompt. For the filtered version,
    also carries the Wilson CI bounds and n for the competence (impression)
    column, since that is the accuracy reported with CIs in the paper's Table 1.
    acc_col selects 'filtered_acc' or 'unfiltered_acc'."""
    frames = {}
    ci_low = ci_high = n_col = None
    for key in [IMPRESSION] + DECISIONS:
        df = _load_acc(key)
        if df is None:
            continue
        frames[key] = df.set_index('model')[acc_col]
        # capture CI + n for the impression column (filtered only — that's what
        # accuracy_both_ways computed a Wilson interval for)
        if key == IMPRESSION and acc_col == 'filtered_acc':
            idx = df.set_index('model')
            if 'filtered_ci_low' in idx.columns:
                ci_low = idx['filtered_ci_low']
                ci_high = idx['filtered_ci_high']
                n_col = idx['filtered_n']

    if not frames:
        print("no accuracy tables found — run the per-prompt analyses first")
        return None

    master = pd.DataFrame(frames)
    master.columns = [f"{c}_acc" for c in master.columns]
    if ci_low is not None:
        master['competence_ci_low'] = ci_low
        master['competence_ci_high'] = ci_high
        master['competence_n'] = n_col
    master = master.reset_index()
    suffix = 'unfiltered' if acc_col == 'unfiltered_acc' else 'filtered'
    if save:
        _save(master, f'master_accuracy_{suffix}.csv')
    print(f"[{suffix}]")
    print(master.to_string(index=False))
    return master


def master_dissociation_table(acc_col='filtered_acc', save=True):
    """For each decision prompt, competence-minus-decision per model, plus a
    sign test across models. acc_col selects filtered or unfiltered accuracy."""
    comp = _load_acc(IMPRESSION)
    if comp is None:
        print("competence accuracy table missing")
        return None
    comp_acc = comp.set_index('model')[acc_col]
    suffix = 'unfiltered' if acc_col == 'unfiltered_acc' else 'filtered'

    summary_rows = []
    detail = {'model': comp_acc.index.tolist(), 'competence_acc': comp_acc.values.round(3)}

    for dec in DECISIONS:
        d = _load_acc(dec)
        if d is None:
            continue
        dec_acc = d.set_index('model')[acc_col]
        # align on shared models
        shared = comp_acc.index.intersection(dec_acc.index)
        diff = (comp_acc[shared] - dec_acc[shared]).dropna()
        detail[f'{dec}_acc'] = dec_acc.reindex(comp_acc.index).round(3).values
        detail[f'comp_minus_{dec}'] = (comp_acc - dec_acc.reindex(comp_acc.index)).round(3).values

        n = len(diff)
        higher = int((diff > 0).sum())
        p = binomtest(higher, n, 0.5, alternative='greater').pvalue if n else None
        summary_rows.append({
            'decision_prompt': dec,
            'n_models': n,
            'competence_higher_in': higher,
            'mean_advantage': round(diff.mean(), 3),
            'sign_test_p': f"{p:.4f}" if p is not None else None,
        })

    detail_df = pd.DataFrame(detail)
    summary_df = pd.DataFrame(summary_rows)
    print(f"[{suffix}] Per-model competence vs each decision:")
    print(detail_df.to_string(index=False))
    print(f"\n[{suffix}] Dissociation summary (competence vs each decision):")
    print(summary_df.to_string(index=False))
    if save:
        _save(detail_df, f'master_dissociation_detail_{suffix}.csv')
        _save(summary_df, f'master_dissociation_summary_{suffix}.csv')
    return detail_df, summary_df


def build_table1(save=True):
    """Assemble the paper's Table 1 for the COMPETENCE (impression) prompt:
    per model — filtered accuracy with Wilson CI, inclusive accuracy, and the
    stable/position-bias/noise reliability breakdown. Pulls the accuracy file
    and the classification file and joins them."""
    acc = _load_acc(IMPRESSION)  # multitrial_accuracy_both_competence_multitrial.csv
    cls_path = RESULTS / "multitrial_classification_competence_multitrial.csv"
    if acc is None:
        print("competence accuracy file missing")
        return None
    if not cls_path.exists():
        print(f"  (missing: {cls_path.name} — run classification_breakdown('competence_multitrial'))")
        cls = None
    else:
        cls = pd.read_csv(cls_path)

    t = acc[['model', 'filtered_acc', 'filtered_ci_low', 'filtered_ci_high',
             'filtered_n', 'unfiltered_acc']].copy()
    # format the CI as a single "acc [lo, hi]" string for the paper
    t['competence_acc_ci'] = t.apply(
        lambda r: f"{r['filtered_acc']:.2f} [{r['filtered_ci_low']:.2f}, {r['filtered_ci_high']:.2f}]"
        if pd.notna(r['filtered_acc']) else "—", axis=1)

    if cls is not None:
        t = t.merge(cls[['model', 'stable_face', 'position_bias', 'noise']],
                    on='model', how='left')

    # tidy column order for the paper
    cols = ['model', 'competence_acc_ci', 'unfiltered_acc', 'filtered_n']
    if cls is not None:
        cols += ['stable_face', 'position_bias', 'noise']
    t = t[cols].rename(columns={
        'competence_acc_ci': 'stable_acc [95% CI]',
        'unfiltered_acc': 'inclusive_acc',
        'filtered_n': 'n_stable',
    })
    if save:
        _save(t, 'table1_competence.csv')
    print(t.to_string(index=False))
    return t


def build_all(save=True):
    """Build every paper-ready table, both filtered and unfiltered, in one call."""
    for acc_col in ['filtered_acc', 'unfiltered_acc']:
        tag = 'UNFILTERED' if acc_col == 'unfiltered_acc' else 'FILTERED'
        print(f"\n{'='*60}\n=== MASTER ACCURACY ({tag}) ===\n{'='*60}")
        master_accuracy_table(acc_col=acc_col, save=save)
        print(f"\n=== DISSOCIATION ({tag}) ===")
        master_dissociation_table(acc_col=acc_col, save=save)
    print(f"\n{'='*60}\n=== TABLE 1 (competence, paper-ready) ===\n{'='*60}")
    build_table1(save=save)
    print("\nDone. Paper-ready tables written (filtered and unfiltered).")

"""
Figure 1 — competence-impression accuracy predicts elections, per model.

A caterpillar / forest plot: one row per model, sorted by filtered accuracy.
  - filtered (consistent-preference) accuracy, shown WITH a Wilson 95% CI
  - unfiltered (inclusive) accuracy, shown as a second marker (no CI; inclusive
    scoring is an average of per-race probabilities, not a clean k/n)
Reference lines:
  - chance (0.50)
  - human benchmark (0.688, Todorov et al. 2005)

The story: every model sits above chance, clustered around the human rate — the
effect is consistent across models, not driven by a few. Reads the per-prompt
accuracy CSV that accuracy_both_ways() writes (multitrial_accuracy_both_<label>.csv),
which carries filtered_ci_low/high.
"""

IMPRESSION_LABEL = "competence_multitrial"
HUMAN_BENCHMARK = 0.688              # Todorov et al. 2005
CHANCE = 0.50

# presentation constants (local to this figure, not config)
FILT_COLOR = "#2b6cb0"
UNFILT_COLOR = "#9aa5b1"
HUMAN_COLOR = "#2e8b57"


def forest_plot(out=FIGURES / "forest_plot.png"):
    path = RESULTS / f"multitrial_accuracy_both_{IMPRESSION_LABEL}.csv"
    df = pd.read_csv(path)
    need = {"model", "filtered_acc", "filtered_ci_low", "filtered_ci_high", "unfiltered_acc"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name} missing columns: {missing}")

    # sort by filtered accuracy so the plot reads as a clean gradient
    df = df.dropna(subset=["filtered_acc"]).sort_values("filtered_acc").reset_index(drop=True)
    y = range(len(df))

    # CHANGED: tighter per-row pitch (0.42 -> 0.30) and smaller base pad (1.2 -> 0.7)
    # 15 models: was 7.5in tall, now ~5.2in. This is the single biggest space win.
    fig, ax = plt.subplots(figsize=(6.0, 0.30 * len(df) + 0.7))

    # reference lines
    ax.axvline(CHANCE, color="#c0392b", lw=1, ls="--", zorder=1)
    ax.axvline(HUMAN_BENCHMARK, color=HUMAN_COLOR, lw=1, ls="--", zorder=1)
    
    # CHANGED: place reference-line labels just ABOVE the top row, not overlapping it
    ax.text(CHANCE - 0.005, len(df) - 0.3, "chance", color="#c0392b", fontsize=7.5,
            ha="right", va="bottom")
    ax.text(HUMAN_BENCHMARK + 0.005, len(df) - 0.3, "human (68.8%)", color=HUMAN_COLOR,
            fontsize=7.5, ha="left", va="bottom")
            
    # filtered accuracy with Wilson CI (asymmetric error bars)
    xerr_low = df["filtered_acc"] - df["filtered_ci_low"]
    xerr_high = df["filtered_ci_high"] - df["filtered_acc"]
    # CHANGED: slightly smaller markers/caps so rows can sit closer without crowding
    ax.errorbar(df["filtered_acc"], y, xerr=[xerr_low, xerr_high],
                fmt="o", color=FILT_COLOR, ecolor=FILT_COLOR, elinewidth=1.0,
                capsize=2.5, markersize=5, zorder=3, label="Consistent-Preference accuracy (95% CI)")

    # unfiltered accuracy as a second marker, no CI
    ax.scatter(df["unfiltered_acc"], y, marker="s", s=32, color=UNFILT_COLOR,
               zorder=2, label="Overall accuracy")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"], fontsize=8)
    # ADDED: clamp y-limits so there isn't a half-row of padding above/below the data
    ax.set_ylim(-0.6, len(df) + 0.3)

    ax.set_xlabel("Competence-impression accuracy (predicting election winner)", fontsize=9)
    ax.set_xlim(0.40, 0.85)
    ax.tick_params(axis="x", labelsize=8)
    #ax.set_title("Competence impressions predict election outcomes, per model",fontsize=10.5, pad=4)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    # quick sanity: how many models' CI lower bound is above chance?
    above = int((df["filtered_ci_low"] > CHANCE).sum())
    print(f"  {above}/{len(df)} models have a 95% CI entirely above chance")
    print(f"  filtered median {df['filtered_acc'].median():.3f}, "
          f"range {df['filtered_acc'].min():.3f}-{df['filtered_acc'].max():.3f}")
    return out

"""
Figure 2 — two panels sharing an x-axis (general capability / MMMU-Pro).

  Panel A: competence-election accuracy vs. capability. HONEST NULL — shows the
           scatter, a flat fit with a wide confidence band, and an annotated
           "no significant trend" with r and p. This is deliberately not dressed
           up: the point is that accuracy does NOT track capability.
  Panel B: alignment with human facial-impression structure vs. capability.
           Rises with capability (the trend that Panel A lacks).

The contrast is in the slopes: B climbs, A does not. The two panels use different
y-quantities (accuracy vs. RSA), so they are stacked with a shared x-axis rather
than sharing a y-scale. Each panel uses its own full model set; the caption
discloses the overlap.

"""

# presentation constants (local to this figure)
A_COLOR = "#c1272d"     # election accuracy (the null panel)
B_COLOR = "#2b6cb0"     # alignment (the rising panel)
BAND_ALPHA = 0.15


def _fit_with_band(ax, x, y, color):
    """Scatter + linear fit line + 95% band (visual aids), but return the
    SPEARMAN r and p as the reported statistic. The line/band are illustrative;
    the reported correlation is rank-based (spearman), matching the monotonic-
    trend claims and robust to small-n outliers."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ax.scatter(x, y, s=48, color=color, edgecolors="white", linewidths=0.5, zorder=3)

    n = len(x)
    # linear fit — ONLY for drawing the line and its band (visual aid)
    slope, intercept, _r_pearson, _p_pearson, se = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = intercept + slope * xs
    ax.plot(xs, ys, color=color, lw=1.4, ls="--", zorder=2)

    tval = stats.t.ppf(0.975, n - 2)
    x_mean = x.mean()
    ss_x = np.sum((x - x_mean) ** 2)
    resid = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(resid ** 2) / (n - 2))
    band = tval * s_err * np.sqrt(1.0 / n + (xs - x_mean) ** 2 / ss_x)
    ax.fill_between(xs, ys - band, ys + band, color=color, alpha=BAND_ALPHA, zorder=1)

    # REPORTED statistic: Spearman (rank-based, matches the monotonic-trend claim)
    r_spearman, p_spearman = stats.spearmanr(x, y)
    return r_spearman, p_spearman


def capability_dissociation_plt(out= FIGURES / "capability_dissociation.png"):
    
    # --- Panel A data: competence-election accuracy per model ---
    accA = pd.read_csv(RESULTS / "multitrial_accuracy_both_competence_multitrial.csv")
    accA["cap"] = accA["model"].map(MMMU_PRO)
    A = accA.dropna(subset=["cap", "unfiltered_acc"])          # use inclusive acc; swap if you prefer filtered

    # --- Panel B data: per-model alignment (RSA) ---
    # EXPECTED: a CSV with columns ['model', 'rsa'] (or similar). Adjust the name/col.
    alignB = pd.read_csv(RESULTS / "rsa_scores_main.csv")   # <-- point at your real file
    rsa_col = "rsa" if "rsa" in alignB.columns else alignB.columns[-1]
    alignB["cap"] = alignB["model"].map(MMMU_PRO)
    B = alignB.dropna(subset=["cap", rsa_col])

    #fig, (axA, axB) = plt.subplots(2, 1, figsize=(5.6, 6.4), sharex=True)
    # CHANGED: 1x2 (side by side) instead of 2x1; wide-and-short figsize;
    #          no sharex (each panel autoscales to its own model set + gets its own x-label)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.4))


    # Panel A — the honest null
    rA, pA = _fit_with_band(axA, A["cap"], A["unfiltered_acc"], A_COLOR)
    axA.axhline(0.5, color="#bbbbbb", lw=0.8, ls=":", zorder=0)
    axA.set_ylabel("Election-prediction accuracy")
    axA.set_title("A  Competence-based election prediction", loc="left", fontsize=10)
    axA.text(0.03, 0.06, f"(ρ = {rA:.2f} n = {len(A)})",
             transform=axA.transAxes, fontsize=8, color="#555555")
    for s in ["top", "right"]:
        axA.spines[s].set_visible(False)

    # Panel B — alignment rises
    rB, pB = _fit_with_band(axB, B["cap"], B[rsa_col], B_COLOR)
    axB.set_ylabel("Alignment with human\nimpressions (RSA)")
    axB.set_title("B  General facial-impression alignment", loc="left", fontsize=10)
    #star = "***" if pB < 0.001 else "**" if pB < 0.01 else "*" if pB < 0.05 else "n.s."
    axB.text(0.03, 0.90, f"(ρ = {rB:.2f}, n = {len(B)})",
             transform=axB.transAxes, fontsize=8, color="#555555", va="top")
    axB.set_xlabel("General capability (MMMU-Pro)")
    for s in ["top", "right"]:
        axB.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  Panel A: r={rA:.3f}, p={pA:.3f}, n={len(A)}")
    print(f"  Panel B: r={rB:.3f}, p={pB:.3f}, n={len(B)}")
    return out


"""
Figure 3 — election-prediction accuracy across the four prompt conditions.

Each model contributes one faint line connecting its accuracy on the four prompts;
a bold overlay shows the mean (with a 95% CI across models) per condition. The
x-axis is ordered to group the two higher-predicting prompts (competence, forecast)
and the two lower-predicting ones (vote, recommendation), so the pattern
"competence ~ prediction > vote ~ recommendation" reads directly.

Reads the per-prompt accuracy CSVs written by accuracy_both_ways()
(multitrial_accuracy_both_<label>.csv), using inclusive (unfiltered) accuracy to
match the conservative main-text scoring. Switch ACC_COL to 'filtered_acc' if the
text leads with consistent-preference accuracy.
"""

ACC_COL = "unfiltered_acc"     # or 'filtered_acc'

# order + display names for the four conditions (grouped: high, high, low, low)
CONDITIONS = [
    ("competence_multitrial",  "Competence\n(impression)"),
    ("prediction_multitrial",  "Prediction\n(forecast)"),
    ("electability_multitrial","Vote"),
    ("advisor_direct_multitrial","Recommendation"),
]

LINE_COLOR = "#b0b7c0"     # faint per-model lines
MEAN_COLOR = "#2b6cb0"     # bold summary


def _load(label):
    df = pd.read_csv(RESULTS / f"multitrial_accuracy_both_{label}.csv")
    return df.set_index("model")[ACC_COL]


def prompt_comparison_plt(out=FIGURES / "prompt_comparison.png"):
    # build a model x condition accuracy matrix
    series = {name: _load(label) for label, name in CONDITIONS}
    mat = pd.DataFrame(series)                      # index = model, cols = conditions
    mat = mat.dropna(how="any")                    # models with all four prompts
    names = [name for _, name in CONDITIONS]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    # faint per-model lines
    for _, row in mat.iterrows():
        ax.plot(x, row[names].values, color=LINE_COLOR, lw=0.8, alpha=0.6,
                marker="o", markersize=3, zorder=2)

    '''
    Switched from t interval to bootstrap over models (below)
    # bold mean +/- 95% CI across models
    means = mat[names].mean()
    sems = mat[names].sem()
    ci = sems * stats.t.ppf(0.975, len(mat) - 1)
    '''

    rng = np.random.default_rng(0)   # fixed seed = reproducible
    N_BOOT = 10000
    means = mat[names].mean()
    lower, upper = {}, {}
    for name in names:
        vals = mat[name].dropna().values          # the 15 per-model accuracies
        boot_means = [rng.choice(vals, size=len(vals), replace=True).mean()
                    for _ in range(N_BOOT)]
        lower[name] = np.percentile(boot_means, 2.5)
        upper[name] = np.percentile(boot_means, 97.5)

    # asymmetric error bars for ax.errorbar: yerr = [[below], [above]]
    yerr_low = [means[n] - lower[n] for n in names]
    yerr_high = [upper[n] - means[n] for n in names]
  
    '''
    ax.errorbar(x, means.values, yerr=ci.values, color=MEAN_COLOR, lw=2.2,
                marker="o", markersize=7, capsize=4, zorder=4, label="Mean (95% CI)")
    '''
    ax.errorbar(x, means.values, yerr=[yerr_low, yerr_high],
            color=MEAN_COLOR, lw=2.2, marker="o", markersize=7,
            capsize=4, zorder=4, label="Mean (95% CI)")

    ax.axhline(0.5, color="#cccccc", lw=0.8, ls=":", zorder=0)
    ax.text(len(names) - 0.5, 0.505, "chance", color="#999999", fontsize=7.5,
            ha="right", va="bottom")

    # a light divider between the "high" pair and the "low" pair
    ax.axvline(1.5, color="#e5e5e5", lw=1, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(f"Election-prediction accuracy ({ACC_COL.replace('_',' ')})", fontsize=9)
    ax.set_title("Accuracy across prompt conditions, per model", fontsize=11, pad=8)
    
    # CHANGED: top tracks the highest actual data point (per-model lines + CI tops),
    # not just the means, so there's no dead space above the tallest line.
    data_top = max(mat[names].values.max(), max(upper.values()))
    ax.set_ylim(0.48, data_top + 0.015)
    
    ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}  ({len(mat)} models with all four prompts)")
    for name in names:
        print(f"  {name.replace(chr(10),' '):28s} mean {means[name]:.3f}")
    return out


"""
OLD: Figure 3 — the dissociation, as a per-model scatter.
 
x-axis: competence-impression accuracy (per model)
y-axis: decision-prompt accuracy (per model), one series per decision prompt
diagonal: y = x. Points BELOW the diagonal mean the impression predicts better
than that decision output (x > y).
 
The story the figure tells: vote and recommendation points sit below the diagonal
(impression wins); prediction points sit ON the diagonal (impression and forecast
are the same assessment). This is the assessment-vs-action dissociation in one view.
 
Reads the consolidated master_accuracy_{unfiltered,filtered}.csv produced by
consolidate_results.build_all(). Defaults to unfiltered (the conservative,
lead-with version), matching the main-text numbers.
"""
 
'''
SCORING = "unfiltered"          # 'unfiltered' (lead) or 'filtered'

# each decision prompt: (column, label, color, marker)
DECISIONS_LABELS = [
    ("electability_acc",   "Vote (\u201cwho would you vote for\u201d)",       "#c1272d", "o"),
    ("advisor_direct_acc", "Recommendation (\u201cwho should I vote for\u201d)", "#e07b39", "s"),
    ("prediction_acc",     "Prediction (\u201cwho will win\u201d)",         "#2b6cb0", "^"),
]

def election_main_figure(scoring=SCORING, out= FIGURES / "figure1_dissociation.png"):
    path = RESULTS / f"master_accuracy_{scoring}.csv"
    df = pd.read_csv(path)
    if "competence_acc" not in df.columns:
        raise SystemExit(f"expected 'competence_acc' in {path.name}; got {list(df.columns)}")
 
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
 
    # diagonal (y = x): the line of no dissociation
    lo, hi = 0.45, 0.80
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", zorder=1)
    ax.text(hi - 0.005, hi - 0.03, "impression = decision", color="#888888",
            fontsize=8, ha="right", rotation=45, rotation_mode="anchor")
 
    # chance reference lines
    ax.axhline(0.5, color="#cccccc", lw=0.8, ls=":", zorder=0)
    ax.axvline(0.5, color="#cccccc", lw=0.8, ls=":", zorder=0)
 
    x = df["competence_acc"]
    for col, label, color, marker in DECISIONS_LABELS:
        if col not in df.columns:
            print(f"  (skipping missing column {col})")
            continue
        y = df[col]
        ax.scatter(x, y, s=55, c=color, marker=marker, edgecolors="white",
                   linewidths=0.6, label=label, zorder=3, alpha=0.9)
 
    ax.set_xlabel("Competence-impression accuracy", fontsize=10)
    ax.set_ylabel("Decision-prompt accuracy", fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
 
    # annotation of what below-diagonal means
    ax.text(0.62, 0.475, "below line:\nimpression predicts better",
            fontsize=7.5, color="#555555", ha="center", style="italic")
 
    ax.legend(fontsize=7.5, loc="upper left", frameon=True, framealpha=0.9)
    ax.set_title("Impression vs. each decision output, per model",
                 fontsize=10.5, pad=8)
 
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
 
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    # quick sanity: how many models below diagonal per decision
    for col, label, *_ in DECISIONS_LABELS:
        if col in df.columns:
            below = int((df["competence_acc"] > df[col]).sum())
            print(f"  {label}: impression higher in {below}/{len(df)} models")
    return out
'''

def forest_plot_v2(out=FIGURES / "forest_plot_v2.png"):
    path = RESULTS / f"multitrial_accuracy_both_{IMPRESSION_LABEL}.csv"
    df = pd.read_csv(path)
    need = {"model", "filtered_acc", "filtered_ci_low", "filtered_ci_high", "unfiltered_acc"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path.name} missing columns: {missing}")

    # sort by filtered accuracy so the plot reads as a clean gradient
    df = df.dropna(subset=["filtered_acc"]).sort_values("filtered_acc").reset_index(drop=True)
    y = list(range(len(df)))
    OFFSET = 0.16   # vertical separation between the two series

    fig, ax = plt.subplots(figsize=(6.0, 0.30 * len(df) + 0.7))

    # reference lines
    ax.axvline(CHANCE, color="#c0392b", lw=1, ls="--", zorder=1)
    ax.axvline(HUMAN_BENCHMARK, color=HUMAN_COLOR, lw=1, ls="--", zorder=1)

    ax.text(CHANCE - 0.005, len(df) - 0.3, "chance", color="#c0392b", fontsize=7.5,
            ha="right", va="bottom")
    ax.text(HUMAN_BENCHMARK + 0.005, len(df) - 0.3, "human (68.8%)", color=HUMAN_COLOR,
            fontsize=7.5, ha="left", va="bottom")


    # filtered accuracy with Wilson CI — nudged UP
    y_filt = [yi + OFFSET for yi in y]
    xerr_low = df["filtered_acc"] - df["filtered_ci_low"]
    xerr_high = df["filtered_ci_high"] - df["filtered_acc"]
    ax.errorbar(df["filtered_acc"], y_filt, xerr=[xerr_low, xerr_high],
                fmt="o", color=FILT_COLOR, ecolor=FILT_COLOR, elinewidth=1.0,
                capsize=2.5, markersize=5, zorder=3,
                label="Consistent-preference accuracy (95% Wilson CI)")

    # unfiltered accuracy with bootstrap CI — nudged DOWN
    y_unfilt = [yi - OFFSET for yi in y]
    if {"unfiltered_ci_low", "unfiltered_ci_high"} <= set(df.columns) \
            and df["unfiltered_ci_low"].notna().any():
        u_low = df["unfiltered_acc"] - df["unfiltered_ci_low"]
        u_high = df["unfiltered_ci_high"] - df["unfiltered_acc"]
        ax.errorbar(df["unfiltered_acc"], y_unfilt, xerr=[u_low, u_high],
                    fmt="s", color=UNFILT_COLOR, ecolor=UNFILT_COLOR, elinewidth=1.0,
                    capsize=2.5, markersize=5, zorder=2,
                    label="Overall accuracy (95% bootstrap CI)")
    else:
        ax.scatter(df["unfiltered_acc"], y_unfilt, marker="s", s=32,
                   color=UNFILT_COLOR, zorder=2, label="Overall accuracy")

    for i, yi in enumerate(y):
        if i % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#f5f5f5", zorder=0)

    # y-ticks stay at the ORIGINAL integer positions (centered between the two series)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=8)
    ax.set_ylim(-0.6, len(df) + 0.3)

    ax.set_xlabel("Competence-impression accuracy (predicting election winner)", fontsize=9)
    ax.set_xlim(0.40, 0.85)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    above = int((df["filtered_ci_low"] > CHANCE).sum())
    print(f"  {above}/{len(df)} models have a 95% CI entirely above chance")
    print(f"  filtered median {df['filtered_acc'].median():.3f}, "
          f"range {df['filtered_acc'].min():.3f}-{df['filtered_acc'].max():.3f}")
    return out