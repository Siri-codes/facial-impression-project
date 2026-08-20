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
from pathlib import Path
 
from scipy.stats import binomtest

from config import ELECTION_MODELS, RESULTS, FIGURES
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
Figure 1 — the dissociation, as a per-model scatter.
 
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
 
 

SCORING = "unfiltered"          # 'unfiltered' (lead) or 'filtered'
OUT = FIGURES / "figure1_dissociation.png"

# each decision prompt: (column, label, color, marker)
DECISIONS_LABELS = [
    ("electability_acc",   "Vote (\u201cwho would you vote for\u201d)",       "#c1272d", "o"),
    ("advisor_direct_acc", "Recommendation (\u201cwho should I vote for\u201d)", "#e07b39", "s"),
    ("prediction_acc",     "Prediction (\u201cwho will win\u201d)",         "#2b6cb0", "^"),
]

def election_main_figure(scoring=SCORING, out=OUT):
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