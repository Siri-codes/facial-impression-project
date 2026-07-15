import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from rdm import get_unique_correlations
from config import FIGURES, ATTRIBUTES

def plot_dissimilarities(rdm_1, rdm_2, name_1, name_2, save=True):
    d1 = get_unique_correlations(rdm_1)
    d2 = get_unique_correlations(rdm_2)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(d1, d2, alpha=0.5, s=12)
    lo, hi = min(d1.min(), d2.min()), max(d1.max(), d2.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)   # y=x reference
    ax.set_xlabel(f"{name_1} RDM dissimilarity")
    ax.set_ylabel(f"{name_2} RDM dissimilarity")
    ax.set_title(f"{name_1} vs {name_2}: attribute-pair dissimilarities")

    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        slug = f"{name_1}_{name_2}".lower().replace(' ', '_')
        fig.savefig(FIGURES / f"dissimilarities_{slug}.png", bbox_inches="tight")
    return fig

def plot_rdm_comparison(human_rdm, model_rdm, model_label, save=True):
    """
    Plots rdms for human and model ratings.
    """
    vmin = min(human_rdm.min(), model_rdm.min())
    vmax = max(human_rdm.max(), model_rdm.max())

    fig, axes = plt.subplots(2, 1, figsize=(14, 26))

    for ax, rdm, title in zip(axes, [human_rdm, model_rdm], ["Human RDM", f"{model_label} RDM"]):
        im = ax.imshow(rdm, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(ATTRIBUTES)))
        ax.set_yticks(range(len(ATTRIBUTES)))
        ax.set_xticklabels(ATTRIBUTES, rotation=90, fontsize=9)
        ax.set_yticklabels(ATTRIBUTES, fontsize=9)
        ax.set_title(title, fontsize=14)

    fig.colorbar(im, ax=axes, shrink=0.5, label="Dissimilarity (1 - correlation)")

    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        slug = model_label.lower().replace(' ', '_')
        fig.savefig(FIGURES / f"rdm_comparison_{slug}.png", dpi=150, bbox_inches="tight")
    return fig

def plot_model_comparison(df_results, model_name, show_self_reliability=False, save=True):
    """
    Plots model performance vs human reliability (vs. self-reliability) for each attribute.
    """
    
    fig, ax = plt.subplots(figsize=(10, len(df_results) * 0.4 + 2), dpi=150)
    y_positions = np.arange(len(df_results))

    ax.barh(y_positions, df_results['human_reliability_r2'], color='#fcdcdb',
            edgecolor='none', label='Model Shortfall', height=0.5)
    ax.barh(y_positions, df_results['ai_performance_r2'], color='black',
            edgecolor='none', label='Model', height=0.5)

    for idx, row in df_results.iterrows():
        human_tick_range = [idx - 0.18, idx + 0.18] if show_self_reliability else [idx - 0.25, idx + 0.25]
        ax.plot([row['human_reliability_r2'], row['human_reliability_r2']],
                human_tick_range, color='#d62728', linewidth=2.5, zorder=3)

        model_pct = int(round(row['ai_performance_r2'] * 100))
        human_pct = int(round(row['human_reliability_r2'] * 100))

        ax.text(row['ai_performance_r2'] + 0.01, idx, f"{model_pct}",
                va='center', ha='left', fontsize=9, color='black')

        if show_self_reliability:
            ax.plot([row['self_reliability_r2'], row['self_reliability_r2']],
                    [idx - 0.18, idx + 0.18], color='#1f77b4', linewidth=2.5, zorder=4)
            self_pct = int(round(row['self_reliability_r2'] * 100))
            ax.text(row['human_reliability_r2'] + 0.01, idx - 0.15, f"{human_pct}",
                    va='center', ha='left', fontsize=8, color='#d62728', fontweight='bold')
            ax.text(row['self_reliability_r2'] + 0.01, idx + 0.15, f"{self_pct}",
                    va='center', ha='left', fontsize=8, color='#1f77b4', fontweight='bold')
        else:
            ax.text(row['human_reliability_r2'] + 0.01, idx, f"{human_pct}",
                    va='center', ha='left', fontsize=9, color='#d62728', fontweight='bold')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([attr.replace('_', ' ') for attr in df_results['attribute']], fontsize=10)
    ax.set_xlabel('Out-of-sample $R^2$', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    legend_elements = [
        Line2D([0], [0], color='#d62728', lw=2.5, label='Human Split-Half Ceiling'),
        Line2D([0], [0], color='black', lw=6, label=f'{model_name} Model'),
        Line2D([0], [0], color='#fcdcdb', lw=6, label='Model Shortfall')
    ]
    if show_self_reliability:
        legend_elements.insert(1, Line2D([0], [0], color='#1f77b4', lw=2.5, label='Model Self-Reliability'))

    ax.legend(handles=legend_elements, loc='lower right', frameon=False, bbox_to_anchor=(1, 0.1))
    title_suffix = " and Self-Reliability" if show_self_reliability else ""
    plt.title(f'Model Performance vs. Human Reliability{title_suffix} ({model_name})',
              fontsize=12, pad=20, fontstyle='italic')
    plt.tight_layout()

    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)
        slug = model_name.lower().replace(' ', '_')
        fig.savefig(FIGURES / f'model_comparison_{slug}_{'with_self' if show_self_reliability else 'base'}.png', dpi=150, bbox_inches="tight")
    return fig