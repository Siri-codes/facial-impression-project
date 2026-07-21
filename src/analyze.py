# analyze.py
from pathlib import Path
import pandas as pd
from config import MODELS, MODEL_DIR, ATTRIBUTES, HUMAN_MEANS, RESULTS, ALL_MODEL_NAMES, REP_SUBSET_SIZE, SUBSAMPLE_SEED
from data_io import load_ratings, load_human_means, load_human_raw, common_stimuli
from rdm import build_rdm, compare_rdms
from pca import fit_pca, match_components
from metrics import build_evaluation_dataset
from plots import plot_model_comparison, plot_pca_loadings
import numpy as np
from metrics import pearsonr_correlation
   
def _save(df, name):
    (RESULTS).mkdir(exist_ok=True)
    df.to_csv(RESULTS / name, index=False)
    return df

def rsa_scores():
    """RDM Spearman, human vs each model, direct condition."""
    
    human = load_human_means()
    models = {l: load_ratings(MODEL_DIR/f/"direct_main.csv") for l, f in MODELS.items()}
    dfs = common_stimuli(human, *models.values())
    human_rdm = build_rdm(dfs[0][ATTRIBUTES].to_numpy())     # you have build_rdm
    rows = []
    for (label, _), df in zip(models.items(), dfs[1:]):
        rdm = build_rdm(df[ATTRIBUTES].to_numpy())
        r, p = compare_rdms(human_rdm, rdm)                   # you have compare_rdms
        rows.append({'model': label, 'spearman': round(r,3)})
    return _save(pd.DataFrame(rows), 'rsa_scores.csv')

def main_pca_comparison():
    """n=1004 direct: match each model's PCs to human, report |r| + variance."""

    human = load_human_means()
    models = {l: load_ratings(MODEL_DIR/f/"direct_main.csv") for l, f in MODELS.items()}
    dfs = common_stimuli(human, *models.values())
    h_pca = fit_pca(dfs[0][ATTRIBUTES].to_numpy())

    rows = []
    for (label, _), df in zip(models.items(), dfs[1:]):
        m_pca = fit_pca(df[ATTRIBUTES].to_numpy())
        for i, j, r, vr, vm in match_components(h_pca, m_pca, k=3):
            rows.append({'model': label, 'human_pc': i+1,
                         'human_var': round(vr,3), 'model_pc': j+1,
                         'model_var': round(vm,3), 'abs_r': round(r,3)})
    return _save(pd.DataFrame(rows), 'main_pca_comparison.csv')

def prompt_condition_comparison():
    """
    Compare biased / predict_human / direct prompts against human structure,
    on the shared 100-stimulus pilot set. Shows the biased prompt matched
    human structure WORSE than direct.
    """
    # 1. Load humans, and get the pilot 100 stimulus ids
    human = load_human_means()
    human_subset = human.sample(n=REP_SUBSET_SIZE, random_state=SUBSAMPLE_SEED)
   
    # 2. Build a dict of the three condition files for one model:
    condition_files = {'biased': 'predict_human_biased_main.csv',
                       'predict_human': 'predict_human_pilot.csv',
                       'direct': 'direct_pilot.csv'}
    
    rows = []
    for model, folder in MODELS.items():
        ratings = {condition: load_ratings(MODEL_DIR/folder/path) for condition, path in condition_files.items()}
       
        # 3. For each model: common_stimuli(human_subsetted_to_pilot, *the three dfs)
        #    -> everything on the same 100 faces
        dfs = common_stimuli(human_subset, *ratings.values())
        
        # 4. fit_pca on the aligned human df -> h_pca
        h_pca = fit_pca(dfs[0][ATTRIBUTES].to_numpy())

        # 5. For each condition df: fit_pca, then match_components(h_pca, m_pca, k=3)
        #    Collect rows: model, condition, human_pc, abs_r, model_var

        for (label, _), df in zip(ratings.items(), dfs[1:]):
            m_pca = fit_pca(df[ATTRIBUTES].to_numpy())
            for i, j, r, vr, vm in match_components(h_pca, m_pca, k=3):
                rows.append({'model': model, 'condition': label, 'human_pc': i+1,
                            'human_var': round(vr,3), 'model_pc': j+1,
                            'model_var': round(vm,3), 'abs_r': round(r,3)})
    
    return _save(pd.DataFrame(rows), 'prompt_comparison.csv')


def priming_comparison(n_boot=1000, seed=0):
    """direct vs direct_primed: mean per-attribute R² delta + bootstrap CI."""
    rng = np.random.default_rng(seed)
    human = load_human_means()

    rows = []
    for label, folder in MODELS.items():
        # 1. load the two condition files: direct_pilot.csv  and  direct_primed_pilot.csv
        direct = load_ratings(MODEL_DIR/folder/"direct_pilot.csv")
        primed = load_ratings(MODEL_DIR/folder/"direct_primed_pilot.csv")

        # 2. common_stimuli(human, direct, primed) -> h, d, p  (same faces)
        h, d, p = common_stimuli(human, direct, primed)

        # 3. point estimate:
        def mean_delta(hh, dd, pp):
            out = []
            for attr in ATTRIBUTES:
                r_d = np.corrcoef(hh[attr], dd[attr])[0, 1]
                r_p = np.corrcoef(hh[attr], pp[attr])[0, 1]
                if np.isnan(r_d) or np.isnan(r_p):
                    continue
                out.append(r_p**2 - r_d**2)
            return np.mean(out)
        
        point = mean_delta(h, d, p)
        
        # 4. bootstrap/resample over stimuli (rows)
        boot = []
        for _ in range(n_boot):
            idx = rng.choice(len(h), len(h), replace=True)
            boot.append(mean_delta(h.iloc[idx], d.iloc[idx], p.iloc[idx]))
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

        rows.append({'model': label, 'mean_delta': round(point, 3),
                     'ci_low': round(ci_low, 3), 'ci_high': round(ci_high, 3)})
        
    return _save(pd.DataFrame(rows), 'priming_comparison.csv')

def reliability_replication(k=3):
    """Does PCA structure replicate across 3 collections of the same prompt?
    Reads predict_human_biased_main (=rep1, subset to 100), rep2, rep3."""
    rows = []
    rep_paths = {1: 'predict_human_biased_main.csv',   # rep1 = main, subset to 100
                2: 'predict_human_biased_rep2.csv',
                3: 'predict_human_biased_rep3.csv'}
                
    for model, folder in MODELS.items():
        ratings = {rep: load_ratings(MODEL_DIR/folder/path) for rep, path in rep_paths.items()}
        dfs = common_stimuli(*ratings.values())  # all on the same 100 faces
        pcas = [fit_pca(df[ATTRIBUTES].to_numpy()) for df in dfs]

        for (a, b) in [(0,1), (0,2), (1,2)]:
            for i, j, r, vr, vm in match_components(pcas[a], pcas[b], k=k):
                    rows.append({'model': model,
                                 'pair': f'rep{a+1}v{b+1}',
                               'pc': i+1, 'abs_r': round(r,3)})

    return _save(pd.DataFrame(rows), 'reliability_replication.csv')

def pca_loading_figures(pc_x=0, pc_y=1):
    '''
    Loads PCA scatterplots for human and model ratings
    '''
    human = load_human_means()
    models = {l: load_ratings(MODEL_DIR/f/"direct_main.csv") for l, f in MODELS.items()}
    dfs = common_stimuli(human, *models.values())
    judges = {'Humans': dfs[0], **dict(zip(models, dfs[1:]))}
    figs = {}
    for label, df in judges.items():
        pca = fit_pca(df[ATTRIBUTES].to_numpy())
        figs[label] = plot_pca_loadings(pca, label, pc_x, pc_y, save=True)
    return figs 