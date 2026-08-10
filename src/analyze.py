# analyze.py
from pathlib import Path
import pandas as pd
from config import MODELS, MODEL_DIR, ATTRIBUTES, HUMAN_MEANS, RESULTS, REP_SUBSET_SIZE, SUBSAMPLE_SEED, MMMU_PRO
from data_io import load_ratings, load_human_means, load_human_raw, common_stimuli
from rdm import build_rdm, compare_rdms
from pca import fit_pca, match_components
from metrics import build_evaluation_dataset, compute_self_reliability
from plots import plot_model_comparison, plot_pca_loadings
import numpy as np
from scipy.stats import spearmanr
   
def _save(df, name):
    (RESULTS).mkdir(exist_ok=True)
    df.to_csv(RESULTS / name, index=False)
    return df

def usable_models(suffix, sensitive_only=True, min_constant=3):
    """
    Returns (usable, excluded) dicts of {label: folder}.
    A model is excluded if it gives constant ratings on >= min_constant
    sensitive attributes (i.e. it's refusing/neutralizing).
    """
    sensitive = ['asian','black','white','hispanic','middle-eastern','islander',
                 'native','gay','privileged','liberal']
    check_attrs = sensitive if sensitive_only else ATTRIBUTES

    usable, excluded = {}, {}
    for label, folder in MODELS.items():
        path = MODEL_DIR/folder/f"direct_{suffix}.csv"
        if not path.exists():
            excluded[label] = 'missing_file'
            continue
        df = load_ratings(path)
        n_const = sum(df[a].std() == 0 for a in check_attrs if a in df)
        if n_const >= min_constant:
            excluded[label] = f'{n_const}_constant_sensitive_attrs'
        else:
            usable[label] = folder
    return usable, excluded

def reliability_bars(condition="direct"):
    """Per-attribute accuracy vs human ceiling and model self-reliability.
    All three quantities on the same prompt condition."""
    human_means = load_human_means()
    human_raw   = load_human_raw()

    rows = []
    for label, folder in MODELS.items():
        # 1. accuracy + human ceiling
        main = load_ratings(MODEL_DIR/folder/f"{condition}_main.csv")
        results = build_evaluation_dataset(human_means, human_raw, main)
        # -> columns: attribute, human_reliability_r2, ai_performance_r2

        # 2. self-reliability from the three reps
        rep1 = load_ratings(MODEL_DIR/folder/f"{condition}_pilot.csv")
        rep2 = load_ratings(MODEL_DIR/folder/f"{condition}_pilot_rep2.csv")
        rep3 = load_ratings(MODEL_DIR/folder/f"{condition}_pilot_rep3.csv")

        r1, r2, r3 = common_stimuli(rep1, rep2, rep3)
        self_rel = compute_self_reliability(r1, r2, r3)
        # -> columns: attribute, self_reliability_r2

        # 3. merge results + self on 'attribute'
        merged = pd.merge(results, self_rel, on='attribute')

        # 4. plot
        fig = plot_model_comparison(merged, label, show_self_reliability=True)

        # 5. labeling
        rows.append(merged.assign(model=label))

    return _save(pd.concat(rows), f'reliability_bars_{condition}.csv')

def rsa_scores(suffix="main"):
    """RDM Spearman, human vs each model."""
    human = load_human_means()
    usable, excluded = usable_models(suffix)
    if excluded:
        print("Excluded (refusing/neutralizing):")
        for label, reason in excluded.items():
            print(f"  {label}: {reason}")

    models = {l: load_ratings(MODEL_DIR/MODELS[l]/f"direct_{suffix}.csv") for l in usable}
    dfs = common_stimuli(human, *models.values())
    human_rdm = build_rdm(dfs[0][ATTRIBUTES].to_numpy())
    rows = []
    for label, df in zip(models.keys(), dfs[1:]):
        rdm = build_rdm(df[ATTRIBUTES].to_numpy())
        r, p = compare_rdms(human_rdm, rdm)
        rows.append({'model': label, 'spearman': round(r,3)})
    return _save(pd.DataFrame(rows), f'rsa_scores_{suffix}.csv')

def main_pca_comparison(suffix="main"):
    """n=1004 direct: match each model's PCs to human, report |r| + variance."""
    human = load_human_means()
    usable, excluded = usable_models(suffix)
    if excluded:
        print("Excluded (refusing/neutralizing):")
        for label, reason in excluded.items():
            print(f"  {label}: {reason}")

    models = {l: load_ratings(MODEL_DIR/MODELS[l]/f"direct_{suffix}.csv") for l in usable}
    dfs = common_stimuli(human, *models.values())
    h_pca = fit_pca(dfs[0][ATTRIBUTES].to_numpy())
    rows = []
    for (label, _), df in zip(models.items(), dfs[1:]):
        m_pca = fit_pca(df[ATTRIBUTES].to_numpy())
        for i, j, r, vr, vm in match_components(h_pca, m_pca, k=3):
            rows.append({'model': label, 'human_pc': i+1, 'human_var': round(vr,3),
                         'model_pc': j+1, 'model_var': round(vm,3), 'abs_r': round(r,3)})
    return _save(pd.DataFrame(rows), f'main_pca_comparison_{suffix}.csv')

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

def reliability_replication(k=3,condition="direct"):
    """Does PCA structure replicate across 3 collections of the same prompt?
    Reads predict_human_biased_main (=rep1, subset to 100), rep2, rep3."""
    rows = []
    rep_paths = {1: f'{condition}_main.csv',   # rep1 = main, subset to 100
                2: f'{condition}_rep2.csv',
                3: f'{condition}_rep3.csv'}
                
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

def capability_plots(suffix="pilot"):
    from config import MMMU_PRO
    from plots import plot_capability_scatter

    # RSA vs capability
    rsa = pd.read_csv(RESULTS / f'rsa_scores_{suffix}.csv')
    rsa['mmmu'] = rsa['model'].map(MMMU_PRO)
    plot_capability_scatter(rsa, 'spearman', 'RSA (Spearman ρ)',
                            'Structural alignment vs capability')
    
    pca = pd.read_csv(RESULTS / f'main_pca_comparison_{suffix}.csv')

    # Valence axis (PC1) fidelity vs capability:
    val = pca[pca['human_pc'] == 1].copy()
    val['mmmu'] = val['model'].map(MMMU_PRO)
    plot_capability_scatter(val, 'abs_r', 'Valence-Dominance-axis fidelity (|r|)',
                            'Valence-Dominance-axis vs capability')

    # Race-axis 
    race = pca[pca['human_pc'] == 2].copy()      # race axis = human PC2
    race['mmmu'] = race['model'].map(MMMU_PRO)
    plot_capability_scatter(race, 'abs_r', 'Race-axis fidelity (|r|)',
                            'Race-axis vs capability')
    
    # Competence axis (PC3)
    pc3 = pca[pca['human_pc'] == 3].copy()
    pc3['mmmu'] = pc3['model'].map(MMMU_PRO)
    plot_capability_scatter(pc3, 'abs_r', 'Competence-axis fidelity (|r|)',
                            'Competence-axis vs capability')
   

def capability_correlations(suffix="main"):
    #loading results files
    pca = pd.read_csv(RESULTS / f'main_pca_comparison_{suffix}.csv')
    rsa = pd.read_csv(RESULTS / f'rsa_scores_{suffix}.csv')

    rows = []

    for name, pc in [("valence-dominance", 1), ("race", 2), ("competence", 3)]: #for each PC
        d = pca[pca['human_pc'] == pc].copy() #filtering for current axis
        d['mmmu'] = d['model'].map(MMMU_PRO) #set mmmu score based on dict in config
        d = d.dropna(subset=['mmmu', 'abs_r']) #drop rows with missing values
        r, p = spearmanr(d['mmmu'], d['abs_r']) #calculate spearman correlation between capability and fidelity accross models
        print(f"{name} (PC{pc}): r={r:.3f}, p={p:.3f} (n={len(d)})")
        
        rows.append({'measure': name, 'pc': pc, 'r': round(r, 3),
                     'p': round(p, 3), 'n': len(d)}) #append results to rows


    rsa['mmmu'] = rsa['model'].map(MMMU_PRO)
    d = rsa.dropna(subset=['mmmu', 'spearman'])
    r, p = spearmanr(d['mmmu'], d['spearman'])
    print(f"rsa: r={r:.3f}, p={p:.3f} (n={len(d)})")

    rows.append({'measure': 'rsa', 'pc': None, 'r': round(r, 3),
                 'p': round(p, 3), 'n': len(d)})

    return _save(pd.DataFrame(rows), f'capability_correlations_{suffix}.csv')