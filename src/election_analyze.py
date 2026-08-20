from config import ELECTION_MODELS, MODEL_DIR
import pandas as pd
from scipy.stats import binomtest
from pathlib import Path
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest, spearmanr
from config import ELECTION_MODELS, MODEL_DIR, RESULTS, ELECTION_RESULTS_PATH
from collect import get_filepath
from election_collect import clean_election_df
from analyze import _save
from collections import Counter
import re
import matplotlib.pyplot as plt
import numpy as np

'''
def check_recognition_match(raw_response, real_last_name):
    """
    Classify the model's recognition attempt against the real candidate.
    Returns 'declined', 'wrong_guess', or 'correct'.
    """
    if raw_response is None or not isinstance(raw_response, str):
        return None
    resp = raw_response.lower()

    # model declined to identify
    if 'unknown' in resp[:30]:
        return 'declined'

    # did the model name the actual candidate? (last name appears in response)
    if isinstance(real_last_name, str) and real_last_name.strip():
        last = real_last_name.lower().strip()
        # word-boundary match so "Hunt" doesn't match "hunting", etc.
        if re.search(rf'\b{re.escape(last)}\b', resp):
            return 'correct'      # genuinely recognized the real person
    return 'wrong_guess'          # said something, but not the right name

@deprecated("Use election_analyze_multitrial instead")
def election_results_table(mode="electability"):
    rows = []
    for model, folder in ELECTION_MODELS.items():
        path = MODEL_DIR / folder / f"{mode}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        total = len(df)
        clean = df[df['status'] == 'ok']
        n = len(clean)
        pos_bias = df['status'].str.startswith('POSITION_BIAS').sum()
        refusal  = df['status'].str.startswith('REFUSE').sum()
        other_err = df['status'].str.startswith('ERROR').sum()
        if n > 0:
            hits = int(clean['model_correct'].sum())
            p = binomtest(hits, n, 0.5, alternative='greater').pvalue
            acc = hits / n
        else:
            hits, p, acc = 0, None, None
        rows.append({
            'model': folder, 'clean_n': n, 'total': total,
            'accuracy': round(acc, 3) if acc is not None else None,
            'p_value': f"{p:.1e}" if p is not None else None,
            'hits': hits, 'position_bias': int(pos_bias),
            'position_bias_rate': round(pos_bias / total, 3),
            'refusals': int(refusal),
            'other error': int(other_err),
        })
    return _save(pd.DataFrame(rows), f'election_results_table_{mode}.csv')  # mode in name


@deprecated("Use election_analyze_multitrial instead")
def election_validity_table(mode="electability"):
    """Per-model: overall vs open-race accuracy + contamination rate."""
    _, elec = clean_election_df(ELECTION_RESULTS_PATH)
    incumbent_by_race = (
        elec.assign(is_inc=elec['Incumbent?'].astype(str).str.lower().str.strip().eq('yes'))
            .groupby('Election ID')['is_inc'].any())

    rows = []
    for model, folder in ELECTION_MODELS.items():
        epath = MODEL_DIR / folder / "{mode}.csv"
        if not epath.exists():
            continue
        df = pd.read_csv(epath)
        df = df[df['status'] == 'ok'].copy()
        df['has_incumbent'] = df['election_id'].map(incumbent_by_race)

        def acc(sub):
            n = len(sub)
            return (round(sub['model_correct'].sum()/n, 3), n) if n else (None, 0)

        all_acc, all_n   = acc(df)
        open_acc, open_n = acc(df[df['has_incumbent'] == False])

        # contamination rate from recognition file (if present)
        rpath = MODEL_DIR / folder / "recognition.csv"
        #updated_rpath = MODEL_DIR / folder / "updated_recognition.csv"
        contam = None
        if rpath.exists():
            rec = pd.read_csv(rpath)
            comp_names = elec.set_index('Full Label')['Candidate']  # last names
            rec['real_name'] = rec['candidate_label'].map(comp_names)
            rec['match'] = rec.apply(
                lambda r: check_recognition_match(r['raw_response'], r['real_name']), axis=1)
            contam = round((rec['match'] == 'correct').mean(), 3)         

        rows.append({
            'model': folder,
            'accuracy_all': all_acc, 'n_all': all_n,
            'accuracy_open': open_acc, 'n_open': open_n,
            'contamination_rate': contam,
            #'updated_contamination_rate': updated_contam
        })
    return _save(pd.DataFrame(rows), f'election_validity_table_{mode}.csv')

@deprecated("Use election_analyze_multitrial instead")
def human_agreement(pooled=True, mode="electability"):
    """
    Agreement between model's forced-choice pick and the human competence pick.
    Returns per-model rates and a single pooled rate (vs 50% chance).
    """
    _, elec = clean_election_df(ELECTION_RESULTS_PATH)
    comp = elec.set_index('Full Label')['Competency']

    rows = []
    all_agree = 0; all_n = 0
    for folder in ELECTION_MODELS:
        path = MODEL_DIR / folder / f"{mode}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df['status'] == 'ok']

        agree = 0; n = 0
        for _, r in df.iterrows():
            ca, cb = comp.get(r['candidate_a']), comp.get(r['candidate_b'])
            if pd.isna(ca) or pd.isna(cb) or ca == cb:
                continue
            human_pick = r['candidate_a'] if ca > cb else r['candidate_b']
            if r['model_chose_label'] == human_pick:
                agree += 1
            n += 1
        rows.append({'model': folder, 'n': n,
                     'agreement': round(agree/n, 3) if n else None})
        all_agree += agree; all_n += n

    table = pd.DataFrame(rows)
    _save(table, f'human_agreement_by_model_{mode}.csv')

    # single pooled number
    pooled_rate = all_agree / all_n
    p = binomtest(all_agree, all_n, 0.5, alternative='greater').pvalue
    print(f"POOLED model-human agreement: {all_agree}/{all_n} = {pooled_rate:.1%}, p={p:.1e}")
    _save(pd.DataFrame([{'pooled_agreement': round(pooled_rate,3),
                         'n': all_n, 'p_value': f"{p:.1e}"}]),
          f'human_agreement_pooled_{mode}.csv')
    return table

@deprecated("Use election_analyze_multitrial instead")
def bridge_plot(suffix="main", mode="electability", max_position_bias=0.4):
    """
    Bridge analysis: election accuracy vs. competence-axis fidelity.
    Tests whether predictive validity tracks representational fidelity.
    A FLAT relationship = the dissociation (fidelity scales, validity doesn't).
    """
   
    # 1. competence-axis fidelity (PC3) from the scaling study
    pca = pd.read_csv(RESULTS / f'main_pca_comparison_{suffix}.csv')
    comp_fid = (pca[pca['human_pc'] == 3]
                .set_index('model')['abs_r'])   # model label -> competence fidelity

    # 2. election accuracy per model (clean responses, excluding high-position-bias)
    rows = []
    for label, folder in ELECTION_MODELS.items():   # label must match pca 'model' column
        
        full_path = MODEL_DIR / folder / f"{mode}.csv"

        if not full_path.exists():
            continue
        e = pd.read_csv(full_path)
        total = len(e)
    
        if total == 0:
            continue

        pos_bias_rate = e['status'].str.startswith('POSITION_BIAS').mean()
        clean = e[e['status'] == 'ok']
       
        if len(clean) == 0 or pos_bias_rate > max_position_bias:
            continue   # skip refusers and high-position-bias artifacts (Mistral, etc.)
        
        acc = clean['model_correct'].mean()
        fid = comp_fid.get(label)
        
        if fid is None:
            print(f"no fidelity for {label}"); continue
        
        rows.append({'model': label, 'accuracy': acc, 'competence_fidelity': fid,
                     'n': len(clean)})

    df = pd.DataFrame(rows)
    _save(df, f'bridge_data_{suffix}_{mode}.csv')

    # 3. correlation: does accuracy track fidelity?
    r, p = spearmanr(df['competence_fidelity'], df['accuracy'])
    print(f"election-accuracy vs competence-fidelity: r={r:.3f}, p={p:.3f} (n={len(df)})")

    # 4. plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['competence_fidelity'], df['accuracy'], s=100)
    for _, row in df.iterrows():
        ax.annotate(row['model'], (row['competence_fidelity'], row['accuracy']),
                    fontsize=8, xytext=(5, 5), textcoords='offset points')
    ax.axhline(0.688, ls='--', color='gray', alpha=0.7, label='Human baseline (68.8%)')
    ax.axhline(0.5, ls=':', color='red', alpha=0.4, label='Chance (50%)')
    ax.set_xlabel('Competence-axis fidelity (|r|)')
    ax.set_ylabel('Election prediction accuracy')
    
    ax.set_title(f'Bridge: predictive validity vs. representational fidelity ({mode})')
    
    ax.legend()
    fig.savefig(RESULTS / f'bridge_plot_{suffix}_{mode}.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    return df

@deprecated("Use election_analyze_multitrial instead")
def impression_decision_comparison(min_n=100, max_pos_bias=0.4,
                                    comp_label="competence", vote_label="electability"):
    """
    Core finding: does the competence IMPRESSION predict elections better than
    the model's own VOTE decision? Per-model + pooled, saved to file.
    """
    rows = []
    for model, folder in ELECTION_MODELS.items():
        cpath = get_filepath(model_folder=folder, prompt_label=comp_label)
        vpath = get_filepath(model_folder=folder, prompt_label=vote_label)
        if not (cpath.exists() and vpath.exists()):
            continue
        c = pd.read_csv(cpath); v = pd.read_csv(vpath)
        # position-bias check on each prompt
        c_pb = c['status'].str.startswith('POSITION_BIAS').mean()
        v_pb = v['status'].str.startswith('POSITION_BIAS').mean()
        c_clean = c[c['status'] == 'ok']; v_clean = v[v['status'] == 'ok']
        if (len(c_clean) < min_n or len(v_clean) < min_n
                or c_pb > max_pos_bias or v_pb > max_pos_bias):
            continue
        comp_acc = c_clean['model_correct'].mean()
        vote_acc = v_clean['model_correct'].mean()
        rows.append({
            'model': model,
            'comp_acc': round(comp_acc, 3), 'comp_n': len(c_clean),
            'vote_acc': round(vote_acc, 3), 'vote_n': len(v_clean),
            'comp_minus_vote': round(comp_acc - vote_acc, 3),
        })
    table = pd.DataFrame(rows)

    # pooled/consistency stats
    from scipy.stats import binomtest
    n_models = len(table)
    n_comp_higher = (table['comp_minus_vote'] > 0).sum()
    sign_p = binomtest(n_comp_higher, n_models, 0.5, alternative='greater').pvalue
    print(f"competence > vote in {n_comp_higher}/{n_models} models (sign test p={sign_p:.4f})")
    print(f"mean(comp - vote) = {table['comp_minus_vote'].mean():+.3f}")

    _save(table, 'impression_decision_by_model.csv')
    _save(pd.DataFrame([{
        'n_models': n_models, 'n_comp_higher': n_comp_higher,
        'sign_test_p': f"{sign_p:.4f}",
        'mean_comp_minus_vote': round(table['comp_minus_vote'].mean(), 3),
    }]), 'impression_decision_summary.csv')
    return table

@deprecated("Use election_analyze_multitrial instead")
def disagreement_analysis(min_n=100, max_pos_bias=0.4,
                           comp_label="competence", vote_label="electability"):
    """
    When competence-pick and vote-pick DISAGREE: are those closer races,
    and which pick is more accurate? Saves per-race disagreement data + summary.
    """
    from scipy.stats import binomtest, mannwhitneyu

    _, elec = clean_election_df(ELECTION_RESULTS_PATH)
    margin = (elec.groupby('Election ID')
              .apply(lambda g: abs(g['Vote Share'].iloc[0] - g['Vote Share'].iloc[1])
                     if len(g) == 2 else None))

    # first find clean-on-both models (reuse the same filter)
    clean = impression_decision_comparison(min_n, max_pos_bias, comp_label, vote_label)
    clean_models = set(clean['model'])

    all_disagree = []   # per-race records across clean models
    agree_margins = []
    for model, folder in ELECTION_MODELS.items():
        if model not in clean_models:
            continue
        c = pd.read_csv(get_filepath(model_folder=folder, prompt_label=comp_label))
        v = pd.read_csv(get_filepath(model_folder=folder, prompt_label=vote_label))
        c = c[c['status']=='ok'][['election_id','model_chose_label','model_correct']]
        v = v[v['status']=='ok'][['election_id','model_chose_label','model_correct']]
        c = c.rename(columns={'model_chose_label':'c_pick','model_correct':'c_correct'})
        v = v.rename(columns={'model_chose_label':'v_pick','model_correct':'v_correct'})
        m = c.merge(v, on='election_id')
        m['margin'] = m['election_id'].map(margin)
        m['model'] = model
        dis = m[m['c_pick'] != m['v_pick']]
        agree_margins += m[m['c_pick'] == m['v_pick']]['margin'].dropna().tolist()
        all_disagree.append(dis)

    disagree = pd.concat(all_disagree, ignore_index=True)
    _save(disagree, 'disagreement_races.csv')   # the raw per-race data

    # summary stats
    comp_correct = disagree['c_correct'].sum()
    total = disagree['c_correct'].notna().sum()
    pooled_p = binomtest(int(comp_correct), int(total), 0.5).pvalue
    dis_margins = disagree['margin'].dropna().tolist()
    u, margin_p = mannwhitneyu(dis_margins, agree_margins)

    print(f"disagreements: {total} races across {len(clean_models)} models")
    print(f"competence correct on disagreements: {comp_correct:.0f}/{total} "
          f"= {comp_correct/total:.1%} (p={pooled_p:.4f})")
    print(f"disagree margin {np.mean(dis_margins):.3f} vs agree {np.mean(agree_margins):.3f} "
          f"(Mann-Whitney p={margin_p:.4f})")

    _save(pd.DataFrame([{
        'n_disagreements': int(total),
        'comp_correct_pct': round(comp_correct/total, 3),
        'vote_correct_pct': round(1 - comp_correct/total, 3),
        'pooled_p': f"{pooled_p:.4f}",
        'disagree_mean_margin': round(np.mean(dis_margins), 3),
        'agree_mean_margin': round(np.mean(agree_margins), 3),
        'margin_mannwhitney_p': f"{margin_p:.4f}",
    }]), 'disagreement_summary.csv')
    return disagree

@deprecated("Use election_analyze_multitrial instead")
def age_confound_check(comp_label="competence"):
    """Does the competence impression just track candidate age? Saves result."""
    _, elec = clean_election_df(ELECTION_RESULTS_PATH)
    ages = elec.set_index('Full Label')['Age']   # real ages

    picks_older = 0; total = 0
    for model, folder in ELECTION_MODELS.items():
        cpath = get_filepath(model_folder=folder, prompt_label=comp_label)
        if not cpath.exists():
            continue
        c = pd.read_csv(cpath)
        c = c[c['status'] == 'ok']
        for _, r in c.iterrows():
            a_age = ages.get(r['candidate_a']); b_age = ages.get(r['candidate_b'])
            if pd.isna(a_age) or pd.isna(b_age) or a_age == b_age:
                continue
            older = r['candidate_a'] if a_age > b_age else r['candidate_b']
            if r['model_chose_label'] == older:
                picks_older += 1
            total += 1
    pct = picks_older / total if total else None
    print(f"competence pick = older candidate: {pct:.1%} (n={total})")
    _save(pd.DataFrame([{
        'competence_picks_older_pct': round(pct, 3) if pct else None,
        'n': total,
    }]), 'age_confound_check.csv')
    return pct
'''