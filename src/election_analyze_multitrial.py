"""
Multi-trial election analysis.

Trial-level schema (one row per race x ordering x trial):
    election_id, trial, slotA_face, chosen_slot, chosen_face, status, raw

Recommended order:
    verify_pipeline(label)          # run ONCE — confirms winner mapping is sound
    classification_breakdown(label) # reliability profile: stable / bias / noise
    accuracy_both_ways(label)       # filtered AND unfiltered accuracy, with CIs
    disagreement_analysis(a, b)     # mechanism: where impression and vote diverge
    refusal_rates(labels)           # refusal comparison across prompts

Face convention: face 'a' is the FIRST candidate of a race (race.iloc[0]),
face 'b' the second. verify_pipeline() checks this holds against the data.
"""

import pandas as pd
import numpy as np
from scipy.stats import binomtest, mannwhitneyu, norm

from config import ELECTION_MODELS, ELECTION_RESULTS_PATH
from election_collect import clean_election_df, get_filepath
from analyze import _save

import re

def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion (k of n successes).
    Same result as statsmodels proportion_confint(method='wilson'), no extra dependency"""
    if not n:
        return (None, None)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5)) / denom
    return (center - half, center + half)


# ======================================================================
# Ground truth: which face won each race, and the winning margin.
# Keyed on Election ID, which must match the collected 'election_id'.
# ======================================================================
def _race_truth():
    """Return {election_id: {'winner': 'a'|'b', 'margin': float}}.
    'a' is the first candidate (race.iloc[0]); margin is |vote share difference|."""
    _, elec = clean_election_df(ELECTION_RESULTS_PATH)
    truth = {}
    for eid, g in elec.groupby('Election ID'):
        if len(g) != 2:
            continue
        first_won = g.iloc[0]['Winner/Loser'] == 'Winner'
        margin = abs(g.iloc[0]['Vote Share'] - g.iloc[1]['Vote Share'])
        truth[eid] = {'winner': 'a' if first_won else 'b', 'margin': margin}
    return truth


# ======================================================================
# VERIFY FIRST. Confirms the winner mapping is internally consistent
# before any accuracy number is trusted. Catches the two failure modes
# that would silently invert results: id-type mismatch, and
# Winner/Loser disagreeing with Vote Share.
# ======================================================================
def verify_pipeline(prompt_label):
    truth = _race_truth()
    _, elec = clean_election_df(ELECTION_RESULTS_PATH)

    # (1) Winner/Loser must agree with Vote Share on who won.
    disagree_rows = []
    for eid, g in elec.groupby('Election ID'):
        if len(g) != 2:
            continue
        by_label = g.iloc[0]['Winner/Loser'] == 'Winner'
        by_votes = g.iloc[0]['Vote Share'] > g.iloc[1]['Vote Share']
        if by_label != by_votes:
            disagree_rows.append({
                'election_id': eid,
                'cand_0': g.iloc[0]['Full Label'],
                'cand_0_label': g.iloc[0]['Winner/Loser'],
                'cand_0_vote': g.iloc[0]['Vote Share'],
                'cand_1': g.iloc[1]['Full Label'],
                'cand_1_label': g.iloc[1]['Winner/Loser'],
                'cand_1_vote': g.iloc[1]['Vote Share'],
            })
    print(f"Winner/Loser vs Vote Share disagreements: {len(disagree_rows)}")
    if disagree_rows:
        print("  WARNING: label and vote share disagree on these races — "
              "investigate before trusting winners:\n")
        dis_df = pd.DataFrame(disagree_rows)
        # show each disagreeing race clearly
        for _, r in dis_df.iterrows():
            print(f"  race {r['election_id']}:")
            print(f"    {r['cand_0']}: label={r['cand_0_label']}, vote={r['cand_0_vote']}")
            print(f"    {r['cand_1']}: label={r['cand_1_label']}, vote={r['cand_1_vote']}")
        _save(dis_df, 'winner_voteshare_disagreements.csv')
        print(f"\n  (saved the {len(dis_df)} disagreeing races to winner_voteshare_disagreements.csv)")

    # (2) election_id types must match between truth map and collected data.
    sample_folder = next(iter(ELECTION_MODELS.values()))
    path = get_filepath(model_folder=sample_folder, prompt_label=prompt_label)
    if path.exists():
        df = pd.read_csv(path)
        truth_key_type = type(next(iter(truth)))
        data_key_type = type(df['election_id'].iloc[0])
        overlap = set(df['election_id']) & set(truth)
        print(f"truth id type: {truth_key_type.__name__}, "
              f"data id type: {data_key_type.__name__}")
        print(f"races in data that map to a known winner: "
              f"{len(overlap)} / {df['election_id'].nunique()}")
        if not overlap:
            print("  WARNING: no id overlap — election_id types likely mismatched. "
                  "Cast them to the same type before analysis.")
    else:
        print(f"(no collected file for '{prompt_label}' to check ids against)")


# ======================================================================
# Per-race classification from its trials.
# ======================================================================
def classify_races(df, stable_threshold=0.75):
    """Label each race from its ok trials:
        stable_face   - one face chosen >= threshold of the time (real preference)
        position_bias - one slot chosen >= threshold (tracks position, not face)
        noise         - neither dominant (genuine inconsistency)
        insufficient  - fewer than 2 ok trials
    Also returns p_a (fraction of ok trials that chose face 'a'), which the
    unfiltered accuracy uses so noisy races contribute a graded value."""
    out = []
    for eid, race in df.groupby('election_id'):
        ok = race[race['status'] == 'ok']
        n_ok = len(ok)
        if n_ok < 2:
            out.append({'election_id': eid, 'classification': 'insufficient',
                        'chosen_face': None, 'p_a': None, 'n_ok': n_ok})
            continue
        p_a = (ok['chosen_face'] == 'a').mean()      # graded preference for face a
        p_slot_a = (ok['chosen_slot'] == 'A').mean()  # graded preference for slot A

        if p_a >= stable_threshold:
            label, face = 'stable_face', 'a'
        elif p_a <= 1 - stable_threshold:
            label, face = 'stable_face', 'b'
        elif p_slot_a >= stable_threshold or p_slot_a <= 1 - stable_threshold:
            label, face = 'position_bias', None
        else:
            label, face = 'noise', None

        out.append({'election_id': eid, 'classification': label,
                    'chosen_face': face, 'p_a': round(p_a, 3), 'n_ok': n_ok})
    return pd.DataFrame(out)


# ======================================================================
# Reliability profile: how much of each model's behaviour is a stable
# preference vs. position bias vs. noise. This is a result in itself.
# ======================================================================
def classification_breakdown(prompt_label, stable_threshold=0.75, save=True):
    rows = []
    for model, folder in ELECTION_MODELS.items():
        path = get_filepath(model_folder=folder, prompt_label=prompt_label)
        if not path.exists():
            continue
        cls = classify_races(pd.read_csv(path), stable_threshold)
        counts = cls['classification'].value_counts()
        total = len(cls)
        rows.append({
            'model': model,
            'total_races': total,
            'stable_face': int(counts.get('stable_face', 0)),
            'position_bias': int(counts.get('position_bias', 0)),
            'noise': int(counts.get('noise', 0)),
            'insufficient': int(counts.get('insufficient', 0)),
            'pct_stable': round(counts.get('stable_face', 0) / total, 3) if total else None,
        })
    table = pd.DataFrame(rows)
    if save:
        _save(table, f'multitrial_classification_{prompt_label}.csv')
    return table


# ======================================================================
# Accuracy, reported BOTH ways:
#   filtered   - stable-preference races only, scored 0/1 (content of a
#                committed preference); comes with a Wilson CI and a sign test
#                against chance.
#   unfiltered - every race with >=2 ok trials contributes its probability of
#                choosing the winning face (position-bias races land near 0.5,
#                noisy races contribute a graded value). Nothing is discarded.
# The gap between the two reflects how much a model's inconsistency costs it.
# ======================================================================
def accuracy_both_ways(prompt_label, stable_threshold=0.75, save=True):
    truth = _race_truth()
    rows = []
    for model, folder in ELECTION_MODELS.items():
        path = get_filepath(model_folder=folder, prompt_label=prompt_label)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        cls = classify_races(df, stable_threshold)

        filt_correct = filt_n = 0
        unfilt_scores = []
        for _, r in cls.iterrows():
            t = truth.get(r['election_id'])
            if t is None or r['classification'] == 'insufficient':
                continue
            winner = t['winner']

            # unfiltered: P(choose winning face) across this race's ok trials
            ok = df[(df['election_id'] == r['election_id']) & (df['status'] == 'ok')]
            unfilt_scores.append((ok['chosen_face'] == winner).mean())

            # filtered: stable races only, 0/1 on whether the preferred face won
            if r['classification'] == 'stable_face':
                filt_n += 1
                filt_correct += int(r['chosen_face'] == winner)

        filt_acc = filt_correct / filt_n if filt_n else None
        if filt_n:
            lo, hi = wilson_ci(filt_correct, filt_n, alpha=0.05)
            filt_p = binomtest(filt_correct, filt_n, 0.5).pvalue
        else:
            lo = hi = filt_p = None
        unfilt_acc = float(np.mean(unfilt_scores)) if unfilt_scores else None

        rows.append({
            'model': model,
            'filtered_acc': round(filt_acc, 3) if filt_acc is not None else None,
            'filtered_ci_low': round(lo, 3) if lo is not None else None,
            'filtered_ci_high': round(hi, 3) if hi is not None else None,
            'filtered_n': filt_n,
            'filtered_p': f"{filt_p:.1e}" if filt_p is not None else None,
            'unfiltered_acc': round(unfilt_acc, 3) if unfilt_acc is not None else None,
            'unfiltered_n': len(unfilt_scores),
        })
    table = pd.DataFrame(rows)
    if save:
        _save(table, f'multitrial_accuracy_both_{prompt_label}.csv')
    return table


# ======================================================================
# Impression vs. decision, the paper's headline. Compares a competence
# label against a decision label (vote / advise / predict) on the SAME
# races, so the contrast is paired. Reports both filtered and unfiltered.
# ======================================================================
def compare_prompts(comp_label, decision_label, stable_threshold=0.75, save=True):
    truth = _race_truth()

    def per_model_unfiltered(folder, label):
        path = get_filepath(model_folder=folder, prompt_label=label)
        if not path.exists():
            return None
        df = pd.read_csv(path)
        scores = {}
        for eid, race in df.groupby('election_id'):
            t = truth.get(eid)
            ok = race[race['status'] == 'ok']
            if t is None or len(ok) < 2:
                continue
            scores[eid] = (ok['chosen_face'] == t['winner']).mean()
        return scores

    rows = []
    for model, folder in ELECTION_MODELS.items():
        comp = per_model_unfiltered(folder, comp_label)
        dec = per_model_unfiltered(folder, decision_label)
        if not comp or not dec:
            continue
        shared = set(comp) & set(dec)           # SAME races, for a paired contrast
        if not shared:
            continue
        comp_acc = np.mean([comp[e] for e in shared])
        dec_acc = np.mean([dec[e] for e in shared])
        rows.append({
            'model': model,
            'comp_acc': round(comp_acc, 3),
            'decision_acc': round(dec_acc, 3),
            'comp_minus_decision': round(comp_acc - dec_acc, 3),
            'n_shared_races': len(shared),
        })
    table = pd.DataFrame(rows)

    if len(table):
        n = len(table)
        higher = int((table['comp_minus_decision'] > 0).sum())
        p = binomtest(higher, n, 0.5, alternative='greater').pvalue
        print(f"competence > decision in {higher}/{n} models "
              f"(sign test p = {p:.4f})")
        print(f"mean(competence - decision) = "
              f"{table['comp_minus_decision'].mean():+.3f}")
    if save:
        _save(table, f'multitrial_compare_{comp_label}_vs_{decision_label}.csv')
    return table


# ======================================================================
# Mechanism: on races where the competence pick and the decision pick
# disagree, are those the closer races, and which pick is more often right?
# Uses each race's majority face per prompt (stable or not) so the two
# picks are always defined.
# ======================================================================
def disagreement_analysis(label_a, label_b, save=True, verbose=True):
    """On races where the two prompts' majority picks DISAGREE: which prompt is
    more accurate, and are those the closer races? Output columns and printout
    use the labels passed, so swapping arguments cannot mislabel the result."""
    truth = _race_truth()
 
    def majority_face(df):
        """Per race: the face chosen on a majority of ok trials, or None if tied."""
        out = {}
        for eid, race in df.groupby('election_id'):
            ok = race[race['status'] == 'ok']
            if len(ok) < 2:
                continue
            p_a = (ok['chosen_face'] == 'a').mean()
            if p_a > 0.5:
                out[eid] = 'a'
            elif p_a < 0.5:
                out[eid] = 'b'
            # exactly 0.5 -> tied, leave undefined
        return out
 
    disagree_margins, agree_margins = [], []
    a_right = b_right = n_disagree = 0
    n_models_used = 0
    n_files_missing = 0
 
    for model, folder in ELECTION_MODELS.items():
        ap = get_filepath(model_folder=folder, prompt_label=label_a)
        bp = get_filepath(model_folder=folder, prompt_label=label_b)
        if not (ap.exists() and bp.exists()):
            n_files_missing += 1
            if verbose:
                miss = [l for l, p in [(label_a, ap), (label_b, bp)] if not p.exists()]
                print(f"  {model}: missing {', '.join(miss)}")
            continue
        pick_a = majority_face(pd.read_csv(ap))
        pick_b = majority_face(pd.read_csv(bp))
        shared = set(pick_a) & set(pick_b)
        if verbose:
            print(f"  {model}: {label_a}={len(pick_a)}, {label_b}={len(pick_b)}, "
                  f"shared={len(shared)}")
        n_models_used += 1
        for eid in shared:
            t = truth.get(eid)
            if t is None:
                continue
            if pick_a[eid] == pick_b[eid]:
                agree_margins.append(t['margin'])
            else:
                disagree_margins.append(t['margin'])
                n_disagree += 1
                a_right += int(pick_a[eid] == t['winner'])
                b_right += int(pick_b[eid] == t['winner'])
 
    if verbose:
        print(f"\nmodels used: {n_models_used}, files missing: {n_files_missing}, "
              f"disagreements: {n_disagree}, agreements: {len(agree_margins)}")
 
    summary = {
        'n_disagreements': n_disagree,
        f'{label_a}_correct_on_disagree': round(a_right / n_disagree, 3) if n_disagree else None,
        f'{label_b}_correct_on_disagree': round(b_right / n_disagree, 3) if n_disagree else None,
        'disagree_mean_margin': round(np.mean(disagree_margins), 3) if disagree_margins else None,
        'agree_mean_margin': round(np.mean(agree_margins), 3) if agree_margins else None,
    }
    if n_disagree:
        summary[f'{label_a}_vs_{label_b}_p'] = \
            f"{binomtest(a_right, n_disagree, 0.5).pvalue:.4f}"
    if disagree_margins and agree_margins:
        summary['margin_mannwhitney_p'] = \
            f"{mannwhitneyu(disagree_margins, agree_margins).pvalue:.4f}"
 
    table = pd.DataFrame([summary])
    if save:
        _save(table, f'multitrial_disagreement_{label_a}_vs_{label_b}.csv')
    return table


# ======================================================================
# Refusal / non-answer rates per prompt, per model. A "non-answer" is any
# trial whose status is not 'ok' and not 'ERROR' (ERROR = transport failure,
# not a model refusal). Used to test whether a prompt is refused more.
# ======================================================================
def refusal_rates(prompt_labels, save=True):
    rows = []
    for model, folder in ELECTION_MODELS.items():
        row = {'model': model}
        for label in prompt_labels:
            path = get_filepath(model_folder=folder, prompt_label=label)
            if not path.exists():
                row[label] = None
                continue
            df = pd.read_csv(path)
            answerable = df[df['status'] != 'ERROR']   # exclude transport errors
            if len(answerable) == 0:
                row[label] = None
                continue
            refused = (answerable['status'] != 'ok').mean()
            row[label] = round(refused, 3)
        rows.append(row)
    table = pd.DataFrame(rows)
    if save:
        _save(table, 'multitrial_refusal_rates.csv')
    return table

# ======================================================================
# RECOGNITION / CONTAMINATION CHECK
#
# Ground truth contains only surnames, so we classify conservatively: any
# response containing the correct surname counts as "potentially recognized"
# and is EXCLUDED from the clean subset, even though some surname matches may
# be coincidental. This over-counts recognition, which is the safe direction:
# it yields a stricter, genuinely-unrecognized subset. The election findings
# are then recomputed on that subset. Common surnames (which are the ones most
# likely to match coincidentally) are flagged so they can be audited.
# ======================================================================
 
# surnames common enough that a bare last-name match may be coincidental;
# matches on these are flagged for optional manual review.
_COMMON_SURNAMES = {
    'smith', 'johnson', 'williams', 'brown', 'jones', 'davis', 'miller',
    'wilson', 'moore', 'taylor', 'anderson', 'thomas', 'white', 'harris',
    'martin', 'thompson', 'young', 'walker', 'hall', 'allen', 'king',
    'wright', 'hill', 'green', 'baker', 'nelson', 'clark', 'bush', 'gore',
    'kennedy', 'carter', 'reed', 'cook', 'bell', 'ward', 'cox', 'long',
}
 
 
def _candidate_surnames():
    """Map candidate Full Label -> real surname (lowercased).
    Assumes clean_election_df exposes a surname/last-name column; adjust the
    column name below to match the data."""
    _, elec = clean_election_df(ELECTION_RESULTS_PATH)
    surnames = {}
    for _, row in elec.iterrows():
        lbl = row['Full Label']
        # ADJUST: point this at whatever column holds the candidate's surname.
        name = row.get('Last Name') or row.get('Candidate') or ''
        if isinstance(name, str) and name.strip():
            # take the last whitespace-separated token as the surname
            surnames[lbl] = name.strip().split()[-1].lower()
    return surnames
 
 
def classify_recognition(prompt_label='recognition_v2', save=True):
    """Per candidate face, classify the model's recognition attempt as
    'declined', 'wrong_guess', or 'correct' (conservative surname match).
    A 'correct' or common-surname match marks the face as potentially
    recognized -> excluded from the clean subset downstream."""
    surnames = _candidate_surnames()
    rows = []
    for model, folder in ELECTION_MODELS.items():
        path = get_filepath(model_folder=folder, prompt_label=prompt_label)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            raw = r.get('raw_response')
            lbl = r.get('candidate_label')
            surname = surnames.get(lbl)
            cls, common = _match_one(raw, surname)
            rows.append({
                'model': model, 'candidate_label': lbl,
                'election_id': r.get('election_id'),
                'recognition': cls,               # declined / wrong_guess / correct
                'common_surname': common,         # True if match is on a common surname
                'recognized': cls == 'correct',   # conservative: correct-surname => recognized
            })
    table = pd.DataFrame(rows)
    if save:
        _save(table, 'recognition_classified.csv')
    # quick summary
    if len(table):
        rate = table.groupby('model')['recognized'].mean().round(3)
        print("recognition rate (conservative) by model:")
        print(rate.to_string())
        flagged = int(table[(table['recognition'] == 'correct') & table['common_surname']].shape[0])
        print(f"\n{flagged} 'correct' matches are on common surnames — audit if desired "
              f"(raw responses saved).")
    return table
 
 
def _match_one(raw, surname):
    """Return (classification, is_common_surname).
    classification in {'declined','wrong_guess','correct', None}."""
    if not isinstance(raw, str) or raw.startswith('ERROR'):
        return None, False
    resp = raw.strip().lower()
    # declined: response is essentially 'unknown'
    if resp == 'unknown' or resp.startswith('unknown'):
        return 'declined', False
    if not isinstance(surname, str) or not surname:
        return 'wrong_guess', False
    # correct: the real surname appears as a whole word
    if re.search(rf'\b{re.escape(surname)}\b', resp):
        return 'correct', surname in _COMMON_SURNAMES
    return 'wrong_guess', False
 
 
def dissociation_on_unrecognized(label_a, label_b,
                                 recognition_label='recognition_v2', save=True):
    """Compare two prompts (label_a vs label_b) on the subset of races where
    NEITHER candidate was recognized by the model, ruling out that the effect
    is driven by the model recalling known outcomes. The printout and output
    columns use the labels you pass, so swapping arguments never mislabels the
    result. Uses per-race P(pick winner) across ok trials (inclusive scoring)."""
    truth = _race_truth()
    recog = classify_recognition(recognition_label, save=False)
 
    # per model, races where at least one candidate was (conservatively) recognized
    recognized_races = {}
    for model, g in recog.groupby('model'):
        recognized_races[model] = set(g[g['recognized']]['election_id'])
 
    def per_model_scores(folder, label, exclude):
        path = get_filepath(model_folder=folder, prompt_label=label)
        if not path.exists():
            return None
        df = pd.read_csv(path)
        scores = {}
        for eid, race in df.groupby('election_id'):
            if eid in exclude:
                continue                       # drop races with a recognized face
            t = truth.get(eid)
            ok = race[race['status'] == 'ok']
            if t is None or len(ok) < 2:
                continue
            scores[eid] = (ok['chosen_face'] == t['winner']).mean()
        return scores
 
    rows = []
    for model, folder in ELECTION_MODELS.items():
        exclude = recognized_races.get(model, set())
        a = per_model_scores(folder, label_a, exclude)
        b = per_model_scores(folder, label_b, exclude)
        if not a or not b:
            continue
        shared = set(a) & set(b)
        if not shared:
            continue
        a_acc = np.mean([a[e] for e in shared])
        b_acc = np.mean([b[e] for e in shared])
        rows.append({
            'model': model,
            f'{label_a}_acc': round(a_acc, 3),
            f'{label_b}_acc': round(b_acc, 3),
            f'{label_a}_minus_{label_b}': round(a_acc - b_acc, 3),
            'n_unrecognized_races': len(shared),
            'n_excluded_recognized': len(exclude),
        })
    table = pd.DataFrame(rows)
    if len(table):
        n = len(table)
        diff_col = f'{label_a}_minus_{label_b}'
        higher = int((table[diff_col] > 0).sum())
        p = binomtest(higher, n, 0.5, alternative='greater').pvalue
        print(f"\nOn UNRECOGNIZED faces: {label_a} > {label_b} in {higher}/{n} "
              f"models (sign test p = {p:.4f})")
        print(f"mean({label_a} - {label_b}) = {table[diff_col].mean():+.3f}")
    if save:
        _save(table, f'dissociation_unrecognized_{label_a}_vs_{label_b}.csv')
    return table
 
 
# ======================================================================
# Recognition RATE (paper-ready). Reports only the contamination-relevant
# number: the fraction of faces where the model named the ACTUAL candidate
# (surname match). The contamination argument rests on this being low —
# the models rarely identify the real candidate, so they cannot be recalling
# outcomes. Guards against a silent surname-lookup failure that would make
# every rate a misleading 0%.
# ======================================================================
def recognition_rates(prompt_label='recognition_v2', save=True):
    # guard: if surnames can't be resolved, correct_rate is meaningless (all 0)
    surnames = _candidate_surnames()
    if not surnames:
        print("WARNING: _candidate_surnames() returned nothing — correct_rate would be a\n"
              "misleading 0% for every model. Fix the surname column in _candidate_surnames()\n"
              "before trusting these numbers.")
        return None
 
    cls = classify_recognition(prompt_label, save=False)
    if cls is None or len(cls) == 0:
        print("no recognition data found for label", prompt_label)
        return None
 
    rows = []
    for model, g in cls.groupby('model'):
        n = len(g)
        correct = int((g['recognition'] == 'correct').sum())
        # per-RACE exposure: fraction of races with >=1 recognized candidate.
        # This is the contamination-relevant number — a race can be affected if
        # EITHER candidate is recognized — and is higher than the per-face rate.
        recog_by_race = g[g['recognition'] == 'correct']['election_id'].dropna().unique()
        all_races = g['election_id'].dropna().unique()
        n_races = len(all_races)
        races_with_recog = len(set(recog_by_race) & set(all_races))
        rows.append({
            'model': model,
            'n_faces': n,
            'correct': correct,
            'per_face_rate': round(correct / n, 3) if n else None,
            'n_races': n_races,
            'races_with_recognized': races_with_recog,
            'per_race_exposure': round(races_with_recog / n_races, 3) if n_races else None,
        })
    table = pd.DataFrame(rows).sort_values('per_race_exposure', ascending=False)
    print(table.to_string(index=False))
    print(f"\nmean per-face recognition rate: {table['per_face_rate'].mean():.1%}")
    print(f"mean per-race exposure (>=1 recognized candidate): "
          f"{table['per_race_exposure'].mean():.1%}")
    if save:
        _save(table, 'recognition_rates.csv')
    return table

 
# ======================================================================
# VALIDITY CHECKS
#   1. Does the competence->winner effect persist on OPEN (non-incumbent)
#      races? If it only worked where an incumbent ran, it might be tracking
#      incumbency cues rather than a general competence signal.
#   2. How often does the competence pick coincide with the OLDER candidate?
#      If competence just tracks apparent age, that would undercut the claim.
# Both need per-candidate metadata (incumbency, age/birth year). The functions
# check for the needed columns and report what's missing rather than crashing.
# ======================================================================
def _election_meta():
    """Return the cleaned election dataframe. Column names confirmed from the
    dataset: incumbency = 'Incumbent?' (value 'yes'), age = 'Age'."""
    _, elec = clean_election_df(ELECTION_RESULTS_PATH)
    return elec
 
 
def _race_faces():
    """{election_id: {'a': row0, 'b': row1}} in iloc order — the SAME grouping
    and ordering as _race_truth, so validity checks that read candidate metadata
    (incumbency, age) line up with how 'a'/'b' is interpreted everywhere else."""
    elec = _election_meta()
    faces = {}
    for eid, g in elec.groupby('Election ID'):
        if len(g) != 2:
            continue
        faces[eid] = {'a': g.iloc[0], 'b': g.iloc[1]}
    return faces
 
 
def _majority_pick(race_df):
    """Face ('a'/'b') chosen on a majority of ok trials, or None if tied/empty."""
    ok = race_df[race_df['status'] == 'ok']
    if len(ok) < 2:
        return None
    p_a = (ok['chosen_face'] == 'a').mean()
    if p_a > 0.5:
        return 'a'
    if p_a < 0.5:
        return 'b'
    return None
 
 
def open_race_ids():
    """Election IDs for OPEN races (no incumbent among the two candidates)."""
    faces = _race_faces()
    if not faces:
        return None
    if 'Incumbent?' not in next(iter(faces.values()))['a'].index:
        print("  no 'Incumbent?' column found — cannot identify open races")
        return None
    open_ids = set()
    for eid, f in faces.items():
        a_inc = str(f['a']['Incumbent?']).lower().strip() == 'yes'
        b_inc = str(f['b']['Incumbent?']).lower().strip() == 'yes'
        if not (a_inc or b_inc):
            open_ids.add(eid)
    print(f"  {len(open_ids)} open (non-incumbent) races of {len(faces)} total")
    return open_ids
 
 
def accuracy_on_open_races(prompt_label='competence_multitrial'):
    """Recompute inclusive competence accuracy on OPEN races only, per model.
    If the effect persists here, it is not merely tracking incumbency."""
    truth = _race_truth()
    open_ids = open_race_ids()
    if open_ids is None:
        return None
 
    rows = []
    for model, folder in ELECTION_MODELS.items():
        path = get_filepath(model_folder=folder, prompt_label=prompt_label)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        scores = []
        for eid, race in df.groupby('election_id'):
            if eid not in open_ids:
                continue
            t = truth.get(eid)
            ok = race[race['status'] == 'ok']
            if t is None or len(ok) < 2:
                continue
            scores.append((ok['chosen_face'] == t['winner']).mean())
        if scores:
            rows.append({
                'model': model,
                'open_race_acc': round(np.mean(scores), 3),
                'n_open_races': len(scores),
            })
    table = pd.DataFrame(rows)
    if len(table):
        print(f"\nmean competence accuracy on open races: {table['open_race_acc'].mean():.3f}")
    _save(table, 'validity_open_races.csv')
    return table
 
 
def age_confound_check(prompt_label='competence_multitrial'):
    """How often does the competence pick coincide with the OLDER candidate?
    ~50% means competence is not merely tracking apparent age. Uses the shared
    _race_faces mapping (face 'a' = race.iloc[0]) so the age comparison lines up
    exactly with how chosen_face is interpreted. 'Age' larger = older."""
    faces = _race_faces()
    if not faces:
        return None
    if 'Age' not in next(iter(faces.values()))['a'].index:
        print("  no 'Age' column found — cannot run the age-confound check")
        return None
 
    older_face = {}
    for eid, f in faces.items():
        a_age, b_age = f['a']['Age'], f['b']['Age']
        if pd.isna(a_age) or pd.isna(b_age) or a_age == b_age:
            continue
        older_face[eid] = 'a' if a_age > b_age else 'b'
 
    picks_older = total = 0
    for model, folder in ELECTION_MODELS.items():
        path = get_filepath(model_folder=folder, prompt_label=prompt_label)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for eid, race in df.groupby('election_id'):
            if eid not in older_face:
                continue
            pick = _majority_pick(race)
            if pick is None:
                continue
            picks_older += int(pick == older_face[eid])
            total += 1
 
    if total:
        pct = picks_older / total
        print(f"\ncompetence pick selects the OLDER candidate in {pct:.1%} of races "
              f"({picks_older}/{total})")
        print("(~50% indicates the competence signal is not merely apparent age)")
        _save(pd.DataFrame([{'competence_picks_older_pct': round(pct, 3), 'n': total}]),
              'validity_age_confound.csv')
        return pct
    print("  no races with usable age data")
    return None