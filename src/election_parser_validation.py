"""
Parser validation — hand-label a sample of raw responses and measure how well
extract_choice() agrees with human labels. Produces the number for the appendix:
"the parser agreed with manual labels on N/M responses."

Outputs are written to an explicit, stable directory (PARSER_VAL_DIR) with atomic
writes, so the hand-labeled data (expensive to produce) is never lost or dropped
in an ambiguous working directory.

Workflow:
    sample = build_validation_sample(n=100)      # pull random raw responses
    save_sample_for_labeling(sample)             # writes a blank-label CSV to fill
    # ... hand-label the 'human' column (a/b/r/u) in the CSV, OR use label_interactive()
    report = score_validation()                  # compares parser vs human, saves report
"""

import random
from pathlib import Path

import pandas as pd

from config import ELECTION_MODELS, PARSER_VAL_DIR
from election_collect import get_filepath
from election_collect_multitrial import extract_choice


def _atomic_write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def build_validation_sample(n=100, seed=42, labels=None, stratify=False,
                            exclude_already_labeled=True, sample_file="validation_sample.csv"):
    """Pull a random sample of raw responses across models/prompts.

    stratify=True: oversample responses that required the parser's HEURISTIC
        rules (marker / starts_with / ends_with / sole_token) rather than the
        trivial 'clean' single-letter case, so the validation actually tests the
        rules that could fail — this fixes the 'undiverse batch' problem.
    exclude_already_labeled=True: skip responses already in the labeled sample
        file, so calling this again pulls NEW responses to ADD to your set.
    """
    labels = labels or ['competence_multitrial', 'electability_multitrial',
                        'prediction_multitrial']

    # responses already labeled (so we can pull new ones to append)
    already = set()
    if exclude_already_labeled:
        try:
            prev = pd.read_csv(PARSER_VAL_DIR/ sample_file)
            already = set(prev['raw'].astype(str))
        except Exception:
            pass

    pool = []
    for model, folder in ELECTION_MODELS.items():
        for label in labels:
            path = get_filepath(model_folder=folder, prompt_label=label)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                raw = r.get('raw')
                if not (isinstance(raw, str) and str(r.get('status')) != 'ERROR'):
                    continue
                if raw[:300] in already or raw in already:
                    continue
                choice, reason = extract_choice(raw)
                pool.append({'model': model, 'label': label, 'raw': raw,
                             'parser_reason': reason})

    if not pool:
        print("no new responses to sample (all already labeled?)")
        return pd.DataFrame()

    random.seed(seed)
    if stratify:
        # split into trivially-clean vs. heuristic-handled, sample more heuristics
        clean = [p for p in pool if p['parser_reason'] in ('clean', 'REFUSE', 'EMPTY')]
        heuristic = [p for p in pool if p['parser_reason'] not in
                     ('clean', 'REFUSE', 'EMPTY', 'UNPARSEABLE')]
        unparse = [p for p in pool if p['parser_reason'] == 'UNPARSEABLE']
        # aim for ~half heuristic cases, the rest clean/unparseable
        n_heur = min(len(heuristic), n // 2)
        n_rest = n - n_heur
        picks = (random.sample(heuristic, n_heur)
                 + random.sample(clean + unparse, min(n_rest, len(clean) + len(unparse))))
        random.shuffle(picks)
        sample = picks
        print(f"stratified: {n_heur} heuristic-rule cases + {len(sample)-n_heur} clean/unparseable")
    else:
        sample = random.sample(pool, min(n, len(pool)))
    return pd.DataFrame(sample)[['model', 'label', 'raw']]


def save_sample_for_labeling(sample_df, filename="validation_sample.csv", append=True):
    """Write the sample with a blank 'human' column to hand-fill (a/b/r/u).
    append=True: ADD these rows to any existing labeled file (so you can grow the
    validation set over time); already-labeled rows are preserved."""
    rows = []
    for _, r in sample_df.iterrows():
        choice, reason = extract_choice(r['raw'])
        parser_label = choice.lower() if choice in ('A', 'B') else (
            'r' if reason == 'REFUSE' else 'u')
        rows.append({
            'model': r['model'], 'label': r['label'],
            'raw': r['raw'][:300],
            'human': '',                    # <-- you fill this: a / b / r / u
            'parser': parser_label,
            'parser_reason': reason,
        })
    new_df = pd.DataFrame(rows)
    out = PARSER_VAL_DIR / filename

    if append:
        try:
            existing = pd.read_csv(out)
            # keep existing (labeled) rows, add only genuinely new raws
            existing_raws = set(existing['raw'].astype(str))
            new_df = new_df[~new_df['raw'].astype(str).isin(existing_raws)]
            combined = pd.concat([existing, new_df], ignore_index=True)
            _atomic_write(combined, out)
            print(f"appended {len(new_df)} new rows ({len(combined)} total) to {out}")
            print(f"{combined['human'].astype(str).str.strip().isin(['a','b','r','u']).sum()} already labeled, "
                  f"{len(combined) - combined['human'].astype(str).str.strip().isin(['a','b','r','u']).sum()} to go")
            return out
        except FileNotFoundError:
            pass

    _atomic_write(new_df, out)
    print(f"wrote {len(new_df)} rows to {out}")
    print("Fill the 'human' column with a / b / r (refuse) / u (unparseable), then run score_validation().")
    return out


def label_interactive(filename="validation_sample.csv"):
    """Optional: label in-notebook instead of editing the CSV. Saves after each
    entry so an interruption never loses your progress."""
    path = PARSER_VAL_DIR / filename
    df = pd.read_csv(path)
    for i, row in df.iterrows():
        if str(row.get('human', '')).strip() in ('a', 'b', 'r', 'u'):
            continue   # already labeled — resume where you left off
        print(f"\n[{i+1}/{len(df)}] {row['raw'][:200]}")
        ans = input("  label (a/b/r/u, or 's' to stop): ").strip().lower()
        if ans == 's':
            break
        df.at[i, 'human'] = ans
        _atomic_write(df, path)   # save after every label — no lost work
    print(f"saved to {path}")
    return path


def score_validation(filename="validation_sample.csv"):
    """Compare parser vs human labels; save a report and print the headline number."""
    path = PARSER_VAL_DIR  / filename
    df = pd.read_csv(path)
    labeled = df[df['human'].astype(str).str.strip().isin(['a', 'b', 'r', 'u'])].copy()
    if len(labeled) == 0:
        print("No rows labeled yet — fill the 'human' column first.")
        return None

    labeled['human'] = labeled['human'].str.strip().str.lower()
    labeled['agree'] = labeled['human'] == labeled['parser']
    n = len(labeled)
    agree = int(labeled['agree'].sum())
    print(f"Parser–human agreement: {agree}/{n} = {agree/n:.1%}")

    disagreements = labeled[~labeled['agree']]
    if len(disagreements):
        print(f"\n{len(disagreements)} disagreements:")
        for _, r in disagreements.iterrows():
            print(f"  human={r['human']} parser={r['parser']} ({r['parser_reason']}): {r['raw'][:70]!r}")

    _atomic_write(labeled, PARSER_VAL_DIR / "validation_scored.csv")
    print(f"\nsaved scored results to {PARSER_VAL_DIR / 'validation_scored.csv'}")
    return agree / n