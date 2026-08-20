"""
Multi-trial election forced-choice collection.

Stores trial-level data (one row per race x ordering x trial) so that:
  - position bias vs. noise can be separated at analysis time
  - more trials can be added later without recollecting (call with higher target_trials)

Usage from a notebook:
    from multitrial_collect import backup_existing, run_pilot, run_full

    backup_existing()                       # always back up first
    run_pilot(prompt_key="electability")    # watch this, confirm clean
    # then, once confirmed:
    run_full(prompt_key="electability")
    run_full(prompt_key="advisor_direct") etc.

prompt_key must be a key in prompts.PROMPTS. Output is written to a file
labelled "<prompt_key>_multitrial" (pilot: "<prompt_key>_multitrial_TEST"),
so it never overwrites single-trial data.
"""

import re
import time
import random
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config import (ELECTION_MODELS, SENATE_PATH, GOVERNOR_PATH,
                    ELECTION_RESULTS_PATH, MODEL_SNAPSHOTS, REASONING_EFFORT)
from collect import encode_image, get_client
from election_collect import clean_label, clean_election_df, get_filepath

# Uses the PROMPTS dict already defined in prompts.py:
#   PROMPTS = {"electability": ELECTION_PROMPT, "advisor_direct": ADVISOR_PROMPT_DIRECT, ...}
# The prompt_key selects both the prompt AND the output label, so they can never be mismatched. 
# Output files are labelled "<key>_multitrial".
from prompts import ELECTION_INSTRUCTION, PROMPTS


def _resolve(prompt_key):
    """Return (question_prompt, output_label) for a prompt_key in PROMPTS."""
    if prompt_key not in PROMPTS:
        raise KeyError(f"'{prompt_key}' not in prompts.PROMPTS. "
                       f"Available: {list(PROMPTS)}")
    return PROMPTS[prompt_key], f"{prompt_key}_multitrial"


# ------------------------------------------------------------------
# Retry wrapper: retries transient errors (429 / rate limit / 5xx)
# with exponential backoff, so a rate limit doesn't become lost data.
# ------------------------------------------------------------------
def call_with_retry(fn, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = any(k in msg for k in
                            ['429', 'rate', 'timeout', 'overloaded', '503', '502'])
            if not transient or attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))


# ------------------------------------------------------------------
# Robust A/B extraction. Handles markdown ("**B**"), quotes, and
# answer markers ("Answer: B") without false-positiving on prose
# refusals ("I can't judge a face"). Returns ('A'|'B'|None, reason).
# ------------------------------------------------------------------
def extract_choice(raw):
    if not raw or not raw.strip():
        return None, "EMPTY"
    text = raw.strip()
    low = text.lower()
 
    # 0. REFUSAL FIRST. Prose refusals often begin with 'A' ("As an AI...")
    #    or contain a stray A/B, so this must run BEFORE any letter extraction,
    #    or those refusals get misread as a slot-A vote. This ordering is the
    #    fix for the false-positive contamination; do not move it below.
    refusal_terms = [
        "as an ai", "as an artificial", "as a language model", "as a large language",
        "i cannot", "i can't", "i can not", "i won't", "i will not",
        "i'm not able", "i am not able", "i'm unable", "i am unable", "unable to",
        "i do not", "i don't", "not appropriate", "inappropriate",
        "cannot judge", "can't judge", "cannot determine", "can't determine",
        "cannot make", "not able to", "decline", "unknown", "neither",
        "both equally", "no basis", "cannot provide", "can't provide",
        "not comfortable", "should not", "cannot assist", "won't be able",
    ]
    if any(t in low for t in refusal_terms):
        return None, "REFUSE"
 
    # 1. whole response is essentially just the letter: "A", "**B**", '"A"', "B.", "A)"
    stripped = re.sub(r'[^A-Za-z]', '', text).upper()
    if stripped in ('A', 'B'):
        return stripped, "clean"
 
    # 2. explicit answer markers: "Answer: B", "I choose A", "Candidate B", "vote for A"
    marker = re.search(
        r'(?:answer|choose|choice|select|pick|vote for|candidate|option|prefer)'
        r'\s*(?:is|:)?\s*["\*\(]*\b([AB])\b', text, re.IGNORECASE)
    if marker:
        return marker.group(1).upper(), "marker"
 
    # 3. starts with the letter clearly set off as an answer, THEN explains.
    #    e.g. "**A**. This candidate...", "A) looks more...", "A -- because...".
    #    Not length-limited: a response that opens with a set-off A/B is answering,
    #    even if it justifies at length afterward. Requires the letter to be
    #    followed by a boundary marker (punctuation/space) so "Alabama" can't match.
    start = re.match(r'^["\*\(\s]*([AB])["\*\)]*\s*[\.\)\:\-–—,\s]', text)
    if start:
        return start.group(1).upper(), "starts_with"
 
    # 4. short response ending with a standalone letter: "...therefore B"
    if len(text) <= 40:
        end = re.search(r'\b([AB])["\*\.\)]*\s*$', text)
        if end:
            return end.group(1).upper(), "ends_with"
 
    # 5. short response with exactly one standalone A or B token, nothing ambiguous
    if len(text) <= 40:
        a_count = len(re.findall(r'\bA\b', text))
        b_count = len(re.findall(r'\bB\b', text))
        if a_count == 1 and b_count == 0:
            return 'A', "sole_token"
        if b_count == 1 and a_count == 0:
            return 'B', "sole_token"
 
    # 6. can't confidently extract -> NOT a fake answer
    return None, "UNPARSEABLE"


# ------------------------------------------------------------------
# One trial = the two counterbalanced orderings for a single race.
# Records, per call: which face was in slot A, and which slot the
# model picked -> lets us classify face-preference vs slot-bias later.
# ------------------------------------------------------------------
def run_single_trial(image_a, image_b, model_folder, question, race_id, trial_num):
    snapshot = MODEL_SNAPSHOTS[model_folder]
    effort = REASONING_EFFORT.get(model_folder)
    extra = {"reasoning": {"effort": effort}} if effort else {}
    rows = []
    for slotA_face, slotB_face, faceA_id, faceB_id in [
        (image_a, image_b, 'a', 'b'), #two different orderings
        (image_b, image_a, 'b', 'a'),
    ]:
        content = [
            {"type": "text", "text": f"{question}\n\nCandidate A:"},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{encode_image(slotA_face)}"}},
            {"type": "text", "text": "Candidate B:"},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{encode_image(slotB_face)}"}},
            {"type": "text", "text": ELECTION_INSTRUCTION},
        ]
        try:
            resp = call_with_retry(lambda: get_client().chat.completions.create(
                model=snapshot, messages=[{"role": "user", "content": content}],
                temperature=0.1, max_tokens=2000, extra_body=extra,
                timeout=60))   # timeout so a hung call becomes a retryable error, not a freeze
            raw = resp.choices[0].message.content.strip()
            letter, reason = extract_choice(raw)   # robust parser (handles **B**, "Answer: B", etc.)
            # map the chosen SLOT (A/B) back to which FACE it was
            if letter == 'A':
                chosen_face = faceA_id           # slot A held faceA_id this ordering
            elif letter == 'B':
                chosen_face = faceB_id           # slot B held faceB_id
            else:
                chosen_face = None
            rows.append({'election_id': race_id, 'trial': trial_num,
                         'slotA_face': faceA_id, 'chosen_slot': letter,
                         'chosen_face': chosen_face,
                         'status': 'ok' if letter else reason,   # REFUSE/UNPARSEABLE/EMPTY
                         'raw': raw[:300]})   # keep more raw so re-parsing is possible
        except Exception as e:
            rows.append({'election_id': race_id, 'trial': trial_num,
                         'slotA_face': faceA_id, 'chosen_slot': None,
                         'chosen_face': None, 'status': 'ERROR', 'raw': str(e)[:80]})
    return rows


# ------------------------------------------------------------------
# Collect one model up to target_trials. Additive + resume-safe:
# counts trials already present per race, only collects the missing.
# ------------------------------------------------------------------
def collect_trials(usable_df, model_folder, img_map, question, prompt_label,
                   target_trials=3, max_workers=4, save_every=20):
    """Collect up to target_trials per race. Saves incrementally (every
    `save_every` completed trial-runs) with atomic writes, so an interrupt
    or crash loses at most `save_every` trial-runs rather than the whole model.
    Resume counts only COMPLETE trials (both orderings present)."""
    output_csv = get_filepath(model_folder=model_folder, prompt_label=prompt_label)
    existing = pd.read_csv(output_csv) if output_csv.exists() else pd.DataFrame()

    # resume: count a (race, trial) as done only if BOTH orderings are present.
    # (a half-saved trial has 1 row; we want to re-collect it, not skip it.)
    trials_done = {}
    if len(existing):
        per = existing.groupby(['election_id', 'trial']).size()   # rows per (race,trial)
        complete = per[per >= 2].reset_index()                    # both orderings present
        trials_done = complete.groupby('election_id')['trial'].nunique().to_dict()

    tasks = []
    for eid, race in usable_df.groupby('Election ID'):
        if len(race) != 2:
            continue
        a, b = race.iloc[0]['Full Label'], race.iloc[1]['Full Label']
        if a not in img_map or b not in img_map:
            continue
        for t in range(trials_done.get(eid, 0), target_trials):
            tasks.append((eid, img_map[a], img_map[b], t))

    print(f"  {model_folder}: {len(tasks)} trial-runs to reach {target_trials}/race")

    # main thread is the only writer (no concurrent-write corruption)
    accumulated = existing.to_dict('records') if len(existing) else []
    since_save = 0

    def _flush():
        tmp = output_csv.with_suffix('.tmp')
        pd.DataFrame(accumulated).to_csv(tmp, index=False)
        tmp.replace(output_csv)   # atomic: never leaves a half-written file

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_single_trial, ia, ib, model_folder, question, eid, t)
                for eid, ia, ib, t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            accumulated.extend(fut.result())   # both orderings appended together
            since_save += 1
            if since_save >= save_every:
                _flush()
                since_save = 0
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(tasks)}")

    _flush()   # final save
    new_count = len(accumulated) - (len(existing) if len(existing) else 0)
    if new_count > 0:
        newdf = pd.DataFrame(accumulated).tail(new_count)
        print(f"    new statuses: {dict(newdf['status'].value_counts())}")
    return output_csv


# ------------------------------------------------------------------
# Helpers (notebook calls)
# ------------------------------------------------------------------
def build_img_map():
    all_imgs = list(SENATE_PATH.glob("*.jpg")) + list(GOVERNOR_PATH.glob("*.jpg"))
    img_map = {}
    for p in all_imgs:
        lbl = clean_label(p)
        if lbl not in img_map or len(p.name) < len(img_map[lbl].name):
            img_map[lbl] = p
    return img_map


def _load_complete(img_map):
    _, complete = clean_election_df(ELECTION_RESULTS_PATH)
    complete['image_path'] = complete['Full Label'].astype(str).map(img_map)
    return complete[complete['image_path'].notna()].copy()


def backup_existing(dest="backup_before_multitrial"):
    """Back up existing single-trial data before any collection."""
    backup = Path(dest)
    backup.mkdir(exist_ok=True)
    for model, folder in ELECTION_MODELS.items():
        for lbl in ['electability', 'competence']:
            src = get_filepath(model_folder=folder, prompt_label=lbl)
            if src.exists():
                safe = folder.replace('/', '__')   # slashes in model names -> valid filename
                shutil.copy(src, backup / f"{safe}_{lbl}.csv")
    print(f"backed up existing data to {dest}/")


def run_pilot(prompt_key, n_races=10, target_trials=3, max_workers=4):
    """
    Watched pilot on ONE model, writing to a *_TEST label so it can never
    touch real data. Run this and inspect the output before run_full.
    """
    question, label = _resolve(prompt_key)
    label = label + "_TEST"
    img_map = build_img_map()
    complete = _load_complete(img_map)
    test_model = list(ELECTION_MODELS.values())[0]
    pilot_ids = complete['Election ID'].unique()[:n_races]
    pilot_df = complete[complete['Election ID'].isin(pilot_ids)]
    print(f"PILOT: {prompt_key} prompt, model={test_model}, label={label}")
    out = collect_trials(pilot_df, test_model, img_map, question, label,
                         target_trials=target_trials, max_workers=max_workers)
    print(f">>> inspect {out} — confirm 3 trials/race, clean statuses — before run_full <<<")
    return out


def run_full(prompt_key, target_trials=3, model_workers=3, race_workers=4):
    """
    Full collection across all models for one prompt. Writes to the
    prompt's *_multitrial label (NOT the single-trial files).
    Runs models concurrently; each model runs its trials concurrently.
    """
    question, label = _resolve(prompt_key)
    img_map = build_img_map()
    complete = _load_complete(img_map)
    print(f"FULL RUN: {prompt_key} prompt -> label '{label}', {target_trials} trials/race")

    def _one(item):
        model, folder = item
        print(f"\n=== {folder} ===")
        collect_trials(complete, folder, img_map, question, label,
                       target_trials=target_trials, max_workers=race_workers)

    with ThreadPoolExecutor(max_workers=model_workers) as ex:
        list(ex.map(_one, ELECTION_MODELS.items()))
    print("\nFULL RUN complete.")