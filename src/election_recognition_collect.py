"""
Recognition collection — a contamination check for the election analyses.

For each candidate face we ask the model to name the person. A face counts as
"potentially recognized" if the model's response contains the candidate's real
surname (see recognition analysis in multitrial_analysis_clean.py). The election
findings are then recomputed on the faces the model did NOT recognize, ruling out
the possibility that the model recalls known outcomes rather than judging the face.

Per-face (not per-race), single call, no counterbalancing or multi-trial:
recognition is not position-dependent. Saves the raw response so the match rule
can be re-run or audited.
"""

import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config import ELECTION_MODELS, MODEL_SNAPSHOTS, REASONING_EFFORT
from collect import encode_image, get_client
from election_collect import get_filepath   # shared helper
from prompts import RECOGNITION_PROMPT


# Forced-guess, name-first. Forcing a guess is the CONSERVATIVE choice for a
# contamination check: it gives faint recognition a chance to surface as the
# correct name, rather than letting the model hide behind "unknown". Wrong
# guesses are harmless — the surname-match rule filters them out. Name-first +
# low max_tokens prevents the truncation seen with rambling variants.


def _call_with_retry(fn, max_retries=5, base_delay=2):
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


def check_recognition(image_path, model_folder):
    """Ask the model to name one face using the default RECOGNITION_PROMPT."""
    return check_recognition_with_prompt(image_path, model_folder, RECOGNITION_PROMPT)


def check_recognition_with_prompt(image_path, model_folder, prompt, max_tokens=30):
    """Ask the model to name one face using an arbitrary prompt. Used by the
    pilot to compare prompt variants on the same faces."""
    snapshot = MODEL_SNAPSHOTS[model_folder]
    effort = REASONING_EFFORT.get(model_folder)
    extra = {"reasoning": {"effort": effort}} if effort else {}
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url":
            {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
    ]
    try:
        resp = _call_with_retry(lambda: get_client().chat.completions.create(
            model=snapshot, messages=[{"role": "user", "content": content}],
            temperature=0.0, max_tokens=max_tokens, extra_body=extra, timeout=60))
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)[:80]}"


def collect_recognition(usable_df, model_folder, img_map, max_workers=6, save_every=20):
    """Collect one recognition response per candidate face. Faces are fetched
    concurrently; only the main thread writes (safe incremental save). Resume-safe:
    skips faces already collected. Saves the raw response for later matching."""
    label = "recognition_v2"   # distinct from old 'recognition'/'recognition_updated'
    output_csv = get_filepath(model_folder=model_folder, prompt_label=label)

    done = set()
    rows = []
    if output_csv.exists():
        prev = pd.read_csv(output_csv)
        done = set(prev['candidate_label'])
        rows = prev.to_dict('records')

    # one row per unique candidate face (dedupe by Full Label)
    seen = set()
    todo = []
    for _, cand in usable_df.iterrows():
        lbl = cand['Full Label']
        if lbl in seen or lbl in done or lbl not in img_map:
            continue
        seen.add(lbl)
        todo.append((lbl, cand['Election ID']))

    print(f"  {model_folder}: {len(todo)} faces to recognize")
    if not todo:
        return output_csv

    def _flush():
        tmp = output_csv.with_suffix('.tmp')
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(output_csv)

    def _one(item):
        lbl, eid = item
        raw = check_recognition(img_map[lbl], model_folder)
        return {'candidate_label': lbl, 'election_id': eid, 'raw_response': raw}

    since_save = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, item) for item in todo]
        for i, fut in enumerate(as_completed(futs)):
            rows.append(fut.result())     # only the main thread appends/writes
            since_save += 1
            if since_save >= save_every:
                _flush()
                since_save = 0
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(todo)}")
    _flush()
    print(f"    {model_folder}: {len(rows)} total faces saved")
    return output_csv


def run_recognition_all(usable_df, img_map, model_workers=3, face_workers=6):
    """Collect recognition for every model, models running concurrently and each
    model fetching faces concurrently."""
    def _one(item):
        model, folder = item
        print(f"\n=== {folder} ===")
        collect_recognition(usable_df, folder, img_map, max_workers=face_workers)

    with ThreadPoolExecutor(max_workers=model_workers) as ex:
        list(ex.map(_one, ELECTION_MODELS.items()))
    print("\nRecognition collection complete.")