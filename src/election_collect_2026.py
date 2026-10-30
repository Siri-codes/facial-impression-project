"""
election_collect_2026.py — prospective forced-choice collection for the 2026
midterms. Reuses the retrospective forced-choice ENGINE unchanged, so the method
is identical to the published study.

ALL PATHS come from config_2026.py — there are no hardcoded paths in this file.
If a run can't find your files, run `import config_2026; config_2026.describe()`
to see what resolved and where it looked.

What is reused UNCHANGED (from election_collect_multitrial):
    collect_trials  -> resume-safe, atomic, multi-trial forced-choice collection
                       (internally run_single_trial -> extract_choice, both orderings)
    _resolve        -> prompt_key -> (question, label)
and the model set (config.ELECTION_MODELS). Nothing about HOW a pair is queried
changes versus the published collection.

What changes, and why: the race set is the CLEAN CSVs (outcome-blind, frozen for
the prereg), and images are matched to candidates by the `stem` column. Winners
are added only later, at analysis time, after the election.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# --- the UNCHANGED forced-choice engine from the retrospective study ---
from election_collect_multitrial import collect_trials, _resolve

# --- same model set as the retrospective (preregister these snapshots) ---
from config import ELECTION_MODELS

# --- ALL paths + settings come from config_2026 (one source of truth) ---
from config_2026 import RACE_CSVS, FACES_ROOT, IMAGE_EXTS, describe as _describe_paths


def _check_paths():
    """Fail early with a helpful message if config_2026 didn't resolve a path."""
    problems = []
    if FACES_ROOT is None or not Path(FACES_ROOT).exists():
        problems.append(f"FACES_ROOT (nobg_gray folder): {FACES_ROOT}")
    for c in RACE_CSVS:
        if c is None or not Path(c).exists():
            problems.append(f"race CSV: {c}")
    if problems:
        print("\n[config_2026 could not resolve these paths]")
        for p in problems:
            print("   -", p)
        print()
        _describe_paths()
        raise SystemExit(
            "Fix the paths above. Easiest: upload the files, or set the env vars "
            "FACES_ROOT / SEN_CLEAN_CSV / GOV_CLEAN_CSV, then RESTART the kernel.")


def build_img_map_2026(faces_root=None):
    """{filename stem -> image path} over every image under faces_root (recursive).
    Duplicate stems are a data error and are reported."""
    faces_root = faces_root or FACES_ROOT
    root = Path(faces_root)
    if not root.exists():
        _describe_paths()
        raise SystemExit(f"FACES_ROOT does not exist: {root}  (see resolved paths above)")
    img_map = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            if p.stem in img_map:
                print(f"  WARNING: duplicate stem {p.stem} ({p} vs {img_map[p.stem]}) — keeping first")
                continue
            img_map[p.stem] = p
    return img_map


def build_races_2026(race_csvs=None, img_map=None, strict=True, verbose=True):
    """Build the race table from the CLEAN CSVs and cross-check against images.
    Returns a df with 'Election ID' + 'Full Label' (2 rows/race) — the columns
    collect_trials expects — plus office/state/party for reference."""
    race_csvs = race_csvs or RACE_CSVS
    frames = []
    for csv in race_csvs:
        df = pd.read_csv(csv)
        need = {"election_id", "stem"}
        if not need.issubset(df.columns):
            raise SystemExit(f"{csv} missing columns {need - set(df.columns)}")
        df["office"] = df["election_id"].str.split("_").str[0]
        frames.append(df)
    races = pd.concat(frames, ignore_index=True)

    sizes = races.groupby("election_id").size()
    bad = sizes[sizes != 2]
    if len(bad):
        raise SystemExit(f"these races are not 2-candidate (fix the clean CSV): {dict(bad)}")

    if img_map is not None:
        have = set(img_map)
        need_stems = set(races["stem"])
        for office, g in races.groupby("office"):
            o_need = set(g["stem"])
            print(f"{office}: need {len(o_need)}, have {len(o_need & have)} | "
                  f"missing {sorted(o_need - have)}")
        extra = sorted(have - need_stems)
        if extra:
            print(f"stray images not in any CSV (ignored): {extra}")
        missing_all = sorted(need_stems - have)
        if missing_all and strict:
            raise SystemExit(f"{len(missing_all)} candidates have no image — cannot start: {missing_all}")

    out = races.rename(columns={"election_id": "Election ID", "stem": "Full Label"})
    if verbose:
        n = out["Election ID"].nunique()
        print(f"{n} races / {len(out)} candidates "
              f"({(out['office']=='senate').sum()} senate + "
              f"{(out['office']=='governor').sum()} governor)")
    return out[["Election ID", "Full Label", "office", "state", "party"]]


def run_pilot_2026(prompt_key, n_races=10, target_trials=3, max_workers=4,
                   race_csvs=None, faces_root=None):
    """Watched pilot: ONE model, a few races, a *_TEST label. Inspect before full."""
    _check_paths()
    question, base = _resolve(prompt_key)
    label = base + "_2026_TEST"
    img_map = build_img_map_2026(faces_root)
    complete = build_races_2026(race_csvs, img_map=img_map)
    test_model = list(ELECTION_MODELS.values())[0]
    pilot_ids = complete["Election ID"].unique()[:n_races]
    pilot_df = complete[complete["Election ID"].isin(pilot_ids)]
    print(f"\nPILOT 2026: '{prompt_key}', model={test_model}, {len(pilot_ids)} races, label={label}")
    out = collect_trials(pilot_df, test_model, img_map, question, label,
                         target_trials=target_trials, max_workers=max_workers)
    print(f">>> inspect {out} — confirm {target_trials} trials/race, both orderings, "
          f"clean statuses — before run_full_2026 <<<")
    return out


def run_full_2026(prompt_key, target_trials=3, model_workers=3, race_workers=4,
                  race_csvs=None, faces_root=None):
    """Full collection across all ELECTION_MODELS for one prompt. Writes to the
    '<prompt_key>_multitrial_2026' label. Resume-safe: re-run to continue."""
    _check_paths()
    question, base = _resolve(prompt_key)
    label = base + "_2026"
    img_map = build_img_map_2026(faces_root)
    complete = build_races_2026(race_csvs, img_map=img_map)
    n_races = complete["Election ID"].nunique()
    print(f"\nFULL 2026 RUN: '{prompt_key}' -> label '{label}', "
          f"{n_races} races, {target_trials} trials/race, {len(ELECTION_MODELS)} models")

    def _one(item):
        model, folder = item
        print(f"\n=== {folder} ===")
        collect_trials(complete, folder, img_map, question, label,
                       target_trials=target_trials, max_workers=race_workers)

    with ThreadPoolExecutor(max_workers=model_workers) as ex:
        list(ex.map(_one, ELECTION_MODELS.items()))
    print("\nFULL 2026 RUN complete. Freeze + hash the predictions before Election Day.")