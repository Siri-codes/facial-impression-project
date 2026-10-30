"""
freeze_and_hash_2026.py — turn the collected forced-choice data into the FROZEN
preregistered predictions file + its SHA-256 hash. Run this AFTER run_full_2026,
BEFORE Election Day.

================================ PREDICTION RULE ================================
Preregister this verbatim. For each (model, race):
  among status=='ok' presentations, the face chosen in the MAJORITY of trials
  (3 trials x 2 orderings = up to 6) is that model's predicted winner.
  A tie (equal votes, e.g. 3-3) is recorded as 'tie' = no prediction; ties are
  handled at analysis time by the preregistered tie rule (default: excluded).

a/b -> candidate mapping:
  'a' = the FIRST candidate row for that election_id in the clean CSV,
  'b' = the SECOND. (Your clean CSVs list Democrat then Republican per race.)
  >>> The script PRINTS this mapping for a few races. Verify ONCE that it matches
      how the collector labeled slots, before you trust the hash. <<<

================================ WHY THE HASH ===================================
The whole point of a prospective study is that the models can't have seen the
outcomes. Registering a SHA-256 of the frozen predictions BEFORE the polls close
makes them tamper-evident: anyone can later recompute the hash from the same file
and confirm the predictions existed, unchanged, before any result was known.

Determinism (so the hash is reproducible): rows sorted by (model, election_id);
integer fields only; UTF-8; '\n' line endings; NO timestamp inside the hashed
file. Re-running on the same raw CSVs reproduces the identical file and hash.
"""

import csv
import hashlib
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---- paths: default from config_2026, but everything is overridable ----
def _defaults():
    root = label = None
    sen = gov = None
    try:
        import config_2026 as c
        sen, gov = c.SEN_CLEAN, c.GOV_CLEAN
        root = getattr(c, "RESULTS_2026", None)
    except Exception:
        pass
    return sen, gov, root


LABEL = "competence_multitrial_2026"     # run_full_2026's output label
FROZEN_NAME = "predictions_2026_frozen.csv"
HASH_NAME = "predictions_2026.sha256"

FIELDS = ["model", "election_id", "office", "state",
          "cand_a", "party_a", "cand_b", "party_b",
          "n_ok", "votes_a", "votes_b", "predicted_face",
          "predicted_stem", "predicted_party"]


def load_race_map(race_csvs):
    """election_id -> {a:(stem,party), b:(stem,party), office, state}.
    'a' = first CSV row for the race, 'b' = second (the preregistered rule)."""
    racemap = {}
    for csv_path in race_csvs:
        df = pd.read_csv(csv_path)
        df["office"] = df["election_id"].str.split("_").str[0]
        for eid, g in df.groupby("election_id", sort=False):
            g = g.reset_index(drop=True)
            if len(g) != 2:
                raise SystemExit(f"{eid} has {len(g)} candidates (need 2) in {csv_path}")
            racemap[eid] = {
                "a": (g.loc[0, "stem"], g.loc[0, "party"]),
                "b": (g.loc[1, "stem"], g.loc[1, "party"]),
                "office": g.loc[0, "office"], "state": g.loc[0, "state"],
            }
    return racemap


def find_result_files(results_root, label=LABEL):
    """Find each model's collected CSV. Returns {model_name: path}.
    A model's file is <label>.csv; its model name is the parent folder (how the
    engine writes per-model outputs). TEST/pilot files are skipped."""
    root = Path(results_root)
    found = {}
    for p in sorted(root.rglob("*.csv")):
        name = p.stem.lower()
        if label.lower() not in name or "test" in name:
            continue
        model = p.parent.name            # engine writes <model_folder>/<label>.csv
        found[model] = p
    return found


def aggregate(result_files, racemap):
    """Build the frozen rows: one per (model, race) with vote counts + prediction."""
    rows = []
    for model, path in result_files.items():
        d = pd.read_csv(path)
        d = d[d["status"] == "ok"]
        for eid, g in d.groupby("election_id"):
            if eid not in racemap:
                print(f"  WARNING: {model} has race {eid} not in the clean CSVs — skipped")
                continue
            va = int((g["chosen_face"] == "a").sum())
            vb = int((g["chosen_face"] == "b").sum())
            if va > vb:
                face = "a"
            elif vb > va:
                face = "b"
            else:
                face = "tie"
            rm = racemap[eid]
            stem, party = (rm[face] if face in ("a", "b") else ("", ""))
            rows.append({
                "model": model, "election_id": eid,
                "office": rm["office"], "state": rm["state"],
                "cand_a": rm["a"][0], "party_a": rm["a"][1],
                "cand_b": rm["b"][0], "party_b": rm["b"][1],
                "n_ok": va + vb, "votes_a": va, "votes_b": vb,
                "predicted_face": face,
                "predicted_stem": stem, "predicted_party": party,
            })
    rows.sort(key=lambda r: (r["model"], r["election_id"]))   # deterministic order
    return rows


def write_frozen(rows, out_path):
    """Write the canonical frozen CSV with fixed, reproducible formatting."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(results_root=None, race_csvs=None, label=LABEL, out_dir=None):
    sen, gov, root_default = _defaults()
    race_csvs = race_csvs or [sen, gov]
    if any(c is None for c in race_csvs):
        raise SystemExit("Clean CSV paths not resolved — pass race_csvs=[senate, governor].")
    results_root = results_root or root_default
    if results_root is None:
        raise SystemExit("Pass results_root= (the folder run_full_2026 wrote into).")
    out_dir = Path(out_dir) if out_dir else Path(results_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    racemap = load_race_map(race_csvs)
    result_files = find_result_files(results_root, label)
    if not result_files:
        raise SystemExit(f"No '{label}' result CSVs found under {results_root} "
                         f"(did run_full_2026 finish? is the label right?)")

    print(f"models found: {len(result_files)}")
    for m, p in result_files.items():
        print(f"  {m:40s} {p}")

    # a/b -> candidate mapping spot-check (verify against the collector once!)
    print("\na/b -> candidate mapping (VERIFY this matches the collector):")
    for eid in list(racemap)[:4]:
        rm = racemap[eid]
        print(f"  {eid}:  a = {rm['a'][0]} ({rm['a'][1]})   |   b = {rm['b'][0]} ({rm['b'][1]})")

    rows = aggregate(result_files, racemap)
    frozen_path = out_dir / FROZEN_NAME
    write_frozen(rows, frozen_path)
    digest = sha256_of(frozen_path)

    n_models = len({r["model"] for r in rows})
    n_races = len({r["election_id"] for r in rows})
    n_ties = sum(1 for r in rows if r["predicted_face"] == "tie")

    # manifest: the hash + metadata; the timestamp lives HERE, not in the hashed file
    manifest = out_dir / HASH_NAME
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("# 2026 prospective election predictions — preregistration hash\n")
        f.write(f"sha256: {digest}\n")
        f.write(f"file: {FROZEN_NAME}\n")
        f.write(f"rows: {len(rows)}\n")
        f.write(f"models: {n_models}\n")
        f.write(f"races: {n_races}\n")
        f.write(f"ties: {n_ties}\n")
        f.write(f"generated_utc: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("prediction_rule: majority chosen face per (model,race); "
                "equal votes = 'tie' (excluded by default at analysis)\n")

    print(f"\nfrozen predictions -> {frozen_path}  ({len(rows)} rows, "
          f"{n_models} models x {n_races} races, {n_ties} ties)")
    print("\n" + "=" * 68)
    print(f"  SHA-256:  {digest}")
    print("=" * 68)
    print(f"\nmanifest -> {manifest}")
    print("REGISTER this hash (OSF, timestamped) BEFORE polls close on Nov 3.")
    return frozen_path, digest


if __name__ == "__main__":
    # usage: python freeze_and_hash_2026.py <results_root> [senate_csv governor_csv]
    rr = sys.argv[1] if len(sys.argv) > 1 else None
    rc = sys.argv[2:4] if len(sys.argv) > 3 else None
    freeze(results_root=rr, race_csvs=rc)
