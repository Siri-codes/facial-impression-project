"""
config_2026.py — all paths for the prospective 2026 study.

NO directory scanning. It only constructs Path objects from the repo root, so
importing it is instant/cheap regardless of the mount speed.

Layout assumed (code in src/, data in data/):
    <repo>/data/
        senate_races_clean.csv
        governor_races_clean.csv
        faces_2026/nobg_gray/{senate,governor}/

If a path is wrong, describe() prints what's actually in data/ so you can see
the real names. Fix DATA / the three paths below to match (or set PROJECT_ROOT).
"""

import os
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

from config import DATA_DIR
DATA_2026 = DATA_DIR / "prospective_2026"            # <-- 2026 data lives here

# ---- explicit paths (built, never searched) ----
SEN_CLEAN   = DATA_2026 /  "senate_races_clean.csv"
GOV_CLEAN   = DATA_2026 / "governor_races_clean.csv"
STUDY_FACES = DATA_2026 / "faces_2026" / "nobg_gray"

RACE_CSVS  = [SEN_CLEAN, GOV_CLEAN]
FACES_ROOT = STUDY_FACES

RESULTS_2026 = DATA_2026 / "results_2026"
RAW_FACES    = DATA_2026 / "faces_2026" / "raw"
STD_FACES    = DATA_2026 / "faces_2026" / "standardized"


def describe():
    """Fast existence check; if something's missing, show what's really in
    data/ (and data/faces_2026/) so the real names are visible. One level
    each (os.listdir) — safe, never a recursive walk."""
    print("config_2026 (no-scan):")
    paths = [("DATA_2026", DATA_2026), ("SEN_CLEAN", SEN_CLEAN),
             ("GOV_CLEAN", GOV_CLEAN), ("STUDY_FACES", STUDY_FACES)]
    for name, p in paths:
        try:
            ok = p.exists()
        except Exception as e:
            ok = f"ERR {e}"
        print(f"  {name:12s} {p}   exists={ok}")

    if not (SEN_CLEAN.exists() and STUDY_FACES.exists()):
        print("\n  -- locating: contents of data/ and data/faces_2026/ --")
        for d in [DATA, DATA / "faces_2026"]:
            try:
                print(f"  {d} -> {sorted(os.listdir(d))}")
            except Exception as e:
                print(f"  {d} -> ERR {e}")


if __name__ == "__main__":
    describe()