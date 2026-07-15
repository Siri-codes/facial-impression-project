import pandas as pd
from config import ATTRIBUTES, MODEL_DIR, HUMAN_MEANS, HUMAN_RATINGS

def load_ratings(path):
    """Load a wide ratings file, enforcing canonical attribute order."""
    df = pd.read_csv(path)
    missing = set(ATTRIBUTES) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing attributes: {sorted(missing)}")
    if df["stimulus"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate stimuli")
    return df[["stimulus"] + ATTRIBUTES]

def load_model(folder, filename):
    """Load one model's wide ratings file from MODEL_DIR."""
    return load_ratings(MODEL_DIR / folder / filename)

def load_human_means():
    return load_ratings(HUMAN_MEANS)

def load_human_raw():
    """Load the LONG per-rater file. Errors on any attribute-set mismatch."""
    df = pd.read_csv(HUMAN_RATINGS)
    expected_cols = {"stimulus","attribute","rating","rating_type"}
    if set(df.columns) != expected_cols:
        raise ValueError(f"{HUMAN_RATINGS.name} columns {set(df.columns)} != {expected_cols}")
    found = set(df["attribute"].unique())
    expected = set(ATTRIBUTES)
    if found != expected:
        raise ValueError(
            f"attribute mismatch in {HUMAN_RATINGS.name}: "
            f"missing {sorted(expected - found)}, unexpected {sorted(found - expected)}"
        )
    return df

def common_stimuli(*dfs):
    """Restrict all dfs to stimuli present in every df, in identical row order."""
    shared = set(dfs[0]["stimulus"])
    for df in dfs[1:]:
        shared &= set(df["stimulus"])
    return [df[df["stimulus"].isin(shared)].sort_values("stimulus").reset_index(drop=True)
            for df in dfs]
