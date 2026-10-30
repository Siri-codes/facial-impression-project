"""
filter_two_candidate.py — reduce a raw candidate list to the top-two
major-party (one Democrat vs one Republican) matchup per race, implementing
the preregistered selection rule (Option A).

It also CATEGORISES every state so the edge cases are explicit (which the
preregistration must specify how to handle):
    clean       : exactly one D and one R  -> included
    same_party  : e.g. two D or two R and no cross-party (top-two primary states)
    missing     : a placeholder / no candidates yet
    no_major    : a race with no D or no R (independent-dominated, etc.)
    multi_major : >1 D or >1 R (unresolved primary in your data — needs a pick)

Fusion tickets ("Democratic, Working Families Party") are treated by their
FIRST listed major party, so they aren't dropped.

Usage:
    from filter_two_candidate import filter_candidates
    clean, report = filter_candidates("governor_candidates__1_.csv",
                                      out="governor_two_candidate.csv")
    print(report)          # inspect the edge cases before trusting `clean`
"""

from pathlib import Path

import pandas as pd


def _norm(df):
    """Strip the messy quoting/whitespace this data tends to carry."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.strip('"').str.strip()
    return df


def _major_party(party):
    """Map a (possibly fusion) party label to 'D', 'R', or None.

    'Democratic, Working Families Party' -> 'D'
    'Republican, Conservative Party'     -> 'R'
    'Independent' / 'Green' / ...        -> None
    """
    p = str(party).lower()
    # check the FIRST token so fusion tickets keep their major party
    first = p.split(",")[0].strip()
    if first.startswith("democrat"):     # 'democrat' / 'democratic'
        return "D"
    if first.startswith("republican"):
        return "R"
    return None


def filter_candidates(csv_path, out="two_candidate.csv",
                      name_col="name", state_col="state", party_col="party"):
    """Return (clean_df, report_df).

    clean_df : one D and one R per clean race, ready for face scraping.
    report_df: per-state category so you can see/handle every edge case.
    """
    df = _norm(pd.read_csv(csv_path, skipinitialspace=True))
    df["major"] = df[party_col].map(_major_party)

    # drop blank placeholder rows (state but no name)
    df = df[df[name_col].notna() & ~df[name_col].isin(["", "nan"])]

    report_rows = []
    clean_rows = []
    for state, g in df.groupby(state_col):
        dems = g[g["major"] == "D"]
        reps = g[g["major"] == "R"]
        nd, nr = len(dems), len(reps)

        if nd == 1 and nr == 1:
            category = "clean"
            clean_rows.append(dems.iloc[0])
            clean_rows.append(reps.iloc[0])
        elif nd == 0 and nr == 0:
            category = "no_major"      # no D or R at all
        elif nd == 0 or nr == 0:
            category = "no_major"      # missing one side (e.g. safe seat, indep race)
        elif nd > 1 or nr > 1:
            category = "multi_major"   # unresolved: >1 D or >1 R — you must pick the nominee
        else:
            category = "review"

        report_rows.append({
            "state": state, "n_dem": nd, "n_rep": nr,
            "category": category,
            "dem_names": "; ".join(dems[name_col].tolist()),
            "rep_names": "; ".join(reps[name_col].tolist()),
        })

    clean = pd.DataFrame(clean_rows).reset_index(drop=True) if clean_rows else pd.DataFrame()
    report = pd.DataFrame(report_rows).sort_values(["category", "state"]).reset_index(drop=True)

    # summary
    print(f"{csv_path}")
    print(report["category"].value_counts().to_dict())
    n_clean = (report["category"] == "clean").sum()
    print(f"  {n_clean} clean 2-candidate races -> {2*n_clean} candidates for scraping")
    needs = report[report["category"] != "clean"]
    if len(needs):
        print(f"  {len(needs)} states need attention (see report):")
        print(needs[["state", "n_dem", "n_rep", "category"]].to_string(index=False))

    if len(clean):
        clean.to_csv(out, index=False)
        print(f"  wrote clean matchups -> {out}")
    # build the report path from `out`, working for both str and Path inputs
    out_path = Path(out)
    report_path = out_path.with_name(out_path.stem + "_report.csv")
    report.to_csv(report_path, index=False)
    return clean, report


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "governor_candidates__1_.csv"
    filter_candidates(path)