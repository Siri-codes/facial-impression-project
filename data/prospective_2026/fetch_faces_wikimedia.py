"""
Fetch candidate face images from Wikipedia/Wikimedia for the 2026 prospective
election study.

Uses the MediaWiki API (sanctioned for automated access — no scraping of
disallowed pages). For each candidate we:
  1. Search Wikipedia for the person (name + a politics disambiguator),
  2. take the top page hit,
  3. pull that page's lead/infobox image (the "pageimage"),
  4. download the original image file.

CRITICAL: Wikipedia name-matching is NOT reliable enough to trust blindly —
same-name people, missing pages, wrong photos all happen. So this script does
NOT silently accept matches. It writes a manifest CSV logging, per candidate:
the matched page title, the image URL, and the status. YOU then eyeball the
manifest (and the downloaded images) and correct/replace any wrong ones by hand.
Treat this as a first pass that gets ~most faces, not a finished dataset.

Requires: requests, pandas  (both confirmed available).
Be polite: set a real user-agent with contact info, and keep the rate modest.

Usage:
    from fetch_faces_wikimedia import fetch_all
    fetch_all("governor_candidates.csv", out_dir="faces_2026/governor")
    # then OPEN faces_2026/governor/_manifest.csv and verify every row
"""

import time
import re
from pathlib import Path

import requests
import pandas as pd

API = "https://en.wikipedia.org/w/api.php"

# Wikipedia asks automated clients to identify themselves. Put a real contact
# in here (your email or project URL) before running — it's their requested
# etiquette and keeps you in good standing.
HEADERS = {
    "User-Agent": "FacialImpressionResearch/1.0 (prereg study; scox@commschool.org)"
}

REQUEST_PAUSE = 0.5   # seconds between API calls — be gentle


def _clean_name(raw):
    """Strip the leading/trailing spaces your CSV has (', name' -> 'name')."""
    return str(raw).strip().strip('"').strip()


def _search_page(name, disambiguator="politician"):
    """Search Wikipedia; return the best-guess page title or None.

    We bias the query toward the politician sense by appending a disambiguator,
    but we DON'T assume it's right — the manifest lets you verify.
    """
    params = {
        "action": "query", "list": "search", "format": "json",
        "srsearch": f"{name} {disambiguator}", "srlimit": 3,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    hits = r.json().get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def _page_image_url(title):
    """Return the URL of a page's lead image (pageimage), or None."""
    params = {
        "action": "query", "prop": "pageimages", "format": "json",
        "piprop": "original", "titles": title,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for _, page in pages.items():
        original = page.get("original")
        if original and "source" in original:
            return original["source"]
    return None


def _safe_filename(state, name, party):
    """Deterministic, filesystem-safe stem, e.g. AL_Tommy_Tuberville_R."""
    party_code = {"Republican": "R", "Democrat": "D", "Democratic": "D"}.get(
        party.strip(), party.strip()[:3])
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    return f"{state.strip()}_{slug}_{party_code}"


def _download(url, dest):
    """Download an image URL to dest. Returns True on success."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"    download failed: {e}")
        return False


def _load_existing_manifest(man_path):
    """Return the prior manifest as {stem: row_dict}, or {} if none.

    Lets us RESUME: keep prior results (and your 'verified' marks) and only
    fetch candidates we don't already have.
    """
    if not man_path.exists():
        return {}
    prev = pd.read_csv(man_path).fillna("")
    return {r["stem"]: r.to_dict() for _, r in prev.iterrows() if "stem" in r}


def fetch_all(candidate_csv, out_dir="faces_2026",
              name_col="name", state_col="state", party_col="party",
              disambiguator="politician", refetch=False):
    """Fetch a first-pass face image for every candidate in the CSV. RESUMABLE.

    - Skips any candidate whose image file already exists (unless refetch=True),
      so you can add rows to the CSV later and re-run to fetch only the new ones.
    - Merges with the existing manifest, preserving your 'verified' annotations.

    Writes images to out_dir/ and a manifest CSV you MUST verify by hand.
    Skips blank rows (placeholder rows for races not yet filled in).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "_manifest.csv"
    prior = _load_existing_manifest(man_path)   # {stem: row}

    df = pd.read_csv(candidate_csv)
    for c in (name_col, state_col, party_col):
        df[c] = df[c].map(_clean_name)

    manifest = dict(prior)   # start from prior, update/add as we go
    n_new = n_skip = 0

    for _, cand in df.iterrows():
        name, state, party = cand[name_col], cand[state_col], cand[party_col]
        if not name or name.lower() == "nan":     # blank placeholder row
            continue

        stem = _safe_filename(state, name, party)

        # RESUMABLE: skip if we already have an image for this candidate
        existing = [p for p in out.glob(f"{stem}.*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if existing and not refetch:
            n_skip += 1
            # keep prior manifest row if present; otherwise record the file we found
            if stem not in manifest:
                manifest[stem] = {
                    "stem": stem, "state": state, "name": name, "party": party,
                    "wiki_title": "(existing file)", "image_url": "",
                    "saved_path": str(existing[0]), "status": "already_present",
                    "verified": "",
                }
            continue

        print(f"{state:3s} {name:30s} ({party})")
        title = image_url = None
        status = "no_page"
        try:
            title = _search_page(name, disambiguator)
            if title:
                image_url = _page_image_url(title)
                status = "ok" if image_url else "page_no_image"
        except Exception as e:
            status = f"error: {e}"

        saved_path = ""
        if image_url:
            ext = Path(image_url.split("?")[0]).suffix or ".jpg"
            dest = out / f"{stem}{ext}"
            if _download(image_url, dest):
                saved_path = str(dest)
            else:
                status = "download_failed"

        # preserve a prior 'verified' mark if this stem was seen before
        prev_verified = manifest.get(stem, {}).get("verified", "")
        manifest[stem] = {
            "stem": stem, "state": state, "name": name, "party": party,
            "wiki_title": title or "", "image_url": image_url or "",
            "saved_path": saved_path, "status": status,
            "verified": prev_verified,     # <-- YOU fill in: y / n after eyeballing
        }
        n_new += 1
        print(f"    -> {status}  [{title or 'no title'}]")
        time.sleep(REQUEST_PAUSE)

    man_df = pd.DataFrame(list(manifest.values()))
    man_df.to_csv(man_path, index=False)

    ok = (man_df["status"] == "ok").sum() if len(man_df) else 0
    print(f"\nfetched {n_new} new, skipped {n_skip} already-present.")
    print(f"{ok}/{len(man_df)} candidates in manifest have status 'ok'.")
    print(f"Manifest: {man_path}")
    print("NEXT: open the manifest, look at each image, set 'verified' to y/n, "
          "and hand-collect anything marked no_page / page_no_image / n.")
    return man_df


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "governor_candidates.csv"
    fetch_all(csv)