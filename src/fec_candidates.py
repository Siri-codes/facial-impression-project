# Pull 2026 Senate candidates from FEC API
# FEC API is free public government data. Get a key at https://api.data.gov/signup/

import requests
import pandas as pd
import time

FEC_API_KEY = os.environ.get("FEC_API_KEY")   # from api.data.gov/signup — free, instant
BASE = "https://api.open.fec.gov/v1"

def get_senate_candidates_2026():
    """Fetch all 2026 Senate candidates from the FEC API."""
    candidates = []
    page = 1
    while True:
        resp = requests.get(f"{BASE}/candidates/", params={
            "api_key": FEC_API_KEY,
            "election_year": 2026,        # 2026 cycle
            "office": "S",                # S = Senate (H = House, P = President)
            "per_page": 100,
            "page": page,
            "sort": "name",
        })
        resp.raise_for_status()
        data = resp.json()

        for c in data["results"]:
            candidates.append({
                "name": c.get("name"),
                "party": c.get("party_full"),
                "state": c.get("state"),
                "incumbent_challenge": c.get("incumbent_challenge_full"),  # Incumbent/Challenger/Open
                "candidate_id": c.get("candidate_id"),
                "office": c.get("office_full"),
                "active": c.get("candidate_status"),
            })

        # pagination: stop when we've fetched the last page
        pages = data["pagination"]["pages"]
        print(f"  page {page}/{pages} ({len(candidates)} so far)")
        if page >= pages:
            break
        page += 1
        time.sleep(0.5)   # be polite to the API

    return pd.DataFrame(candidates)

