from prompts import ELECTION_PROMPT, ELECTION_INSTRUCTION, COMPETENCE_PROMPT, COMPETENCE_RATING_PROMPT, COMPETENCE_RATING_INSTRUCTION, RECOGNITION_PROMPT
from collect import get_client, encode_image, get_filepath
from config import MODEL_SNAPSHOTS, REASONING_EFFORT, SENATE_PATH, GOVERNOR_PATH, ELECTION_RESULTS_PATH, ELECTION_MODELS
import pandas as pd
import re
import time

# --- strip only the -YYYYMMDD-HHMMSS timestamp, leave real labels intact ---
def clean_label(p):
    return re.sub(r'-\d{8}-\d{6}$', '', p.stem)

def clean_election_df(path):
    '''
    Cleans the election results CSV, returning (raw, usable) dataframes.
    '''
    df = pd.read_csv(path) #read the existing csv

    # 1. coerce numeric columns, junk strings -> NaN
    numeric_cols = ['Vote Share', 'Votes', 'Total Votes in Election',
                    'Competency', 'Attractiveness', 'Age',
                    '% Dem Voters', '% Rep Voters']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # "unknown" -> NaN

    # 2. normalize categoricals
    if 'Incumbent?' in df.columns:
        df['is_incumbent'] = df['Incumbent?'].astype(str).str.lower().str.strip() == 'yes'

    # 3. count usable candidates (complete on what the study absolutely needs)
    required = ['Full Label', 'Vote Share']   # minimum: an img key + an outcome
    complete = df.dropna(subset=required) #remove missing vals
    
    print(f"total: {len(df)}, complete on {required}: {len(complete)}")

    # 4. how many have the covariates you need for confound control (incumbency)?
    with_controls = complete.dropna(subset=['is_incumbent'])  
    print(f"with incumbency: {len(with_controls)}")

    return df, complete

'''
@deprecated("Use election_collect_multitrial instead")
def forced_choice(image_a, image_b, model_folder, question):
    """
    Ask a model to pick between two faces, counterbalanced.
    Returns (winner, raw) where winner is 'a' or 'b' (which face won),
    or None if refused/inconsistent (position bias).
    """
    snapshot = MODEL_SNAPSHOTS[model_folder]
    effort = REASONING_EFFORT.get(model_folder)
    extra = {"reasoning": {"effort": effort}} if effort else {}

    winners = []   # which input-face won each ordering
    raws = []
    # ordering 1: image_a in slot A, image_b in slot B
    # ordering 2: image_b in slot A, image_a in slot B
    for slotA, slotB, (name_A, name_B) in [
        (image_a, image_b, ('a', 'b')),   # slot A holds face a
        (image_b, image_a, ('b', 'a')),   # slot A holds face b
    ]:
        content = [
            {"type": "text", "text": f"{question}\n\nCandidate A:"},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{encode_image(slotA)}"}},
            {"type": "text", "text": "Candidate B:"},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{encode_image(slotB)}"}},
            {"type": "text", "text": ELECTION_INSTRUCTION},
        ]
        try:
            resp = get_client().chat.completions.create(
                model=snapshot, messages=[{"role": "user", "content": content}],
                temperature=0.1, max_tokens=2000, extra_body=extra,
            )
            raw = resp.choices[0].message.content.strip()
            raws.append(raw)
            letter = raw.upper()[0] if raw and raw.upper()[0] in ('A', 'B') else None
            if letter is None:
                return None, f"REFUSE/FAIL: {raw[:50]}"
            # map the chosen SLOT back to which INPUT FACE it was
            winners.append(name_A if letter == 'A' else name_B)
        except Exception as e:
            return None, f"ERROR: {e}"

    # real preference = same input face won BOTH orderings
    if winners[0] == winners[1]:
        return winners[0], f"consistent: {winners}"   # 'a' or 'b'
    else:
        return None, f"POSITION_BIAS: {winners} | {raws}"

@deprecated("Use election_collect_multitrial instead")
def collect_election_choices(usable_df, model_folder, img_map, question, pilot=False):

    # where to save results
    output_csv = get_filepath(model_folder=model_folder, prompt_label="electability")

    #keep track of processed elections
    processed = set()
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        processed = set(existing['election_id'])   # track by race, not candidate

    if pilot:
        first_ids = usable_df['Election ID'].unique()[:10]   # first 10 race IDs
        usable_df = usable_df[usable_df['Election ID'].isin(first_ids)]

    election_race_pairs = usable_df.groupby('Election ID')

    # group by race --- each race is one comparison
    for election_id, race in election_race_pairs:
        time.sleep(1)

        if election_id in processed:
            continue
        if len(race) != 2:  # skip races without exactly 2 candidates
            continue

        # identify the two candidates
        cand_a = race.iloc[0]
        cand_b = race.iloc[1]

        if cand_a['Full Label'] not in img_map or cand_b['Full Label'] not in img_map:
            print("not in image map")
            continue   # skip races where either candidate has no image

        img_a = img_map[cand_a['Full Label']] 
        img_b = img_map[cand_b['Full Label']] 

        winner, raw = forced_choice(img_a, img_b, model_folder, question)

        # which candidate did the model pick? map 'a'/'b' back to Full Label + outcome
        result = {
        'election_id': election_id,
        # the two candidates (always recorded, regardless of choice)
        'candidate_a': cand_a['Full Label'],
        'candidate_b': cand_b['Full Label'],
        'a_won_real': cand_a['Winner/Loser'] == 'Winner',   # ground truth
        'a_vote_share': cand_a['Vote Share'],
        'b_vote_share': cand_b['Vote Share'],
        # the model's choice
        'model_chose': None,          # 'a', 'b', or None
        'model_chose_label': None,    # Full Label of chosen, or None
        'model_correct': None,        # did model pick the real winner?
        'status': None,               # 'ok' / 'REFUSE' / 'POSITION_BIAS' / etc.
        }
        # then fill based on outcome:
        if winner is None:
            result['status'] = raw[:30]
        else:
            result['model_chose'] = winner
            chosen = cand_a if winner == 'a' else cand_b
            result['model_chose_label'] = chosen['Full Label']
            result['model_correct'] = (chosen['Winner/Loser'] == 'Winner')
            result['status'] = 'ok'

        df = pd.DataFrame([result])
        df.to_csv(output_csv, mode='a', index=False,
                                      header=not output_csv.exists())

@deprecated("Use election_collect_multitrial instead")
def rate_competence(image, model_folder, question=COMPETENCE_RATING_PROMPT):
    """Rate a single candidate's competence 0-100. Returns (score, raw)."""
    snapshot = MODEL_SNAPSHOTS[model_folder]
    effort = REASONING_EFFORT.get(model_folder)
    extra = {"reasoning": {"effort": effort}} if effort else {}

    content = [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url":
            {"url": f"data:image/jpeg;base64,{encode_image(image)}"}},
        {"type": "text", "text": COMPETENCE_RATING_INSTRUCTION},
    ]
    try:
        resp = get_client().chat.completions.create(
            model=snapshot, messages=[{"role": "user", "content": content}],
            temperature=0.1, max_tokens=2000, extra_body=extra,
        )
        raw = resp.choices[0].message.content.strip()
        import re
        m = re.search(r'\d+', raw)            # extract first number
        score = int(m.group()) if m else None  # None = refusal/no number
        return score, raw
    except Exception as e:
        return None, f"ERROR: {e}"

@deprecated("Use election_collect_multitrial instead")
def collect_competence_ratings(usable_df, model_folder, img_map, pilot=False):
    output_csv = get_filepath(model_folder=model_folder, prompt_label="competence_rating")

    if pilot and output_csv.exists():
        output_csv.unlink()

    processed = set()
    if output_csv.exists():
        processed = set(pd.read_csv(output_csv)['candidate_label'])

    df = usable_df
    if pilot:
        first_ids = df['Election ID'].unique()[:10]
        df = df[df['Election ID'].isin(first_ids)]

    for _, cand in df.iterrows():
        label = cand['Full Label']
        if label in processed:
            continue
        if label not in img_map:
            continue

        score, raw = rate_competence(img_map[label], model_folder)

        result = {
            'candidate_label': label,
            'election_id': cand['Election ID'],
            'competence_score': score,               # None if refused
            'won_real': cand['Winner/Loser'] == 'Winner',
            'vote_share': cand['Vote Share'],
            'status': 'ok' if score is not None else raw[:40],
        }
        pd.DataFrame([result]).to_csv(
            output_csv, mode='a', index=False, header=not output_csv.exists())

@deprecated("Use election_collect_multitrial instead")
def check_recognition(image, model_folder, updated):
    """Ask if the model recognizes a candidate. Returns (recognized, raw)."""
    snapshot = MODEL_SNAPSHOTS[model_folder]
    prompt = UPDATED_RECOGNITION_PROMPT if updated else RECOGNITION_PROMPT
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url":
            {"url": f"data:image/jpeg;base64,{encode_image(image)}"}},
    ]
    try:
        resp = get_client().chat.completions.create(
            model=snapshot, messages=[{"role": "user", "content": content}],
            temperature=0.1, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # crude auto-flag: did it claim to recognize? (you'll verify manually too)
        recognized = 'unknown' not in raw.lower()[:20]
        return recognized, raw
    except Exception as e:
        return None, f"ERROR: {e}"

@deprecated("Use election_collect_multitrial instead")
def collect_recognition(usable_df, model_folder, img_map, updated=True):
    """Collect recognition responses for each candidate, resume-safe."""
    label = "recognition_updated" if updated else "recognition"
    output_csv = get_filepath(model_folder=model_folder, prompt_label=label)

    processed = set()
    if output_csv.exists():
        processed = set(pd.read_csv(output_csv)['candidate_label'])

    for _, cand in usable_df.iterrows():
        label = cand['Full Label']
        if label in processed:
            continue
        if label not in img_map:
            continue

        recognized, raw = check_recognition(img_map[label], model_folder, updated=updated)

        result = {
            'candidate_label': label,
            'election_id': cand['Election ID'],
            'recognized_flag': recognized,     # crude auto-guess (verify manually)
            'raw_response': raw,               # keep full text for manual review
        }
        pd.DataFrame([result]).to_csv(
            output_csv, mode='a', index=False, header=not output_csv.exists())

@deprecated("Use election_collect_multitrial instead")
def main(pilot=False):
    # --- build label -> filepath map, deduping (prefer non-timestamped copy) ---
    all_imgs = list(SENATE_PATH.glob("*.jpg")) + list(GOVERNOR_PATH.glob("*.jpg"))
    img_map = {}
    for p in all_imgs:
        label = clean_label(p)
        if label not in img_map or len(p.name) < len(img_map[label].name):
            img_map[label] = p   # shorter filename = the clean (non-timestamped) one

    raw, complete = clean_election_df(ELECTION_RESULTS_PATH)   
    
    complete['image_path'] = complete['Full Label'].astype(str).map(img_map)
    complete = complete[complete['image_path'].notna()].copy() #checking to make sure it has an img

    for model, model_folder in ELECTION_MODELS.items():
        print(f"\n=== {model_folder} ===")
        #collect_competence_ratings(complete, model_folder, img_map, pilot=pilot)
        collect_election_choices(complete, model_folder, img_map, question=ELECTION_PROMPT, pilot=pilot)
'''     