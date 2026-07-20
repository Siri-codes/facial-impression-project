import pandas as pd
from pathlib import Path
import os
import json
from openai import OpenAI
import base64

from config import ATTRIBUTES, MODEL_DIR, HUMAN_MEANS, HUMAN_RATINGS, IMAGE_DIR, TEMPERATURE, TOKEN_DIR, OPENROUTER_BASE_URL, REP_SUBSET_SIZE, MODELS, MODEL_SNAPSHOTS, SUBSAMPLE_SEED
from prompts import PROMPTS, INSTRUCTION
from data_io import load_human_means
from context import context_ids, make_context_grid

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _client

def encode_image(image_path):
    """Reads a local file and encodes its raw bytes to a text string."""
    with open(image_path, "rb") as image_file:
        raw_bytes = image_file.read()
        encoded_string = base64.b64encode(raw_bytes).decode("utf-8")
        return encoded_string

def _norm(k):
    return k.strip().lower().replace("_", "-").replace(" ", "-")


def model_predictions(image_path, system_prompt, model_name, context_path=None):
    """Gathers model predictions for a single image."""
    raw_output = None
    try:
        content = []
        if context_path:
            content += [
                {"type": "text", "text":
                    "Below is a grid of 25 example faces sampled from the same set, "
                    "shown to give a sense of the range of faces you will encounter. "
                    "Do not rate these examples."},
                {"type": "image_url", "image_url":
                    {"url": f"data:image/jpeg;base64,{encode_image(context_path)}"}},
                {"type": "text", "text": "Now rate the following face."},
            ]

        content += [
            {"type": "text", "text": INSTRUCTION},
            {"type": "image_url", "image_url":
                {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
        ]

        response = get_client().chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": content}],
            temperature=TEMPERATURE)

        raw_output = response.choices[0].message.content.strip()

        if raw_output.startswith("```json"):
            raw_output = raw_output.removeprefix("```json").removesuffix("```")
        elif raw_output.startswith("```"):
            raw_output = raw_output.removeprefix("```").removesuffix("```")

        output = {_norm(k): v for k, v in json.loads(raw_output).items()}

        missing = [c for c in ATTRIBUTES if c not in output]
        if missing:
            raise KeyError(f"missing after normalization: {missing}")

        ratings = [output[col] for col in ATTRIBUTES]

        return (ratings,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.model)

    except Exception as e:
        print(f"Failure for {image_path}: {e}")
        print(f"Raw output: {raw_output}")
        return None, 0, 0, None

def get_filepath(model_name, prompt_label, base_output_dir=MODEL_DIR):
    """
    Creates necessary directories and returns Path object.
    """
    # Create safe string names for filesystems (no spaces or slashes)
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    safe_prompt = prompt_label.lower().strip().replace(" ", "_")
    
    # Define and create the directory path object
    directory_target = Path(base_output_dir) / safe_model
    directory_target.mkdir(parents=True, exist_ok=True) # Automatically checks and creates folder
 
    name = directory_target / safe_prompt
    final_file_path = f"{name}.csv"
    
    return final_file_path

def collect_data(input_df, system_prompt, model_name, prompt_label, image_dir=Path(IMAGE_DIR), context_path=None):
    '''
    Collects data using the specified LLM engine and prompt strategy, saving results to hierarchical resume-safe file structure.
    '''
    
    safe_model = model_name.replace("/", "_").replace(" ", "_")

    # Compute the path so prompt data goes to separate CSVs
    output_csv = Path(get_filepath(model_name, prompt_label))

    # Initialize empty set to keep track of already processed stimuli.
    processed_stimuli = set()

    if output_csv.exists():
        print(f"Found existing file: {output_csv}. Continuing from where we left off.")

        # 1. Read existing CSV into a temporary dataframe
        existing_df = pd.read_csv(output_csv)

        # 2. Extract the 'stimulus' column from existing_df, convert it to a set of integers,
        # and update 'processed_stimuli' set so know what to skip.
        processed_stimuli = set(existing_df['stimulus'].astype(int))

        print(f"Resuming previous run. Skipping {len(processed_stimuli)} already processed images.")

    else:
        print(f"No existing progress file found. Starting fresh run for {model_name}, {prompt_label}.")

    #start looping through the human ratings csv:
    for idx, row in input_df.iterrows():
        # 1. Extract the stimulus ID from the current row
        stim_id = int(row['stimulus'])

        # Skip if we've already processed this stimulus
        if stim_id in processed_stimuli:
            continue

        # 2. Reconstruct the full path to the image
        image_path = image_dir / f"{stim_id}.jpg"

        print(f"Processing stimulus {stim_id}...")

        # 3. Get ratings for this image
        ratings, prompt_tokens, completion_tokens, resolved_model = model_predictions(image_path, system_prompt, model_name, context_path)

        # If the API failed and returned None values, skip saving
        if ratings is None:
            fail_path = Path(MODEL_DIR) / safe_model / f"{prompt_label}_failures.csv"
            pd.DataFrame([{'stimulus': stim_id, 'reason': 'api_or_parse_failure'}]).to_csv(
                fail_path, mode='a', index=False, header=not fail_path.exists()
            )
            print(f"Skipping save for {stim_id} due to API failure.")
            continue

        # 4. Package results into dictionary
        result_dict = {'stimulus': stim_id} #stimulus column
        for col_name, score in zip(ATTRIBUTES, ratings):
            result_dict[col_name] = score #map rating to correct column

        # Convert dictionary into 1-row DataFrame
        single_row_df = pd.DataFrame([result_dict])

        #5. Update CSV with results
        # Determine if we need to write the header string:
        write_header = not output_csv.exists()
        
        single_row_df.to_csv(
            output_csv, 
            mode='a', 
            index=False, 
            header=write_header
        )
        
        # Store token data
        token_path = Path(TOKEN_DIR)
        token_path.mkdir(parents=True, exist_ok=True)
        token_data = {
            'stimulus': stim_id,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'resolved_model': resolved_model,     
            'timestamp': pd.Timestamp.now().isoformat(),   
        }

        token_header = not (token_path / f"{safe_model}_{prompt_label}_tokens.csv").exists()
        token_df = pd.DataFrame([token_data])
        token_df.to_csv(token_path / f"{safe_model}_{prompt_label}_tokens.csv", mode='a', index=False, header=token_header)

        # Update runtime checklist so don't duplicate efforts
        processed_stimuli.add(stim_id)

def main(prompt_labels, pilot=True, primed=False):
    """
    Collect model ratings.

    prompt_labels: list of keys into PROMPTS, e.g. ["direct"]
    pilot:  True  -> 100-stimulus subsample (SUBSAMPLE_SEED)
            False -> all stimuli (costs real money)
    primed: True  -> prepend a fixed 25-face context grid to every call.
                     Context faces are never rated as targets.
    """
    for label in prompt_labels:
        if label not in PROMPTS:
            raise ValueError(f"unknown prompt: {label!r}. Available: {sorted(PROMPTS)}")

    human = load_human_means()
    pilot_ids = human.sample(n=REP_SUBSET_SIZE, random_state=SUBSAMPLE_SEED)['stimulus']

    ctx, ctx_ids = None, []
    if primed:
        # Exclude the pilot 100 always, so the grid is identical
        # across pilot and main runs.
        ctx_ids = context_ids(human['stimulus'], exclude=pilot_ids)
        ctx = make_context_grid(ctx_ids)
        print("context faces:", list(ctx_ids))

    df = human[human['stimulus'].isin(pilot_ids)] if pilot else human
    df = df[~df['stimulus'].isin(ctx_ids)]      # never rate a context face
    print(f"targets: {len(df)}")

    suffix = ("primed_" if primed else "") + ("pilot" if pilot else "main")

    for label in prompt_labels:
        for folder in MODELS.values():
            print(f"\n=== {folder} / {label}_{suffix} ===")
            collect_data(df, PROMPTS[label], MODEL_SNAPSHOTS[folder],
                         f"{label}_{suffix}", context_path=ctx)


if __name__ == "__main__":
    main(["direct"], pilot=True, primed=True)