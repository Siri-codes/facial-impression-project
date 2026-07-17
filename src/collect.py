import pandas as pd
from pathlib import Path
import os
import json
from openai import OpenAI
import base64

from config import ATTRIBUTES, MODEL_DIR, HUMAN_MEANS, HUMAN_RATINGS, IMAGE_DIR, TEMPERATURE, TOKEN_DIR, OPENROUTER_BASE_URL, REP_SUBSET_SIZE, MODELS, MODEL_SNAPSHOTS, SUBSAMPLE_SEED
from prompts import PROMPTS
from data_io import load_human_means

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

def model_predictions(image_path, system_prompt, model_name):
    """
    Gathers model predictions for a single image.
    """

    base64_image = encode_image(image_path)

    instruction = f"""
    CRITICAL REQUIRED FORMAT: Return ONLY a raw JSON object with exactly these 34 keys,
    each mapped to an integer rating from 0-100: {', '.join(ATTRIBUTES)}
    Do not include conversational phrases, explanations, or markdown code block formatting.
    Example shape: {{"trustworthy": 72, "attractive": 55, ...}}
    """

    raw_output = None
    try:
        response = get_client().chat.completions.create(
            model=model_name,  # e.g. "google/gemini-2.5-flash-lite"
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        { "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }}
                    ],
                },
            ],
            temperature=TEMPERATURE
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # Strip code block decorators if the model drops them in anyway
        if raw_output.startswith("```json"):
            raw_output = raw_output.removeprefix("```json").removesuffix("```")
        elif raw_output.startswith("```"):
            raw_output = raw_output.removeprefix("```").removesuffix("```")

        #replacing `output = json.loads(raw_output)`:
        output = {_norm(k): v for k, v in json.loads(raw_output).items()}

        missing = [c for c in ATTRIBUTES if c not in output]
        if missing:
            raise KeyError(f"missing after normalization: {missing}")

        ratings = [output[col] for col in ATTRIBUTES]
        
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
       
        return ratings, prompt_tokens, completion_tokens, response.model  # ratings should be a list matching ATTRIBUTES order
    
    except Exception as e:
        print(f"API failure for {image_path}: {e}")
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

def collect_data(input_df, system_prompt, model_name, prompt_label, image_dir=Path(IMAGE_DIR)):
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
        ratings, prompt_tokens, completion_tokens, resolved_model = model_predictions(image_path, system_prompt, model_name)

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

def main(prompt_labels, pilot=True):
    for label in prompt_labels:
        if label not in PROMPTS:
            raise ValueError(f"unknown prompt: {label!r}. Available: {sorted(PROMPTS)}")
            
    df = load_human_means() # validated loader, not raw read_csv
    if pilot:
        df = df.sample(n=REP_SUBSET_SIZE, random_state=SUBSAMPLE_SEED)

    labels = prompt_labels 
    suffix = "pilot" if pilot else "main"

    for label in labels:
        for folder in MODELS.values():
            print(f"\n=== {folder} / {label}_{suffix} ===")
            collect_data(df, PROMPTS[label], MODEL_SNAPSHOTS[folder],
                         f"{label}_{suffix}")

if __name__ == "__main__":
    main(["direct", "predict_human"], pilot=True)