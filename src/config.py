"""Central configuration. Constant values only."""
import os
from pathlib import Path

# ---- Paths ----
# Machine-specific — env-overridable
ROOT      = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
IMAGE_DIR = Path(os.environ.get("OMI_IMAGE_DIR", ROOT / "data/omi-main-dataset/images"))

# Project-specific — derived, never env
DATA_DIR  = ROOT / "data"
HUMAN_DIR = DATA_DIR / "human_ratings"
MODEL_DIR = DATA_DIR / "model_ratings"
TOKEN_DIR = DATA_DIR / "tokens"
RESULTS   = ROOT / "results"

HUMAN_MEANS   = HUMAN_DIR / "attribute_means.csv"
HUMAN_RATINGS = HUMAN_DIR / "attribute_ratings.zip"

# ---- Attribute order (enforced on every load) ----
ATTRIBUTES = [
    'trustworthy', 'attractive', 'dominant', 'smart', 'age', 'gender', 'weight',
    'typical', 'happy', 'familiar', 'outgoing', 'memorable', 'well-groomed',
    'long-haired', 'smug', 'dorky', 'skin-color', 'hair-color', 'alert', 'cute',
    'privileged', 'liberal', 'asian', 'middle-eastern', 'hispanic', 'islander',
    'native', 'black', 'white', 'looks-like-you', 'gay', 'electable', 'godly',
    'outdoors'
]

#Labels for interpretation, not inputed into analysis
ATTRIBUTE_GROUPS = {
    'demographic': ['asian', 'white', 'black', 'hispanic', 'middle-eastern', 'islander', 'native', 'gender', 'age'],
    'physical':    ['skin-color', 'hair-color', 'long-haired', 'weight', 'attractive', 'well-groomed'],
    'trait':       ['trustworthy', 'dominant', 'smart', 'happy', 'outgoing', 'smug', 'dorky', 'alert', 'cute', 'privileged', 'liberal', 'gay', 'electable', 'godly'],
    'other':       ['typical', 'familiar', 'memorable', 'looks-like-you', 'outdoors'],
}

GROUP_COLORS = {'demographic': '#d62728', 'physical': '#ff7f0e',
                'trait': '#1f77b4', 'other': '#7f7f7f'}

# ---- Models: label -> folder name ----
ALL_MODEL_NAMES = ['Gemini 2.5 Flash-Lite', 'GPT-5.4 Mini','Claude Sonnet 5']

MODELS = {
    'Gemini 2.5 Flash-Lite': 'google_gemini-2.5-flash-lite',
    'GPT-5.4 Mini':          'openai_gpt-5.4-mini',
    'Claude Sonnet 5':       'anthropic_claude-sonnet-5',
}

# Exact snapshots queried via OpenRouter
MODEL_SNAPSHOTS = {
    'google_gemini-2.5-flash-lite': 'google/gemini-2.5-flash-lite', 
    'openai_gpt-5.4-mini':          'openai/gpt-5.4-mini',
    'anthropic_claude-sonnet-5':    'anthropic/claude-sonnet-5',
}

# OpenRouter base url:
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---- Collection parameters ----
TEMPERATURE = 0.1
N_REPS = 3
REP_SUBSET_SIZE = 100     # reps 2-3 use a 100-image subset; rep 1 is full
SUBSAMPLE_SEED   = 42     # the 100-stimulus subsample: reps 2-3 and all pilots

# ---- Priming/Context ----
CONTEXT_GRID = DATA_DIR / "context_grid.jpg"
CONTEXT_SEED = 55 # distinct from SUBSAMPLE_SEED; meant to mimic distribution of full dataset 

# ---- Analysis parameters ----
SEED = 0
N_STIM = None             # None = full set. Set to 20 ONLY for quick tests.
N_SPLITS = 100            # split-half reliability iterations
N_PERMUTATIONS = 10000   # RSA permutation test
SPEARMAN_BROWN = True     # report SB-corrected ceiling