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
FIGURES = RESULTS / "figures"

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
MODELS = {
    #Frontier: (80-85% MMMU-Pro)
    'Claude Opus 5' : 'anthropic_claude-opus-5',
    'Gemini 3.5 Flash' : 'google_gemini-3.5-flash',
    'GPT-5.6 Sol': 'openai_gpt-5.6-sol', #
    'Grok 4.5': 'xai_grok-4.5',
    
    #High: (71–79%)
    'Qwen3.6 Plus': 'qwen_qwen3.6-plus', #
    'Claude Sonnet 5 High':       'anthropic_claude-sonnet-5-high',
    'Claude Sonnet 5 Max':       'anthropic_claude-sonnet-5-max',
    'Gemini 3.1 Flash Lite': 'google_gemini-3.1-flash-lite',
    'GPT-5.6 Luna': 'openai_gpt-5.6-luna', #
    'Grok 4.3': 'xai_grok-4.3', #
    
    #Upper-Mid: (61–70%):
    'Gemma 4 26B A4B': 'google_gemma-4-26b-a4b',
    'Qwen3 VL 32B Instruct': 'qwen_qwen3-vl-32b-instruct', #
    'Claude Sonnet 4': 'anthropic_claude-sonnet-4',#
    
    #Middle: (52–60%)
    'GPT-5.4 Mini':          'openai_gpt-5.4-mini',
    'Mistral Large 3': 'mistralai_mistral-large-3', #
    'Gemini 2.5 Flash-Lite': 'google_gemini-2.5-flash-lite',
    
    #Low: (40–50%)
    'Qwen3 VL 8B Instruct': 'qwen_qwen3-vl-8b-instruct', #
    'GPT-4o Mini': 'openai_gpt-4o-mini',
    
    #Floor: (26–39%):
    'Ministral 3 3B': 'mistralai_ministral-3-3b',       
    'Claude 3 Haiku': 'anthropic_claude-3-haiku'
}

# Exact snapshots queried via OpenRouter
MODEL_SNAPSHOTS = {
    'anthropic_claude-opus-5' : 'anthropic/claude-opus-5',
    'google_gemini-3.5-flash' : 'google/gemini-3.5-flash',
    'openai_gpt-5.6-sol': 'openai/gpt-5.6-sol',
    'xai_grok-4.5': 'x-ai/grok-4.5',

    'qwen_qwen3.6-plus' : 'qwen/qwen3.6-plus',
    'anthropic_claude-sonnet-5-high':    'anthropic/claude-sonnet-5',
    'anthropic_claude-sonnet-5-max': 'anthropic/claude-sonnet-5',
    'google_gemini-3.1-flash-lite': 'google/gemini-3.1-flash-lite',
    'openai_gpt-5.6-luna': 'openai/gpt-5.6-luna',
    'xai_grok-4.3': 'x-ai/grok-4.3',

    'google_gemma-4-26b-a4b' : 'google/gemma-4-26b-a4b-it:free',
    'qwen_qwen3-vl-32b-instruct': 'qwen/qwen3-vl-32b-instruct',
    'anthropic_claude-sonnet-4': 'anthropic/claude-sonnet-4',

    'openai_gpt-5.4-mini':          'openai/gpt-5.4-mini',
    'mistralai_mistral-large-3': 'mistralai/mistral-large-2512',
    'google_gemini-2.5-flash-lite': 'google/gemini-2.5-flash-lite',
    
    'qwen_qwen3-vl-8b-instruct': 'qwen/qwen3-vl-8b-instruct',
    'openai_gpt-4o-mini' : 'openai/gpt-4o-mini',

    'mistralai_ministral-3-3b' : 'mistralai/ministral-3b-2512',
    'anthropic_claude-3-haiku': 'anthropic/claude-3-haiku'
}

# effort level per model, chosen to match the Artificial Analysis MMMU-Pro config
REASONING_EFFORT = {
    "anthropic_claude-opus-5":       "high",
    "google_gemini-3.5-flash":       "medium",
    "openai_gpt-5.6-sol":            "medium",
    "xai_grok-4.5":                 "high",

    "qwen_qwen3.6-plus":              "medium",
    "anthropic_claude-sonnet-5-high":     "high", 
    'anthropic_claude-sonnet-5-max': "max",
    'google_gemini-3.1-flash-lite': "medium",
    "openai_gpt-5.6-luna":                   "low",
    "xai_grok-4.3":                  "low",
    
    "google_gemma-4-26b-a4b":         "none",
    "qwen_qwen3-vl-32b-instruct":     "none",
    "anthropic_claude-sonnet-4":       "high", 
    
    "openai_gpt-5.4-mini": "none", #confirmed empirically that this was the default
    "mistralai_mistral-large-3": "none",
    "google_gemini-2.5-flash-lite" : "none",
    
    "qwen_qwen3-vl-8b-instruct": "none",
    "openai_gpt-4o-mini": "none",

    "mistralai_ministral-3-3b": "none",
    "anthropic_claude-3-haiku":      "none"  # non-reasoning
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