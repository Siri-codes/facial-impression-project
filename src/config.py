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

PARSER_VAL_DIR = DATA_DIR / "parser_validation"

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
   #'GPT-5.6 Sol': 'openai_gpt-5.6-sol', #
    'Grok 4.5': 'xai_grok-4.5',
    
    #High: (71–79%)
   #'Qwen3.6 Plus': 'qwen_qwen3.6-plus', # Strange error: flags certain images as "inappropriate content"
    'Gemini 3.1 Flash Lite': 'google_gemini-3.1-flash-lite',
   #'GPT-5.6 Luna': 'openai_gpt-5.6-luna', #
    'Grok 4.3': 'xai_grok-4.3', #
    'Claude Sonnet 5':       'anthropic_claude-sonnet-5-none',
    
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
    'google_gemini-3.1-flash-lite': 'google/gemini-3.1-flash-lite',
    'openai_gpt-5.6-luna': 'openai/gpt-5.6-luna',
    'xai_grok-4.3': 'x-ai/grok-4.3',
    'anthropic_claude-sonnet-5-none': 'anthropic/claude-sonnet-5',

    'google_gemma-4-26b-a4b' : 'google/gemma-4-26b-a4b-it:free',
    'qwen_qwen3-vl-32b-instruct': 'qwen/qwen3-vl-32b-instruct',
    'anthropic_claude-sonnet-4': 'anthropic/claude-sonnet-4',

    'openai_gpt-5.4-mini':          'openai/gpt-5.4-mini',
    'mistralai_mistral-large-3': 'mistralai/mistral-large-2512',
    'google_gemini-2.5-flash-lite': 'google/gemini-2.5-flash-lite',
    
    'qwen_qwen3-vl-8b-instruct': 'qwen/qwen3-vl-8b-instruct',
    'openai_gpt-4o-mini' : 'openai/gpt-4o-mini',

    'mistralai_ministral-3-3b' : 'mistralai/ministral-3b-2512',
    'anthropic_claude-3-haiku': 'anthropic/claude-3-haiku',

    'qwen_qwen3-vl-235b-instruct': 'qwen/qwen3-vl-235b-a22b-instruct',
    'mistralai_mistral-small-4': 'mistralai/mistral-small-2603',
    'meta-llama_llama-4-scout': 'meta-llama/llama-4-scout',
    'z-ai_glm-4.5v': 'z-ai/glm-4.5v',
    'moonshot_kimi-k2.5': 'moonshotai/kimi-k2.5', 
    'amazon_nova-2-lite': 'amazon/nova-2-lite-v1',
}

# effort level per model, chosen to match the Artificial Analysis MMMU-Pro config
REASONING_EFFORT = {
    "anthropic_claude-opus-5":       "high",
    "google_gemini-3.5-flash":       "medium",
    "openai_gpt-5.6-sol":            "medium",
    "xai_grok-4.5":                 "high",

    "qwen_qwen3.6-plus":              "medium",
    "google_gemini-3.1-flash-lite": "medium",
    "openai_gpt-5.6-luna":                   "low",
    "xai_grok-4.3":                  "low",
    "anthropic_claude-sonnet-5-none": "none",
    
    "google_gemma-4-26b-a4b":         "none",
    "qwen_qwen3-vl-32b-instruct":     "none",
    "anthropic_claude-sonnet-4":       "high", 
    
    "openai_gpt-5.4-mini": "none", #confirmed empirically that this was the default
    "mistralai_mistral-large-3": "none",
    "google_gemini-2.5-flash-lite" : "none",
    
    "qwen_qwen3-vl-8b-instruct": "none",
    "openai_gpt-4o-mini": "none",

    "mistralai_ministral-3-3b": "none",
    "anthropic_claude-3-haiku":      "none",  # non-reasoning

    "qwen_qwen3-vl-235b-instruct": "none",
    "mistralai_mistral-small-4": "none",
    "meta-llama_llama-4-scout": "none",
    "z-ai_glm-4.5v": "none",
    "moonshot_kimi-k2.5": "none",
    "amazon_nova-2-lite": "none",
}

#MMMU-Pro scores. Source: Artificial Analysis.
MMMU_PRO = {
    "Claude Opus 5":          84,
    "Gemini 3.5 Flash":       84,
    "Grok 4.5":               80,
    "Qwen3.6 Plus":           78,
    "Gemini 3.1 Flash Lite":  76,
    "Grok 4.3":               73,
    "Claude Sonnet 5":        72,   
    "Gemma 4 26B A4B":        67,
    "Qwen3 VL 32B Instruct":  64,
    "Claude Sonnet 4":        62,
    "GPT-5.4 Mini":           60,
    "Mistral Large 3":        56,
    "Gemini 2.5 Flash-Lite":  54,
    "Qwen3 VL 8B Instruct":   47,
    "GPT-4o Mini":            42,
    "Ministral 3 3B":         38,
    "Claude 3 Haiku":         31,

    'Qwen3 VL 235B A22B Instruct': 68,
    'Mistral Small 4': 46,
    'Llama 4 Scout': 53,
    'GLM 4.5V': 43,

    'Kimi K2.5': 73, 
    'Nova 2 Lite': 49,
}

#Humanity's Last Exam scores. Source: Artificial Analysis
HLE = {
    "Claude Opus 5":          52.8,
    "Gemini 3.5 Flash":       41.3,
    "Grok 4.5":               42.7,
    "Qwen3.6 Plus":           27.8,
    "Gemini 3.1 Flash Lite":  17.2,
    "Grok 4.3":               18.4,
    "Claude Sonnet 5":        19.0,   
    "Gemma 4 26B A4B":        11.5,
    "Qwen3 VL 32B Instruct": 6.8,
    "Claude Sonnet 4":        10.7,
    "GPT-5.4 Mini":           5.9,
    "Mistral Large 3":        4.2,
    "Gemini 2.5 Flash-Lite":  3.7,
    "Qwen3 VL 8B Instruct":    2.7,
    "GPT-4o Mini":            4.2,
    "Ministral 3 3B":         5.4,
    "Claude 3 Haiku":         4.1,
}

#For color-coding in plots:
PROVIDER = {
    'Claude Opus 5': 'Anthropic', 'Claude Sonnet 5': 'Anthropic', 'Claude Sonnet 4': 'Anthropic', 'Claude 3 Haiku': 'Anthropic',
    'GPT-5.6 Sol': 'OpenAI', 'GPT-5.6 Luna': 'OpenAI', 'GPT-5.4 Mini': 'OpenAI', 'GPT-4o Mini': 'OpenAI', 
    'Gemini 3.5 Flash': 'Google', 'Gemini 3.1 Flash Lite': 'Google', 'Gemini 2.5 Flash-Lite': 'Google', 'Gemma 4 26B A4B': 'Google',
    'Grok 4.5': 'xAI', 'Grok 4.3': 'xAI',
    'Qwen3.6 Plus': 'Qwen', 'Qwen3 VL 32B Instruct': 'Qwen', 'Qwen3 VL 8B Instruct': 'Qwen',
    'Mistral Large 3': 'Mistral', 'Ministral 3 3B': 'Mistral'
}
PROVIDER_COLORS = {'Anthropic': '#d97757', 'OpenAI': '#000000', 'Google': '#4285f4',
                   'xAI': '#1da1f2', 'Qwen': '#6b3fa0', 'Mistral': '#ff7000'}

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


# ---- Election Follow-Up Study ---
ELECTION_DATASET = DATA_DIR / "election-dataset"
SENATE_PATH   = ELECTION_DATASET / "Senate_all_stimuli"
GOVERNOR_PATH = ELECTION_DATASET / "Governors_all_stimuli"
ELECTION_RESULTS_PATH = ELECTION_DATASET / "election_results.csv"

#based on Todorov et. al. 2005: "competence, intelligence, leadership, honesty, trustworthiness, charisma, and likability"
ELECTION_ATTRIBUTES = ['competent', 'intelligent', 'leader', 'honest', 'trustworthy', 'charismatic', 'likable']

#highest performing matches on human "competence" axis
ELECTION_MODELS = {
    'Grok 4.5': 'xai_grok-4.5',
    'Gemini 3.1 Flash Lite': 'google_gemini-3.1-flash-lite',
    'Grok 4.3': 'xai_grok-4.3',
    'Claude Sonnet 5':       'anthropic_claude-sonnet-5-none',
    #'Claude Sonnet 4': 'anthropic_claude-sonnet-4', #not clean
    'Claude Opus 5' : 'anthropic_claude-opus-5',
    'Mistral Large 3': 'mistralai_mistral-large-3', #not clean
    #'Gemini 3.5 Flash' : 'google_gemini-3.5-flash', #not clean
    'GPT-5.4 Mini':          'openai_gpt-5.4-mini',
    'Ministral 3 3B': 'mistralai_ministral-3-3b',
    'Qwen3 VL 32B Instruct': 'qwen_qwen3-vl-32b-instruct',
    #'Gemma 4 26B A4B': 'google_gemma-4-26b-a4b', #currently erroring (rate limits?)
    'Gemini 2.5 Flash-Lite': 'google_gemini-2.5-flash-lite',
    'Qwen3 VL 8B Instruct': 'qwen_qwen3-vl-8b-instruct',
    #'Claude 3 Haiku': 'anthropic_claude-3-haiku',
    'Llama 4 Scout': 'meta-llama_llama-4-scout',
    'GLM 4.5V': 'z-ai_glm-4.5v',
    'Kimi K2.5': 'moonshot_kimi-k2.5',
    'Nova 2 Lite': 'amazon_nova-2-lite',
}