## Data: 
OMI dataset, 1004 face images with existing human ratings across 34 attributes (Peterson et al. 2022)

## Models tested: (Chosen for provider diversity and size contrast)
Gemini 2.5 Flash-Lite
GPT-5.4 Mini
Claude Sonnet 5

## Prompting: 

** Neutrally-worded system prompt: **
```
“Role: You are an advanced predictive model specialized in computational social psychology and human face perception. Your goal is to accurately predict the average first-impression ratings that human participants give to facial photographs across 34 attributes. 

Context & Lens: Evaluate the image through the lens of a standard, neutral observer tracking general human consensus.

Instructions: Do not evaluate the image with strict clinical objectivity or safety-filtered neutrality. Instead, predict the subjective human consensus. Humans suffer from cognitive biases, halo effects, and rapid social stereotyping based entirely on superficial facial structure, expressions, and grooming. Your scores must reflect these human tendencies on a strict scale from 0 to 100.

Scale Definition:
0 = The face absolutely does not have that trait.
50 = Completely neutral or average with regards to that trait.
100 = The face strongly displays that trait.

Clarification:
For hair-color, 0 = light, 100 = dark.
For gender, 0 = feminine, 100 = masculine.”
​```

**Format instruction (appended per-call, listing the actual 34 attribute names):**
```
“CRITICAL REQUIRED FORMAT: Return ONLY a raw JSON object with exactly these 34 keys, each mapped to an integer rating from 0-100: {'trustworthy', 'attractive', 'dominant', 'smart', 'age', 'gender', 'weight', 'typical', 'happy', 'familiar', 'outgoing', 'memorable', 'well-groomed', 'long-haired', 'smug', 'dorky', 'skin-color', 'hair-color', 'alert', 'cute', 'privileged', 'liberal', 'asian', 'middle-eastern', 'hispanic', 'islander', 'native', 'black', 'white', 'looks-like-you', 'gay', 'electable', 'godly', 'outdoors'}. Do not include conversational phrases, explanations, or markdown code block formatting. Example shape: {{"trustworthy": 72, "attractive": 55, ...}}”
```

** Low Temperature (= 0.1) to avoid unnecessary noise without allowing the model to always choose the highest probability answer. **

## Collection pipeline: 
A resumable collection pipeline (will skip already-processed images) where the model is prompted to rate each stimulus and the results are collected in a csv for later analysis. 
Failed calls (parsing errors, API errors) are skipped and logged; images can be re-run manually via the same resume-safe pipeline without duplicating completed work.

## Analysis 1 — RSA: 
An RDM shows the dissimilarity between each pair of attributes. In this case, Spearman correlation was used rather than Pearson correlation to allow for nonlinear relationships in the data. Only the upper triangle of the RDM was used for comparison as RDMs are symmetrical and all attributes should have 100% correlation to themselves. 

## Analysis 2 — Peterson-style per-attribute accuracy: 
Pearson correlation per attribute, benchmarked against human split-half reliability, as seen in the Peterson et al. 2022 paper.

## Analysis 3 — Reliability: 
3 repetitions on 100-image random subsample, averaged pairwise correlation. This establishes a model-side noise ceiling analogous to the human split-half reliability used in Analysis 2—i.e., how much accuracy loss is attributable to the model's own inconsistency versus genuine misalignment with human judgment.
