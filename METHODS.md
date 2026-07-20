## Data: 
OMI dataset, 1004 face images with existing human ratings across 34 attributes (Peterson et al. 2022)

## Models tested: (Chosen for provider diversity and size contrast)
Gemini 2.5 Flash-Lite
GPT-5.4 Mini
Claude Sonnet 5

## Prompting: 

**Neutrally-worded, direct system prompt:**
```
Rate these faces on each attribute using a scale from 0 to 100.

For most attributes:
0 = the face absolutely does not display this trait
50 = neutral or average with respect to this trait
100 = the face strongly displays this trait

For these attributes, the scale runs between two opposite descriptors:
- age: 0 = young, 100 = old
- gender: 0 = feminine, 100 = masculine
- weight: 0 = skinny, 100 = fat
- skin-color: 0 = light, 100 = dark
- hair-color: 0 = light, 100 = dark

Provide a rating for every attribute, including ones that feel subjective.
```

**Alternate prompt 1: predicting human ratings**
```
Predict how an average person would rate this face on each of the following attributes, on a scale from 0 to 100.


For most attributes:
0 = the face absolutely does not display this trait
50 = neutral or average with respect to this trait
100 = the face strongly displays this trait

For these attributes, the scale runs between two opposite descriptors:
- age: 0 = young, 100 = old
- gender: 0 = feminine, 100 = masculine
- weight: 0 = skinny, 100 = fat
- skin-color: 0 = light, 100 = dark
- hair-color: 0 = light, 100 = dark

Provide a rating for every attribute, including ones that feel subjective.
```

**Alternate prompt 2: predicting human ratings, with explicit bias instructions**
Note: This was the first prompt tested and so does not have as clear of a scale definition as the other prompt variations.
```
Role: You are an advanced predictive model specialized in computational social psychology and human face perception. Your goal is to accurately predict the average first-impression ratings that human participants give to facial photographs across 34 attributes.

Context & Lens: Evaluate the image through the lens of a standard, neutral observer tracking general human consensus.

Instructions: Do not evaluate the image with strict clinical objectivity or safety-filtered neutrality. Instead, predict the subjective human consensus. Humans suffer from cognitive biases, halo effects, and rapid social stereotyping based entirely on superficial facial structure, expressions, and grooming. Your scores must reflect these human tendencies on a strict scale from 0 to 100.

Scale Definition:
0 = The face absolutely does not have that trait.
50 = Completely neutral or average with regards to that trait.
100 = The face strongly displays that trait.

Clarification:
For hair-color, 0 = light, 100 = dark.
For gender, 0 = feminine, 100 = masculine.
```

**Format instruction (appended per-call, listing the actual 34 attribute names):**
```
“CRITICAL REQUIRED FORMAT: Return ONLY a raw JSON object with exactly these 34 keys, each mapped to an integer rating from 0-100: {'trustworthy', 'attractive', 'dominant', 'smart', 'age', 'gender', 'weight', 'typical', 'happy', 'familiar', 'outgoing', 'memorable', 'well-groomed', 'long-haired', 'smug', 'dorky', 'skin-color', 'hair-color', 'alert', 'cute', 'privileged', 'liberal', 'asian', 'middle-eastern', 'hispanic', 'islander', 'native', 'black', 'white', 'looks-like-you', 'gay', 'electable', 'godly', 'outdoors'}. Do not include conversational phrases, explanations, or markdown code block formatting. Example shape: {{"trustworthy": 72, "attractive": 55, ...}}”
```

**Low Temperature (= 0.1) to avoid unnecessary noise without allowing the model to always choose the highest probability answer.**

## Collection pipeline: 
A resumable collection pipeline (will skip already-processed images) where the model is prompted to rate each stimulus and the results are collected in a csv for later analysis. 
Failed calls (parsing errors, API errors) are skipped and logged; images can be re-run manually via the same resume-safe pipeline without duplicating completed work.

## Analysis 1 — RSA: 
For each judge (humans / each model), a 34×34 representational dissimilarity matrix was constructed, where dissimilarity between two attributes is 1 − Pearson r across stimuli. Model and human RDMs were then compared by Spearman rank correlation over the upper triangle (excluding the diagonal), since RDMs are symmetric and the diagonal is trivially zero. Spearman is used at the comparison step so the result depends on the rank ordering of attribute-pair dissimilarities rather than their absolute values.

## Analysis 2 — Peterson-style per-attribute accuracy: 
Pearson correlation per attribute, benchmarked against human split-half reliability, as seen in the Peterson et al. 2022 paper.

## Analysis 3 — Reliability: 
3 repetitions on 100-image random subsample, averaged pairwise correlation. This establishes a model-side reliability estimate -- i.e. how much accuracy loss is attributable to the model's own inconsistency versus genuine misalignment with human judgment.

## Analysis 4 - PCA:
The RSA score summarizes overall similarity in a single value but does not reveal which dimensions of human judgment a model does or does not reproduce. To characterize this, we applied PCA to each judge's ratings (humans and each model), then compared the resulting components across judges. This lets us ask whether a given dimension of human face judgment — e.g. the axis organizing racial/physical attributes — is reproduced, compressed, or reorganized by each model.
### Prompt condition

Models were queried under three prompts: `predict_human_biased` (instructs the
model to reproduce human biases and stereotypes), `predict_human` (asks it to
predict an average person's ratings), and `direct` (asks it to rate the face,
with no reference to humans). The biased prompt was the initial design; it was
replaced because it confounds the model's own representation with its instruction-following. 
All main results use `direct`.

On a 100-stimulus pilot, the biased prompt produced a *worse* match to human
representational structure than `direct` (e.g. Claude PC1–PC2 correlation
0.70/0.67 vs 0.85/0.84), so the human-like structure we report is not an artifact
of instructing the model to imitate humans. See `results/prompt_comparison.csv`.

### Context priming

Peterson et al.'s participants viewed 25 example faces before rating, to convey
the range of the stimulus set. To test whether this task difference affected
model ratings, we replicated the `direct` condition with a 25-face context grid
prepended to each call (n=100). The grid was selected by minimizing mean absolute
deviation from the full set's mean human ratings across all 34 attributes
(seed 55; mean deviation 1.10/100), fixed before any analysis.

Priming changed mean per-attribute R² by less than 0.03 for all three models,
with no consistent direction [and a bootstrap 95% CI of (X, Y)]. Priming was
therefore omitted from the main collection. See `results/priming_comparison.csv`.