## Folder Structure

- `human_ratings/` — human attribute ratings from the OMI dataset (see below)
- `anthropic_claude-sonnet-5/`, `google_gemini-2.5-flash-lite/`, `openai_gpt-5.4-mini/` — one folder per model, each containing that model's ratings
- `tokens/` — per-call token usage logs (prompt/completion token counts), used to verify actual API costs against estimates

## Model Ratings Data

Each row in a CSV contains the index of the image (`stimulus`) and the model's ratings of that image on 34 attributes.

- `neutral.csv` — ratings for all 1004 images in the OMI dataset (main dataset, used in Analyses 1 and 2 — see methods write-up)
- `neutral_rep2.csv`, `neutral_rep3.csv` — ratings for the same 100-image random subset, repeated 3 times total (including `neutral.csv`'s matching subset) to estimate model self-reliability (Analysis 3)

## Images

Face images are not included in this repository due to size and licensing. Download from the OMI dataset source (see citation below) and place in `data/images/` to reproduce the pipeline.

## Reproducing the Notebooks

Notebooks currently reference Deepnote-specific paths (e.g. `/work/outputs/...`). Update these to local `data/` paths before running outside Deepnote.

## Data Sources & License

Human ratings (`human_ratings/attribute_means.csv`) are sourced from the One Million Impressions (OMI) dataset:

Peterson, J. C., Uddenberg, S., Griffiths, T., Todorov, A., & Suchow, J. W. (2022). Deep models of superficial face judgments. PNAS.

The OMI dataset is licensed under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/). This repository's use of that data — and the derived materials in this repo — follow the same license terms: non-commercial use, attribution, and share-alike.

No modifications were made to the original human ratings data.