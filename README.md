DeepSeek Skill Hallucination Detection
---
This repository contains data pipelines, analysis tools and machine‑learning baselines for measuring hallucinations in language‑model outputs about software‑engineering skills.

###

Overview
---
The goal of the project is to detect and analyze hallucinations in answers produced by the DeepSeek language model when asked to list skills or extract skills from job descriptions. The pipeline compares model outputs with ground‑truth skills extracted from real job postings (via the SKILLAB dataset/APIs) and reports hallucination metrics.

###

Usage
---
A typical workflow to measure hallucinations is:
###
1. Prepare ground‑truth skills

 - Run data_export.py to download jobs and skills from SKILLAB.

 - Run skill_count.py to compute skill frequencies and produce an allowed skill list (choose a minimum count threshold).

2. Query the model

 - Write your questions in a text file under input/queries.

 - For unconstrained extraction, run unconstrained_queries_runner.py to call the local DeepSeek model and save answers to JSONL.

 - For constrained extraction (allowed skills), run constrained_queries_runner.py with the allowed skill list to restrict outputs.

 - To query with full job descriptions, run job_sample_prompt.py which returns JSON objects with skill_label, evidence and confidence.

3. Normalise predictions

 - Use skill_normalization.py to parse the model’s answers, strip <think> meta tokens, split bullet or comma lists, normalise skill names and save them into an Excel file with separate sheets for open, closed and job samples.

4. Fuzzy match and label

 - Run fuzzy_match.py to generate candidate matches between predicted skills and the allowed set using rapidfuzz.

 - Run hallucinations_labelling.py to assign high‑confidence non‑hallucination labels (H0) using exact matches, synonyms and fuzzy matches, and export low‑confidence skills for manual labelling.

5. Statistics and EDA

 - Use stats.py to summarise how many skills per query/job were predicted, the proportion of “no matching skills” responses and top‑10 skills.

 - Use hallucination_eda.py to analyse manual labels and distribution of hallucination types.

6. Train classifiers

 - Labelled datasets can be used to train baseline classifiers.

 - The scripts logistic_regression.py, mlp.py, random_forest.py and svm.py load labelled data, split into train/test sets, build a TF‑IDF vectoriser and train models. They report accuracy, macro‑averaged F1 scores and confusion matrices.

 - Random forest and logistic regression scripts also output top features or feature importance lists.

7. Notes

 - The project uses local deployment of DeepSeek via the Ollama server (BASE_URL = http://localhost:11434) for reproducibility. Adjust BASE_URL and MODEL_NAME if calling a different model or API.

 - All Excel outputs are written using openpyxl. Ensure the output/ directory exists or will be created by scripts.

- The code has been tested with Python 3.9 and requires packages such as pandas, scikit‑learn, openpyxl, rapidfuzz and imblearn.
