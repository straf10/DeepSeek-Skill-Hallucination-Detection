"""
Skill‑Comparison Framework
-------------------------------------------------------
• Vocabulary:
    – διαβάζεται από το output.xlsx (sheet "skills").
    – normalise = unidecode ▸ lower ▸ strip ▸ collapse‑spaces.
    – αφαιρούνται διπλότυπα.
    – επεκτείνεται με synonyms.yaml έτσι ώστε τα ακρώνυμα/εναλλακτικά labels
      να οδηγούν στο ίδιο canonical skill.
• Deepseek skills:
    – ίδιο lightweight normalise.
    – πριν το matching εφαρμόζεται synonyms mapping.
• Matching:
    – exact   (μετά τα synonyms).
    – fuzzy   (token_sort_ratio → 90, token_set_ratio → 85, partial_ratio → 80).
    – χωρίς lemmatisation, χωρίς “too_far”.

Αποθηκεύει αποτελέσματα σε «deepseek_skill_comparison.xlsx» και
εκτυπώνει KPI με report().
"""

import logging
import pathlib
import re
import yaml
import pandas as pd
from unidecode import unidecode
from rapidfuzz import process, fuzz

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
EXCEL_PATH = pathlib.Path(r"C:\Python\THESIS\skillab_job_fetcher\output\output.xlsx")
SHEET_NAME = "skills"
DEEPSEEK_CSV = pathlib.Path(r"C:\Python\THESIS\skillab_job_fetcher\output\ds_query_res_normalised.csv")
OUT_PATH    = pathlib.Path("C:\Python\THESIS\skillab_job_fetcher\output\query_skill_comparison.xlsx")
SYN_FILE    = pathlib.Path(r"C:\Python\THESIS\skillab_job_fetcher\synonyms.yaml")

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
_whitespace_re = re.compile(r"\s+")

def normalize(txt: str) -> str:
    """Simple normalisation: unidecode → lower → strip → collapse spaces."""
    if pd.isna(txt):
        return ""
    txt = unidecode(str(txt)).lower().strip()
    return _whitespace_re.sub(" ", txt)

# load synonyms once
syn_raw = yaml.safe_load(SYN_FILE.read_text()) if SYN_FILE.exists() else {}
SYN_MAP = {normalize(k): normalize(v) for k, v in syn_raw.items()}

def apply_synonyms(token_norm: str) -> str:
    return SYN_MAP.get(token_norm, token_norm)


SCORERS = [
    (fuzz.token_sort_ratio, 90),
    (fuzz.token_set_ratio, 85),
    (fuzz.partial_ratio,   80),
]

# ---------------------------------------------------------------------------
# 3. Build vocabulary (norm → canonical label)
# ---------------------------------------------------------------------------
logging.info("Loading vocabulary…")
df_gt = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df_gt["skill_norm"] = df_gt["skill_label"].apply(normalize)

# drop duplicates, keep first appearance
vocab: dict[str, str] = (
    df_gt.drop_duplicates("skill_norm", keep="first")
         .set_index("skill_norm")["skill_label"].to_dict()
)

# expand with synonyms (so acronyms hit exact)
for syn_norm, canon_norm in SYN_MAP.items():
    if canon_norm in vocab:
        vocab[syn_norm] = vocab[canon_norm]

logging.info("Vocabulary size: %d", len(vocab))

# ---------------------------------------------------------------------------
# 4. Load Deepseek skills
# ---------------------------------------------------------------------------
logging.info("Loading Deepseek skills…")
df_pred = pd.read_csv(DEEPSEEK_CSV)
if "skill" not in df_pred.columns:
    raise ValueError("Column 'skill' not found in ds_query_res_normalised.csv")

df_pred["skill_norm"] = df_pred["skill"].apply(normalize)

# ---------------------------------------------------------------------------
# 5. Matching logic (no lemmatiser, no too_far)
# ---------------------------------------------------------------------------

def match_skill(key_norm: str, vocab):
    key_norm = apply_synonyms(key_norm)
    if not key_norm:
        return "none", None

    # 1. Exact
    if key_norm in vocab:
        return "exact", vocab[key_norm]

    key_tokens = key_norm.split()
    if not key_tokens:                      # all spaces / punctuation
        return "none", None

    best, best_score = None, 0
    for scorer, thr in SCORERS:
        cand, score, _ = process.extractOne(key_norm, vocab.keys(), scorer=scorer)
        if cand is None:        # vocab κενό
            continue

        cand_tokens = cand.split()

        # 💡 ➊ very-short candidate vs long query
        if (len(cand) <= 2 or len(cand_tokens) <= 1) and len(key_tokens) > 2:
            continue

        # 💡 ➋ Jaccard overlap
        jac = len(set(key_tokens) & set(cand_tokens)) / max(len(key_tokens), len(cand_tokens))
        if jac < 0.50:
            continue

        if score >= thr and score > best_score:
            best, best_score = cand, score

    if best is None:            # δεν βρέθηκε τίποτα ικανοποιητικό
        return "none", None

    best_val = vocab[best]

    # promotion σε exact
    if best_score >= 95 or best == key_norm:
        return "exact", best_val

    return "approximate", best_val



# ---------------------------------------------------------------------------
# 6. Apply matcher & save
# ---------------------------------------------------------------------------
logging.info("Matching skills…")
df_pred[["match_type", "matched_vocab"]] = (
    df_pred["skill_norm"]
          .apply(lambda s: pd.Series(match_skill(s, vocab)))
)

df_pred.to_excel(OUT_PATH, index=False)
logging.info("Saved comparison to %s", OUT_PATH)

# ---------------------------------------------------------------------------
# 7. Reporting
# ---------------------------------------------------------------------------

def report(df: pd.DataFrame) -> float:
    total = len(df)
    counts = df["match_type"].value_counts().reindex(
        ["exact", "approximate", "none"], fill_value=0
    )
    for cat, n in counts.items():
        logging.info("%s: %d/%d  →  %.1f%%", cat.capitalize(), n, total, n / total * 100)
    return counts["exact"] / total

if __name__ == "__main__":
    exact_rate = report(df_pred)
    # optional threshold warning
    if exact_rate < 0.70:
        logging.warning("Exact match rate %.1f%% below threshold 70%%", exact_rate * 100)
