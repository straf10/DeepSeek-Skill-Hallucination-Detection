"""
Prompt-Skill Comparison Framework
-------------------------------------------------------
Παρόμοιο με το skill_comparison_framework, αλλά ταιριάζει
τα skills από τα prompts (με job_id) με το API-database vocabulary.
"""

import logging
import pathlib
import re
import yaml
import pandas as pd
from unidecode import unidecode
from rapidfuzz import process, fuzz

logging.basicConfig(level=logging.INFO)


# ─── 1. Config  ───────────────────────────────
# Excel με το API-database skills (sheet "skills", στήλη "skill_label")
EXCEL_PATH         = pathlib.Path(r"C:\Python\THESIS\skillab_job_fetcher\output\output.xlsx")
SHEET_NAME         = "skills"

# Excel με τα skills extracted από prompts:
# αναμένεται στήλες: job_id, skill
PROMPT_SKILLS_PATH = pathlib.Path(r"C:\Python\THESIS\skillab_job_fetcher\output\ds_prompt_res_normalised.xlsx")

# YAML με synonyms (προαιρετικό)
SYN_FILE           = pathlib.Path(r"C:\Python\THESIS\skillab_job_fetcher\synonyms.yaml")

# Τελικό output
OUT_PATH           = pathlib.Path(r"C:\Python\THESIS\skillab_job_fetcher\output\prompt_skill_comparison.xlsx")
# ────────────────────────────────────────────────────────────────────────────

# ─── 2. Helpers ────────────────────────────────────────────────────────────
_whitespace_re = re.compile(r"\s+")

def normalize(txt: str) -> str:
    """unidecode → lower → strip → collapse spaces."""
    if pd.isna(txt):
        return ""
    txt = unidecode(str(txt)).lower().strip()
    return _whitespace_re.sub(" ", txt)

# Φόρτωμα synonyms
syn_raw = {}
if SYN_FILE.exists():
    syn_raw = yaml.safe_load(SYN_FILE.read_text(encoding="utf-8"))

SYN_MAP: dict[str, str] = {}
for canon, variants in syn_raw.items():
    canon_norm = normalize(canon)
    # αν δεν είναι list, μετατρέπουμε σε list για εύκολη επεξεργασία
    if not isinstance(variants, list):
        variants = [variants]
    # για κάθε variant, φτιάχνουμε mapping → canonical
    for v in variants:
        v_norm = normalize(v)
        SYN_MAP[v_norm] = canon_norm
# (προαιρετικά) αν θέλεις να θεωρείς και το ίδιο το canonical ως synonym
for canon in syn_raw.keys():
    SYN_MAP[normalize(canon)] = normalize(canon)

def apply_synonyms(token_norm: str) -> str:
    return SYN_MAP.get(token_norm, token_norm)

# Scorers & thresholds
SCORERS = [
    (fuzz.token_sort_ratio, 85),
    (fuzz.token_set_ratio,   80),
    (fuzz.partial_ratio,     75),
]


def match_skill(key_norm: str, vocab: dict[str,str]):
    """
    Επιστρέφει (match_type, matched_label):
      - exact       → ακριβές match στο vocab
      - approximate → fuzzy match πάνω από thresholds
      - none        → δεν βρέθηκε τίποτα
    """
    key_norm = apply_synonyms(key_norm)
    if not key_norm:
        return "none", None

    # 1. Exact
    if key_norm in vocab:
        return "exact", vocab[key_norm]

    key_tokens = key_norm.split()
    best, best_score = None, 0
    for scorer, thr in SCORERS:
        cand, score, _ = process.extractOne(key_norm, vocab.keys(), scorer=scorer)
        if cand is None or score < thr:
            continue

        # simple Jaccard for token overlap
        cand_tokens = cand.split()
        jac = len(set(key_tokens)&set(cand_tokens)) / max(len(key_tokens), len(cand_tokens))
        if jac < 0.5:
            continue

        if score > best_score:
            best, best_score = cand, score

    if best is None:
        return "none", None

    # promote to exact αν πολύ ψηλή ομοιότητα
    if best_score >= 95 or best == key_norm:
        return "exact", vocab[best]
    return "approximate", vocab[best]


# ─── 3. Φόρτωμα Vocabulary (API-db) ────────────────────────────────────────
logging.info("Loading API-database vocabulary…")
df_vocab = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")
if "skill_label" not in df_vocab.columns:
    raise ValueError(f"Column 'skill_label' not found in {EXCEL_PATH}")
df_vocab["skill_norm"] = df_vocab["skill_label"].apply(normalize)

# drop dupes, keep first
vocab = (
    df_vocab
    .drop_duplicates("skill_norm", keep="first")
    .set_index("skill_norm")["skill_label"]
    .to_dict()
)

# expand with synonyms
for syn_norm, canon_norm in SYN_MAP.items():
    if canon_norm in vocab:
        vocab[syn_norm] = vocab[canon_norm]

logging.info("Vocabulary size: %d entries", len(vocab))


# ─── 4. Φόρτωμα Prompt-Skills ─────────────────────────────────────────────
logging.info("Loading prompt-extracted skills…")
df_pred = pd.read_excel(PROMPT_SKILLS_PATH, engine="openpyxl")
if not {"job_id","skill"}.issubset(df_pred.columns):
    raise ValueError(f"Expected columns ['job_id','skill'] in {PROMPT_SKILLS_PATH}")
df_pred["skill_norm"] = df_pred["skill"].apply(normalize)


# ─── 5. Matching & Save ──────────────────────────────────────────────────
logging.info("Matching skills…")
matched = df_pred["skill_norm"].apply(lambda s: pd.Series(match_skill(s, vocab)))
matched.columns = ["match_type","matched_vocab"]

df_out = pd.concat([df_pred, matched], axis=1)


# ─── 5a. Analysis of unmatched & weak matches ───────────────────────────
# 1) Top 20 skills που δεν matchάρουν καν (match_type == "none")
unmatched = df_out[df_out.match_type=="none"]["skill"]
print("\n=== Top 20 unmatched skills ===")
print(unmatched.value_counts().head(20))

# 2) Skills με fuzzy score 80–89% (πολύ “weak” approximate)
from rapidfuzz import fuzz

def best_score(s):
    # επιστρέφει το μέγιστο score απέναντι σε όλα τα canonical keys
    return max([fuzz.token_sort_ratio(s, k) for k in vocab.keys()] + [0])

# προσθέτουμε στήλη με το καλύτερο score
df_out["best_score"] = df_out["skill_norm"].apply(best_score)

# φιλτράρουμε για scores μεταξύ 80 (inclusive) και 90 (exclusive)
weak = df_out[(df_out.best_score >= 70) & (df_out.best_score < 86)]
print("\n=== Top 20 skills with score 70–85 ===")
print(weak["skill"].value_counts().head(20))

df_out.to_excel(OUT_PATH, index=False, engine="openpyxl")
logging.info("Saved comparison to %s", OUT_PATH)


# ─── 6. Report KPI ────────────────────────────────────────────────────────
def report(df: pd.DataFrame):
    total = len(df)
    counts = df["match_type"].value_counts().reindex(
        ["exact","approximate","none"], fill_value=0
    )
    for cat, count in counts.items():
        pct = count/total*100
        logging.info(f"{cat:12s}: {count:4d}/{total} → {pct:5.1f}%")
    return counts["exact"]/total

if __name__ == "__main__":
    rate = report(df_out)
    if rate < 0.7:
        logging.warning("Exact match rate %.1f%% below 70%%", rate*100)
