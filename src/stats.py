# -------------------------------------------------------------
# Τυπώνει περιγραφικά στατιστικά για τα 3 sheets του normalised.xlsx:
# 1) Συνολικός αριθμός skills & πλήθος queries/job samples
# 2) Μέσος όρος skills ανά query/job sample
# 3) % “no matching skills” (άδειες γραμμές ή "- (no matching skills)")  → μόνο για open/closed
# 4) % exact matching με allowed list (κυρίως για closed mode)
# 5) Top‑10 συχνότερα skills ανά sheet
#
# Απαιτεί:
#  - Το αρχείο που γράφει το skill_normalization.py: normalised.xlsx
#      • open_mode:  columns = [question, skills]  (skills = "a;b;c")
#      • closed_mode: columns = [question, skills]
#      • job_samples: columns = [job_id, skill_label, confidence, evidence] (μία γραμμή ανά skill)
#  - Το αρχείο skill_counts.xlsx (από data_export/skill_count.py) για το allowed list
#      • sheets: 'filtered_skills' (αν υπάρχει) ή 'skill_counts' (fallback με φίλτρο count>=1)
#
# Σημειώσεις:
#  - Στο sheet job_samples δεν υπάρχουν γραμμές “χωρίς skills”, γιατί τα κενά jobs
#    δεν γράφτηκαν στο normalised.xlsx (δεν μπορούμε να μετρήσουμε % no‑match εδώ).
#  - Exact matching γίνεται με απλό lowercase/strip/space-compress.
# -------------------------------------------------------------

import re
import pandas as pd
from pathlib import Path
from typing import Iterable

# ---------- Ρυθμίσεις Μονοπατιών ----------
NORMALISED_XLSX = r"C:\Python\THESIS\skillab_job_fetcher\output\normalised.xlsx"
COUNTS_XLSX     = r"C:\Python\THESIS\skillab_job_fetcher\output\skill_counts.xlsx"

# ---------- Βοηθητικά ----------
WS_RE = re.compile(r"\s+")

NO_MATCH_TOKENS = {
    "(no matching skills)",
    "no matching skills",
    "- (no matching skills)",
}

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = WS_RE.sub(" ", s)
    return s

def is_no_match_token(label: str) -> bool:
    """Έλεγχος αν ένα label είναι placeholder 'no matching skills' (με ή χωρίς '-')"""
    return norm(label) in NO_MATCH_TOKENS

def load_allowed_set(path: str | Path) -> set[str]:
    """Φορτώνει allowed skills από 'filtered_skills' (αν υπάρχει) ή από 'skill_counts'."""
    p = Path(path)
    if not p.exists():
        return set()
    xls = pd.ExcelFile(p, engine="openpyxl")
    sheet = "filtered_skills" if "filtered_skills" in xls.sheet_names else "skill_counts"
    df = pd.read_excel(xls, sheet_name=sheet)
    col = "skill_label" if "skill_label" in df.columns else df.columns[0]
    skills = [norm(x) for x in df[col].astype(str).tolist()]
    return {s for s in skills if s}

def explode_semicolon_series(series: pd.Series, *, drop_no_match=True) -> pd.Series:
    """
    Παίρνει στήλη με 'a;b;c' και επιστρέφει Series (μία γραμμή/skill).
    Αν drop_no_match=True, πετάει placeholders τύπου '(no matching skills)'.
    """
    s = (
        series.fillna("")
              .astype(str)
              .apply(lambda s: [norm(x) for x in s.split(";") if norm(x)])
              .explode()
              .dropna()
              .astype(str)
    )
    if drop_no_match:
        s = s[~s.apply(is_no_match_token)]
    return s

def print_banner(title: str) -> None:
    line = "─" * max(10, len(title) + 2)
    print(f"\n{line}\n{title}\n{line}")

def pct(numer: int, denom: int) -> str:
    return "0.0%" if denom == 0 else f"{(numer/denom)*100:.1f}%"

# ---------- Φόρτωση δεδομένων ----------
xls = pd.ExcelFile(NORMALISED_XLSX, engine="openpyxl")
required = {"open_mode", "closed_mode", "job_samples"}
missing = required.difference(set(xls.sheet_names))
if missing:
    raise SystemExit(f"Λείπουν sheets από το '{NORMALISED_XLSX}': {', '.join(sorted(missing))}")

open_df   = pd.read_excel(xls, sheet_name="open_mode")
closed_df = pd.read_excel(xls, sheet_name="closed_mode")
jobs_df   = pd.read_excel(xls, sheet_name="job_samples")

allowed_set = load_allowed_set(COUNTS_XLSX)

# ---------- Υπολογισμοί: open_mode / closed_mode ----------
def summarize_qs_skills(df: pd.DataFrame, sheet_name: str):
    # Πλήθος queries
    n_queries = len(df)

    # Συνολικός αριθμός skills
    skills_series = explode_semicolon_series(df.get("skills", pd.Series(dtype=str)))
    total_skills = int(skills_series.shape[0])

    # Μέσος όρος skills ανά query
    avg_skills_per_q = 0.0 if n_queries == 0 else total_skills / n_queries

    # % no matching (άδειες ή "- (no matching skills)")
    def is_no_match_cell(cell: str) -> bool:
        items = [norm(x) for x in str(cell or "").split(";") if norm(x)]
        if not items:
            return True  # εντελώς άδειο
        # no‑match αν ΟΛΑ τα items είναι placeholders
        return all(is_no_match_token(x) for x in items)

    no_match_rows = int(df.get("skills", pd.Series(dtype=str)).fillna("").astype(str).apply(is_no_match_cell).sum())
    no_match_pct = pct(no_match_rows, n_queries)

    # % exact matching με allowed list (ανά skill)
    if allowed_set:
        exact_matches = int(skills_series.isin(allowed_set).sum())
        exact_match_pct = pct(exact_matches, total_skills)
    else:
        exact_matches = 0
        exact_match_pct = "— (no allowed list)"

    # Top‑10 συχνότερα
    top10 = skills_series.value_counts().head(10)

    # Εκτύπωση
    print_banner(f"[{sheet_name}] Summary")
    print(f"Queries: {n_queries}")
    print(f"Συνολικά skills: {total_skills}")
    print(f"Μ.Ο. skills/query: {avg_skills_per_q:.2f}")
    print(f'% "no matching skills" (σε επίπεδο query): {no_match_pct}  ({no_match_rows}/{n_queries})')
    print(f"% exact matching (σε επίπεδο skill): {exact_match_pct}  ({exact_matches}/{total_skills})")
    print("\nTop‑10 skills:")
    if top10.empty:
        print("  (κανένα)")
    else:
        for i, (label, count) in enumerate(top10.items(), 1):
            print(f"  {i:>2}. {label}  —  {count}")

# ---------- Υπολογισμοί: job_samples ----------
def summarize_job_samples(df: pd.DataFrame):
    # job_samples: μία γραμμή ανά (job_id, skill_label)
    if "job_id" not in df.columns or "skill_label" not in df.columns:
        raise SystemExit("Το sheet 'job_samples' πρέπει να έχει στήλες 'job_id' και 'skill_label'.")

    # Πλήθος job samples (unique jobs)
    n_jobs = int(df["job_id"].astype(str).nunique())

    # Συνολικός αριθμός skills
    skill_series = df["skill_label"].fillna("").astype(str).apply(norm)
    skill_series = skill_series[skill_series.str.len() > 0]
    total_skills = int(skill_series.shape[0])

    # Μέσος όρος skills ανά job sample
    avg_skills_per_job = 0.0 if n_jobs == 0 else total_skills / n_jobs

    # % no matching skills: δεν είναι μετρήσιμο από αυτό το αρχείο (βλ. σημείωση κεφαλίδας)
    no_match_text = "— (μη διαθέσιμο: jobs χωρίς skills δεν καταγράφονται σε αυτό το sheet)"

    # % exact matching με allowed list
    if allowed_set:
        exact_matches = int(skill_series.isin(allowed_set).sum())
        exact_match_pct = pct(exact_matches, total_skills)
    else:
        exact_matches = 0
        exact_match_pct = "— (no allowed list)"

    # Top‑10
    top10 = skill_series.value_counts().head(10)

    # Εκτύπωση
    print_banner("[job_samples] Summary")
    print(f"Job samples (unique job_id): {n_jobs}")
    print(f"Συνολικά skills: {total_skills}")
    print(f"Μ.Ο. skills/job sample: {avg_skills_per_job:.2f}")
    print(f'% "no matching skills": {no_match_text}')
    print(f"% exact matching (σε επίπεδο skill): {exact_match_pct}  ({exact_matches}/{total_skills})")
    print("\nTop‑10 skills:")
    if top10.empty:
        print("  (κανένα)")
    else:
        for i, (label, count) in enumerate(top10.items(), 1):
            print(f"  {i:>2}. {label}  —  {count}")

# ---------- Εκτέλεση ----------
if __name__ == "__main__":
    summarize_qs_skills(open_df,   "open_mode")
    summarize_qs_skills(closed_df, "closed_mode")
    summarize_job_samples(jobs_df)
