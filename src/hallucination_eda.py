import pandas as pd
from collections import OrderedDict

# ================== HARD-CODED PATHS==================
XLSX_PATH = r"C:\Python\THESIS\skillab_job_fetcher\output\data.xlsx"

COL_SKILL = "original_skill"
COL_LABEL = "manual_label"
COL_SOURCE = "source"

LABELS = ["H0", "H1a", "H1b", "H1c", "H1d", "H1e"]

SOURCE_NORMALIZE = {
    "open": "open_mode",
    "open_mode": "open_mode",
    "open mode": "open_mode",
    "closed": "closed_mode",
    "closed_mode": "closed_mode",
    "closed mode": "closed_mode",
    "jobsample": "job_sample",
    "job_sample": "job_sample",
    "job sample": "job_sample",
}

BLOCKS = ["TOTAL", "open_mode", "closed_mode", "job_sample"]

# ================== Helpers ==================
def print_header(title: str, width: int = 80, char: str = "="):
    print(f"\n{char*width}\n{title}\n{char*width}")

def print_subheader(title: str, width: int = 80, char: str = "-"):
    print(f"\n{title}\n{char*width}")

def format_row(cols, widths):
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

def normalize_source(s: str) -> str:
    if pd.isna(s):
        return "unknown"
    key = str(s).strip().lower()
    return SOURCE_NORMALIZE.get(key, key)


def print_distribution(df: pd.DataFrame, title: str):
    print_subheader(f"[Labels] {title}")
    n = len(df)
    counts = df[COL_LABEL].value_counts().reindex(LABELS).fillna(0).astype(int)
    pct = (counts / n * 100).round(2) if n > 0 else counts.astype(float)

    headers = ["Label", "Count", "% per total"]
    widths = [8, 8, 14]
    print(format_row(headers, widths))
    print(format_row(["-"*w for w in widths], widths))
    for lab in LABELS:
        print(format_row([lab, counts.get(lab, 0), f"{pct.get(lab, 0):.2f}%"], widths))
    print(format_row(["TOTAL", counts.sum(), "100.00%" if n > 0 else "0.00%"], widths))


def print_top_k_for_label(df: pd.DataFrame, label: str, k: int, title: str):
    sub = df[df[COL_LABEL] == label]
    n = len(sub)
    print_subheader(f"[Top-{k} {label}] {title}  (base: {n} with label {label})")
    if n == 0:
        print("— No registers for this label —")
        return
    top = (
        sub[COL_SKILL]
        .astype(str).str.strip()
        .value_counts()
        .head(k)
        .reset_index()
    )
    top.columns = [COL_SKILL, "count"]
    top[f"% επί {label}"] = (top["count"] / n * 100).round(2)

    headers = ["#", "Skill", "Count", f"% επί {label}"]
    widths  = [4, 50, 8, 12]
    print(format_row(headers, widths))
    print(format_row(["-"*w for w in widths], widths))
    for i, row in top.iterrows():
        print(format_row([i+1, row[COL_SKILL], row["count"], f'{row[f"% επί {label}"]:.2f}%'], widths))


def run_block(df: pd.DataFrame, block_title: str):

    df = df[df[COL_LABEL].isin(LABELS)].copy()
    print_distribution(df, block_title)
    print_top_k_for_label(df, "H1a", 10, block_title)
    print_top_k_for_label(df, "H1d", 10, block_title)


# ================== MAIN ==================
def main():
    print_header("EDA: Hallucination Labels (DeepSeek vs Skillab)")

    df = pd.read_excel(XLSX_PATH)

    missing = [c for c in (COL_SKILL, COL_LABEL, COL_SOURCE) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Found: {list(df.columns)}"
        )

    df = df.copy()
    df[COL_SKILL] = df[COL_SKILL].astype(str).str.strip()
    df[COL_LABEL] = df[COL_LABEL].astype(str).str.strip()
    df[COL_SOURCE] = df[COL_SOURCE].apply(normalize_source)

    run_block(df, "Total")

    # --- Per source (open_mode, closed_mode, job_samples) ---
    for source_name in ["open_mode", "closed_mode", "job_sample"]:
        sub = df[df[COL_SOURCE] == source_name]
        if sub.empty:
            print_subheader(f"[WARNING] No registers found '{source_name}'. Skipped!")
            continue
        run_block(sub, source_name)

    print("\nΟΚ — Completed statistic evaluation.\n")

if __name__ == "__main__":
    main()
