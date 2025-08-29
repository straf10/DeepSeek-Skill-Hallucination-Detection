import pandas as pd
from collections import Counter
from itertools import combinations

# ---------- CONFIG ----------
INPUT_XLSX = r"C:\Python\THESIS\skillab_job_fetcher\output\output.xlsx"

def _print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def _print_table(df: pd.DataFrame, index=False, max_rows=30):
    if df.empty:
        print("(no data)")
        return
    if len(df) > max_rows:
        df = df.head(max_rows)
        tail_note = f"... ({len(df)} of {len(df)} shown)"
    else:
        tail_note = ""
    print(df.to_string(index=index))
    if tail_note:
        print(tail_note)

def eda_skillab(top_n: int = 10, co_threshold: int = 5, show_co_pairs: int = 30):
    jobs_df      = pd.read_excel(INPUT_XLSX, sheet_name="jobs")
    skills_df    = pd.read_excel(INPUT_XLSX, sheet_name="skills")
    jobskills_df = pd.read_excel(INPUT_XLSX, sheet_name="job_skills")

    # : Hard vs Soft
    _print_section("1) Skill categorization (Hard vs Soft) — counts & percentages")
    skills_df["skill_label"] = skills_df["skill_label"].astype(str).str.strip()
    skills_df["skill_type"]  = skills_df["skill_type"].astype(str).str.strip()

    type_counts = skills_df["skill_type"].value_counts().rename_axis("skill_type").reset_index(name="count")
    total_types = int(type_counts["count"].sum()) if not type_counts.empty else 0
    if total_types > 0:
        type_counts["pct"] = (type_counts["count"] / total_types * 100).round(2)
    _print_table(type_counts)

    #  Experience level (entry/mid/senior)
    _print_section("2) Experience level distribution (%)")
    exp_series = (
        jobs_df["experience_level"]
        .astype(str).str.strip()
        .replace({"None": "", "nan": ""})
    )
    exp_counts = exp_series.value_counts(dropna=True)
    exp_pct = ((exp_counts / exp_counts.sum()) * 100).round(2)
    exp_df = pd.DataFrame({"experience_level": exp_pct.index, "pct": exp_pct.values})
    _print_table(exp_df)

    # Job postings per occupation (URI)
    _print_section("3) Job postings per occupation (ISCO URI) — totals")
    jobs_df["occupation_ids"] = jobs_df["occupation_ids"].astype(str)
    occ_counts = (
        jobs_df["occupation_ids"]
        .str.split(";").explode()
        .str.strip()
        .value_counts()
        .rename_axis("occupation_uri")
        .reset_index(name="count")
    )
    _print_table(occ_counts)

    jobskills_df = jobskills_df.copy()
    jobskills_df["skill_labels"] = jobskills_df["skill_labels"].astype(str).fillna("")
    exploded = (
        jobskills_df
        .assign(skill=jobskills_df["skill_labels"].str.split(";"))
        .explode("skill")
    )
    exploded["skill"] = exploded["skill"].astype(str).str.strip()
    exploded = exploded[exploded["skill"] != ""]
    skills_map = skills_df[["skill_label", "skill_type"]].drop_duplicates()
    merged = exploded.merge(
        skills_map,
        left_on="skill",
        right_on="skill_label",
        how="left"
    )

    # Top‑N Skills
    _print_section(f"4) Top-{top_n} skills (overall)")
    top_all = (
        exploded["skill"].value_counts()
        .rename_axis("skill").reset_index(name="count")
        .head(top_n)
    )
    _print_table(top_all)

    # Top‑N Hard / Top‑N Soft
    _print_section(f"5) Top-{top_n} HARD skills")
    hard_counts = (
        merged[merged["skill_type"].str.lower() == "hard"]["skill"]
        .value_counts()
        .rename_axis("skill").reset_index(name="count")
        .head(top_n)
    )
    _print_table(hard_counts)

    _print_section(f"6) Top-{top_n} SOFT skills")
    soft_counts = (
        merged[merged["skill_type"].str.lower() == "soft"]["skill"]
        .value_counts()
        .rename_axis("skill").reset_index(name="count")
        .head(top_n)
    )
    _print_table(soft_counts)

    # Skill coverage
    _print_section("7) Skill coverage (appearances per job) — top 20")
    total_jobs = len(jobs_df) if len(jobs_df) > 0 else 1
    coverage = (
        exploded["skill"].value_counts() / total_jobs
    ).rename("coverage").reset_index().rename(columns={"index": "skill"}).sort_values("coverage", ascending=False)
    coverage["coverage"] = coverage["coverage"].round(4)
    _print_table(coverage.head(20))

    # Co‑occurrence
    _print_section(f"8) Skill co-occurrence (pairs with count >= {co_threshold}) — top {show_co_pairs}")
    co_counter = Counter()
    jobskills_df["skill_list"] = (
        jobskills_df["skill_labels"]
        .astype(str)
        .apply(lambda s: sorted(set([x.strip() for x in s.split(";") if x.strip()])))
    )
    for skill_list in jobskills_df["skill_list"]:
        if isinstance(skill_list, list) and len(skill_list) >= 2:
            for a, b in combinations(skill_list, 2):
                # canonical ordering
                a2, b2 = (a, b) if a <= b else (b, a)
                co_counter[(a2, b2)] += 1

    co_rows = [(a, b, c) for (a, b), c in co_counter.items() if c >= co_threshold]
    co_df = pd.DataFrame(co_rows, columns=["skill_1", "skill_2", "count"]).sort_values("count", ascending=False)
    _print_table(co_df.head(show_co_pairs))

    print("\n[OK] EDA (console) completed.")

if __name__ == "__main__":
    eda_skillab(top_n=10, co_threshold=5, show_co_pairs=30)
