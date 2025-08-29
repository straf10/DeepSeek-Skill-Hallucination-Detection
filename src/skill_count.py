"""
    Reads the given Excel file, counts occurrences of each skill across job postings,
    filters by a minimum count threshold, computes coverage, optionally
    reintegrates whitelisted skills, and writes results to a new Excel file
    with two sheets: 'skill_counts' (full list) and 'filtered_skills'.

    :param input_xlsx: Path to the input Excel file with skills per job.
    :param output_xlsx: Path where the result Excel file will be saved.
    :param sheet_name: Name of the sheet holding per-job skill lists.
    :param col_name: Name of the column containing semicolon-separated skill labels.
    :param min_count: Minimum number of occurrences to keep a skill.
    :param whitelist: List of skills to always include, regardless of count.
    """

import pandas as pd
import re
from typing import Optional, List

WS_RE = re.compile(r"\s+")

def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = WS_RE.sub(" ", s)
    return s


def count_and_filter_skills(
    input_xlsx: str,
    output_xlsx: str,
    sheet_name: str = 'job_skills',
    col_name: str = 'skill_labels',
    min_count: int = 2,
    whitelist: Optional[List[str]] = None
) -> None:
    # Load
    df = pd.read_excel(input_xlsx, sheet_name=sheet_name, dtype=str, engine="openpyxl")
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in sheet '{sheet_name}'")

    # fill + split + strip + lower + explode
    skill_series = (
        df[col_name].fillna('')
          .astype(str)
          .str.split(';')
          .apply(lambda xs: [_normalize_label(x) for x in xs if _normalize_label(x)])
          .explode()
    )

    if skill_series.empty:
        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            pd.DataFrame(columns=['skill_label','count','cum_coverage']).to_excel(writer, sheet_name='skill_counts', index=False)
            pd.DataFrame(columns=['skill_label','count','cum_coverage']).to_excel(writer, sheet_name='filtered_skills', index=False)
        print("No skills found. Empty result written.")
        return

    counts = (
        skill_series.value_counts()
        .rename_axis('skill_label')
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
    )

    # Cumulative coverage (safe division)
    total_occurrences = int(counts['count'].sum())
    counts['cum_coverage'] = counts['count'].cumsum() / total_occurrences if total_occurrences else 0.0

    # Filter ( > min_count)
    filtered = counts[counts['count'] > min_count].copy()

    # Whitelist
    if whitelist:
        wl = [_normalize_label(w) for w in whitelist if _normalize_label(w)]
        if wl:
            wl_df = counts[counts['skill_label'].isin(wl)]
            filtered = (
                pd.concat([filtered, wl_df], ignore_index=True)
                  .drop_duplicates(subset=['skill_label'])
                  .sort_values(by='count', ascending=False)
            )

    # Write
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        counts.to_excel(writer, sheet_name='skill_counts', index=False)
        filtered.to_excel(writer, sheet_name='filtered_skills', index=False)

    # Summary
    coverage = (filtered['count'].sum() / total_occurrences) if total_occurrences else 0.0
    print(f"Total unique skills: {len(counts)}")
    print(f"Skills with count > {min_count}: {len(filtered)}")
    print(f"Coverage by filtered skills: {coverage:.1%}")
    print(f"Results written to '{output_xlsx}'")



if __name__ == "__main__":
    input_xlsx  = r'C:\Python\THESIS\skillab_job_fetcher\output\output.xlsx'
    output_xlsx = r'C:\Python\THESIS\skillab_job_fetcher\output\skill_counts.xlsx'
    # Example whitelist for must-have skills
    whitelist = ['docker', 'aws', 'react']
    count_and_filter_skills(
        input_xlsx=input_xlsx,
        output_xlsx=output_xlsx,
        sheet_name='job_skills',
        col_name='skill_labels',
        min_count=2,
        whitelist=whitelist
    )
