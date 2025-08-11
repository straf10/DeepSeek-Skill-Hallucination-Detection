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

def count_and_filter_skills(
    input_xlsx: str,
    output_xlsx: str,
    sheet_name: str = 'job_skills',
    col_name: str = 'skill_labels',
    min_count: int = 2,
    whitelist: list = None
) -> None:

    # Load the sheet
    df = pd.read_excel(input_xlsx, sheet_name=sheet_name, dtype=str)
    df[col_name] = df[col_name].fillna('')

    # Split, strip and explode into individual skill labels
    skill_series = (
        df[col_name]
          .str.split(';')
          .apply(lambda skills: [s.strip() for s in skills if s.strip()])
          .explode()
    )

    # Count occurrences
    counts = (
        skill_series
        .value_counts()
        .rename_axis('skill_label')
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
    )

    # Compute cumulative coverage
    total_occurrences = counts['count'].sum()
    counts['cum_coverage'] = counts['count'].cumsum() / total_occurrences

    # Filter by threshold
    filtered = counts[counts['count'] > min_count].copy()

    # Re-add any whitelisted skills
    if whitelist:
        wl_df = counts[counts['skill_label'].isin(whitelist)]
        filtered = pd.concat([filtered, wl_df]).drop_duplicates(subset=['skill_label'])

    # Write to new Excel with two sheets
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        counts.to_excel(writer, sheet_name='skill_counts', index=False)
        filtered.to_excel(writer, sheet_name='filtered_skills', index=False)

    # Print a brief summary
    print(f"Total unique skills: {len(counts)}")
    print(f"Skills with count > {min_count}: {len(filtered)}")
    print(f"Coverage by filtered skills: {filtered['count'].sum() / total_occurrences:.1%}")
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
