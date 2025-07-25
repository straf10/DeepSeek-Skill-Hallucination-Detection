"""
Simple EDA Script
Generates bar charts for:
1) Distribution of experience levels
2) Number of postings per occupation URI
3) Geographic distribution of postings (by country)
4) Top 10 skills overall (by label)
Loads data from a fixed Excel path and saves figures to 'figures' directory.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

OCC_URIS = [
    "http://data.europa.eu/esco/isco/C2511",
    "http://data.europa.eu/esco/isco/C2512",
    "http://data.europa.eu/esco/isco/C2513",
    "http://data.europa.eu/esco/isco/C2514",
]
OCC_LABELS = {
    "http://data.europa.eu/esco/isco/C2511": "Προγραμματιστής Λογισμικού (C2511)",
    "http://data.europa.eu/esco/isco/C2512": "Μηχανικός Λογισμικού (C2512)",
    "http://data.europa.eu/esco/isco/C2513": "Προγραμματιστής Ιστού/Πολυμέσων (C2513)",
    "http://data.europa.eu/esco/isco/C2514": "Ελεγκτής Λογισμικού (C2514)",
}


def main():
    # Fixed input and output paths
    excel_path = r"/output/output.xlsx"
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)

    # Load data sheets
    jobs_df = pd.read_excel(excel_path, sheet_name="jobs", dtype=str)
    skills_df = pd.read_excel(excel_path, sheet_name="skills", dtype=str)
    job_skills_df = pd.read_excel(excel_path, sheet_name="job_skills", dtype=str)

    # 1) Experience level distribution
    exp_counts = jobs_df['experience_level'].value_counts()
    plt.figure()
    exp_counts.plot(kind='bar', edgecolor='black')
    plt.xlabel('Experience Level')
    plt.ylabel('Number of Postings')
    plt.title('Distribution of Experience Levels')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experience_distribution.png'))
    plt.close()

    # 2) Number of postings per occupation URI
    jobs_df['occupation_list'] = jobs_df['occupation_ids'].str.split(';')
    jobs_expanded = jobs_df.explode('occupation_list')
    occ_filtered = jobs_expanded[jobs_expanded['occupation_list'].isin(OCC_URIS)]
    occ_counts = occ_filtered['occupation_list'].value_counts().reindex(OCC_URIS).fillna(0)
    values = [int(occ_counts[uri]) for uri in OCC_URIS]
    labels = [OCC_LABELS.get(uri, uri) for uri in OCC_URIS]
    plt.figure()
    plt.bar(range(len(labels)), values, edgecolor='black')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.xlabel('Occupation')
    plt.ylabel('Number of Postings')
    plt.title('Number of Postings per Selected Occupation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'occupation_counts.png'))
    plt.close()

    # 3) Geographic distribution (by country)
    countries = jobs_df['location'].dropna().apply(lambda x: x.split(',')[-1].strip())
    country_counts = countries.value_counts().head(10)
    plt.figure()
    country_counts.plot(kind='bar', edgecolor='black')
    plt.xlabel('Country')
    plt.ylabel('Number of Postings')
    plt.title('Top Countries by Number of Postings')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'geo_distribution.png'))
    plt.close()

    # 4) Top 10 skills overall
    jobs_df['skill_list'] = jobs_df['skill_ids'].str.split(';')
    skill_counts = jobs_df['skill_list'].explode().dropna().value_counts().head(10)
    skill_map = skills_df.set_index('skill_uri')['skill_label'].to_dict()
    labels = [skill_map.get(uri, uri) for uri in skill_counts.index]
    plt.figure()
    plt.bar(range(len(labels)), skill_counts.values, edgecolor='black')
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.xlabel('Skill')
    plt.ylabel('Frequency')
    plt.title('Top 10 Skills Overall')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top10_skills.png'))
    plt.close()

    print(f"All plots saved to '{output_dir}'.")

if __name__ == '__main__':
    main()
