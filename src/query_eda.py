import re
import pandas as pd
from unidecode import unidecode

BOLD_RE = re.compile(r'^\s*\*\*(.+?)\*\*')   # "**skill**: definition"
BULLET_RE = re.compile(r'^[\s\-•\d\.\)]+')    # αρχικά bullets/νούμερα

def clean_skill(text: str) -> str | None:
    """
    Παίρνει μια ωμή γραμμή από Deepseek και επιστρέφει ΜΟΝΟ το label της δεξιότητας.
    """
    if not text.strip():
        return None

    # 1) Σβήσε bullets / νούμερα
    text = BULLET_RE.sub('', text).strip()

    # 2) Αν ξεκινά με **...**, πάρε το μέσα στο bold
    m = BOLD_RE.match(text)
    if m:
        return m.group(1).strip()

    # 3) Αν έχει ':' πάρε ό,τι προηγείται (π.χ. "agile methodologies: use scrum...")
    if ':' in text:
        return text.split(':', 1)[0].strip()

    # 4) Fallback – επέστρεψε ολόκληρο το καθαρισμένο string
    return text.strip()

def parse_response_to_list(resp: str) -> list[str]:
    """
    Επιστρέφει λίστα ΜΟΝΟ με τα skill labels.
    """
    lines = resp.splitlines()
    skills = []
    for line in lines:
        skill = clean_skill(line)
        if skill:
            # έξτρα κανονικοποίηση (lower, χωρίς τόνους, trim)
            skill = unidecode(skill).lower().strip()
            skills.append(skill)
    return skills

# ————————————————
# Κύριο μέρος του EDA script:
df = pd.read_json("../output/ds_query_results.jsonl", lines=True)

# Προσθέτουμε στήλη με τη λίστα των skills
df['skill_list'] = df['response'].apply(parse_response_to_list)

# «Εκρήγνυται» σε ξεχωριστή γραμμή ανά skill
df_exploded = df.explode('skill_list')[['id','query','skill_list']].rename(
    columns={'skill_list':'skill'}
)

print(df_exploded.head(20))
# Σώζεις για επόμενη ανάλυση
df_exploded.to_csv("ds_query_res_normalised.csv", index=False)
