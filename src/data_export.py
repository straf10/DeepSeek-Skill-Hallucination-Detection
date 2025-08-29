import os
import sys
import urllib3
import requests as req
import pandas as pd
import logging
import itertools
# import re

from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE = "https://skillab-tracker.csd.auth.gr/api"
VERIFY_SSL = False

ISCO_URIS = [
    "http://data.europa.eu/esco/isco/C2511",
    "http://data.europa.eu/esco/isco/C2512",
    "http://data.europa.eu/esco/isco/C2513",
    "http://data.europa.eu/esco/isco/C2514",
]

# ***CONFIG***
PAGE_SIZE = 100
MAX_PAGES = 7
CHUNK = 50
MAX_WORKERS = 5

load_dotenv()
USER, PASSWORD = os.getenv("SKILLAB_USER"), os.getenv("SKILLAB_PASS")

if not USER or not PASSWORD:
    sys.exit("ERROR: Missing username / password in .env")

else:
    print("Successful login")

_SESSION = req.Session()
_SESSION.verify = VERIFY_SSL
_TOKEN:str | None = None

log = logging.getLogger(__name__)

def main():
    log.info("File: %s", os.path.abspath(__file__))
    log.info("Executable: %s", sys.executable)
    log.info("Python: %s", sys.version)

    logging.basicConfig(level=logging.DEBUG)

    jobs = fetch_jobs()
    log.info("Fetched %d unique jobs", len(jobs))
    export_excel(jobs, outdir=Path("../output"))

def _refresh_token() -> None:
    """Refresh JWT token and input in headers of session."""
    global _TOKEN
    resp = _SESSION.post(f"{API_BASE}/login", json={"username": USER, "password": PASSWORD}, timeout=10)
    resp.raise_for_status()
    _TOKEN = resp.text.strip('"')
    _SESSION.headers.update({"Authorization": f"Bearer {_TOKEN}"})
    log.info("Authenticated and token refreshed.")

def _api_post_form(endpoint: str, form: List[Tuple[str, str]], params: Optional[Dict[str, str]] = None, timeout: int = 30) -> req.Response:
    """POST x-www-form-urlencoded with automated retry and refresh JWT if error 401."""

    if _TOKEN is None:
        _refresh_token()

    url = f"{API_BASE}{endpoint}"
    headers = _SESSION.headers.copy()
    headers["Content-Type"] = "application/x-www-form-urlencoded"

    log.debug("Sending POST to %s, params=%s, form=%s", url, params, form)
    resp = _SESSION.post(url, params=params, data=form,
                          headers=headers, timeout=timeout)

    if resp.status_code == 401:
        log.info("Token expired, refreshing and retrying once!")
        _refresh_token()
        headers = _SESSION.headers.copy()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        resp = _SESSION.post(url, params=params, data=form, headers=headers, timeout=timeout)

    resp.raise_for_status()
    return resp

def _chunks(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def batch_labels(endpoint: str, uris: Set[str]) -> Dict[str, str]:
    """Batch look-up , returns dict uri→label."""
    if not uris:
        return {}

    mapping: Dict[str, str] = {}
    uri_list = list(uris)

    def fetch(chunk: List[str]) -> Dict[str, str]:
        log.debug("Fetching labels for %d URIs…", len(chunk))
        resp = _api_post_form(
            f"/{endpoint}",
            [("ids", u) for u in chunk]
        )
        items = resp.json().get("items", [])
        return {it["id"]: it.get("label", "") for it in items}

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch, c): c for c in _chunks(uri_list, CHUNK)}
        for fut in as_completed(futures):
            chunk = futures[fut]
            try:
                mapping.update(fut.result())
                log.info("Fetched %d labels for chunk of size %d", len(fut.result()), len(chunk))
            except Exception as e:
                log.error("Error fetching chunk %s: %s", chunk, e)

    return mapping

def fetch_jobs(*, page_size: int = PAGE_SIZE, max_pages: int = MAX_PAGES, limit: int | None = None, req_timeout: int = 30) -> List[Dict]:
    """Returns unique job postings"""

    log.debug("→ Entering fetch_jobs: page_size=%s max_pages=%s limit=%s", page_size, max_pages, limit)
    form_body = [("occupation_ids_logic", "or")] + [("occupation_ids", uri) for uri in ISCO_URIS]
    jobs_raw: List[dict] = []

    pages = list(range(1, max_pages + 1))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # Submit κάθε σελίδα ως ξεχωριστό task
        futures = {
            pool.submit(_api_post_form, "/jobs", form_body, {"page": page, "page_size": page_size}, req_timeout):
                page for page in pages
        }

        for fut, page in futures.items():
            log.debug("  page %d → future %s", page, fut)

        for fut in as_completed(futures):
            page = futures[fut]
            log.info("Requesting page %d", page)
            try:
                batch = fut.result().json().get("items", [])
                if not batch:
                    log.info("Page %d returned no items, skipping.", page)
                    continue
                # Αφαιρώ jobs χωρίς τίτλο
                valid = [j for j in batch if j.get("title", "").strip()]
                log.info("Page %d → %d valid jobs", page, len(valid))
                jobs_raw.extend(valid)
            except Exception as e:
                log.error("Error fetching page %d: %s", page, e)

        # Secure unique Id
        seen: Set[str] = set()
        uniq_jobs: List[dict] = []
        for job in jobs_raw:
            jid = job.get("id")
            if not jid or jid in seen:
                continue
            seen.add(jid)
            uniq_jobs.append(job)
            if limit and len(uniq_jobs) >= limit:
                break

        log.info("Total unique jobs fetched: %d", len(uniq_jobs))
        return uniq_jobs

def export_excel(jobs: List[Dict], *, outdir: Path) -> None:
    """Writes jobs.xlsx, skills.xlsx, job_skills.xlsx."""

    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "output.xlsx"

    jobs_df = pd.DataFrame([
        {
            "job_id": j.get("id"),
            "title": j.get("title", ""),
            "occupation_ids": ";".join(j.get("occupations", [])),
            "skill_ids": ";".join(j.get("skills", [])),
            "experience_level": j.get("experience_level", ""),
            "location": j.get("location", ""),
            "source": j.get("source", ""),
            "upload_date": j.get("upload_date", ""),
            "description": j.get("description", "")
        }
        for j in jobs
    ]).drop_duplicates()

    skill_uris: Set[str] = set()
    for j in jobs:
        skill_uris.update(j.get("skills", []))

    skill_map = batch_labels("skills", skill_uris)
    skills_df = pd.DataFrame([
        {"skill_uri": uri, "skill_label": label}
        for uri, label in skill_map.items()
    ])

    rows = []
    for j in jobs:
        title = j.get("title", "").strip()
        labels = [skill_map[u] for u in j.get("skills", []) if u in skill_map]
        rows.append({
            "job_title": title,
            "skill_labels": ";".join(labels)
        })
    js_df = pd.DataFrame(rows)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        jobs_df.to_excel(writer, sheet_name="jobs", index=False)
        skills_df.to_excel(writer, sheet_name="skills", index=False)
        js_df.to_excel(writer, sheet_name="job_skills", index=False)

    log.info("[OK] Excel file written → %s", output_path.resolve())

def report_missing_values(excel_path):
    """
    Reads Excel file and reports NaN values and empty strings.
    """
    df = pd.read_excel(excel_path)

    missing = {}
    for col in df.columns:
        na_count = df[col].isna().sum()
        if df[col].dtype == object:
            blank_count = df[col].astype(str).str.strip().eq('').sum()
        else:
            blank_count = 0
        total_missing = na_count + blank_count
        missing[col] = {
            'NaN': int(na_count),
            'Blank': int(blank_count),
            'Total': int(total_missing)
        }

    log.info("Missing values report for '%s':", excel_path)
    for col, stats in missing.items():
        if stats['Total'] > 0:
            log.info("- %s: %d NaN, %d blank → %d total",
                     col, stats['NaN'], stats['Blank'], stats['Total'])

    total_cells = df.size
    total_missing_cells = sum(stats['Total'] for stats in missing.values())
    pct_missing = total_missing_cells / total_cells * 100
    log.info("Overall: %d/%d missing (%.2f%%)", total_missing_cells, total_cells, pct_missing)

if __name__ == "__main__":
    main()
