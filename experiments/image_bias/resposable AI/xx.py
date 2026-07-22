#!/usr/bin/env python3
"""
job_fetcher.py
Simple job fetcher with two modes:
  1) Remotive API (default) – no API key needed
  2) Apify Web Scraper actor – requires APIFY token and a target listing URL

Usage examples:
  # Remotive (default)
  python job_fetcher.py --query "software engineer" --category "software-dev" --limit 10 --csv jobs.csv

  # Apify (scrape a page you’re allowed to extract)
  python job_fetcher.py --mode apify --apify-token "apify_api_xxx" \
         --url "https://remoteok.com/remote-software-engineer-jobs" --limit 10 --csv jobs.csv
"""

import argparse
import csv
import html
import os
import re
import sys
from datetime import datetime
from typing import List, Dict, Any

import requests


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    # strip HTML tags (very simple)
    s = re.sub(r"<[^>]+>", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def iso_or_blank(s: str | None) -> str:
    if not s:
        return ""
    try:
        # Try common formats; Remotive gives ISO8601
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.isoformat()
    except Exception:
        return clean_text(s)


# ---------- MODE 1: Remotive API (no key) ----------
def fetch_remotive(query: str, category: str | None, limit: int) -> List[Dict[str, Any]]:
    base = "https://remotive.com/api/remote-jobs"
    params = {}
    if query:
        params["search"] = query
    if category:
        params["category"] = category

    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    jobs = r.json().get("jobs", [])
    out: List[Dict[str, Any]] = []
    for j in jobs[:limit]:
        out.append({
            "source": "remotive",
            "title": clean_text(j.get("title")),
            "company": clean_text(j.get("company_name")),
            "location": clean_text(j.get("candidate_required_location")),
            "posted_at": iso_or_blank(j.get("publication_date")),
            "link": j.get("url") or "",
            "description": clean_text(j.get("description")),
        })
    return out


# ---------- MODE 2: Apify Web Scraper actor ----------
def fetch_apify(apify_token: str, page_url: str, limit: int) -> List[Dict[str, Any]]:
    """
    Calls the official Apify Web Scraper actor apify~web-scraper in a single request
    and returns dataset items (JSON). You MUST have permission to extract the target page.
    """
    if not apify_token:
        raise ValueError("Missing --apify-token")
    if not page_url:
        raise ValueError("Missing --url for Apify mode")

    endpoint = (
        "https://api.apify.com/v2/acts/apify~web-scraper/"
        "run-sync-get-dataset-items?format=json"
    )
    headers = {"Authorization": f"Bearer {apify_token}"}

    # This pageFunction targets RemoteOK’s layout as an example.
    # Change selectors for other sites you’re allowed to extract.
    page_function = (
        "async function pageFunction(context){"
        " const $=context.jQuery; const out=[];"
        " $('tr.job').each((_,el)=>{"
        "  const title=$('h2',el).text().trim();"
        "  const href=$('a.preventLink',el).attr('href')||'';"
        "  const link=href?new URL(href,'https://remoteok.com').toString():'';"
        "  const company=$('.companyLink h3',el).text().trim();"
        "  const location=$('.location',el).text().trim();"
        "  if(title&&link) out.push({title,link,company,location});"
        " });"
        " for(const it of out) await context.pushData(it);"
        "}"
    )

    payload = {
        "startUrls": [{"url": page_url}],
        "useRequestQueue": False,
        "useJQuery": True,
        "pageFunction": page_function,
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    # Normalize to the same schema as Remotive
    out: List[Dict[str, Any]] = []
    for j in data[:limit]:
        out.append({
            "source": "apify:web-scraper",
            "title": clean_text(j.get("title")),
            "company": clean_text(j.get("company")),
            "location": clean_text(j.get("location")),
            "posted_at": "",  # not extracted by this example function
            "link": j.get("link") or "",
            "description": "",  # not extracted in this minimal example
        })
    return out


def print_jobs(jobs: List[Dict[str, Any]]) -> None:
    if not jobs:
        print("No jobs found.")
        return
    for i, j in enumerate(jobs, 1):
        print(f"{i}. {j.get('title','(no title)')}")
        print(f"   Company : {j.get('company','')}")
        print(f"   Location: {j.get('location','')}")
        print(f"   Posted  : {j.get('posted_at','')}")
        print(f"   Link    : {j.get('link','')}")
        desc = j.get("description", "")
        if desc:
            print(f"   Desc    : {desc[:160]}{'…' if len(desc)>160 else ''}")
        print("-" * 60)


def save_csv(jobs: List[Dict[str, Any]], path: str) -> None:
    if not jobs:
        print("Nothing to save.")
        return
    cols = ["source", "title", "company", "location", "posted_at", "link", "description"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for j in jobs:
            w.writerow({k: j.get(k, "") for k in cols})
    print(f"Saved {len(jobs)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch jobs from Remotive or via Apify Web Scraper.")
    parser.add_argument("--mode", choices=["remotive", "apify"], default="remotive",
                        help="Data source. 'remotive' (default) or 'apify'.")
    parser.add_argument("--query", default="data science",
                        help="Search query (Remotive only).")
    parser.add_argument("--category", default="software-dev",
                        help="Category (Remotive only), e.g., software-dev, data, design.")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max jobs to print/save.")
    parser.add_argument("--csv", default="",
                        help="Optional path to save results as CSV.")
    # Apify-specific
    parser.add_argument("--apify-token", default=os.environ.get("APIFY_TOKEN", ""),
                        help="Apify API token (env APIFY_TOKEN also supported).")
    parser.add_argument("--url", default="",
                        help="Target listing URL for Apify mode (you must have permission to extract).")

    args = parser.parse_args()

    try:
        if args.mode == "remotive":
            jobs = fetch_remotive(args.query, args.category, args.limit)
        else:
            jobs = fetch_apify(args.apify_token, args.url, args.limit)

        print_jobs(jobs)

        if args.csv:
            save_csv(jobs, args.csv)

    except requests.HTTPError as e:
        print(f"HTTP error: {e}\nResponse text: {getattr(e.response, 'text', '')[:300]}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


