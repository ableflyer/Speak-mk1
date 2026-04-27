"""
scrape_pubmed_slp.py
Scrapes PubMed Central open-access articles on speech therapy.
Output: pmc_slp_articles.jsonl  (one article per line)
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import os
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
OUT_FILE   = Path("../Data/SLP/pmc_slp_articles.jsonl")

# Broad SLP queries — PMC open-access only
QUERIES = [
    "articulation therapy children[tiab]",
    "speech sound disorder intervention[tiab]",
    "phonological therapy children[tiab]",
    "speech-language pathology pediatric treatment[tiab]",
    "childhood apraxia speech treatment[tiab]",
    "stuttering intervention children[tiab]",
    "augmentative alternative communication children[tiab]",
]

MAX_PER_QUERY = 200   # PMC open-access articles per query
DELAY_S       = 0.35  # NCBI allows ~3 req/s without API key; 0.35s = safe


# ── Step 1: Search → get PMCIDs ─────────────────────────────────────────────
def search_pmc(query: str, retmax: int = 200) -> list[str]:
    params = {
        "db":         "pmc",
        "term":       query + " AND open access[filter]",
        "retmax":     retmax,
        "usehistory": "y",
        "retmode":    "json",
    }
    r = requests.get(BASE_URL + "esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    ids = data["esearchresult"]["idlist"]
    print(f"  Query: '{query}' → {len(ids)} results")
    return ids


# ── Step 2: Fetch full text XML for a batch of PMCIDs ───────────────────────
def fetch_full_text(pmcids: list[str]) -> str:
    """Fetch up to 20 articles at a time (NCBI batch limit recommendation)."""
    params = {
        "db":      "pmc",
        "id":      ",".join(pmcids),
        "rettype": "full",
        "retmode": "xml",
    }
    r = requests.get(BASE_URL + "efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.text


# ── Step 3: Parse XML → extract title + abstract + body ─────────────────────
def parse_articles(xml_text: str) -> list[dict]:
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        return articles

    for article in root.findall(".//article"):
        try:
            # Title
            title_el = article.find(".//article-title")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            # Abstract
            abstract_parts = []
            for ab in article.findall(".//abstract//p"):
                abstract_parts.append("".join(ab.itertext()).strip())
            abstract = " ".join(abstract_parts)

            # Body text (sections)
            body_parts = []
            for p in article.findall(".//body//p"):
                text = "".join(p.itertext()).strip()
                if text:
                    body_parts.append(text)
            body = " ".join(body_parts)

            # PMCID
            pmcid = ""
            for aid in article.findall(".//article-id"):
                if aid.get("pub-id-type") == "pmc":
                    pmcid = aid.text or ""

            if not title and not abstract:
                continue

            full_text = f"{title}\n\n{abstract}\n\n{body}".strip()
            # Skip very short articles (likely metadata-only)
            if len(full_text) < 300:
                continue

            articles.append({
                "pmcid":    pmcid,
                "title":    title,
                "abstract": abstract,
                "body":     body[:50_000],  # cap at 50k chars to avoid giant reviews
                "full_text": full_text[:52_000],
            })
        except Exception as e:
            print(f"  Skipped article due to: {e}")
            continue

    return articles


# ── Step 4: Deduplicate by PMCID ─────────────────────────────────────────────
def load_seen_ids(out_file: Path) -> set[str]:
    seen = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("pmcid"):
                        seen.add(obj["pmcid"])
                except Exception:
                    pass
    return seen


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    seen_ids = load_seen_ids(OUT_FILE)
    print(f"Already have {len(seen_ids)} articles in {OUT_FILE}")

    total_new = 0
    BATCH_SIZE = 20  # fetch 20 at a time from efetch

    with open(OUT_FILE, "a") as fout:
        for query in QUERIES:
            pmcids = search_pmc(query, retmax=MAX_PER_QUERY)
            time.sleep(DELAY_S)

            # Filter already-seen
            new_ids = [pid for pid in pmcids if pid not in seen_ids]
            print(f"  {len(new_ids)} new IDs to fetch")

            for i in range(0, len(new_ids), BATCH_SIZE):
                batch = new_ids[i : i + BATCH_SIZE]
                try:
                    xml_text = fetch_full_text(batch)
                    time.sleep(DELAY_S)
                    articles = parse_articles(xml_text)
                    for art in articles:
                        fout.write(json.dumps(art) + "\n")
                        seen_ids.add(art["pmcid"])
                        total_new += 1
                    fout.flush()
                    print(f"    Batch {i//BATCH_SIZE + 1}: fetched {len(articles)} articles "
                          f"(total new: {total_new})")
                except requests.RequestException as e:
                    print(f"  Network error on batch {i//BATCH_SIZE + 1}: {e} — skipping")
                    time.sleep(2.0)

    print(f"\nDone. {total_new} new articles saved to {OUT_FILE}")
    print(f"Total unique articles: {len(seen_ids)}")


if __name__ == "__main__":
    main()