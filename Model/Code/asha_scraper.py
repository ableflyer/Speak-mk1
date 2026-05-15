"""
This has not been used
"""

import time
import json
import random
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

BASE_URL    = "https://apps.asha.org/EvidenceMaps/"
ROBOTS_URL  = "https://www.asha.org/robots.txt"
OUTPUT_DIR  = Path("./asha_data")
DELAY_MIN   = 3.0   # seconds between requests (be polite)
DELAY_MAX   = 6.0
HEADERS     = {
    # Honest identification — do NOT spoof as googlebot or browser
    "User-Agent": "SpeechTherapyResearchBot/1.0 (academic research; "
                  "contact: your@email.com)",
    "Accept": "text/html,application/xhtml+xml",
}

# ─── STEP 1: CHECK ROBOTS.TXT ─────────────────────────────────────────────────

def check_robots_allowed(url: str) -> bool:
    rp = RobotFileParser()
    rp.set_url(ROBOTS_URL)
    try:
        rp.read()
        allowed = rp.can_fetch(HEADERS["User-Agent"], url)
        if not allowed:
            log.warning(f"robots.txt DISALLOWS: {url}")
        return allowed
    except Exception as e:
        log.error(f"Could not read robots.txt: {e}")
        # Conservative: if can't read robots.txt, stop
        return False

# ─── STEP 2: POLITE FETCHER ───────────────────────────────────────────────────

def polite_get(url: str, session: requests.Session) -> requests.Response | None:
    # Always check robots first
    if not check_robots_allowed(url):
        log.warning(f"Skipping disallowed URL: {url}")
        return None
    
    # Polite delay
    delay = random.uniform(DELAY_MIN, DELAY_MAX)
    log.info(f"Waiting {delay:.1f}s before fetching: {url}")
    time.sleep(delay)
    
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        log.error(f"Failed to fetch {url}: {e}")
        return None

# ─── STEP 3: PARSE EVIDENCE MAP LIST ─────────────────────────────────────────

def get_evidence_map_links(session: requests.Session) -> list[dict]:
    log.info("Fetching evidence map index...")
    resp = polite_get(BASE_URL, session)
    if not resp:
        return []
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    maps = []
    
    # ASHA evidence maps are listed as links — adjust selector after inspecting
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text(strip=True)
        
        # Filter to evidence map links only
        if '/EvidenceMaps/' in href and text:
            full_url = urljoin(BASE_URL, href)
            maps.append({
                'title': text,
                'url':   full_url
            })
    
    # Deduplicate
    seen = set()
    unique_maps = []
    for m in maps:
        if m['url'] not in seen:
            seen.add(m['url'])
            unique_maps.append(m)
    
    log.info(f"Found {len(unique_maps)} evidence maps")
    return unique_maps

# ─── STEP 4: PARSE SINGLE EVIDENCE MAP ───────────────────────────────────────

def parse_evidence_map(url: str, title: str, 
                        session: requests.Session) -> dict | None:
    resp = polite_get(url, session)
    if not resp:
        return None
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    data = {'title': title, 'url': url, 'sections': {}}
    
    # Extract main content sections
    # ASHA maps have: External Scientific Evidence, Clinical Expertise, 
    # Client Perspectives — adjust selectors after inspecting actual HTML
    
    # Generic content extraction
    main_content = soup.find('main') or soup.find('div', {'id': 'main'}) \
                   or soup.find('div', {'class': 'content'})
    
    if main_content:
        # Extract all text blocks with their headers
        current_section = 'general'
        for elem in main_content.find_all(['h1','h2','h3','h4','p','li']):
            text = elem.get_text(strip=True)
            if not text:
                continue
                
            if elem.name in ['h1','h2','h3','h4']:
                current_section = text
                data['sections'][current_section] = []
            else:
                if current_section not in data['sections']:
                    data['sections'][current_section] = []
                data['sections'][current_section].append(text)
    
    # Extract paper citations if present
    citations = []
    for cite in soup.find_all(['cite', 'blockquote']):
        citations.append(cite.get_text(strip=True))
    if citations:
        data['citations'] = citations
    
    return data

# ─── STEP 5: MAIN LOOP ───────────────────────────────────────────────────────

def scrape_asha_evidence_maps():
    OUTPUT_DIR.mkdir(exist_ok=True)
    index_path  = OUTPUT_DIR / "index.json"
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # Load existing progress (resume-friendly)
    scraped = set()
    if index_path.exists():
        with open(index_path) as f:
            existing = json.load(f)
        scraped = {e['url'] for e in existing}
        log.info(f"Resuming: {len(scraped)} already scraped")
        all_data = existing
    else:
        all_data = []
    
    # Get map list
    maps = get_evidence_map_links(session)
    if not maps:
        log.error("No maps found. Check selectors or robots.txt block.")
        return
    
    for i, m in enumerate(maps):
        if m['url'] in scraped:
            log.info(f"[{i+1}/{len(maps)}] Skipping (already done): {m['title']}")
            continue
        
        log.info(f"[{i+1}/{len(maps)}] Scraping: {m['title']}")
        data = parse_evidence_map(m['url'], m['title'], session)
        
        if data:
            all_data.append(data)
            scraped.add(m['url'])
            
            # Save after every map (safe against crashes)
            with open(index_path, 'w') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            
            # Also save individual file
            safe_name = "".join(c if c.isalnum() else '_' for c in m['title'])[:80]
            with open(OUTPUT_DIR / f"{safe_name}.json", 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    log.info(f"Done. {len(all_data)} maps saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    # STOP if robots.txt blocks us
    if not check_robots_allowed(BASE_URL):
        print("STOPPED: robots.txt disallows scraping. Respect this.")
        print("Contact ASHA at ncep@asha.org for data access instead.")
        exit(0)
    
    scrape_asha_evidence_maps()