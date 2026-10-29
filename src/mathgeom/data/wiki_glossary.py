"""Fetch Wikipedia glossary / topic-list pages at pinned revisions and extract concept names.

Global context: concepts for the field map (H-A1) come from Wikipedia "Glossary of …"
pages (CC BY-SA 4.0), one or more per mathematical field (configs/fields.yaml). Pages
change over time, so the first fetch records each page's revision id in
data/wiki_revisions.json and every later fetch requests exactly that revision (`oldid`).

Local context: uses the MediaWiki `action=parse` API
(https://www.mediawiki.org/wiki/API:Parsing_wikitext) and BeautifulSoup to read glossary
terms, which Wikipedia's {{term}} template renders as <dt> elements
(https://en.wikipedia.org/wiki/Template:Glossary).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from wordfreq import zipf_frequency

log = logging.getLogger(__name__)
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "mathgeom-research/0.1 (https://github.com/BurnyCoder/geometry-of-representations-of-mathematics-in-llms)"}


def fetch_page(title: str, raw_dir: Path, lock_path: Path) -> tuple[int, str]:
    """Return (revid, html) for `title`, pinned via the lock file; caches HTML on disk."""
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    params = {"action": "parse", "format": "json", "formatversion": 2, "prop": "text|revid", "redirects": 1}
    if title in lock:  # reproduce the exact revision used originally
        cached = raw_dir / f"{lock[title]}.html"
        if cached.exists():
            return lock[title], cached.read_text(encoding="utf-8")
        params["oldid"] = lock[title]
    else:
        params["page"] = title
    r = requests.get(API, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    js = r.json()["parse"]
    revid, html = int(js["revid"]), js["text"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{revid}.html").write_text(html, encoding="utf-8")
    lock[title] = revid
    lock_path.write_text(json.dumps(lock, indent=1, sort_keys=True))
    log.info("fetched %r rev %d (%d chars)", title, revid, len(html))
    return revid, html


_PAREN = re.compile(r"\s*\([^)]*\)")  # "(mathematics)" style qualifiers
_FOOT = re.compile(r"\[[^\]]*\]")  # "[1]" footnote markers


def clean_term(t: str, lowercase_usage: set[str] | None = None) -> str | None:
    """Normalize a raw term; return None if it is not a usable concept name.

    Keeps 1–4-word alphabetic names (hyphens/apostrophes allowed); drops symbols, single
    letters, math-notation entries, prefixes ("pseudo-") and single adverbs ("faithfully").
    Capitalization: many glossaries use sentence case, so a capitalized first word is a proper
    name only if the page never uses it in lowercase mid-sentence (`lowercase_usage`) and it is
    not an ordinary English word (wordfreq Zipf ≥ 3). "Homeomorphism" → "homeomorphism";
    "Drinfeld" stays capitalized so the eponym flag can see it.
    """
    t = _FOOT.sub("", _PAREN.sub("", t)).strip().strip(".,;:")
    t = t.split(",")[0].split(";")[0].strip()  # "compact, compactness" → "compact"
    if not t or len(t) < 3:
        return None
    if not re.fullmatch(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\- ]*", t) or t.endswith("-"):  # letters only; no prefixes
        return None
    words = t.split()
    if not 1 <= len(words) <= 4 or t.isupper():  # acronyms like "CW" are too ambiguous
        return None
    if len(words) == 1 and (t.lower().endswith("ly") or len(t) <= 4):  # adverbs, fragments ("dom", "adic")
        return None
    if any(len(w) <= 1 for w in re.split(r"[\s\-']+", t) if w):  # single letters: "n galaxy", "player i's"
        return None
    first = words[0]
    if first[:1].isupper() and not first[1:2].isupper():
        common = zipf_frequency(first.lower(), "en") >= 3.0
        if common or (lowercase_usage is not None and first.lower() in lowercase_usage):
            t = t[0].lower() + t[1:]
    return t


def extract_terms(html: str, is_list_page: bool) -> list[str]:
    """Glossary pages → <dt> texts; 'List of … topics' pages → first link text of each <li>."""
    soup = BeautifulSoup(html, "html.parser")
    raw = []
    if is_list_page:
        for li in soup.select("div.mw-parser-output > ul > li"):
            a = li.find("a")
            if a and a.get("href", "").startswith("/wiki/") and ":" not in a["href"]:
                raw.append(a.get_text())
    else:
        raw = [dt.get_text(" ", strip=True) for dt in soup.find_all("dt")]
        if not raw:  # some glossaries (e.g. Riemannian and metric geometry) use "<p><b>term</b> definition"
            for p in soup.select("div.mw-parser-output > p"):
                b = p.find("b")
                if b and p.get_text().strip().startswith(b.get_text().strip()):
                    raw.append(b.get_text(" ", strip=True))
    text = soup.get_text(" ")
    lowercase_usage = set(re.findall(r"(?<=[a-z,;:] )([a-z][a-z\-]+)", text))  # words used mid-sentence in lowercase
    seen, out = set(), []
    for r in raw:
        c = clean_term(r, lowercase_usage)
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out
