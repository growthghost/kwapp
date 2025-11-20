# crawler.py
# Fast, clean crawler that extracts only the structural signals we need:
# slug, title, h1, h2, h3 (body intentionally ignored)

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Union

# Polite crawler header
HEADERS = {
    "User-Agent": "OutrankIQ-Mapper/2.0 (SEO research tool; +hello@yourdomain.com)"
}
TIMEOUT = (8, 15)  # connect, read


def _normalize_url(raw: str) -> str:
    """Make sure we have a proper https:// URL without trailing slash issues"""
    raw = raw.strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw.lstrip("/")
    parsed = urlparse(raw)
    if not parsed.netloc:
        return ""
    # Remove trailing slash unless it's the homepage
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def extract_profile(url: str) -> Dict[str, str]:
    """
    Fetch one page and return only the structural SEO signals we care about.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return {"slug": "", "title": "", "h1": "", "h2": "", "h3": "", "meta": "", "body": ""}
    except Exception:
        return {"slug": "", "title": "", "h1": "", "h2": "", "h3": "", "meta": "", "body": ""}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # H1 (first one only)
    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(" ", strip=True) if h1_tag else ""

    # All H2s concatenated
    h2 = " ".join(h.get_text(" ", strip=True) for h in soup.find_all("h2"))

    # All H3s concatenated
    h3 = " ".join(h.get_text(" ", strip=True) for h in soup.find_all("h3"))

    # Meta description (fallback to og:description)
    meta_tag = soup.find("meta", attrs={"name": "description"}) or \
               soup.find("meta", property="og:description")
    meta = meta_tag["content"].strip() if meta_tag and meta_tag.get("content") else ""

    # Slug – last meaningful part of the path
    final_url = resp.url
    path = urlparse(final_url).path or "/"
    slug = path.strip("/").split("/")[-1] if path.strip("/") else "home"

    return {
        "slug": slug,
        "title": title,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "meta": meta,
        "body": ""  # intentionally empty – we don’t use body text anymore
    }


def fetch_profiles(urls: Union[str, List[str]]) -> Dict[str, Dict[str, str]]:
    """
    Accept one URL or many (comma/newline separated), limit to 10.
    Always crawl the homepage once per domain + the specific pages provided.
    """
    profiles: Dict[str, Dict[str, str]] = {}

    # Convert single string → list and clean
    if isinstance(urls, str):
        raw_list = [u.strip() for u in urls.replace(",", "\n").split("\n") if u.strip()]
    else:
        raw_list = [u.strip() for u in urls if u.strip()]

    raw_list = raw_list[:10]  # hard limit

    seen_domains = set()

    for raw in raw_list:
        norm = _normalize_url(raw)
        if not norm:
            continue

        parsed = urlparse(norm)
        domain_url = f"{parsed.scheme}://{parsed.netloc}/"

        # Crawl homepage once per domain
        if domain_url not in seen_domains:
            profiles[domain_url] = extract_profile(domain_url)
            seen_domains.add(domain_url)

        # Crawl the specific page (skip if it's the same as homepage)
        if norm != domain_url and norm not in profiles:
            profiles[norm] = extract_profile(norm)

    return profiles