# crawler.py
# This file handles crawling a site and extracting signals
# (slug, title, H1, H2/H3, meta). Paragraph text is excluded.

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Dict


def extract_profile(url: str) -> Dict[str, str]:
    """
    Fetch and parse a single page, extracting signals:
    slug, title, H1, H2/H3, and meta description.
    Paragraph (<p>) content is excluded by design.
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception:
        return {"slug": "", "title": "", "h1": "", "h2": "", "h3": "", "meta": ""}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Slug from URL path
    slug = urlparse(url).path.strip("/").split("/")[-1]

    # Title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # First H1
    h1 = ""
    h1_tag = soup.find("h1")
    if h1_tag:
        h1 = h1_tag.get_text(strip=True)

    # Collect all H2 text
    h2_tags = [h.get_text(" ", strip=True) for h in soup.find_all("h2")]
    h2 = " | ".join(h2_tags)

    # Collect all H3 text
    h3_tags = [h.get_text(" ", strip=True) for h in soup.find_all("h3")]
    h3 = " | ".join(h3_tags)

    # Meta description
    meta = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta = meta_tag["content"].strip()

    return {
        "slug": slug,
        "title": title,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "meta": meta
    }


def fetch_profiles(base_url: str, include_subdomains: bool = True) -> Dict[str, Dict[str, str]]:
    """
    Crawl starting from base_url and extract signals for each discovered page.
    For now, this is a simple version: only fetches the base URL.
    (Can be expanded later to full crawl.)
    """
    profiles = {}

    # Normalize URL
    if not base_url.startswith("http"):
        base_url = "https://" + base_url.strip("/")

    # Right now, just fetch the base URL
    profiles[base_url] = extract_profile(base_url)

    return profiles
