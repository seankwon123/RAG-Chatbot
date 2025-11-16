#!/usr/bin/env python

from __future__ import annotations

import os
import re
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
import psycopg2
import psycopg2.extras

from models import RAGConfig


log = logging.getLogger("scrape_articles")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def get_db_conn(cfg: RAGConfig):
    conn = psycopg2.connect(
        host=cfg.DB_HOST,
        port=cfg.DB_PORT,
        dbname=cfg.DB_NAME,
        user=cfg.DB_USER,
        password=cfg.DB_PASSWORD,
    )
    conn.autocommit = True
    return conn


def get_base_url() -> str:
    # Set this in .env:
    # BLOG_BASE_URL=https://example.com
    base = os.getenv("BLOG_BASE_URL")
    if not base:
        raise RuntimeError("BLOG_BASE_URL env var is not set")
    return base.rstrip("/")

def generate_page_urls() -> List[str]:
    base = get_base_url()
    max_pages = int(os.getenv("BLOG_MAX_PAGES", "41"))

    urls = [f"{base}/blog"]
    for i in range(2, max_pages + 1):
        urls.append(f"{base}/blog/page/{i}")
    log.info("Generated %d listing pages", len(urls))
    return urls

def fetch(url: str, timeout: int = 30) -> Optional[str]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (n8n-migration-scraper)"},
            timeout=timeout,
        )
        if resp.status_code >= 400:
            log.warning("HTTP %s for %s", resp.status_code, url)
            return None
        return resp.text
    except Exception as e:
        log.warning("Error fetching %s: %s", url, e)
        return None


def extract_article_urls(listing_html: str, listing_url: str) -> List[str]:
    soup = BeautifulSoup(listing_html, "lxml")
    anchors = soup.select("a[href*='/blog/']")
    urls: List[str] = []

    for a in anchors:
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(listing_url, href)

        # Filters similar to n8n workflow
        if "/blog/topic/" in abs_url:
            continue
        if "/blog/page/" in abs_url:
            continue
        if "#" in abs_url:
            continue

        # old date-style paths like /blog/2013/06/...
        if re.search(r"/blog/\d{4}/\d{2}/", abs_url):
            continue

        urls.append(abs_url)

    return urls


def normalize_url_for_dedup(url: str) -> str:
    u = url.strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = re.sub(r"/$", "", u)
    u = u.split("#", 1)[0]
    return u


def canonicalize_url(raw: Optional[str], base: str) -> Optional[str]:
    if not raw and not base:
        return None
    try:
        abs_url = urljoin(base, raw or "")
        parsed = urlparse(abs_url)
        scheme = "https"
        host = parsed.hostname or ""
        host = host.lower().replace("www.", "")

        # clean path
        path = parsed.path or "/"
        path = path.lower()
        path = re.sub(r"/index\.html$", "", path)
        path = re.sub(r"\.html$", "", path)
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        clean = urlunparse((scheme, host, path or "/", "", "", ""))
        return clean
    except Exception:
        return None


def cheap_hash(text: str) -> str:
    # simple 64-char hash similar to your JS cheapHash
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def to_iso(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.isoformat()
    except Exception:
        try:
            dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
            return dt.isoformat()
        except Exception:
            return None


def extract_article_fields(html: str, page_url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    # canonical candidates
    canonical_link = soup.select_one('link[rel="canonical"]')
    og_url = soup.select_one('meta[property="og:url"]')

    canonical_href = canonical_link["href"] if canonical_link and canonical_link.has_attr("href") else None
    og_url_val = og_url["content"] if og_url and og_url.has_attr("content") else None

    url_abs = og_url_val or page_url
    url_canonical = canonicalize_url(canonical_href, page_url) or canonicalize_url(url_abs, url_abs)

    # title
    title_tag = (
        soup.select_one("h1")
        or soup.select_one("article h1")
        or soup.select_one(".post-title")
        or soup.select_one(".entry-title")
        or soup.title
    )
    title_text = (title_tag.get_text(strip=True) if title_tag else "Untitled")[:500]

    # content
    content_root = (
        soup.select_one("article") or
        soup.select_one(".post-content") or
        soup.select_one(".entry-content") or
        soup.select_one("main article") or
        soup.select_one("main")
    )

    content_text = ""
    if content_root:
        content_text = content_root.get_text(" ", strip=True)
    content_text = re.sub(r"\s+", " ", content_text).strip()

    # excerpt
    meta_desc = soup.select_one("meta[name='description']")
    excerpt = None
    if meta_desc and meta_desc.has_attr("content"):
        excerpt = meta_desc["content"].strip()
    if not excerpt and content_text:
        snippet = content_text[:500]
        excerpt = snippet + ("..." if len(content_text) > 500 else "")

    # author
    author_meta = soup.select_one('meta[name="author"]')
    author_name = None
    if author_meta and author_meta.has_attr("content"):
        author_name = author_meta["content"].strip()

    if not author_name:
        author_text = (
            soup.select_one(".author-name")
            or soup.select_one(".by-author a")
            or soup.select_one(".author-link")
            or soup.select_one("span.author")
        )
        if author_text:
            author_name = author_text.get_text(strip=True)

    if author_name and len(author_name) > 200:
        author_name = author_name[:200]

    # published date
    published_meta = soup.select_one('meta[property="article:published_time"]')
    published_time_tag = soup.select_one("time[datetime]")
    published_str = None
    if published_meta and published_meta.has_attr("content"):
        published_str = published_meta["content"].strip()
    elif published_time_tag and published_time_tag.has_attr("datetime"):
        published_str = published_time_tag["datetime"].strip()

    published_iso = to_iso(published_str)

    # tags
    tag_candidates = []
    for el in soup.select("[rel='tag'], .tag, .post-tag, .category, a[href*='/blog/topic/']"):
        txt = el.get_text(strip=True)
        if txt:
            tag_candidates.append(txt)

    tags = []
    seen = set()
    for t in tag_candidates:
        cleaned = re.sub(r"\s+", " ", t).strip()
        if not cleaned:
            continue
        if cleaned.lower().startswith("tag for "):
            cleaned = cleaned[8:].strip()
        if cleaned in seen:
            continue
        seen.add(cleaned)
        tags.append(cleaned)

    tags_array = tags if tags else None

    word_count = len(content_text.split()) if content_text else None

    # Always have a hash, even if content_text is empty so it doesn't stop upsert
    hash_source = content_text or title_text or (url_abs or "")
    content_hash = cheap_hash(hash_source)

    return {
        "title": title_text,
        "url": url_abs[:1000] if url_abs else None,
        "content": content_text or None,
        "excerpt": excerpt or None,
        "author": author_name or None,
        "published_date": published_iso,
        "tags": tags_array,
        "word_count": word_count,
        "content_hash": content_hash,
        "url_canonical": url_canonical[:1000] if url_canonical else None,
    }



def upsert_articles(rows: List[Dict[str, Any]], cfg: RAGConfig) -> None:
    if not rows:
        return

    conn = get_db_conn(cfg)
    cur = conn.cursor()

    sql = """
    INSERT INTO articles (
      title,
      url,
      content,
      excerpt,
      author,
      published_date,
      tags,
      word_count,
      content_hash,
      url_canonical
    ) VALUES (
      %(title)s,
      %(url)s,
      %(content)s,
      %(excerpt)s,
      %(author)s,
      %(published_date)s::timestamptz,
      %(tags)s,
      %(word_count)s,
      %(content_hash)s,
      %(url_canonical)s
    )
    ON CONFLICT (url)
    DO UPDATE SET
      content       = EXCLUDED.content,
      excerpt       = EXCLUDED.excerpt,
      author        = EXCLUDED.author,
      published_date= EXCLUDED.published_date,
      tags          = EXCLUDED.tags,
      word_count    = EXCLUDED.word_count,
      content_hash  = EXCLUDED.content_hash,
      url_canonical = EXCLUDED.url_canonical,
      updated_at    = NOW();
    """
    psycopg2.extras.execute_batch(cur, sql, rows, page_size=50)
    cur.close()
    conn.close()


def main() -> None:
    cfg = RAGConfig

    listing_urls = generate_page_urls()
    all_article_urls: List[str] = []

    for url in listing_urls:
        html = fetch(url)
        if not html:
            continue
        urls = extract_article_urls(html, url)
        all_article_urls.extend(urls)

    # dedupe
    seen_norm = set()
    deduped: List[str] = []
    for u in all_article_urls:
        norm = normalize_url_for_dedup(u)
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        deduped.append(u)

    log.info("Found %d raw article URLs, %d unique after normalization", len(all_article_urls), len(deduped))

    articles: List[Dict[str, Any]] = []
    for i, url in enumerate(deduped, start=1):
        html = fetch(url)
        if not html:
            continue
        data = extract_article_fields(html, url)
        if not data.get("url"):
            continue
        articles.append(data)
        if i % 10 == 0:
            log.info("Processed %d/%d articles", i, len(deduped))

    log.info("Upserting %d articles into Postgres", len(articles))
    upsert_articles(articles, cfg)
    log.info("Done.")


if __name__ == "__main__":
    main()