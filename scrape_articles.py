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
    # Set this in your .env, for example:
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