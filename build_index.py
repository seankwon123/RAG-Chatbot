#!/usr/bin/env python
from __future__ import annotations

import os
import math
import logging

import psycopg2
import psycopg2.extras

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from models import RAGConfig
from rag_service import get_embedder

log = logging.getLogger("build_index")
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


def main() -> None:
    cfg = RAGConfig

    log.info("Connecting to Postgres at %s:%s / db=%s", cfg.DB_HOST, cfg.DB_PORT, cfg.DB_NAME)
    conn = get_db_conn(cfg)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """
        SELECT
            id,
            title,
            url,
            excerpt,
            content,
            author,
            published_date
        FROM articles
        ORDER BY id ASC;
        """
    )
    rows = cur.fetchall()
    if not rows:
        log.warning("No rows found in articles table. Did you run scrape_articles.py?")
        return

    log.info("Fetched %d articles.", len(rows))

    embedder = get_embedder()
    test_vec = embedder.embed("test vector dimension")
    dim = len(test_vec)
    log.info("Embedding dimension detected: %d", dim)

    qdrant_host = getattr(cfg, "QDRANT_HOST", os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port = int(getattr(cfg, "QDRANT_PORT", os.getenv("QDRANT_PORT", "6333")))
    collection_name = getattr(cfg, "QDRANT_COLLECTION", "articles_collection")

    log.info("Connecting to Qdrant at %s:%s", qdrant_host, qdrant_port)
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    log.info("Recreating collection %r ...", collection_name)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    batch_size = 64
    total = len(rows)
    num_batches = math.ceil(total / batch_size)
    log.info("Indexing in batches of %d (%d batches total)...", batch_size, num_batches)

    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, total)
        batch = rows[start:end]

        texts = []
        for r in batch:
            title = r.get("title") or ""
            excerpt = r.get("excerpt") or ""
            content = r.get("content") or ""
            text = f"{title}\n\n{excerpt}\n\n{content}"
            texts.append(text[:4000])

        vectors = [embedder.embed(t) for t in texts]

        points = []
        for row, vec in zip(batch, vectors):
            pid = int(row["id"])
            payload = {
                "title": row.get("title") or "",
                "url": row.get("url") or "",
                "excerpt": row.get("excerpt") or "",
                "author": row.get("author"),
                "published_date": row["published_date"].isoformat()
                if row.get("published_date") is not None
                else None,
            }
            points.append(PointStruct(id=pid, vector=vec, payload=payload))

        client.upsert(collection_name=collection_name, points=points)
        log.info("Indexed batch %d/%d (%d points).", b + 1, num_batches, len(points))

    log.info("Done. Indexed %d articles into %s.", total, collection_name)


if __name__ == "__main__":
    main()
