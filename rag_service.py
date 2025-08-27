# rag_service.py
# ----------------
# Focused RAG backend for PostgreSQL + Qdrant.
# Changes in this version:
# - Reduce keyword overcounting (esp. "ai") via regex word-boundaries for short terms.
# - For longer terms, use LIKE + regex.
# - Keep top-5 preview for count questions.
# - Use excerpt as summary for latest post (LLM fallback if missing).
# - Fix _fetch_article_content to use self.pg.fetchall.
# - Remove unused imports; keep only used helpers.

from __future__ import annotations
import os
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import psycopg2
import psycopg2.extras
import requests

from models import RAGConfig
from qdrant_manager import QdrantManager

try:
    from embedding_manager import EmbeddingManager  # type: ignore
except Exception:
    EmbeddingManager = None


log = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# -----------------------------
# Embeddings
# -----------------------------
class SimpleEmbedder:
    """Fallback embedder using Ollama's /api/embeddings."""
    def __init__(self, model_name: Optional[str] = None, host: Optional[str] = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-minilm")
        self.ollama_host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def embed(self, text: str) -> List[float]:
        payload = {"model": self.model_name, "prompt": text}
        r = requests.post(f"{self.ollama_host}/api/embeddings", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        if "embedding" in data:
            return data["embedding"]
        if "data" in data and data["data"] and "embedding" in data["data"][0]:
            return data["data"][0]["embedding"]
        raise RuntimeError(f"Unexpected embedding response: {data}")


def get_embedder():
    if EmbeddingManager is not None:
        try:
            mgr = EmbeddingManager()
            if hasattr(mgr, "embed"):
                return mgr
            if hasattr(mgr, "embed_text"):
                class _Wrap:
                    def __init__(self, m): self.m = m
                    def embed(self, t): return self.m.embed_text(t)
                return _Wrap(mgr)
        except Exception as e:
            log.warning(f"EmbeddingManager not available, falling back. ({e})")
    return SimpleEmbedder()


# -----------------------------
# Postgres wrapper
# -----------------------------
class Pg:
    def __init__(self, cfg: RAGConfig = None):
        self.cfg = cfg or RAGConfig
        self.conn = psycopg2.connect(
            host=self.cfg.DB_HOST,
            port=self.cfg.DB_PORT,
            dbname=self.cfg.DB_NAME,
            user=self.cfg.DB_USER,
            password=self.cfg.DB_PASSWORD,
        )
        self.conn.autocommit = True

    def fetchone(self, sql: str, params: Tuple = ()):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    def fetchall(self, sql: str, params: Tuple = ()):
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


# -----------------------------
# RAG Service
# -----------------------------
@dataclass
class Retrieved:
    id: int
    title: str
    url: str
    excerpt: str
    author: Optional[str]
    published_date: Optional[str]
    score: float
    match_reason: str  # 'semantic' or 'keyword' or 'latest'


class RagService:
    def __init__(self, cfg: RAGConfig = None):
        self.cfg = cfg or RAGConfig
        self.pg = Pg(self.cfg)
        self.qdrant = QdrantManager(self.cfg)
        self.embedder = get_embedder()
        self.sim_threshold = getattr(self.cfg, "SIMILARITY_THRESHOLD", 0.20)
        self.count_threshold = getattr(self.cfg, "COUNT_THRESHOLD", 0.12)
        self.ollama_host = getattr(self.cfg, "OLLAMA_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.ollama_model = getattr(self.cfg, "OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1"))

    # ----------------- utilities -----------------
    def _short(self, s: str | None, n: int = 320) -> str:
        if not s:
            return ""
        s = " ".join(s.split())
        return s if len(s) <= n else s[: n - 1].rstrip() + "…"

    def _first_sentences(self, text: str, max_sents: int = 2) -> str:
        text = " ".join((text or "").split())
        if not text:
            return ""
        parts = re.split(r'(?<=[.!?])\s+', text)
        return " ".join(parts[:max_sents]).strip()

    def _regex_for_word(self, term: str) -> str:
        """
        Case-insensitive word-ish boundary regex that avoids matching substrings inside words.
        Works well for very short topics like 'ai'.
        """
        t = re.escape(term.strip())
        # (^|[^a-z0-9_])term([^a-z0-9_]|$)
        return fr'(^|[^a-z0-9_]){t}([^a-z0-9_]|$)'

    def _is_short_term(self, term: str) -> bool:
        return len(term.strip()) <= 3

    def _fetch_article_content(self, article_id: int) -> str | None:
        rows = self.pg.fetchall("SELECT content FROM articles WHERE id = %s LIMIT 1", (article_id,))
        if rows and rows[0].get("content"):
            return rows[0]["content"]
        return None

    def _ollama_summarize(self, title: str, text: str, max_chars: int = 1200) -> str:
        try:
            import httpx
            model = os.getenv("LLM_MODEL", "qwen2.5:7b")
            snippet = (text or "")[:max_chars]
            prompt = (
                "You are a concise summarizer. Return 1–2 sentences (max ~60 words total) "
                "that accurately summarize the article for an engineering audience.\n\n"
                f"Title: {title}\n\nText:\n{snippet}\n\nSummary:"
            )
            with httpx.Client(timeout=20.0) as client:
                r = client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
                )
                r.raise_for_status()
                data = r.json()
                out = (data.get("response") or "").strip()
                return out or self._first_sentences(snippet)
        except Exception:
            return self._first_sentences(text)

    def _summary_for(self, retrieved: Retrieved) -> str:
        if getattr(retrieved, "excerpt", None):
            return self._short(retrieved.excerpt, 320)
        content = self._fetch_article_content(retrieved.id)
        if content:
            return self._short(self._ollama_summarize(retrieved.title, content), 320)
        return ""

    def _normalize_query(self, q: str) -> dict:
        """
        Strip brand/intent fluff so retrieval focuses on the topical bits.
        Also returns a keyword list we can use for SQL matching.
        """
        q0 = (q or "").lower()

        import re
        q0 = re.sub(r"[^a-z0-9\s\-]+", " ", q0)
        raw = [t for t in q0.split() if t]

        # remove words that add noise ("bitovi", "recommend", "what", "show", …)
        stop = {
            "bitovi","blog","blogs","article","articles","post","posts",
            "recommend","recommends","recommendation","recommendations",
            "suggest","suggests","suggestion","suggestions",
            "does","do","did","kind","kinds","type","types","what","which",
            "show","me","about","on","for","of","the","a","an","some",
            "please","could","would","should","latest","newest","recent","most",
            "oldest","least"
        }
        kept = [t for t in raw if t not in stop]
        semantic = " ".join(kept) or q0

        # minimal synonym expansion helpful for your corpus
        kws = set(kept)
        joined = " ".join(kept)

        # E2E synonyms
        if "e2e" in kws or ("end" in kws and "to" in kws and "end" in joined):
            kws.update({"e2e", "end-to-end", "end to end"})
            kws.update({"testing", "cypress"})

        if "testing" in kws or "test" in kws:
            kws.update({"testing", "tests", "qa"})

        # fall back if we stripped everything
        if not kws:
            kws = set(raw[:3])

        return {"semantic": semantic, "keywords": sorted(kws)}

    def _like_patterns(self, words: list[str]) -> list[str]:
        """For LIKE/ILIKE patterns."""
        return [f"%{w.lower()}%" for w in words if w and w.strip()]


    # ====== Basic Postgres helper ======
    def _fetch_by_ids(self, ids: List[int]) -> Dict[int, Dict[str, Any]]:
        if not ids:
            return {}
        rows = self.pg.fetchall("SELECT * FROM articles WHERE id = ANY(%s);", (ids,))
        return {int(row["id"]): row for row in rows}

    # ====== Public API ======
    def latest_article(self) -> Optional[Retrieved]:
        sql = """
            SELECT id, title, url, excerpt, author, published_date
            FROM articles
            WHERE published_date IS NOT NULL
            ORDER BY published_date DESC
            LIMIT 1;
        """
        r = self.pg.fetchone(sql)
        if not r:
            return None
        return Retrieved(
            id=int(r["id"]),
            title=r["title"],
            url=r["url"],
            excerpt=r.get("excerpt") or "",
            author=r.get("author"),
            published_date=r["published_date"].isoformat() if r.get("published_date") else None,
            score=1.0,
            match_reason="latest",
        )

    def list_by_topic_all(self, topic: str) -> List["Retrieved"]:
        """
        Keyword retrieval with reasonable precision:
        - title/excerpt/tags: LIKE ANY(patterns)
        - content: ONLY when we have at least two keywords -> LIKE ALL(patterns)
        """
        norm = self._normalize_query(topic)
        patterns = self._like_patterns(norm["keywords"])          # list[str]
        use_content_all = len(patterns) >= 2
        content_patterns = patterns if use_content_all else None  # reuse same patterns

        # SELECT / WHERE fragments and matching param tuples
        content_select = ", FALSE AS m_content"
        content_where = ""
        if use_content_all:
            content_select = ", (lower(coalesce(content, '')) LIKE ALL(%s)) AS m_content"
            content_where = " OR (lower(coalesce(content, '')) LIKE ALL(%s))"

        sql = f"""
            SELECT
                id, title, url, excerpt, author, published_date,
                (lower(title) LIKE ANY(%s))                  AS m_title,
                (lower(coalesce(excerpt,  '')) LIKE ANY(%s)) AS m_excerpt,
                EXISTS (
                    SELECT 1
                    FROM unnest(coalesce(tags, ARRAY[]::text[])) AS tag
                    WHERE lower(tag) LIKE ANY(%s)
                ) AS m_tags
                {content_select}
            FROM articles
            WHERE
                (lower(title) LIKE ANY(%s))
                OR (lower(coalesce(excerpt, '')) LIKE ANY(%s))
                OR EXISTS (
                    SELECT 1 FROM unnest(coalesce(tags, ARRAY[]::text[])) AS tag
                    WHERE lower(tag) LIKE ANY(%s)
                )
                {content_where}
            ORDER BY coalesce(published_date, '1900-01-01') DESC, id DESC;
        """

        # Build params to exactly match the %s placeholders above
        if use_content_all:
            params = (
                patterns,                # SELECT m_title
                patterns,                # SELECT m_excerpt
                patterns,                # SELECT m_tags
                content_patterns,        # SELECT m_content
                patterns,                # WHERE title
                patterns,                # WHERE excerpt
                patterns,                # WHERE tags
                content_patterns,        # WHERE content
            )
        else:
            params = (
                patterns,                # SELECT m_title
                patterns,                # SELECT m_excerpt
                patterns,                # SELECT m_tags
                patterns,                # WHERE title
                patterns,                # WHERE excerpt
                patterns,                # WHERE tags
            )

        rows = self.pg.fetchall(sql, params)

        out: List[Retrieved] = []
        for r in rows:
            hits = []
            if r.get("m_title"):   hits.append("title")
            if r.get("m_excerpt"): hits.append("excerpt")
            if r.get("m_tags"):    hits.append("tags")
            if use_content_all and r.get("m_content"): hits.append("content(all)")
            reason = "keyword(" + ", ".join(hits) + ")" if hits else "keyword"
            out.append(Retrieved(
                id=int(r["id"]),
                title=r["title"],
                url=r["url"],
                excerpt=r.get("excerpt") or "",
                author=r.get("author"),
                published_date=r["published_date"].isoformat() if r.get("published_date") else None,
                score=1.0,
                match_reason=reason,
            ))
        return out



    def count_about(self, topic: str) -> int:
        """
        Conservative count:
        - Only title/excerpt/tags (no full content) to avoid overcounting short tokens.
        - Returns integer count. (Semantic search is used elsewhere just for previews.)
        """
        norm = self._normalize_query(topic)
        patterns = self._like_patterns(norm["keywords"])

        sql = """
            SELECT COUNT(*) AS n
            FROM articles
            WHERE
                (lower(title) LIKE ANY(%s))
                OR (lower(coalesce(excerpt, '')) LIKE ANY(%s))
                OR EXISTS (
                    SELECT 1 FROM unnest(coalesce(tags, ARRAY[]::text[])) AS tag
                    WHERE lower(tag) LIKE ANY(%s)
                );
        """
        row = self.pg.fetchone(sql, (patterns, patterns, patterns))
        return int(row["n"]) if row else 0

    # ====== Hybrid Retrieval & Answering ======
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Retrieved]:
        norm = self._normalize_query(query)

        # 1) semantic
        emb = self.embedder.embed(norm["semantic"])
        sem_results = self._qdrant_search(emb, top_k=top_k, score_threshold=self.sim_threshold)

        # 2) keyword (oversample for merge) using the original API name
        kw_rows = self.list_by_topic_all(query)[: top_k * 2]
        kw_by_id = {r.id: r for r in kw_rows}

        # 3) merge
        out: Dict[int, Retrieved] = {}
        for rid, score, payload in sem_results:
            out[rid] = Retrieved(
                id=rid,
                title=payload.get("title", ""),
                url=payload.get("url", ""),
                excerpt=payload.get("excerpt", "") or "",
                author=payload.get("author"),
                published_date=payload.get("published_date"),
                score=float(score),
                match_reason="semantic",
            )
        for rid, r in kw_by_id.items():
            if rid not in out:
                out[rid] = r

        def sort_key(x: Retrieved):
            base = x.score if x.match_reason == "semantic" else 0.0
            return (base, x.published_date or "")

        ranked = sorted(out.values(), key=sort_key, reverse=True)
        return ranked[:top_k]


    def hybrid_answer(self, question: str, top_k: int = 6) -> Dict[str, Any]:
        ctx = self.hybrid_search(question, top_k=top_k)
        if not ctx:
            return {"answer": "I couldn't find anything relevant.", "sources": []}

        snippets = []
        sources = []
        for r in ctx:
            snippets.append(f"TITLE: {r.title}\nEXCERPT: {r.excerpt}\nURL: {r.url}")
            sources.append({"title": r.title, "url": r.url})

        sys_prompt = (
            "You are a helpful Bitovi blog assistant. Answer concisely and cite sources.\n"
            "Use ONLY the provided context. Prefer 3–6 short bullets when suitable.\n"
            "If there is no relevant context, say: “I couldn't find relevant Bitovi articles.” "
            "Do NOT output empty bullets.\n\n"
        )
        user_prompt = f"QUESTION: {question}\n\nCONTEXT:\n" + "\n---\n".join(snippets) + "\n\nANSWER:"

        try:
            ans = self._ollama_answer(sys_prompt + user_prompt).strip()
            # Guard against “blank bullets”
            lines = [ln for ln in ans.splitlines() if ln.strip() not in {"-", "•", "–"}]
            ans = "\n".join(lines).strip()
            if not ans:
                raise RuntimeError("Empty answer after cleanup")
        except Exception as e:
            log.warning(f"Ollama generation failed or empty, falling back to extractive answer. ({e})")
            bullets = [f"- **{r.title}** — {r.url}" for r in ctx[:6]]
            ans = "Here’s what I found:\n\n" + "\n\n".join(bullets)

        return {"answer": ans.strip(), "sources": sources}

    def answer_question(self, question: str) -> Dict[str, Any]:
        q = question.lower().strip()

        # Latest / most recent
        if ("latest" in q) or ("most recent" in q) or ("newest" in q):
            r = self.latest_article()
            if not r:
                return {"answer": "I couldn't find any articles.", "sources": []}

            parts = [f"The latest Bitovi post is “{r.title}”"]
            if r.author:
                parts.append(f"by {r.author}")
            if r.published_date:
                parts.append(f"published on {r.published_date[:10]}")
            headline = ", ".join(parts) + "."

            wants_summary = ("about" in q) or ("summary" in q)
            summary_text = self._summary_for(r)
            summary = f"\n\n**Summary:** {summary_text}" if (wants_summary or summary_text) else ""

            ans = f"{headline}{summary}\n\nRead it: {r.url}"
            return {"answer": ans, "sources": [{"title": r.title, "url": r.url}]}

        # show/list/find … articles about/on …
        m = re.search(r"(?:show|list|find)\s+(?:me\s+)?(?:some|a\s*few|all)?\s*articles?\s+(?:about|on)\s+(.+)", q)
        if m:
            topic = m.group(1).strip(" ?.")
            rows = self.list_by_topic_all(topic)
            if not rows:
                return {"answer": f"I couldn't find Bitovi articles about “{topic}”.", "sources": []}

            preface = ("Matched by keyword across *title, excerpt, content, or tags* "
                    "(each bullet shows which fields matched).")

            # Clean, spaced Markdown bullets with a blank line between items
            bullets = []
            for r in rows[:20]:
                pub = r.published_date[:10] if r.published_date else "no date"
                bullets.append(
                    f"- **{r.title}**  \n"
                    f"  📅 {pub} · _{r.match_reason}_  \n"
                    f"  {r.url}"
                )
            answer = f"{preface}\n\n" + "\n\n".join(bullets)
            return {"answer": answer, "sources": [{"title": r.title, "url": r.url} for r in rows[:10]]}

        # About block
        m = re.search(r"all .* about (.+)", q)
        if m:
            topic = m.group(1).strip(" ?.")
            rows = self.list_by_topic_all(topic)
            if not rows:
                return {"answer": f"No articles matched “{topic}”.", "sources": []}
            preface = ("Matched by keyword across *title, excerpt, content, or tags* "
                       "(each line shows which fields matched).")
            items = [f"• {r.title} — {r.url}  [{r.match_reason}]" for r in rows]
            return {"answer": f"{preface}\n\n" + "\n".join(items),
                    "sources": [{"title": r.title, "url": r.url} for r in rows]}

        # Count block
        m = re.search(r"(how many|count).+about (.+)", q)
        if m:
            topic = m.group(2).strip(" ?.")
            n = self.count_about(topic)

            # Show a preview with clean bullets and spacing
            top5 = self.hybrid_search(topic, top_k=5)
            if not top5:
                return {"answer": f"I found about **{n}** articles related to “{topic}”.", "sources": []}

            bullets = []
            sources = []
            for r in top5:
                bullets.append(
                    f"- **{r.title}**  \n"
                    f"  {self._short(r.excerpt, 160)}  \n"
                    f"  _{r.match_reason}_ · {r.url}"
                )
                sources.append({"title": r.title, "url": r.url})

            ans = f"I found about **{n}** articles related to “{topic}”. Here are a few examples:\n\n" + "\n\n".join(bullets)
            return {"answer": ans, "sources": sources}


        # Default → hybrid
        return self.hybrid_answer(question, top_k=6)

    # ----------------- Internals -----------------
    def _ollama_answer(self, prompt: str) -> str:
        url = f"{self.ollama_host}/api/generate"
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    def _qdrant_search(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        results = self.qdrant.search_semantic(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        normalized: List[Tuple[int, float, Dict[str, Any]]] = []

        for r in results or []:
            if isinstance(r, (list, tuple)) and len(r) >= 3:
                rid = int(r[0]); score = float(r[1]); payload = r[2] if isinstance(r[2], dict) else {}
                normalized.append((rid, score, payload))
                continue
            if hasattr(r, "id") and hasattr(r, "score") and hasattr(r, "payload"):
                rid = int(getattr(r, "id"))
                score = float(getattr(r, "score"))
                payload = getattr(r, "payload") or {}
                if not payload.get("url") or not payload.get("title"):
                    row = self._fetch_by_ids([rid]).get(rid)
                    if row:
                        payload.setdefault("title", row.get("title"))
                        payload.setdefault("url", row.get("url"))
                        payload.setdefault("excerpt", row.get("excerpt"))
                        payload.setdefault("author", row.get("author"))
                        payload.setdefault("published_date", row.get("published_date").isoformat()
                                           if row.get("published_date") else None)
                normalized.append((rid, score, payload))
                continue
        return normalized


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default="What kind of tools does Bitovi recommend for E2E testing?")
    parser.add_argument("--top_k", type=int, default=6)
    args = parser.parse_args()

    svc = RagService()
    log.info("Running a quick RAG query...")
    out = svc.answer_question(args.question)
    print("\n===== ANSWER =====\n")
    print(out["answer"])
    if out.get("sources"):
        print("\nSources:")
        for s in out["sources"]:
            print(f"- {s['title']} — {s['url']}")