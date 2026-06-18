# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1 · Researcher
"""The internet pull — how the researcher reaches out of the model and into the world.

The brain (Qwen) knows what it was trained on. To *research* — to "pull up every
possible knowledge it can from the Internet" — Kioku needs hands that touch live
pages. This module is those hands:

  search(query) -> [SearchHit(url, title, snippet)]      find sources
  fetch(url)     -> str                                   read a page as text
  gather(query)  -> [Source(url, title, text)]            find + read, bounded

Two providers, auto-selected, no config required for the default:
  • **Tavily** — if ``TAVILY_API_KEY`` is set, a clean research-grade search API
    that returns extracted content directly (best quality).
  • **DuckDuckGo HTML** — the free fallback, no key. Scrapes the lite HTML
    endpoint, decodes the redirect links, then fetches and extracts page text
    with lxml.

If the network is unreachable (offline demo, locked-down host), ``gather``
returns ``[]`` and the researcher degrades honestly to model-only knowledge,
clearly labelled — it never fabricates a source.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

log = logging.getLogger("kioku.research.web")

# Polite, real-browser-ish UA — many hosts 403 a bare python client.
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 Kioku-Researcher/1.0"
)
_TAVILY = "https://api.tavily.com/search"

# Bounds so one research run can't hammer the network or blow the context.
MAX_PAGE_CHARS = 6000          # per fetched page, after extraction
FETCH_TIMEOUT_S = 12.0
SEARCH_TIMEOUT_S = 15.0
DEFAULT_RESULTS = 4            # sources gathered per sub-question
_FETCH_CONCURRENCY = 6


@dataclass(frozen=True, slots=True)
class SearchHit:
    url: str
    title: str
    snippet: str = ""


@dataclass(slots=True)
class Source:
    """One read source: where it came from, and what it said."""

    url: str
    title: str
    text: str = ""
    snippet: str = ""

    @property
    def domain(self) -> str:
        try:
            return urlparse(self.url).netloc or self.url
        except ValueError:
            return self.url

    def cite(self) -> str:
        return f"{self.title or self.domain} <{self.url}>"


def _provider() -> str:
    return "tavily" if os.environ.get("TAVILY_API_KEY") else "duckduckgo"


# ── HTML → text (lxml, already a project dep) ────────────────────────────────

def html_to_text(html: str) -> str:
    """Strip a page to readable text. Best-effort; never raises."""
    try:
        from lxml import html as lxml_html  # local import: keeps module import cheap

        doc = lxml_html.fromstring(html)
        for bad in doc.xpath("//script | //style | //noscript | //nav | //footer | //header | //form"):
            bad.getparent().remove(bad)
        text = doc.text_content()
    except Exception:  # noqa: BLE001 — malformed HTML must not kill a run
        text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()[:MAX_PAGE_CHARS]



class WebSearch:
    """Find and read sources from the live web. One per researcher run."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
            follow_redirects=True,
            timeout=httpx.Timeout(FETCH_TIMEOUT_S, connect=8.0),
        )
        self.provider = _provider()
        self._sem = asyncio.Semaphore(_FETCH_CONCURRENCY)

    # -- search -----------------------------------------------------------

    async def search(self, query: str, k: int = DEFAULT_RESULTS) -> list[SearchHit]:
        try:
            if self.provider == "tavily":
                return await self._tavily(query, k)
            return await self._ddg(query, k)
        except (httpx.HTTPError, OSError) as e:
            log.warning("search failed for %r (%s): %s", query[:60], self.provider, e)
            return []

    async def _tavily(self, query: str, k: int) -> list[SearchHit]:
        resp = await self._client.post(
            _TAVILY,
            json={
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "max_results": k,
                "search_depth": "advanced",
                "include_raw_content": True,
            },
            timeout=SEARCH_TIMEOUT_S,
        )
        resp.raise_for_status()
        out: list[SearchHit] = []
        for r in (resp.json().get("results") or [])[:k]:
            out.append(SearchHit(url=r.get("url", ""), title=r.get("title", ""), snippet=r.get("content", "")[:600]))
        return [h for h in out if h.url]

    async def _ddg(self, query: str, k: int) -> list[SearchHit]:
        # The ddgs library handles session management and bot-detection correctly.
        # It's synchronous, so run it in the default threadpool executor.
        loop = asyncio.get_running_loop()

        def _run() -> list[SearchHit]:
            from ddgs import DDGS  # local import: keep startup cheap
            hits: list[SearchHit] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=k):
                    url = r.get("href") or r.get("url", "")
                    title = r.get("title", "")
                    snippet = r.get("body", "")[:600]
                    if url.startswith("http") and title:
                        hits.append(SearchHit(url=url, title=title, snippet=snippet))
                    if len(hits) >= k:
                        break
            return hits

        return await loop.run_in_executor(None, _run)

    # -- fetch ------------------------------------------------------------

    async def fetch(self, url: str) -> str:
        async with self._sem:
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if "html" not in ctype and "text" not in ctype and ctype:
                    return ""  # skip PDFs/binaries — out of scope for the text pull
                return html_to_text(resp.text)
            except (httpx.HTTPError, OSError) as e:
                log.info("fetch failed %s: %s", url[:80], e)
                return ""

    # -- the one call the researcher uses --------------------------------

    async def gather(self, query: str, k: int = DEFAULT_RESULTS) -> list[Source]:
        """Search, then read the top pages concurrently. Tavily already returns
        content, so we keep it; for DDG we fetch each page's text."""
        hits = await self.search(query, k)
        if not hits:
            return []
        if self.provider == "tavily":
            # Tavily's raw content is already the page text.
            return [
                Source(url=h.url, title=h.title, text=(h.snippet or "")[:MAX_PAGE_CHARS], snippet=h.snippet[:300])
                for h in hits
            ]
        texts = await asyncio.gather(*(self.fetch(h.url) for h in hits))
        sources: list[Source] = []
        for h, text in zip(hits, texts):
            body = text or h.snippet  # fall back to the search snippet if the page blocked us
            if body:
                sources.append(Source(url=h.url, title=h.title, text=body[:MAX_PAGE_CHARS], snippet=h.snippet[:300]))
        return sources

    async def aclose(self) -> None:
        await self._client.aclose()
