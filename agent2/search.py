"""Search provider abstraction.

Agent 2 does not scrape websites itself. It issues web searches through a
configurable provider (Tavily by default) and reads only the results the
provider returns - exact URLs, titles and snippets. Those results are the
evidence that backs every contact detail.

``mock`` provider: deterministic sample data, used for tests and for trying
Agent 2 without any API key. It never touches the network.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings


@dataclass
class SearchResult:
    """One search hit exactly as returned by the provider."""

    url: str
    title: str
    snippet: str = ""
    score: float = 0.0
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "title": self.title,
                "snippet": self.snippet, "score": self.score}


@dataclass
class SearchResponse:
    """Outcome of one search query."""

    query: str
    results: list[SearchResult]
    error: str = ""


class TavilySearchProvider:
    """Search via the Tavily API."""

    name = "tavily"

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self._client = client
        self._owns_client = client is None

    # -- plumbing ----------------------------------------------------------
    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.settings.search_timeout_seconds,
                headers={"Authorization": f"Bearer {self.settings.tavily_api_key}"},
            )
            self._owns_client = True
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    # -- core --------------------------------------------------------------
    def search(self, query: str) -> SearchResponse:
        last_error = ""
        for attempt in range(1, self.settings.search_max_retries + 1):
            try:
                resp = self._http().post(
                    f"{self.settings.tavily_api_url.rstrip('/')}/search",
                    json={
                        "query": query,
                        "max_results": self.settings.tavily_max_results,
                        "include_answer": False,
                        "search_depth": "basic",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results = [
                    SearchResult(
                        url=str(item.get("url", "")).strip(),
                        title=str(item.get("title", "")).strip(),
                        snippet=str(item.get("content", "")).strip(),
                        score=float(item.get("score", 0.0) or 0.0),
                        raw=item,
                    )
                    for item in data.get("results", [])
                    if item.get("url", "").strip()
                ]
                return SearchResponse(query=query, results=results)
            except httpx.HTTPStatusError as exc:
                last_error = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
                if exc.response.status_code in (400, 401, 403, 404):
                    break  # non-retryable
            except httpx.HTTPError as exc:
                last_error = f"network error: {exc}"
            except Exception as exc:  # pragma: no cover - defensive
                last_error = f"unexpected error: {exc}"

            if attempt < self.settings.search_max_retries:
                time.sleep(2 ** attempt * self.settings.search_delay_seconds)

        return SearchResponse(query=query, results=[], error=last_error)


class MockSearchProvider:
    """Deterministic offline provider for tests and demos.

    The dataset deliberately contains a person whose email was changed in a
    later source, so the dedupe logic can be exercised.
    """

    name = "mock"

    # (keywords, results) - matched by substring on the query
    _PAGES = [
        (
            ("about", "company", "team"),
            [
                SearchResult("https://examplecorp.dev/about", "About ExampleCorp", "We build software. Founder: Ada Lovelace.", 0.9),
                SearchResult("https://examplecorp.dev/team", "ExampleCorp Team", "Our leadership: Ada Lovelace (Founder), Grace Hopper (CTO), Alan Turing (Head of Engineering).", 0.8),
                SearchResult("https://examplecorp.dev/", "ExampleCorp Home", "Custom software development for the UK market.", 0.7),
            ],
        ),
        (
            ("ceo", "founder"),
            [
                SearchResult("https://examplecorp.dev/about", "About ExampleCorp", "Founded by Ada Lovelace in 2020.", 0.9),
                SearchResult("https://www.linkedin.com/in/ada-lovelace", "Ada Lovelace | Founder at ExampleCorp | LinkedIn", "Ada Lovelace - Founder, ExampleCorp, London, UK.", 0.8),
                SearchResult("https://examplecorp.dev/vendor", "ExampleCorp - Vendors & Partners", "Contact our vendor team: vendors@examplecorp.dev", 0.5),
            ],
        ),
        (
            ("cto",),
            [
                SearchResult("https://www.linkedin.com/in/grace-hopper", "Grace Hopper | CTO | LinkedIn", "Grace Hopper, CTO at ExampleCorp, United Kingdom.", 0.8),
                SearchResult("https://examplecorp.dev/team", "ExampleCorp Team", "Grace Hopper (CTO) - grace@examplecorp.dev", 0.7),
            ],
        ),
        (
            ("engineering", "head of"),
            [
                SearchResult("https://www.linkedin.com/in/alan-turing", "Alan Turing | Head of Engineering | LinkedIn", "Alan Turing, Head of Engineering, ExampleCorp.", 0.8),
            ],
        ),
        (
            ("security", "ciso"),
            [
                SearchResult("https://examplecorp.dev/team", "ExampleCorp Team", "Our leadership: Ada Lovelace (Founder), Grace Hopper (CTO), Alan Turing (Head of Engineering). No security roles listed.", 0.6),
            ],
        ),
        (
            ("contact", "email", "press"),
            [
                SearchResult("https://examplecorp.dev/contact", "Contact ExampleCorp", "General enquiries: hello@examplecorp.dev", 0.8),
            ],
        ),
    ]

    def search(self, query: str) -> SearchResponse:
        q = query.lower()
        hits: list[SearchResult] = []
        for keywords, results in self._PAGES:
            if any(k in q for k in keywords):
                hits.extend(r for r in results if r.url not in {h.url for h in hits})
        if not hits:
            hits = [SearchResult("https://examplecorp.dev/", "ExampleCorp Home", "Company website.", 0.2)]
        return SearchResponse(query=query, results=hits[: self._settings_max()])

    def _settings_max(self) -> int:
        return getattr(self, "max_results", 8)

    def close(self) -> None:
        pass


def build_search_provider(settings: Settings):
    """Instantiate the configured search provider."""
    if settings.search_provider == "mock":
        provider = MockSearchProvider()
        provider.max_results = settings.tavily_max_results
        return provider
    return TavilySearchProvider(settings)


def query_variants(company_name: str, website: str, roles: list[str], max_queries: int) -> list[str]:
    """Build a small, ordered set of search queries for one company.

    Individuals' names and details are never guessed in queries - we ask the
    provider for *public leadership and contact pages*, then take whatever real
    people the returned sources name.
    """
    domain = ""
    if website:
        m = re.match(r"https?://([^/]+)", website.strip())
        if m:
            domain = m.group(1)
    queries: list[str] = []
    if company_name:
        queries += [
            f"{company_name} leadership team",
            f"{company_name} CEO CTO contact",
        ]
        for role in roles[:3]:
            queries.append(f"{company_name} {role}")
    if domain:
        queries += [
            f"site:{domain} about team contact",
            f"{domain} email contact",
        ]
    # dedupe, limit
    seen: list[str] = []
    for q in queries:
        if q not in seen:
            seen.append(q)
    return seen[:max_queries]