import aiohttp
import re


class WebSearchService:
    def __init__(self, api_key: str | None = None, timeout_seconds: int = 8) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def web_search(self, query: str) -> list[str]:
        """Searches the web through Tavily when configured, with a small public fallback."""
        search_query = self._kompas_query(query)
        if self.api_key:
            tavily_results = await self._search_tavily(search_query)
            if tavily_results:
                return tavily_results

        return await self._search_duckduckgo(search_query)

    async def _search_tavily(self, query: str) -> list[str]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 3,
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
        except Exception:
            return []

        snippets: list[str] = []
        answer = self._clean_text(data.get("answer") or "")
        if answer:
            snippets.append(f"[web] {answer}")

        for item in data.get("results") or []:
            title = self._clean_text(item.get("title") or "")
            content = self._clean_text(item.get("content") or "")
            url = (item.get("url") or "").strip()
            if content:
                label = f"{title}: " if title else ""
                suffix = f" ({url})" if url else ""
                snippets.append(f"[web] {label}{content}{suffix}")
            if len(snippets) >= 4:
                break

        return snippets[:4]

    async def _search_duckduckgo(self, query: str) -> list[str]:
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        url = "https://api.duckduckgo.com/"

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
        except Exception:
            return []

        snippets: list[str] = []
        abstract = self._clean_text(data.get("AbstractText") or "")
        abstract_url = (data.get("AbstractURL") or "").strip()
        if abstract:
            suffix = f" ({abstract_url})" if abstract_url else ""
            snippets.append(f"[web] {abstract}{suffix}")

        related = data.get("RelatedTopics") or []
        for item in related:
            text = self._clean_text(item.get("Text") or "") if isinstance(item, dict) else ""
            first_url = (item.get("FirstURL") or "").strip() if isinstance(item, dict) else ""
            if text:
                suffix = f" ({first_url})" if first_url else ""
                snippets.append(f"[web] {text}{suffix}")
            if len(snippets) >= 3:
                break

        return snippets[:3]

    def _kompas_query(self, query: str) -> str:
        lower = query.lower()
        if "компас" in lower or "kompas" in lower:
            return query
        return f"{query} КОМПАС-3D"

    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"Please enable JavaScript to view this site\.?", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()
