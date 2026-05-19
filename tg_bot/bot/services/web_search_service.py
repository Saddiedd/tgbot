import aiohttp


class WebSearchService:
    def __init__(self, timeout_seconds: int = 8) -> None:
        self.timeout_seconds = timeout_seconds

    async def web_search(self, query: str) -> list[str]:
        """Выполняет web-поиск через DuckDuckGo Instant Answer API."""
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
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            snippets.append(f"[web] {abstract}")

        related = data.get("RelatedTopics") or []
        for item in related:
            text = (item.get("Text") or "").strip() if isinstance(item, dict) else ""
            if text:
                snippets.append(f"[web] {text}")
            if len(snippets) >= 3:
                break

        return snippets[:3]
