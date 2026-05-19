class WebSearchService:
    async def web_search(self, query: str) -> list[str]:
        return [f"[web] Актуальные результаты для '{query}' недоступны в демо-режиме."]
