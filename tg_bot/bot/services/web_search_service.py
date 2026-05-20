class WebSearchService:
    async def web_search(self, query: str) -> list[str]:
        return [
            f"Интернет-поиск по запросу: '{query}'.",
            "В этой сборке подключена демо-версия поиска; подключите Tavily API для реальных источников.",
        ]
