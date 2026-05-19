from bot.services.lancedb_service import LanceDBService
from bot.services.web_search_service import WebSearchService


class ReActAgent:
    def __init__(self, memory_service) -> None:
        self.memory_service = memory_service
        self.docs = LanceDBService()
        self.web = WebSearchService()

    async def run(self, user_id: int, query: str) -> str:
        lower = query.lower()
        if any(word in lower for word in ["сегодня", "новости", "актуально", "интернет"]):
            observations = await self.web.web_search(query)
            source = "web_search"
        elif any(word in lower for word in ["помнишь", "ранее", "история"]):
            observations = await self.memory_service.search(user_id, query)
            source = "memory_search"
        else:
            observations = await self.docs.search_kompas_docs(query)
            source = "search_kompas_docs"

        context = "\n".join(observations) if observations else "Контекст не найден"
        return f"Инструмент: {source}\n\nОтвет:\n{context}"
