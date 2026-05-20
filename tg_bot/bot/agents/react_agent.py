from bot.services.lancedb_service import LanceDBService
from bot.services.llm_service import LLMService
from bot.services.web_search_service import WebSearchService


class ReActAgent:
    def __init__(self, memory_service) -> None:
        self.memory_service = memory_service
        self.docs = LanceDBService()
        self.web = WebSearchService()
        self.llm = LLMService()

    async def run(self, user_id: int, query: str) -> str:
        lower = query.lower().strip()

        if lower in {"вопрос по компас-3d", "поиск в документации", "спросить react-агента"}:
            lower = "как создать деталь в компас-3d"
            query = lower

        if any(word in lower for word in ["сегодня", "новости", "актуально", "интернет"]):
            source = "web_search"
            observations = await self.web.web_search(query)
        elif any(word in lower for word in ["помнишь", "ранее", "история"]):
            source = "memory_search"
            observations = await self.memory_service.search(user_id, query)
        else:
            source = "search_kompas_docs"
            observations = await self.docs.search_kompas_docs(query)

        final = await self.llm.generate_answer(query=query, context=observations, source=source)
        return f"Инструмент: {source}\n\n{final}"
