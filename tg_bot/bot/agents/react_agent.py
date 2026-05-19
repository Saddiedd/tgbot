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

        if not observations:
            return (
                f"Инструмент: {source}\n\n"
                "Не нашёл релевантный контекст. Уточните запрос: добавьте тему, команду или версию КОМПАС-3D."
            )

        summary = self._compose_summary(query=query, observations=observations)
        evidence = "\n".join(f"- {item}" for item in observations)
        return f"Инструмент: {source}\n\n{summary}\n\nНайденные факты:\n{evidence}"

    def _compose_summary(self, query: str, observations: list[str]) -> str:
        key_points = []
        for item in observations:
            cleaned = item.split("]", 1)[-1].strip()
            if cleaned and cleaned not in key_points:
                key_points.append(cleaned)
            if len(key_points) == 2:
                break

        if not key_points:
            return f"По запросу '{query}' не удалось сформировать краткий вывод."

        return f"По запросу '{query}' рекомендую опираться на: " + "; ".join(key_points) + "."
