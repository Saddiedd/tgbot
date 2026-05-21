import json
from dataclasses import dataclass

from langchain_core.tools import StructuredTool

from bot.services.context_analysis_service import ContextAnalysisService
from bot.services.lancedb_service import LanceDBService
from bot.services.web_search_service import WebSearchService


@dataclass
class ToolOutput:
    source: str
    items: list[str]


class AgentToolbox:
    def __init__(
        self,
        docs: LanceDBService,
        web: WebSearchService,
        analyzer: ContextAnalysisService,
        memory_service,
    ) -> None:
        self.docs = docs
        self.web = web
        self.analyzer = analyzer
        self.memory_service = memory_service

    def build(self, user_id: int) -> dict[str, StructuredTool]:
        async def search_kompas_docs(query: str) -> str:
            items = await self.docs.search_kompas_docs(query)
            return pack_tool_output("search_kompas_docs", items)

        async def web_search(query: str) -> str:
            items = await self.web.web_search(query)
            return pack_tool_output("web_search", items)

        async def memory_search(query: str) -> str:
            items = await self.memory_service.search(user_id, query)
            return pack_tool_output("memory_search", items)

        async def analyze_context(query: str, source: str, items_json: str, memory_json: str = "[]") -> str:
            items = unpack_items(items_json)
            memory = unpack_items(memory_json)
            answer = await self.analyzer.analyze(query=query, source=source, items=items, memory=memory)
            return pack_tool_output("analyze_context", [answer] if answer else [])

        return {
            "search_kompas_docs": StructuredTool.from_function(
                coroutine=search_kompas_docs,
                name="search_kompas_docs",
                description="Ищет релевантные фрагменты в локальной LanceDB-базе документации КОМПАС-3D.",
            ),
            "web_search": StructuredTool.from_function(
                coroutine=web_search,
                name="web_search",
                description="Ищет актуальные сведения по КОМПАС-3D в интернете через Tavily или fallback-поиск.",
            ),
            "memory_search": StructuredTool.from_function(
                coroutine=memory_search,
                name="memory_search",
                description="Ищет похожие сообщения в истории диалога текущего пользователя.",
            ),
            "analyze_context": StructuredTool.from_function(
                coroutine=analyze_context,
                name="analyze_context",
                description="Анализирует найденные фрагменты и превращает их в короткий учебный ответ по КОМПАС-3D.",
            ),
        }


def pack_tool_output(source: str, items: list[str]) -> str:
    return json.dumps({"source": source, "items": items}, ensure_ascii=False)


def pack_items(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def unpack_items(payload: str) -> list[str]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return [item for item in data if isinstance(item, str)] if isinstance(data, list) else []


def parse_tool_output(payload: str, fallback_source: str) -> ToolOutput:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ToolOutput(source=fallback_source, items=[payload] if payload else [])

    source = data.get("source") if isinstance(data, dict) else None
    items = data.get("items") if isinstance(data, dict) else None
    clean_items = [item for item in items if isinstance(item, str)] if isinstance(items, list) else []
    return ToolOutput(source=source or fallback_source, items=clean_items)
