import logging
import re
from dataclasses import dataclass

from bot.services.lancedb_service import LanceDBService
from bot.services.llm_service import LLMService
from bot.services.embedding_service import EmbeddingService
from bot.services.web_search_service import WebSearchService
from bot.services.context_analysis_service import ContextAnalysisService
from bot.agents.tools import AgentToolbox, pack_items, parse_tool_output


@dataclass(frozen=True)
class AgentStep:
    phase: str
    detail: str


class ReActAgent:
    def __init__(self, memory_service, settings) -> None:
        self.memory_service = memory_service
        embedding_service = EmbeddingService(
            api_key=settings.mistral_api_key,
            model=settings.mistral_embedding_model,
        )
        self.docs = LanceDBService(
            db_path=settings.lancedb_path,
            table_name=settings.lancedb_table,
            embedding_service=embedding_service,
        )
        self.web = WebSearchService(api_key=settings.tavily_api_key)
        self.analyzer = ContextAnalysisService(
            provider=settings.analysis_provider,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_chat_model,
            mistral_api_key=settings.mistral_api_key,
            mistral_model=settings.mistral_chat_model,
        )
        self.llm = LLMService()
        self.toolbox = AgentToolbox(
            docs=self.docs,
            web=self.web,
            analyzer=self.analyzer,
            memory_service=memory_service,
        )

    async def run(self, user_id: int, query: str, mode: str = "react") -> str:
        query = query.strip()
        if not query:
            return "Напишите вопрос по КОМПАС-3D: команду, задачу моделирования, ошибку API или то, что нужно найти."

        trace: list[AgentStep] = [
            AgentStep("thought", f"mode={mode}; plan the next tool call"),
        ]

        memory_context = await self.memory_service.summary(user_id)
        trace.append(AgentStep("observation", f"memory_summary returned {len(memory_context)} item(s)"))

        memory_observations: list[str] = []
        if mode in {"auto", "react"}:
            memory_source, memory_observations = await self._call_tool(
                user_id=user_id,
                tool_name="memory_search",
                query=query,
            )
            trace.append(AgentStep("observation", f"{memory_source} returned {len(memory_observations)} item(s)"))

        tool_name = await self._plan_tool(mode=mode, query=query, memory_context=memory_context)
        trace.append(AgentStep("action", tool_name))
        if tool_name == "memory_search" and memory_observations:
            source, observations = "memory_search", memory_observations
        else:
            source, observations = await self._call_tool(user_id=user_id, tool_name=tool_name, query=query)

        trace.append(AgentStep("observation", f"{source} returned {len(observations)} item(s)"))

        if mode in {"auto", "react"} and source == "search_kompas_docs" and not observations:
            trace.append(AgentStep("thought", "documentation returned no context; try web search"))
            trace.append(AgentStep("action", "web_search"))
            source, observations = await self._call_tool(user_id=user_id, tool_name="web_search", query=query)
            trace.append(AgentStep("observation", f"{source} returned {len(observations)} item(s)"))

        history = await self.memory_service.tail(user_id)
        should_analyze = mode in {"auto", "react", "web", "memory"} and (observations or source == "memory_search")
        if should_analyze:
            trace.append(AgentStep("thought", "analyze observations before final answer"))
            trace.append(AgentStep("action", "analyze_context"))
            source_before_analysis = source
            observations_before_analysis = observations
            analysis_source = "react_context" if mode in {"auto", "react"} and source != "memory_search" else source
            analysis_observations = self._merge_react_observations(
                source=source,
                observations=observations,
                memory_observations=memory_observations,
                react_mode=mode in {"auto", "react"},
            )
            analyzed_source, analyzed_items = await self._call_analyze_tool(
                user_id=user_id,
                query=query,
                source=analysis_source,
                observations=analysis_observations,
                memory_context=memory_context,
            )
            trace.append(AgentStep("observation", f"{analyzed_source} returned {len(analyzed_items)} item(s)"))
            if analyzed_items:
                if source_before_analysis == "web_search":
                    analyzed_items[0] = self._attach_web_sources(
                        answer=analyzed_items[0],
                        observations=observations_before_analysis,
                    )
                observations = analyzed_items
                source = analyzed_source

        trace.append(AgentStep("final_answer", f"source={source}; history_items={len(history)}"))
        logging.info(
            "ReAct trace for user %s: %s",
            user_id,
            " -> ".join(f"{step.phase}:{step.detail}" for step in trace),
        )
        return await self.llm.generate_answer(query=query, context=observations, source=source, history=history)

    async def _plan_tool(self, mode: str, query: str, memory_context: list[str]) -> str:
        if mode in {"auto", "react"}:
            tool_name = await self.analyzer.plan_tool(query=query, memory=memory_context)
            if tool_name == "search_kompas_docs" and memory_context and not self.docs.has_domain_signal(query):
                return "memory_search"
            return tool_name
        return self._select_tool(mode=mode)

    def _select_tool(self, mode: str) -> str:
        if mode == "web":
            return "web_search"
        if mode == "memory":
            return "memory_search"
        return "search_kompas_docs"

    async def _call_tool(self, user_id: int, tool_name: str, query: str) -> tuple[str, list[str]]:
        tools = self.toolbox.build(user_id=user_id)
        tool = tools[tool_name]
        payload = await tool.ainvoke({"query": query})
        output = parse_tool_output(str(payload), fallback_source=tool_name)
        return output.source, output.items

    async def _call_analyze_tool(
        self,
        user_id: int,
        query: str,
        source: str,
        observations: list[str],
        memory_context: list[str],
    ) -> tuple[str, list[str]]:
        tools = self.toolbox.build(user_id=user_id)
        payload = await tools["analyze_context"].ainvoke(
            {
                "query": query,
                "source": source,
                "items_json": pack_items(observations),
                "memory_json": pack_items(memory_context),
            }
        )
        output = parse_tool_output(str(payload), fallback_source="analyze_context")
        return output.source, output.items

    def _merge_react_observations(
        self,
        source: str,
        observations: list[str],
        memory_observations: list[str],
        react_mode: bool,
    ) -> list[str]:
        if not react_mode or source == "memory_search":
            return observations

        merged = []
        merged.extend(f"[memory] {item}" for item in memory_observations[:4])
        merged.extend(observations)
        return merged

    def _attach_web_sources(self, answer: str, observations: list[str]) -> str:
        urls = self._extract_urls(observations)
        if not urls:
            return answer
        if "Источники:" in answer:
            return answer

        source_lines = "\n".join(f"{idx + 1}. {url}" for idx, url in enumerate(urls[:3]))
        return f"{answer.rstrip()}\n\nИсточники:\n{source_lines}"

    def _extract_urls(self, observations: list[str]) -> list[str]:
        urls: list[str] = []
        for observation in observations:
            for url in re.findall(r"https?://[^\s)]+", observation):
                clean_url = url.rstrip(".,;:!?)]}")
                if clean_url and clean_url not in urls:
                    urls.append(clean_url)
        return urls
