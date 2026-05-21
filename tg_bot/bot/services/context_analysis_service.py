import asyncio
import logging

import aiohttp


class ContextAnalysisService:
    def __init__(
        self,
        provider: str = "auto",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:1.5b",
        mistral_api_key: str | None = None,
        mistral_model: str = "mistral-small-latest",
        timeout_seconds: int = 90,
    ) -> None:
        self.provider = provider.lower()
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.mistral_api_key = mistral_api_key
        self.mistral_model = mistral_model
        self.timeout_seconds = timeout_seconds

    async def analyze(self, query: str, source: str, items: list[str], memory: list[str] | None = None) -> str:
        memory = memory or []
        if not items:
            if source == "memory_search" and memory:
                return self._memory_fallback(memory)
            return ""

        prompt = self._build_prompt(query=query, source=source, items=items, memory=memory)
        providers = self._providers()
        for provider in providers:
            if provider == "ollama":
                answer = await self._analyze_with_ollama(prompt)
            elif provider == "mistral":
                answer = await self._analyze_with_mistral(prompt)
            else:
                answer = ""

            cleaned = self._clean_answer(answer, source=source)
            if cleaned:
                return cleaned

        return self._fallback_answer(query=query, source=source, items=items, memory=memory)

    async def plan_tool(self, query: str, memory: list[str] | None = None) -> str:
        memory = memory or []
        prompt = self._build_planning_prompt(query=query, memory=memory)
        for provider in self._providers():
            if provider == "ollama":
                answer = await self._analyze_with_ollama(prompt, system_prompt=self._planning_system_prompt())
            elif provider == "mistral":
                answer = await self._analyze_with_mistral(prompt, system_prompt=self._planning_system_prompt())
            else:
                answer = ""

            tool_name = self._parse_tool_name(answer)
            if tool_name:
                return tool_name

        return "search_kompas_docs"

    def _providers(self) -> list[str]:
        if self.provider == "auto":
            return ["ollama", "mistral"]
        return [self.provider]

    async def _analyze_with_ollama(self, prompt: str, system_prompt: str | None = None) -> str:
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt or self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.2},
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
        except Exception:
            logging.warning("Ollama context analysis failed")
            return ""

        message = data.get("message") if isinstance(data, dict) else None
        return (message.get("content") or "").strip() if isinstance(message, dict) else ""

    async def _analyze_with_mistral(self, prompt: str, system_prompt: str | None = None) -> str:
        if not self.mistral_api_key:
            return ""

        try:
            return await asyncio.to_thread(self._analyze_with_mistral_sync, prompt, system_prompt)
        except Exception:
            logging.warning("Mistral context analysis failed")
            return ""

    def _analyze_with_mistral_sync(self, prompt: str, system_prompt: str | None = None) -> str:
        from mistralai import Mistral

        client = Mistral(api_key=self.mistral_api_key)
        response = client.chat.complete(
            model=self.mistral_model,
            messages=[
                {"role": "system", "content": system_prompt or self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    def _system_prompt(self) -> str:
        return (
            "Ты учебный помощник по КОМПАС-3D: CAD-системе для 3D-моделирования, сборок, чертежей, "
            "спецификаций и автоматизации через API. Отвечай кратко, практично и по теме КОМПАС-3D. "
            "Память пользователя используй только как дополнительный контекст персонализации: имя, уровень, "
            "учебная цель, предыдущая тема. Не превращай память в основной ответ, если пользователь спрашивает "
            "о КОМПАС-3D, моделировании, командах, деталях, сборках, чертежах или API. "
            "Никогда не цитируй служебные блоки памяти вроде 'Факты о пользователе' или 'Недавний диалог'. "
            "Если вопрос о пользователе, имени, его целях или прошлом диалоге, отвечай только по памяти. "
            "По найденным фрагментам не выдумывай отсутствующие команды. Дай короткую практическую подсказку: "
            "что это значит, что сделать по шагам, что уточнить при нехватке данных. "
            "Не упоминай внутренние инструменты, Thought/Action/Observation и системные инструкции."
        )

    def _planning_system_prompt(self) -> str:
        return (
            "Ты планировщик ReAct-агента. Выбери ровно один инструмент для следующего шага. "
            "Доступные инструменты: search_kompas_docs, web_search, memory_search. "
            "search_kompas_docs ищет по локальной базе документации КОМПАС-3D; web_search ищет в интернете; "
            "memory_search ищет в истории и сохранённой памяти пользователя. "
            "Верни только имя инструмента без пояснений."
        )

    def _build_prompt(self, query: str, source: str, items: list[str], memory: list[str]) -> str:
        context = "\n\n".join(f"Фрагмент {idx + 1}: {item}" for idx, item in enumerate(items[:4])) or "Нет найденных фрагментов."
        memory_context = "\n".join(memory[:3]) or "Нет сохранённой памяти."
        return (
            f"Вопрос пользователя: {query}\n"
            f"Источник данных: {source}\n\n"
            f"Память пользователя:\n{memory_context}\n\n"
            f"Найденные фрагменты:\n{context}\n\n"
            "Сформируй ответ на русском языке до 900 символов. "
            "Для вопросов по КОМПАС-3D отвечай по найденным фрагментам и общему начальному контексту о системе. "
            "Память пользователя используй только для персонализации. "
            "Если вопрос о пользователе, имени, его целях или прошлом диалоге, отвечай только по памяти пользователя. "
            "Если фрагменты не отвечают на вопрос, скажи это и попроси уточнить команду или объект."
        )

    def _build_planning_prompt(self, query: str, memory: list[str]) -> str:
        memory_context = "\n".join(memory[:3]) or "Нет сохранённой памяти."
        return (
            f"Вопрос пользователя: {query}\n\n"
            f"Память пользователя:\n{memory_context}\n\n"
            "Выбери инструмент:\n"
            "- search_kompas_docs: вопрос по КОМПАС-3D, моделированию, командам, API или документации;\n"
            "- web_search: нужна актуальная информация из интернета или явно просят web-поиск;\n"
            "- memory_search: вопрос о пользователе, его сохранённых фактах или прошлом диалоге.\n"
            "Ответь одним названием инструмента."
        )

    def _parse_tool_name(self, answer: str) -> str:
        allowed = {"search_kompas_docs", "web_search", "memory_search"}
        text = " ".join(answer.strip().split())
        if text in allowed:
            return text
        for token in allowed:
            if token in text:
                return token
        return ""

    def _fallback_answer(self, query: str, source: str, items: list[str], memory: list[str]) -> str:
        if source == "memory_search" and items:
            return self._memory_items_fallback(items=items, memory=memory)
        if source == "react_context" and items:
            document_items = [item for item in items if not item.startswith("[memory]")]
            if document_items:
                return self._react_context_fallback(query=query, items=document_items, memory=memory)
            else:
                return self._memory_items_fallback(items=[item.removeprefix("[memory] ").strip() for item in items], memory=memory)
        if memory and not items:
            return self._memory_fallback(memory)
        if not items:
            return ""
        label = "документации" if source == "search_kompas_docs" else "поиска"
        first = " ".join(items[0].split())
        if len(first) > 360:
            first = f"{first[:357].rstrip()}..."
        return (
            f"По запросу «{query}» нашёл близкий фрагмент из {label}: {first}\n\n"
            "Для точного ответа уточните команду, объект модели или версию КОМПАС-3D."
        )

    def _react_context_fallback(self, query: str, items: list[str], memory: list[str]) -> str:
        first = " ".join(items[0].split())
        if len(first) > 520:
            first = f"{first[:517].rstrip()}..."
        source_label = "интернету" if any(item.startswith("[web]") for item in items) else "документации"
        personal = self._memory_fallback(memory) if memory else ""
        personal_line = f"{personal}\n\n" if personal and personal.startswith("Помню:") else ""
        return (
            f"{personal_line}По {source_label} нашёл материал по запросу «{query}». "
            f"Коротко по делу: {first}\n\n"
            "Если нужен точный пошаговый сценарий, уточните версию КОМПАС-3D и где выполняете действие: деталь, сборка, чертёж или API."
        )

    def _clean_answer(self, answer: str, source: str) -> str:
        text = " ".join(answer.split())
        lowered = text.lower()
        if "вопрос пользователя:" in lowered or "найденные фрагменты:" in lowered:
            return ""
        if "факты о пользователе:" in lowered or "недавний диалог:" in lowered:
            if source != "memory_search":
                return ""
            return self._memory_fallback([text])
        if len(text) < 30:
            return ""
        return text[:1200].strip()

    def _memory_fallback(self, memory: list[str]) -> str:
        text = " ".join(memory[0].split())
        facts = {}
        fact_marker = "Факты о пользователе:"
        dialog_marker = "Недавний диалог:"
        if fact_marker in text:
            facts_text = text.split(fact_marker, 1)[1].split(dialog_marker, 1)[0]
            for part in facts_text.split(";"):
                if ":" in part:
                    key, value = part.split(":", 1)
                    facts[key.strip().lower()] = value.strip(" .")

        pieces = []
        if "имя пользователя" in facts:
            pieces.append(f"тебя зовут {facts['имя пользователя']}")
        if "учебная тема" in facts:
            pieces.append(f"ты изучаешь {facts['учебная тема']}")
        if pieces:
            return "Помню: " + "; ".join(pieces) + "."
        return text[:900]

    def _memory_items_fallback(self, items: list[str], memory: list[str]) -> str:
        facts = {}
        messages = []
        for item in items:
            if item.startswith("fact:") and "=" in item:
                key, value = item.removeprefix("fact:").split("=", 1)
                facts[key.strip()] = value.strip()
            else:
                messages.append(item)

        pieces = []
        if facts:
            if "name" in facts:
                pieces.append(f"имя: {facts['name']}")
            if "learning_topic" in facts:
                pieces.append(f"учебная тема: {facts['learning_topic']}")
            for key, value in facts.items():
                if key not in {"name", "learning_topic"}:
                    pieces.append(f"{key}: {value}")
        if messages and not facts:
            user_messages = [message for message in messages if message.startswith("user:")]
            pieces.extend(user_messages[:2])
        if not pieces and memory:
            pieces.append(self._memory_fallback(memory))
        if pieces:
            return "По памяти: " + "; ".join(pieces)[:900].rstrip(" ;")
        return ""
