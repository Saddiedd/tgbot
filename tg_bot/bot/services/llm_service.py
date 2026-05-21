import re


class LLMService:
    """Final fallback formatter for tool outputs.

    The actual reasoning and answer synthesis live in ContextAnalysisService.
    This class deliberately avoids intent detection by user-query keywords.
    """

    def __init__(self, max_items: int = 3) -> None:
        self.max_items = max_items

    async def generate_answer(
        self,
        query: str,
        context: list[str],
        source: str,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        del history

        if source == "analyze_context" and context:
            text = self._sanitize_text(context[0])
            if text:
                return self._fit_message(text)

        items = self._clean_items(context, limit=self.max_items)
        if items:
            return self._fit_message(self._format_items(source=source, items=items))

        return self._not_enough_data(query=query, source=source)

    def _format_items(self, source: str, items: list[str]) -> str:
        labels = {
            "web_search": "По интернет-поиску:",
            "search_kompas_docs": "По документации КОМПАС-3D:",
            "memory_search": "По памяти:",
        }
        answer = labels.get(source, "По найденным данным:")
        answer += "\n" + "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(items))
        if source == "web_search":
            answer += "\n\nИнтернет-выдачу стоит перепроверить по официальной справке АСКОН или документации вашей версии."
        return answer

    def _clean_items(self, context: list[str], limit: int) -> list[str]:
        items: list[str] = []
        for item in context:
            text = item.strip()
            if text.startswith("[") and "]" in text:
                text = text.split("]", 1)[1].strip()
            text = self._sanitize_text(text)
            clipped = self._clip(text)
            if clipped and not self._is_duplicate(clipped, items):
                items.append(clipped)
            if len(items) >= limit:
                break
        return items

    def _is_duplicate(self, text: str, items: list[str]) -> bool:
        normalized = self._dedupe_key(text)
        return any(normalized[:90] == self._dedupe_key(item)[:90] for item in items)

    def _dedupe_key(self, text: str) -> str:
        return "".join(char for char in text.lower().replace("ё", "е") if char.isalnum() or char.isspace())

    def _sanitize_text(self, text: str) -> str:
        cleaned = text
        patterns = [
            r"Please enable JavaScript to view this site\.?",
            r"\[Способы вызова команды\]\(javascript:void\(0\)\)",
            r"\[[^\]]*\]\(javascript:void\(0\)\)",
            r"\[[^\]]*\]\(file:[^)]+\)",
            r"\(file:[^)]+\)",
            r"file:///\S+",
            r"javascript:void\(0\)",
            r"<[^>]+>",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _not_enough_data(self, query: str, source: str) -> str:
        places = {
            "web_search": "в интернете",
            "memory_search": "в памяти диалога",
        }
        place = places.get(source, "в базе документации")
        return (
            f"По запросу «{query}» не нашёл достаточно релевантных данных {place}. "
            "Уточните команду, объект модели, сценарий работы или версию КОМПАС-3D."
        )

    def _clip(self, text: str, max_length: int = 320) -> str:
        text = " ".join(text.split())
        if len(text) <= max_length:
            return text
        cut_at = text.rfind(" ", 0, max_length - 3)
        if cut_at < 160:
            cut_at = max_length - 3
        return f"{text[:cut_at].rstrip()}..."

    def _fit_message(self, text: str, max_length: int = 1400) -> str:
        text = text.strip()
        if len(text) <= max_length:
            return text
        cut_at = text.rfind("\n", 0, max_length - 3)
        if cut_at < 700:
            cut_at = max_length - 3
        return f"{text[:cut_at].rstrip()}..."
