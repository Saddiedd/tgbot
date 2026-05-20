class LLMService:
    """Учебный LLM-сервис.

    Сейчас формирует полезный ответ локально, чтобы бот не застревал на заглушках.
    """

    async def generate_answer(self, query: str, context: list[str], source: str) -> str:
        if source == "memory_search" and context:
            return "Нашёл в истории диалога:\n- " + "\n- ".join(context[:4])

        if source in {"search_kompas_docs", "web_search"} and context:
            return (
                "Вот что удалось найти:\n"
                + "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(context[:3]))
                + "\n\nЕсли нужно, уточните версию КОМПАС-3D и тип операции (деталь/сборка/чертёж)."
            )

        return (
            "Пока не хватает контекста для точного ответа. "
            "Уточните, что именно хотите сделать в КОМПАС-3D: команду, объект и ожидаемый результат."
        )
