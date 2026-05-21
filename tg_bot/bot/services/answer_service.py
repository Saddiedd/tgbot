from bot.agents.react_agent import ReActAgent


class AnswerService:
    def __init__(self, memory_service, settings) -> None:
        self.memory_service = memory_service
        self.agent = ReActAgent(memory_service=memory_service, settings=settings)
        self._next_modes: dict[int, str] = {}

    def set_next_mode(self, user_id: int, mode: str) -> None:
        self._next_modes[user_id] = mode

    async def answer(self, user_id: int, query: str, mode: str | None = None) -> str:
        selected_mode = mode or self._next_modes.pop(user_id, "react")
        await self.memory_service.add(user_id, "user", query)
        answer = await self.agent.run(user_id=user_id, query=query, mode=selected_mode)
        await self.memory_service.add(user_id, "assistant", answer)
        return answer

    async def inline_preview(self, user_id: int, query: str) -> str:
        answer = await self.agent.run(user_id=user_id, query=query)
        return answer[:400]

    async def clear_history(self, user_id: int) -> None:
        await self.memory_service.clear(user_id)
