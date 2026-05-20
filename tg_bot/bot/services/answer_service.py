from bot.agents.react_agent import ReActAgent


class AnswerService:
    def __init__(self, memory_service) -> None:
        self.memory_service = memory_service
        self.agent = ReActAgent(memory_service=memory_service)

    async def answer(self, user_id: int, query: str) -> str:
        await self.memory_service.add(user_id, "user", query)
        answer = await self.agent.run(user_id=user_id, query=query)
        await self.memory_service.add(user_id, "assistant", answer)
        return answer

    async def inline_preview(self, user_id: int, query: str) -> str:
        answer = await self.agent.run(user_id=user_id, query=query)
        return answer[:400]

    async def clear_history(self, user_id: int) -> None:
        await self.memory_service.clear(user_id)
