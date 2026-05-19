from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject


class ServicesMiddleware(BaseMiddleware):
    def __init__(self, answer_service, memory_service) -> None:
        self.answer_service = answer_service
        self.memory_service = memory_service

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        data["answer_service"] = self.answer_service
        data["memory_service"] = self.memory_service
        return await handler(event, data)
