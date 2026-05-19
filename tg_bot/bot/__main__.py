import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession

from bot.config import load_settings
from bot.handlers.chat import router as chat_router
from bot.handlers.inline import router as inline_router
from bot.handlers.menu import router as menu_router
from bot.handlers.start import router as start_router
from bot.middlewares.services import ServicesMiddleware
from bot.services.answer_service import AnswerService
from bot.services.memory_service import MemoryService


async def main() -> None:
    settings = load_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    session = AiohttpSession(proxy=settings.telegram_proxy_url, timeout=settings.telegram_request_timeout)
    bot = Bot(token=settings.telegram_bot_token, session=session)
    dp = Dispatcher()

    memory_service = MemoryService()
    answer_service = AnswerService(memory_service=memory_service)

    dp.update.middleware(ServicesMiddleware(answer_service=answer_service, memory_service=memory_service))

    dp.include_router(start_router)
    dp.include_router(menu_router)
    dp.include_router(chat_router)
    dp.include_router(inline_router)

    try:
        me = await bot.get_me()
        logging.info("Connected to Telegram as @%s (%s)", me.username, me.id)
        await dp.start_polling(bot)
    except Exception:
        logging.exception(
            "Telegram connection failed. Check internet access to https://api.telegram.org, "
            "VPN/proxy/firewall settings, and TELEGRAM_BOT_TOKEN."
        )
        raise
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
