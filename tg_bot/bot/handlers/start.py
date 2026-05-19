from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.keyboards.main_menu import get_main_menu

router = Router()


@router.message(Command("start"))
async def start_handler(message: Message) -> None:
    await message.answer(
        "Привет! Я помогу изучать 3D-моделирование в КОМПАС-3D с помощью ReAct-агента.",
        reply_markup=get_main_menu(),
    )


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    await message.answer(
        "Доступные команды:\n"
        "/start — запуск бота\n"
        "/help — помощь\n"
        "/menu — открыть меню"
    )
