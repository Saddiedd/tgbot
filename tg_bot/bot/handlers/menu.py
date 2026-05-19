from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.keyboards.main_menu import get_main_menu

router = Router()


@router.message(Command("menu"))
async def menu_handler(message: Message) -> None:
    await message.answer("Открыл меню.", reply_markup=get_main_menu())
