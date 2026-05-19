from aiogram import Router
from aiogram.types import Message

router = Router()


@router.message()
async def chat_handler(message: Message, answer_service) -> None:
    if message.text == "Очистить историю":
        await answer_service.clear_history(message.from_user.id)
        await message.answer("История очищена.")
        return

    answer = await answer_service.answer(user_id=message.from_user.id, query=message.text or "")
    await message.answer(answer)
