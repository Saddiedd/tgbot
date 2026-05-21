from aiogram import Router
from aiogram.types import Message

router = Router()


@router.message()
async def chat_handler(message: Message, answer_service) -> None:
    text = (message.text or "").strip()

    if text == "Очистить историю":
        await answer_service.clear_history(message.from_user.id)
        await message.answer("История очищена.")
        return

    if text == "Справка":
        await message.answer(
            "Я помогаю с КОМПАС-3D: отвечаю по документации, могу искать в интернете и учитывать историю диалога.\n\n"
            "Напишите конкретный вопрос, например: «Как создать деталь через API?» или "
            "«Найди в интернете изменения API КОМПАС-3D»."
        )
        return

    if text == "Поиск в документации":
        answer_service.set_next_mode(message.from_user.id, "docs")
        await message.answer("Напишите, что искать в документации КОМПАС-3D: команду, API-метод, ошибку или сценарий работы.")
        return

    if text == "Поиск в интернете":
        answer_service.set_next_mode(message.from_user.id, "web")
        await message.answer("Напишите запрос для web-поиска.")
        return

    if text == "Вопрос по КОМПАС-3D":
        answer_service.set_next_mode(message.from_user.id, "react")
        await message.answer("Задайте вопрос по КОМПАС-3D. Я сначала проверю документацию, а если ответа не хватит, попробую web-поиск.")
        return

    if text == "Спросить ReAct-агента":
        answer_service.set_next_mode(message.from_user.id, "react")
        await message.answer("Задайте вопрос по КОМПАС-3D. Я сначала проверю документацию, а если ответа не хватит, попробую web-поиск.")
        return

    answer = await answer_service.answer(user_id=message.from_user.id, query=text)
    await message.answer(answer)
