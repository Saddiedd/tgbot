from aiogram import Router
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent

router = Router()


@router.inline_query()
async def inline_handler(query: InlineQuery, answer_service) -> None:
    preview = await answer_service.inline_preview(user_id=query.from_user.id, query=query.query)
    results = [
        InlineQueryResultArticle(
            id="react",
            title="Ответ ReAct-агента",
            description="Сформировать учебный ответ по запросу",
            input_message_content=InputTextMessageContent(message_text=preview),
        )
    ]
    await query.answer(results=results, cache_time=1)
