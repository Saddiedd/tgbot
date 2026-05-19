from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def get_main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Вопрос по КОМПАС-3D")],
            [KeyboardButton(text="Поиск в документации"), KeyboardButton(text="Поиск в интернете")],
            [KeyboardButton(text="Спросить ReAct-агента"), KeyboardButton(text="Очистить историю")],
            [KeyboardButton(text="Справка")],
        ],
        resize_keyboard=True,
    )
