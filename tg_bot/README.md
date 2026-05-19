# tg_bot

Telegram-бот на базе LLM с ReAct-агентом для обучения 3D-моделированию в КОМПАС-3D.

## Требования

- Python 3.10+
- Telegram Bot Token

## Быстрый запуск

1. Перейдите в папку проекта:

```bash
cd tg_bot
```

2. Создайте и активируйте виртуальное окружение:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Установите зависимости:

```bash
pip install -e .
```

4. Настройте переменные окружения в файле `.env`.

Минимально обязательно указать:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

Дополнительно (для следующих этапов интеграции):

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

TAVILY_API_KEY=your_tavily_api_key

LANCEDB_PATH=./data/lancedb
LANCEDB_TABLE=kompas_docs
DATABASE_URL=sqlite+aiosqlite:///./data/bot.db
LOG_LEVEL=INFO
```

5. Запустите бота:

```bash
python -m bot
```

## Доступные команды

- `/start` — запуск и приветствие
- `/help` — справка
- `/menu` — открыть клавиатурное меню

## Текущий статус

Текущая версия — это рабочий каркас:

- подключены handlers / middleware / services / agent;
- реализована базовая маршрутизация запроса в ReAct-агенте;
- внешние интеграции (LLM, LanceDB, web-search) пока в формате заглушек и требуют дальнейшей реализации.
