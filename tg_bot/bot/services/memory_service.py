import re
from pathlib import Path
from urllib.parse import urlparse

import aiosqlite


class MemoryService:
    def __init__(self, database_url: str = "sqlite+aiosqlite:///./data/bot.db") -> None:
        self.db_path = self._database_path(database_url)
        self._initialized = False

    async def add(self, user_id: int, role: str, text: str) -> None:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages(user_id, role, text) VALUES (?, ?, ?)",
                (user_id, role, text),
            )
            if role == "user":
                for key, value in self._extract_user_facts(text).items():
                    await db.execute(
                        """
                        INSERT INTO user_facts(user_id, fact_key, fact_value)
                        VALUES (?, ?, ?)
                        ON CONFLICT(user_id, fact_key)
                        DO UPDATE SET fact_value = excluded.fact_value, updated_at = CURRENT_TIMESTAMP
                        """,
                        (user_id, key, value),
                    )
                await self._update_summary(db=db, user_id=user_id)
            await db.commit()

    async def search(self, user_id: int, query: str) -> list[str]:
        await self._ensure_initialized()
        facts = await self.facts(user_id)
        history = await self.tail(user_id, size=12)
        query_tokens = self._tokens(query)
        results: list[str] = []

        for key, value in facts.items():
            results.append(f"fact:{key}={value}")

        for role, text in reversed(history):
            text_tokens = self._tokens(text)
            if not query_tokens or query_tokens.intersection(text_tokens):
                results.append(f"{role}: {text}")
            if len(results) >= 8:
                break

        return results

    async def tail(self, user_id: int, size: int = 6) -> list[tuple[str, str]]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT role, text
                FROM messages
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, size),
            )
            rows = await cursor.fetchall()
        return [(role, text) for role, text in reversed(rows)]

    async def facts(self, user_id: int) -> dict[str, str]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT fact_key, fact_value FROM user_facts WHERE user_id = ?",
                (user_id,),
            )
            rows = await cursor.fetchall()
        return {key: value for key, value in rows}

    async def summary(self, user_id: int) -> list[str]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT summary FROM memory_summaries WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
        return [row[0]] if row and row[0].strip() else []

    async def clear(self, user_id: int) -> None:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM user_facts WHERE user_id = ?", (user_id,))
            await db.execute("DELETE FROM memory_summaries WHERE user_id = ?", (user_id,))
            await db.commit()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_facts(
                    user_id INTEGER NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(user_id, fact_key)
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_summaries(
                    user_id INTEGER PRIMARY KEY,
                    summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await db.commit()
        self._initialized = True

    async def _update_summary(self, db: aiosqlite.Connection, user_id: int) -> None:
        facts_cursor = await db.execute(
            "SELECT fact_key, fact_value FROM user_facts WHERE user_id = ? ORDER BY fact_key",
            (user_id,),
        )
        fact_rows = await facts_cursor.fetchall()
        message_cursor = await db.execute(
            """
            SELECT role, text
            FROM messages
            WHERE user_id = ? AND role = 'user'
            ORDER BY id DESC
            LIMIT 6
            """,
            (user_id,),
        )
        message_rows = await message_cursor.fetchall()

        parts = []
        facts = {key: value for key, value in fact_rows}
        if facts:
            fact_text = []
            if "name" in facts:
                fact_text.append(f"имя пользователя: {facts['name']}")
            if "learning_topic" in facts:
                fact_text.append(f"учебная тема: {facts['learning_topic']}")
            for key, value in facts.items():
                if key not in {"name", "learning_topic"}:
                    fact_text.append(f"{key}: {value}")
            parts.append("Факты о пользователе: " + "; ".join(fact_text) + ".")

        recent_messages = [
            f"{role}: {' '.join(text.split())[:220]}"
            for role, text in reversed(message_rows)
        ]
        if recent_messages:
            parts.append("Недавний диалог: " + " | ".join(recent_messages))

        summary = "\n".join(parts).strip()
        await db.execute(
            """
            INSERT INTO memory_summaries(user_id, summary)
            VALUES (?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET summary = excluded.summary, updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, summary),
        )

    def _database_path(self, database_url: str) -> Path:
        if database_url.startswith("sqlite+aiosqlite:///"):
            raw_path = database_url.removeprefix("sqlite+aiosqlite:///")
            return Path(raw_path)
        if database_url.startswith("sqlite:///"):
            raw_path = database_url.removeprefix("sqlite:///")
            return Path(raw_path)

        parsed = urlparse(database_url)
        if parsed.scheme.startswith("sqlite"):
            return Path(parsed.path.lstrip("/"))
        return Path("./data/bot.db")

    def _extract_user_facts(self, text: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        patterns = [
            r"\bменя зовут\s+([А-ЯЁA-Z][а-яёa-z-]{1,40})",
            r"\bмо[её]\s+имя\s+([А-ЯЁA-Z][а-яёa-z-]{1,40})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                facts["name"] = match.group(1).strip().capitalize()
                break

        topic_match = re.search(r"\b(?:изучаю|учусь|хочу научиться)\s+([^.!?\n]{4,120})", text, flags=re.IGNORECASE)
        if topic_match:
            facts["learning_topic"] = topic_match.group(1).strip(" .!?")

        return facts

    def _tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower().replace("ё", "е"))
            if len(token) > 2
        }
