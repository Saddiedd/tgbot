class MemoryService:
    def __init__(self) -> None:
        self._history: dict[int, list[tuple[str, str]]] = {}

    async def add(self, user_id: int, role: str, text: str) -> None:
        self._history.setdefault(user_id, []).append((role, text))
        self._history[user_id] = self._history[user_id][-20:]

    async def search(self, user_id: int, query: str) -> list[str]:
        items = self._history.get(user_id, [])
        q = query.lower()
        return [f"{role}: {text}" for role, text in items if q in text.lower()][:5]

    async def tail(self, user_id: int, size: int = 6) -> list[tuple[str, str]]:
        return self._history.get(user_id, [])[-size:]

    async def clear(self, user_id: int) -> None:
        self._history[user_id] = []
