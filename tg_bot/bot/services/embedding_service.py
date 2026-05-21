import asyncio
import logging


class EmbeddingService:
    """Creates query embeddings compatible with the stored Mistral vectors."""

    def __init__(self, api_key: str | None, model: str = "mistral-embed") -> None:
        self.api_key = api_key
        self.model = model
        self._cache: dict[str, list[float]] = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def embed_query(self, text: str) -> list[float] | None:
        query = text.strip()
        if not query or not self.api_key:
            return None

        cached = self._cache.get(query)
        if cached is not None:
            return cached

        try:
            vector = await asyncio.to_thread(self._embed_sync, query)
        except Exception:
            logging.exception("Mistral embedding request failed")
            return None

        if vector:
            self._cache[query] = vector
        return vector

    def _embed_sync(self, text: str) -> list[float]:
        from mistralai import Mistral

        client = Mistral(api_key=self.api_key)
        response = client.embeddings.create(model=self.model, inputs=[text])
        return list(response.data[0].embedding)
