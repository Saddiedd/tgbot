class LanceDBService:
    async def search_kompas_docs(self, query: str) -> list[str]:
        return [
            f"[docs] Найден фрагмент по запросу '{query}': используйте API КомпасDocument3D для создания документа.",
        ]
