class LanceDBService:
    async def search_kompas_docs(self, query: str) -> list[str]:
        q = query.lower().strip()
        mapping = {
            "создать": "Для создания документа используйте интерфейс KompasObject и вызов создания 3D-документа.",
            "деталь": "Новая деталь обычно создаётся через документ типа Part и затем через API эскиза/операций.",
            "эскиз": "Эскиз: выберите плоскость, войдите в режим Sketch, добавьте геометрию и завершите редактирование.",
            "сборк": "Сборка: создайте Assembly-документ, добавляйте компоненты и задавайте сопряжения.",
        }
        results = [text for key, text in mapping.items() if key in q]
        if not results:
            results = [
                "По запросу нет точного совпадения в локальном справочнике. Попробуйте уточнить термин API или команду КОМПАС-3D.",
            ]
        return results
