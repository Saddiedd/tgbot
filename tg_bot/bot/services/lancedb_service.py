from pathlib import Path


class LanceDBService:
    """Локальный поисковый сервис по документации проекта.

    Реализует lightweight-поиск без внешних зависимостей на этапе MVP.
    """

    def __init__(self, docs_dir: str = "./docs") -> None:
        self.docs_dir = Path(docs_dir)

    async def search_kompas_docs(self, query: str) -> list[str]:
        tokens = [token for token in query.lower().split() if len(token) > 2]
        if not tokens:
            return []

        results: list[tuple[int, str]] = []
        for file_path in self._iter_doc_files():
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            snippet = self._best_snippet(text=text, tokens=tokens)
            if snippet is not None:
                score, chunk = snippet
                results.append((score, f"[docs:{file_path.name}] {chunk}"))

        results.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in results[:3]]

    def _iter_doc_files(self) -> list[Path]:
        if not self.docs_dir.exists():
            return []
        return [path for path in self.docs_dir.rglob("*") if path.suffix.lower() in {".md", ".txt"} and path.is_file()]

    def _best_snippet(self, text: str, tokens: list[str]) -> tuple[int, str] | None:
        best_score = 0
        best_line = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            score = sum(1 for token in tokens if token in line.lower())
            if score > best_score:
                best_score = score
                best_line = line

        if best_score == 0:
            return None

        clipped = best_line if len(best_line) <= 280 else f"{best_line[:277]}..."
        return best_score, clipped
