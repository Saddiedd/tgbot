import re
from math import ceil
from pathlib import Path
from typing import Any


class LanceDBService:
    """Local search service for the prepared KOMPAS-3D documentation base."""

    NOISE_PATTERNS = [
        r"Please enable JavaScript to view this site\.?",
        r"\[Способы вызова команды\]\(javascript:void\(0\)\)",
        r"\[[^\]]*\]\(javascript:void\(0\)\)",
        r"\[[^\]]*\]\(file:[^)]+\)",
        r"\(file:[^)]+\)",
        r"file:///\S+",
        r"javascript:void\(0\)",
        r"Источник:\s*https?:/\S*",
    ]

    def __init__(
        self,
        db_path: str = "./data/lancedb",
        table_name: str = "kompas_docs",
        docs_dir: str = "./docs",
        embedding_service: Any | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.docs_dir = Path(docs_dir)
        self.embedding_service = embedding_service

    async def search_kompas_docs(self, query: str) -> list[str]:
        tokens = self._query_tokens(query)
        if not tokens:
            return []

        vector_results = await self._search_vectors(query=query, tokens=tokens)
        if vector_results:
            return vector_results

        if not tokens:
            return []

        lancedb_results = self._search_lancedb_text(query=query, tokens=tokens)
        if lancedb_results:
            return lancedb_results

        return self._search_text_docs(tokens=tokens)

    async def _search_vectors(self, query: str, tokens: list[str]) -> list[str]:
        if not self.embedding_service or not self.db_path.exists():
            return []

        query_vector = await self.embedding_service.embed_query(query)
        if not query_vector:
            return []

        table_results = self._search_lancedb_vectors(query_vector=query_vector, tokens=tokens)
        if table_results:
            return table_results

        return self._search_lance_dataset_vectors(query_vector=query_vector, tokens=tokens)

    def _search_lancedb_vectors(self, query_vector: list[float], tokens: list[str]) -> list[str]:
        try:
            import lancedb
        except ImportError:
            return []

        try:
            db = lancedb.connect(str(self.db_path))
            table_names = set(db.list_tables())
            if not table_names:
                return []

            table_name = self.table_name if self.table_name in table_names else sorted(table_names)[0]
            table = db.open_table(table_name)
            rows = table.search(query_vector).limit(8).to_list()
        except Exception:
            return []

        return self._format_vector_rows(rows=rows, label=table_name, tokens=tokens)

    def _search_lance_dataset_vectors(self, query_vector: list[float], tokens: list[str]) -> list[str]:
        try:
            import lance
        except ImportError:
            return []

        dataset_paths = [path for path in self.db_path.iterdir() if (path / "_versions").exists() and (path / "data").exists()]
        ranked: list[tuple[float, str]] = []
        for dataset_path in dataset_paths:
            try:
                rows = lance.dataset(str(dataset_path)).to_table().to_pylist()
            except Exception:
                continue

            for row in rows:
                vector = row.get("vector")
                text = self._clean_text(self._extract_text(row))
                if not vector or not text:
                    continue
                similarity = self._cosine_similarity(query_vector, list(vector))
                score, snippet = self._best_snippet(text, tokens) if tokens else (0, self._clip(text))
                if tokens and not self._has_required_query_token(snippet, tokens):
                    continue
                ranked.append((self._hybrid_score(similarity, score), f"[docs:{dataset_path.name}] {snippet}"))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return self._take_vector_relevant(ranked, has_query_terms=bool(tokens))

    def _search_lancedb_text(self, query: str, tokens: list[str]) -> list[str]:
        if not self.db_path.exists():
            return []

        try:
            import lancedb
        except ImportError:
            return []

        try:
            db = lancedb.connect(str(self.db_path))
            table_names = set(db.list_tables())
            if not table_names:
                return self._search_lance_datasets(tokens=tokens)

            table_name = self.table_name if self.table_name in table_names else sorted(table_names)[0]
            table = db.open_table(table_name)
            try:
                rows = table.search(query, query_type="fts").limit(12).to_list()
            except Exception:
                rows = table.to_arrow().to_pylist()
        except Exception:
            return self._search_lance_datasets(tokens=tokens)

        results = self._rank_rows(rows=rows, tokens=tokens, label=table_name)
        return results or self._search_lance_datasets(tokens=tokens)

    def _format_vector_rows(self, rows: list[dict], label: str, tokens: list[str]) -> list[str]:
        ranked: list[tuple[float, str]] = []
        for row in rows:
            text = self._clean_text(self._extract_text(row))
            if not text:
                continue
            distance = row.get("_distance")
            similarity = 1.0 - float(distance) if distance is not None else 1.0
            lexical_score, snippet = self._best_snippet(text, tokens) if tokens else (0, self._clip(text))
            if tokens and not self._has_required_query_token(snippet, tokens):
                continue
            ranked.append((self._hybrid_score(similarity, lexical_score), f"[docs:{label}] {snippet}"))
        return self._take_vector_relevant(ranked, has_query_terms=bool(tokens))

    def _hybrid_score(self, similarity: float, lexical_score: int) -> float:
        return similarity + min(lexical_score, 6) * 0.03

    def _take_vector_relevant(self, ranked: list[tuple[float, str]], has_query_terms: bool) -> list[str]:
        if not ranked:
            return []

        top_score = ranked[0][0]
        if top_score < 0.48:
            return []

        cutoff = max(0.58 if has_query_terms else 0.64, top_score - 0.06)
        return [text for score, text in ranked if score >= cutoff][:2]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0

        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def _search_lance_datasets(self, tokens: list[str]) -> list[str]:
        try:
            import lance
        except ImportError:
            return []

        dataset_paths = [path for path in self.db_path.iterdir() if (path / "_versions").exists() and (path / "data").exists()]
        results: list[tuple[int, str]] = []
        for dataset_path in dataset_paths:
            try:
                rows = lance.dataset(str(dataset_path)).to_table().to_pylist()
            except Exception:
                continue
            results.extend(self._rank_row_tuples(rows=rows, tokens=tokens, label=dataset_path.name))

        results.sort(key=lambda item: item[0], reverse=True)
        return self._take_relevant(results)

    def _search_text_docs(self, tokens: list[str]) -> list[str]:
        results: list[tuple[int, str]] = []
        for file_path in self._iter_doc_files():
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            snippet = self._best_snippet(text=text, tokens=tokens)
            score, chunk = snippet
            if score > 0 and chunk:
                results.append((score, f"[docs:{file_path.name}] {chunk}"))

        results.sort(key=lambda item: item[0], reverse=True)
        return self._take_relevant(results)

    def _rank_rows(self, rows: list[dict], tokens: list[str], label: str) -> list[str]:
        ranked = self._rank_row_tuples(rows=rows, tokens=tokens, label=label)
        ranked.sort(key=lambda item: item[0], reverse=True)
        return self._take_relevant(ranked)

    def _rank_row_tuples(self, rows: list[dict], tokens: list[str], label: str) -> list[tuple[int, str]]:
        results: list[tuple[int, str]] = []
        min_score = 2 if len(tokens) == 1 else 3

        for row in rows:
            text = self._clean_text(self._extract_text(row))
            if not text:
                continue

            score, snippet = self._best_snippet(text=text, tokens=tokens)
            if score >= min_score and self._has_required_query_token(snippet, tokens):
                results.append((score, f"[docs:{label}] {snippet}"))

        return results

    def _take_relevant(self, ranked: list[tuple[int, str]]) -> list[str]:
        if not ranked:
            return []

        top_score = ranked[0][0]
        cutoff = top_score if top_score >= 5 else max(2, ceil(top_score * 0.75))
        return [text for score, text in ranked if score >= cutoff][:2]

    def _extract_text(self, row: dict) -> str:
        for key in ("text", "content", "chunk", "page_content", "document"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        parts = []
        for key, value in row.items():
            if key.lower() in {"vector", "embedding", "id"}:
                continue
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        return " ".join(parts)

    def _iter_doc_files(self) -> list[Path]:
        if not self.docs_dir.exists():
            return []
        return [path for path in self.docs_dir.rglob("*") if path.suffix.lower() in {".md", ".txt"} and path.is_file()]

    def _best_snippet(self, text: str, tokens: list[str]) -> tuple[int, str]:
        cleaned = self._clean_text(text)
        if not cleaned:
            return 0, ""

        parts = self._split_parts(cleaned)
        best_score = 0
        best_index = 0
        for index, part in enumerate(parts):
            score = self._score_text(part, tokens)
            if index == 0 and score > 0:
                score += 1
            if score > best_score:
                best_score = score
                best_index = index

        if best_score == 0:
            return 0, ""

        start = best_index
        end = min(len(parts), best_index + 2)
        snippet = " ".join(parts[start:end])
        return best_score, self._clip(snippet)

    def _score_text(self, text: str, tokens: list[str]) -> int:
        lower = text.lower().replace("ё", "е")
        score = 0
        for token in tokens:
            normalized = token.replace("ё", "е")
            if normalized in lower:
                score += 2
                continue
            stem = normalized[: max(5, min(len(normalized), 7))]
            if len(stem) >= 5 and stem in lower:
                score += 1
        return score

    def _has_required_query_token(self, text: str, tokens: list[str]) -> bool:
        required = [
            token for token in tokens
            if token not in {"деталь", "тело", "модель", "модели", "моделирование", "система", "трехмерного", "трёхмерного"}
        ]
        if not required:
            return True

        lower = text.lower().replace("ё", "е")
        return any(self._token_matches_text(token=token, text=lower) for token in required)

    def _token_matches_text(self, token: str, text: str) -> bool:
        normalized = token.replace("ё", "е")
        if len(normalized) <= 4:
            return re.search(rf"\b{re.escape(normalized)}\b", text) is not None

        stem_length = max(5, min(len(normalized), 7))
        return normalized[:stem_length] in text

    def _split_parts(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|(?=\d+\.\s)|(?<=\])\s+", text)
        return [part.strip(" -;:") for part in parts if len(part.strip()) >= 20]

    def _clean_text(self, text: str) -> str:
        cleaned = text
        for pattern in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"\[[^\]]{0,80}\]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _query_tokens(self, query: str) -> list[str]:
        query = self._normalize_query(query)
        stop_words = {
            "как",
            "что",
            "где",
            "кто",
            "для",
            "или",
            "при",
            "это",
            "про",
            "мне",
            "меня",
            "тебя",
            "себя",
            "через",
            "нужно",
            "можно",
            "привет",
            "расскажи",
            "найди",
            "найти",
            "хочу",
            "научиться",
            "изучаю",
            "учусь",
            "зовут",
            "команда",
            "команду",
            "компас",
            "kompas",
            "3d",
            "3д",
        }
        aliases = {
            "запустить": ["запуск", "установка", "setup"],
            "запуска": ["запуск", "установка"],
            "создать": ["создание", "создать"],
            "создание": ["создание", "создать"],
            "объединить": ["объедин"],
            "объединение": ["объедин"],
            "объединения": ["объедин"],
            "детали": ["деталь", "тело"],
            "деталь": ["деталь", "тело"],
            "тела": ["тело"],
            "тел": ["тело"],
            "чертеж": ["чертеж", "чертёж"],
            "чертёж": ["чертеж", "чертёж"],
            "модель": ["модель", "модели", "моделирование"],
            "полилинию": ["полилиния"],
            "полилинии": ["полилиния"],
            "полилинией": ["полилиния"],
        }

        raw_tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", query.lower())
        expanded: list[str] = []
        for token in raw_tokens:
            if len(token) <= 2 or token in stop_words:
                continue
            expanded.extend(aliases.get(token, [token]))

        unique_tokens = []
        for token in expanded:
            if token not in unique_tokens:
                unique_tokens.append(token)
        return unique_tokens

    def has_domain_signal(self, query: str) -> bool:
        normalized = self._normalize_query(query).lower().replace("ё", "е")
        return bool(
            self._query_tokens(query)
            or "компас" in normalized
            or "kompas" in normalized
            or "3d" in normalized
            or "3д" in normalized
        )

    def _normalize_query(self, query: str) -> str:
        normalized = re.sub(
            r"\b(?:меня зовут|мо[её]\s+имя)\s+[А-ЯЁA-Zа-яёa-z-]{1,40}",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\bя\s+хочу\s+научиться\s+(?:моделировать|работать|проектировать)(?:\s+в\s+КОМПАС-?3[ДD])?",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
        return normalized

    def _clip(self, text: str, max_length: int = 260) -> str:
        text = " ".join(text.split())
        if len(text) <= max_length:
            return text
        cut_at = text.rfind(" ", 0, max_length - 3)
        if cut_at < 120:
            cut_at = max_length - 3
        return f"{text[:cut_at].rstrip()}..."
