from __future__ import annotations

from docintel.core.types import RetrievedChunk


class ChunkCache:
    """Per-request chunk text. Keep ids in graph state, not payloads."""

    def __init__(self) -> None:
        self._rows: dict[str, RetrievedChunk] = {}

    def put(self, rows: list[RetrievedChunk]) -> list[str]:
        ids: list[str] = []
        for row in rows:
            cid = row.chunk.chunk_id
            self._rows[cid] = row
            ids.append(cid)
        return ids

    def get(self, chunk_id: str) -> RetrievedChunk | None:
        return self._rows.get(chunk_id)

    def get_many(self, ids: list[str]) -> list[RetrievedChunk]:
        return [self._rows[i] for i in ids if i in self._rows]

    def clear(self) -> None:
        self._rows.clear()
