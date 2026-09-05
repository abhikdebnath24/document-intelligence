from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from docintel.core.types import DocumentRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    agreement_type TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    index_sig TEXT NOT NULL,
    n_chunks INTEGER NOT NULL,
    collection TEXT NOT NULL,
    status TEXT NOT NULL,
    split TEXT,
    updated_at TEXT NOT NULL
)
"""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class DocumentRegistry:
    def __init__(self, db_url: str) -> None:
        if db_url.startswith("sqlite:///"):
            path = db_url[len("sqlite:///") :]
        elif db_url.startswith("sqlite://"):
            path = db_url[len("sqlite://") :]
        else:
            path = db_url
        self.path = Path(path)
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def list_all(self) -> list[DocumentRecord]:
        rows = self._conn.execute(
            "SELECT * FROM documents ORDER BY agreement_type, doc_id"
        ).fetchall()
        return [self._row(r) for r in rows]

    def get(self, doc_id: str) -> DocumentRecord | None:
        row = self._conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        return self._row(row)

    def _row(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            source_path=row["source_path"],
            agreement_type=row["agreement_type"],
            sha256=row["sha256"],
            pipeline_version=row["pipeline_version"],
            index_sig=row["index_sig"],
            n_chunks=int(row["n_chunks"]),
            collection=row["collection"],
            status=row["status"],
            split=row["split"],
        )

    def upsert(self, record: DocumentRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO documents (
                doc_id, source_path, agreement_type, sha256, pipeline_version,
                index_sig, n_chunks, collection, status, split, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_path=excluded.source_path,
                agreement_type=excluded.agreement_type,
                sha256=excluded.sha256,
                pipeline_version=excluded.pipeline_version,
                index_sig=excluded.index_sig,
                n_chunks=excluded.n_chunks,
                collection=excluded.collection,
                status=excluded.status,
                split=excluded.split,
                updated_at=excluded.updated_at
            """,
            (
                record.doc_id,
                record.source_path,
                record.agreement_type,
                record.sha256,
                record.pipeline_version,
                record.index_sig,
                record.n_chunks,
                record.collection,
                record.status,
                record.split,
                _utc_now(),
            ),
        )
        self._conn.commit()

    def is_current(self, doc_id: str, sha256: str, index_sig: str) -> bool:
        rec = self.get(doc_id)
        return (
            rec is not None
            and rec.status == "indexed"
            and rec.sha256 == sha256
            and rec.index_sig == index_sig
        )
