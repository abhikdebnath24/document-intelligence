from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from docintel.config import AppConfig, index_sig
from docintel.config.loader import find_repo_root
from docintel.core.interfaces import BaseChunker, BaseDenseEmbedder, BaseLoader, BaseSparseEncoder
from docintel.core.logging import get_logger
from docintel.core.types import Chunk, Document, DocumentRecord
from docintel.data.corpus import load_manifest
from docintel.evaluation.gold import file_sha256
from docintel.ingestion.chunkers import estimate_tokens
from docintel.ingestion.loaders import agreement_type_for, doc_id_for, validate_against_txt
from docintel.ingestion.qdrant_indexer import QdrantIndexer, collection_name, point_id
from docintel.ingestion.registry_store import DocumentRegistry

log = get_logger(__name__)


@dataclass
class IngestReport:
    profile: str
    index_sig: str
    collection: str
    docs: int = 0
    skipped: int = 0
    indexed: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)
    chars: int = 0
    est_tokens: int = 0
    chunks: int = 0
    timings_s: dict[str, float] = field(default_factory=dict)
    validation: list[dict[str, Any]] = field(default_factory=list)
    eval_failures: list[str] = field(default_factory=list)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


class IngestionPipeline:
    def __init__(
        self,
        config: AppConfig,
        loader: BaseLoader,
        chunker: BaseChunker,
        dense: BaseDenseEmbedder,
        sparse: BaseSparseEncoder,
        store: QdrantIndexer,
        registry: DocumentRegistry,
        *,
        repo_root: Path | None = None,
    ) -> None:
        self.config = config
        self.loader = loader
        self.chunker = chunker
        self.dense = dense
        self.sparse = sparse
        self.store = store
        self.registry = registry
        self.repo_root = repo_root or find_repo_root()
        self.sig = index_sig(config)
        self.collection = collection_name(config.ingestion.collection_prefix, self.sig)

    def run(
        self,
        *,
        paths: list[Path] | None = None,
        only_changed: bool = True,
        report_path: Path | None = None,
    ) -> IngestReport:
        """only_changed=True skips docs whose registry row is indexed with the same
        sha256 + index_sig. Pending / failed / changed rows are always retried.
        only_changed=False re-embeds everything (idempotent point ids)."""
        jobs = self._jobs(paths)
        self.store.ensure_collection(
            self.collection, self.dense.dim, has_sparse=True, index_sig=self.sig
        )
        report = IngestReport(
            profile=self.config.profile,
            index_sig=self.sig,
            collection=self.collection,
            docs=len(jobs),
        )
        timings = {"load": 0.0, "chunk": 0.0, "embed": 0.0, "upsert": 0.0}
        for job in jobs:
            doc_id = job["doc_id"]
            sha = file_sha256(job["path"])
            if only_changed and self.registry.is_current(doc_id, sha, self.sig):
                report.skipped += 1
                continue
            try:
                result = self._ingest_one(job, sha, timings)
            except Exception as exc:
                log.warning("ingest_failed", doc_id=doc_id, error=str(exc))
                self._mark(job, sha, status="failed", n_chunks=0)
                report.failed.append({"doc_id": doc_id, "error": str(exc)})
                if job.get("split") == "index_and_eval":
                    report.eval_failures.append(doc_id)
                continue
            report.indexed += 1
            report.chars += result["chars"]
            report.est_tokens += result["tokens"]
            report.chunks += result["n_chunks"]
            if result["validation"]:
                report.validation.append(result["validation"])
        report.timings_s = {k: round(v, 3) for k, v in timings.items()}
        if report_path is not None:
            report.dump(report_path)
        return report

    def _ingest_one(
        self,
        job: dict[str, Any],
        sha: str,
        timings: dict[str, float],
    ) -> dict[str, Any]:
        doc_id: str = job["doc_id"]
        self._mark(job, sha, status="pending", n_chunks=0)
        t0 = time.perf_counter()
        doc = self.loader.load(job["path"])
        timings["load"] += time.perf_counter() - t0
        doc.doc_id = doc_id
        doc.sha256 = sha
        if job.get("agreement_type"):
            doc.agreement_type = job["agreement_type"]
        txt_path = job.get("txt_path")
        validation = validate_against_txt(doc, txt_path) if txt_path else None
        t0 = time.perf_counter()
        chunks = self.chunker.chunk(doc)
        timings["chunk"] += time.perf_counter() - t0
        if not chunks:
            raise ValueError("no chunks produced")
        texts = [c.text for c in chunks]
        t0 = time.perf_counter()
        dense_vecs = self.dense.embed_documents(texts)
        sparse_vecs = self.sparse.encode_documents(texts)
        timings["embed"] += time.perf_counter() - t0
        if len(dense_vecs) != len(chunks) or len(sparse_vecs) != len(chunks):
            raise ValueError("embedder returned a different count than chunks")
        points = [
            _point(chunk, dense_vecs[i], sparse_vecs[i], sha, self.sig, doc)
            for i, chunk in enumerate(chunks)
        ]
        t0 = time.perf_counter()
        self.store.upsert(points)
        timings["upsert"] += time.perf_counter() - t0
        counted = self.store.count_by_doc_hash(doc_id, sha)
        if counted != len(chunks):
            raise ValueError(f"upsert count {counted} != {len(chunks)}")
        self.store.delete_by_doc(doc_id, except_hash=sha)
        self._mark(job, sha, status="indexed", n_chunks=len(chunks))
        full = "\n".join(p.text for p in doc.pages)
        return {
            "chars": len(full),
            "tokens": estimate_tokens(full),
            "n_chunks": len(chunks),
            "validation": validation,
        }

    def _mark(
        self,
        job: dict[str, Any],
        sha: str,
        *,
        status: Literal["pending", "indexed", "failed"],
        n_chunks: int,
    ) -> None:
        self.registry.upsert(
            DocumentRecord(
                doc_id=job["doc_id"],
                source_path=str(job["path"]),
                agreement_type=job.get("agreement_type") or agreement_type_for(job["path"]),
                sha256=sha,
                pipeline_version=self.config.ingestion.pipeline_version,
                index_sig=self.sig,
                n_chunks=n_chunks,
                collection=self.collection,
                status=status,
                split=job.get("split"),
            )
        )

    def _jobs(self, paths: list[Path] | None) -> list[dict[str, Any]]:
        if paths:
            return [
                {
                    "doc_id": doc_id_for(path),
                    "path": path,
                    "txt_path": None,
                    "split": None,
                    "agreement_type": None,
                }
                for path in paths
            ]
        manifest_path = self.repo_root / self.config.corpus.manifest
        payload = load_manifest(manifest_path)
        docs = list(payload.get("documents") or [])
        limit = self.config.corpus.limit_docs
        if limit is not None:
            docs = docs[: int(limit)]
        pdf_root = self.repo_root / self.config.corpus.pdf_root
        txt_root = self.repo_root / self.config.corpus.txt_root
        jobs: list[dict[str, Any]] = []
        for item in docs:
            rel = item["rel_path"]
            path = pdf_root / rel
            txt_name = item.get("txt_name") or ""
            txt_path = (txt_root / txt_name) if txt_name else None
            jobs.append(
                {
                    "doc_id": item["doc_stem"],
                    "path": path,
                    "txt_path": txt_path,
                    "split": item.get("split"),
                    "agreement_type": item.get("agreement_type"),
                }
            )
        return jobs


def _point(
    chunk: Chunk,
    dense: list[float],
    sparse: object,
    sha: str,
    sig: str,
    doc: Document,
) -> dict[str, Any]:
    return {
        "id": point_id(chunk.doc_id, sha, chunk.chunk_idx, sig),
        "dense": dense,
        "sparse": sparse,
        "payload": {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "chunk_idx": chunk.chunk_idx,
            "text": chunk.text,
            "agreement_type": doc.agreement_type,
            "page_no": chunk.page_start,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "sha256": sha,
            "index_sig": sig,
            "section_header": chunk.section_header,
            "bboxes": [b.model_dump() for b in chunk.bboxes],
        },
    }
