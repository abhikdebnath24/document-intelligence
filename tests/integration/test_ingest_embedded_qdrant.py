from __future__ import annotations

from pathlib import Path

import pytest

from docintel.config import load_config
from docintel.core.errors import CollectionMismatchError, QdrantInUseError
from docintel.core.types import Document
from docintel.ingestion.chunkers import RecursiveChunker
from docintel.ingestion.embedders import HashDenseEmbedder, HashSparseEncoder
from docintel.ingestion.loaders import PyMuPDFLoader
from docintel.ingestion.pipeline import IngestionPipeline
from docintel.ingestion.qdrant_indexer import QdrantIndexer, collection_name
from docintel.ingestion.registry_store import DocumentRegistry

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


def _write_pdf(path: Path, pages: list[str]) -> Path:
    import pymupdf

    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        for text in pages:
            page = doc.new_page()
            page.insert_text((72, 72), text)
        doc.save(path)
    finally:
        doc.close()
    return path


class _BoomEmbedder(HashDenseEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed boom")


def _pipeline(
    tmp_path: Path,
    root: Path,
    *,
    dense: HashDenseEmbedder | None = None,
) -> IngestionPipeline:
    cfg = load_config("dev_cpu", repo_root=root)
    store = QdrantIndexer(path=str(tmp_path / "qdrant"), mode="embedded", on_disk=False)
    registry = DocumentRegistry(f"sqlite:///{tmp_path / 'reg.db'}")
    return IngestionPipeline(
        cfg,
        PyMuPDFLoader(strip_headers_footers=False),
        RecursiveChunker(chunk_tokens=32, overlap_tokens=8),
        dense or HashDenseEmbedder(dim=8),
        HashSparseEncoder(),
        store,
        registry,
        repo_root=root,
    )


def test_ingest_three_pdfs_round_trip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    pdfs = [
        _write_pdf(tmp_path / "a.pdf", ["Alpha termination for convenience in service."]),
        _write_pdf(tmp_path / "b.pdf", ["Bravo confidentiality obligations remain."]),
        _write_pdf(tmp_path / "c.pdf", ["Charlie governing law is Delaware."]),
    ]
    pipe = _pipeline(tmp_path, root)
    report = pipe.run(paths=pdfs, only_changed=False)
    assert report.indexed == 3
    assert report.failed == []
    assert report.chunks >= 3
    hits = pipe.store.search_dense(pipe.dense.embed_query("termination for convenience"), k=3)
    assert hits
    assert hits[0].chunk.doc_id in {"a", "b", "c"}
    fused = pipe.store.search_hybrid(
        pipe.dense.embed_query("termination for convenience"),
        pipe.sparse.encode_query("termination for convenience"),
        k=3,
        fusion="rrf",
    )
    assert fused
    assert fused[0].source == "fused"

    again = pipe.run(paths=pdfs, only_changed=True)
    assert again.indexed == 0
    assert again.skipped == 3
    pipe.store.close()


def test_mid_embed_failure_keeps_old_points(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    pdf = _write_pdf(tmp_path / "keep.pdf", ["Original exclusivity clause stays."])
    pipe = _pipeline(tmp_path, root)
    first = pipe.run(paths=[pdf], only_changed=False)
    assert first.indexed == 1
    old_hash = pipe.registry.get("keep")
    assert old_hash is not None and old_hash.status == "indexed"
    counted = pipe.store.count_by_doc_hash("keep", old_hash.sha256)
    assert counted == first.chunks

    _write_pdf(tmp_path / "keep.pdf", ["Revised exclusivity clause is gone."])
    pipe.dense = _BoomEmbedder()
    report = pipe.run(paths=[pdf], only_changed=True)
    assert any(item["doc_id"] == "keep" for item in report.failed)
    rec = pipe.registry.get("keep")
    assert rec is not None
    assert rec.status == "failed"
    # old version stays queryable
    assert pipe.store.count_by_doc_hash("keep", old_hash.sha256) == counted
    hits = pipe.store.search_dense(HashDenseEmbedder(dim=8).embed_query("exclusivity"), k=1)
    assert hits
    assert "Original exclusivity" in hits[0].chunk.text

    # failed rows are retried on the next --only-changed run, not silently skipped
    pipe.dense = HashDenseEmbedder(dim=8)
    retry = pipe.run(paths=[pdf], only_changed=True)
    assert retry.indexed == 1 and retry.skipped == 0
    rec = pipe.registry.get("keep")
    assert rec is not None and rec.status == "indexed"
    assert pipe.store.count_by_doc_hash("keep", old_hash.sha256) == 0
    hits = pipe.store.search_dense(HashDenseEmbedder(dim=8).embed_query("exclusivity"), k=1)
    assert "Revised exclusivity" in hits[0].chunk.text
    pipe.store.close()


def test_dim_mismatch_rejected(tmp_path: Path) -> None:
    store = QdrantIndexer(path=str(tmp_path / "q"), mode="embedded", on_disk=False)
    name = collection_name("cuad", "a" * 64)
    store.ensure_collection(name, 8, has_sparse=True, index_sig="a" * 64)
    with pytest.raises(CollectionMismatchError, match="dense_dim"):
        store.ensure_collection(name, 16, has_sparse=True, index_sig="a" * 64)
    store.close()


def test_shared_client_survives_other_close(tmp_path: Path) -> None:
    path = str(tmp_path / "share")
    keep = QdrantIndexer(path=path, mode="embedded", on_disk=False)
    keep.ensure_collection("share__sig", 8, has_sparse=True, index_sig="c" * 64)
    extra = QdrantIndexer(path=path, mode="embedded", on_disk=False)
    extra.close()
    keep.ensure_collection("share__sig", 8, has_sparse=True, index_sig="c" * 64)
    keep.close()


def test_second_opener_is_clear_error(tmp_path: Path) -> None:
    path = str(tmp_path / "lockme")
    first = QdrantIndexer(path=path, mode="embedded", on_disk=False)
    first.ensure_collection("lock__sig", 8, has_sparse=True, index_sig="b" * 64)
    with pytest.raises(QdrantInUseError, match="already"):
        QdrantIndexer(path=path, mode="embedded", on_disk=False, force_new=True)
    first.close()


def test_loader_round_trip_pages(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path / "x.pdf", ["Hello clause.", "Second page body."])
    doc: Document = PyMuPDFLoader(strip_headers_footers=False).load(pdf)
    assert doc.doc_id == "x"
    assert len(doc.pages) == 2
    assert "Hello" in doc.pages[0].text
    assert doc.pages[0].blocks
    assert doc.metadata["raw_norm_chars"] == sum(
        len("".join(p.text.lower().split())) for p in doc.pages
    )


def test_manifest_jobs_carry_agreement_type_and_txt(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config("dev_cpu", repo_root=root)
    manifest = root / cfg.corpus.manifest
    if not manifest.is_file():
        pytest.skip("manifest not built")
    pipe = _pipeline(tmp_path, root)
    jobs = pipe._jobs(None)
    assert len(jobs) == cfg.corpus.limit_docs
    assert all(j["agreement_type"] and j["txt_path"] is not None for j in jobs)
    assert all(j["split"] in {"index", "index_and_eval"} for j in jobs)
    pipe.store.close()
