from __future__ import annotations

import math
import re

from docintel.core.interfaces import (
    BaseDenseEmbedder,
    BaseFusion,
    BaseRetriever,
    BaseSparseEncoder,
    BaseVectorStore,
)
from docintel.core.types import RetrievalQuery, RetrievedChunk
from docintel.ingestion.qdrant_indexer import QdrantIndexer

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DenseRetriever(BaseRetriever):
    def __init__(self, embedder: BaseDenseEmbedder, store: BaseVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        vec = self.embedder.embed_query(query.text)
        rows = self.store.search_dense(vec, query.k, query.filters or None)
        return [_tag(row, "dense") for row in rows]


class SparseRetriever(BaseRetriever):
    def __init__(self, encoder: BaseSparseEncoder, store: BaseVectorStore) -> None:
        self.encoder = encoder
        self.store = store

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        vec = self.encoder.encode_query(query.text)
        rows = self.store.search_sparse(vec, query.k, query.filters or None)
        return [_tag(row, "sparse") for row in rows]


class QdrantHybridRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseDenseEmbedder,
        encoder: BaseSparseEncoder,
        store: BaseVectorStore,
        fusion: str = "rrf",
        fusion_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.encoder = encoder
        self.store = store
        self.fusion = fusion
        self.fusion_k = fusion_k

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        qd = self.embedder.embed_query(query.text)
        qs = self.encoder.encode_query(query.text)
        rows = self.store.search_hybrid(
            qd,
            qs,
            query.k,
            self.fusion,
            query.filters or None,
            fusion_k=self.fusion_k,
        )
        return [_tag(row, "fused") for row in rows]


class ClientHybridRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseDenseEmbedder,
        encoder: BaseSparseEncoder,
        store: BaseVectorStore,
        fusion: BaseFusion,
    ) -> None:
        self.dense = DenseRetriever(embedder, store)
        self.sparse = SparseRetriever(encoder, store)
        self.fusion = fusion

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        wide = query.model_copy(update={"k": max(query.k, 20)})
        fused = self.fusion.fuse([self.dense.retrieve(wide), self.sparse.retrieve(wide)])
        return fused[: query.k]


class SparseBm25Inproc(BaseRetriever):
    """In-process BM25 over Qdrant payloads. Used when comparing fusion without native sparse."""

    def __init__(self, store: QdrantIndexer, k1: float = 1.5, b: float = 0.75) -> None:
        self.store = store
        self.k1 = k1
        self.b = b
        self._docs: list[RetrievedChunk] | None = None
        self._tf: list[dict[str, int]] | None = None
        self._df: dict[str, int] | None = None
        self._avgdl = 0.0

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        self._ensure()
        assert self._docs is not None and self._tf is not None and self._df is not None
        q_tokens = _tokenize(query.text)
        if not q_tokens:
            return []
        n = len(self._docs)
        scored: list[RetrievedChunk] = []
        filt = query.filters or {}
        want = filt.get("doc_id")
        allowed = set(want) if isinstance(want, list | tuple | set) else ({want} if want else set())
        for doc, tf in zip(self._docs, self._tf, strict=True):
            if allowed and doc.chunk.doc_id not in allowed:
                continue
            if filt.get("agreement_type") and not _type_matches(doc, str(filt["agreement_type"])):
                continue
            dl = sum(tf.values()) or 1
            score = 0.0
            for tok in set(q_tokens):
                freq = tf.get(tok, 0)
                if not freq:
                    continue
                df = self._df.get(tok, 0)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                score += idf * (freq * (self.k1 + 1.0)) / denom
            if score > 0:
                scored.append(
                    doc.model_copy(
                        update={
                            "score": score,
                            "source": "sparse",
                            "provenance": ["bm25_inproc"],
                        }
                    )
                )
        scored.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
        out: list[RetrievedChunk] = []
        for rank, row in enumerate(scored[: query.k], start=1):
            out.append(row.model_copy(update={"rank": rank}))
        return out

    def _ensure(self) -> None:
        if self._docs is not None:
            return
        docs = self.store.scroll_chunks()
        tfs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        lengths: list[int] = []
        for doc in docs:
            tf: dict[str, int] = {}
            for tok in _tokenize(doc.chunk.text):
                tf[tok] = tf.get(tok, 0) + 1
            tfs.append(tf)
            lengths.append(sum(tf.values()))
            for tok in tf:
                df[tok] = df.get(tok, 0) + 1
        self._docs = docs
        self._tf = tfs
        self._df = df
        self._avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _tag(row: RetrievedChunk, stage: str) -> RetrievedChunk:
    return row.model_copy(update={"provenance": [*row.provenance, stage]})


def _type_matches(row: RetrievedChunk, wanted: str) -> bool:
    meta = row.chunk.metadata or {}
    return str(meta.get("agreement_type") or "") == wanted
