from __future__ import annotations

import uuid
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from docintel.core.errors import CollectionMismatchError, QdrantInUseError
from docintel.core.interfaces import BaseVectorStore
from docintel.core.types import BBox, Chunk, RetrievalSource, RetrievedChunk, SparseVector

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"
FINGERPRINT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "docintel:fingerprint"))
POINT_NS = uuid.uuid5(uuid.NAMESPACE_URL, "docintel:point")
_CLIENTS: dict[str, QdrantClient] = {}
_REFS: dict[str, int] = {}


def point_id(doc_id: str, content_hash: str, chunk_idx: int, index_sig: str) -> str:
    return str(uuid.uuid5(POINT_NS, f"{doc_id}:{content_hash}:{chunk_idx}:{index_sig}"))


def collection_name(prefix: str, sig: str) -> str:
    return f"{prefix}__{sig[:12]}"


def _sparse_model(vec: SparseVector) -> models.SparseVector:
    return models.SparseVector(indices=vec.indices, values=vec.values)


def _match(key: str, value: Any) -> models.FieldCondition:
    if isinstance(value, list | tuple | set):
        return models.FieldCondition(key=key, match=models.MatchAny(any=list(value)))
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


_NOT_FINGERPRINT = _match("is_fingerprint", True)


class QdrantIndexer(BaseVectorStore):
    def __init__(
        self,
        path: str = ".qdrant",
        mode: str = "embedded",
        on_disk: bool = True,
        url: str | None = None,
        force_new: bool = False,
        **_: object,
    ) -> None:
        self.mode = mode
        self.on_disk = on_disk
        self._name = ""
        self._index_sig = ""
        if mode == "server":
            self._client = QdrantClient(url=url or "http://127.0.0.1:6333")
            return
        resolved = str(Path(path).resolve())
        self._path = resolved
        if not force_new and resolved in _CLIENTS:
            self._client = _CLIENTS[resolved]
            _REFS[resolved] = _REFS.get(resolved, 1) + 1
            return
        try:
            client = QdrantClient(path=resolved)
        except RuntimeError as exc:
            raise QdrantInUseError(str(exc)) from exc
        _CLIENTS[resolved] = client
        _REFS[resolved] = 1
        self._client = client

    def alias(self) -> str:
        return "qdrant"

    def close(self) -> None:
        path = getattr(self, "_path", None)
        if path:
            left = _REFS.get(path, 1) - 1
            if left > 0:
                _REFS[path] = left
                return
            _REFS.pop(path, None)
        try:
            self._client.close()
        except Exception:
            pass
        if path:
            _CLIENTS.pop(path, None)

    def ensure_collection(
        self, name: str, dense_dim: int, has_sparse: bool, index_sig: str
    ) -> None:
        self._name = name
        self._index_sig = index_sig
        existing = {c.name for c in self._client.get_collections().collections}
        if name not in existing:
            sparse_cfg = None
            if has_sparse:
                sparse_cfg = {SPARSE_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)}
            self._client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_NAME: models.VectorParams(
                        size=dense_dim,
                        distance=models.Distance.COSINE,
                        on_disk=self.on_disk,
                    )
                },
                sparse_vectors_config=sparse_cfg,
            )
            for field, schema in (
                ("doc_id", models.PayloadSchemaType.KEYWORD),
                ("agreement_type", models.PayloadSchemaType.KEYWORD),
                ("page_no", models.PayloadSchemaType.INTEGER),
            ):
                # local/embedded mode ignores payload indexes; keep the call for server mode
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Payload indexes have no effect")
                    self._client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=schema,
                    )
            self._write_fingerprint(name, dense_dim, has_sparse, index_sig)
            return
        self._assert_fingerprint(name, dense_dim, has_sparse, index_sig)

    def _write_fingerprint(
        self, name: str, dense_dim: int, has_sparse: bool, index_sig: str
    ) -> None:
        vector: dict[str, Any] = {DENSE_NAME: [0.0] * dense_dim}
        if has_sparse:
            vector[SPARSE_NAME] = models.SparseVector(indices=[0], values=[0.0])
        self._client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=FINGERPRINT_ID,
                    vector=vector,
                    payload={
                        "is_fingerprint": True,
                        "index_sig": index_sig,
                        "dense_name": DENSE_NAME,
                        "dense_dim": dense_dim,
                        "sparse_name": SPARSE_NAME if has_sparse else None,
                    },
                )
            ],
        )

    def _assert_fingerprint(
        self, name: str, dense_dim: int, has_sparse: bool, index_sig: str
    ) -> None:
        points = self._client.retrieve(collection_name=name, ids=[FINGERPRINT_ID])
        if not points:
            raise CollectionMismatchError(f"{name} has no fingerprint point")
        payload = points[0].payload or {}
        expected = {
            "index_sig": index_sig,
            "dense_name": DENSE_NAME,
            "dense_dim": dense_dim,
            "sparse_name": SPARSE_NAME if has_sparse else None,
        }
        for key, want in expected.items():
            got = payload.get(key)
            if got != want:
                raise CollectionMismatchError(f"{name} fingerprint {key}={got!r} != {want!r}")

    def upsert(self, points: Sequence[dict[str, Any]]) -> None:
        if not points:
            return
        structs: list[models.PointStruct] = []
        for item in points:
            vector: dict[str, Any] = {DENSE_NAME: list(item["dense"])}
            sparse = item.get("sparse")
            if sparse is not None:
                if isinstance(sparse, SparseVector):
                    vector[SPARSE_NAME] = _sparse_model(sparse)
                else:
                    vector[SPARSE_NAME] = sparse
            structs.append(
                models.PointStruct(
                    id=item["id"],
                    vector=vector,
                    payload=item["payload"],
                )
            )
        self._client.upsert(collection_name=self._name, points=structs)

    def delete_by_doc(self, doc_id: str, *, except_hash: str | None = None) -> int:
        must: list[models.Condition] = [_match("doc_id", doc_id)]
        must_not: list[models.Condition] = [_NOT_FINGERPRINT]
        if except_hash is not None:
            must_not.append(_match("sha256", except_hash))
        before = self._count(models.Filter(must=must, must_not=[_NOT_FINGERPRINT]))
        self._client.delete(
            collection_name=self._name,
            points_selector=models.Filter(must=must, must_not=must_not),
        )
        after = self._count(models.Filter(must=must, must_not=[_NOT_FINGERPRINT]))
        return max(0, before - after)

    def count_by_doc_hash(self, doc_id: str, content_hash: str) -> int:
        must: list[models.Condition] = [_match("doc_id", doc_id), _match("sha256", content_hash)]
        return self._count(models.Filter(must=must))

    def _count(self, filt: models.Filter) -> int:
        result = self._client.count(collection_name=self._name, count_filter=filt, exact=True)
        return int(result.count)

    def _filter(self, filters: dict[str, Any] | None) -> models.Filter:
        must: list[models.Condition] = [_match(k, v) for k, v in (filters or {}).items()]
        return models.Filter(must=must or None, must_not=[_NOT_FINGERPRINT])

    def _to_retrieved(self, hits: Sequence[Any], source: RetrievalSource) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            raw_boxes = payload.get("bboxes") or []
            boxes = [BBox.model_validate(b) for b in raw_boxes]
            chunk = Chunk(
                chunk_id=str(payload.get("chunk_id") or hit.id),
                doc_id=str(payload.get("doc_id") or ""),
                text=str(payload.get("text") or ""),
                page_start=int(payload.get("page_no") or payload.get("page_start") or 1),
                page_end=int(payload.get("page_end") or payload.get("page_no") or 1),
                bboxes=boxes,
                section_header=payload.get("section_header"),
                chunk_idx=int(payload.get("chunk_idx") or 0),
                char_span=None,
                metadata={
                    "sha256": payload.get("sha256"),
                    "agreement_type": payload.get("agreement_type"),
                },
            )
            # scroll() yields Record without .score
            score = float(getattr(hit, "score", 0.0) or 0.0)
            out.append(RetrievedChunk(chunk=chunk, score=score, source=source, rank=rank))
        return out

    def has_collection(self, name: str) -> bool:
        return name in {c.name for c in self._client.get_collections().collections}

    def search_dense(
        self, q: Sequence[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        result = self._client.query_points(
            collection_name=self._name,
            query=list(q),
            using=DENSE_NAME,
            query_filter=self._filter(filters),
            limit=k,
            with_payload=True,
        )
        return self._to_retrieved(result.points, "dense")

    def search_sparse(
        self, q: SparseVector, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        result = self._client.query_points(
            collection_name=self._name,
            query=_sparse_model(q),
            using=SPARSE_NAME,
            query_filter=self._filter(filters),
            limit=k,
            with_payload=True,
        )
        return self._to_retrieved(result.points, "sparse")

    def search_hybrid(
        self,
        qd: Sequence[float],
        qs: SparseVector,
        k: int,
        fusion: str,
        filters: dict[str, Any] | None = None,
        fusion_k: int = 60,
    ) -> list[RetrievedChunk]:
        # qdrant 1.19: RrfQuery for RRF (k is first-class); FusionQuery for DBSF.
        # https://qdrant.tech/documentation/concepts/hybrid-queries/
        pre_k = max(k, 20)
        filt = self._filter(filters)
        prefetch = [
            models.Prefetch(query=list(qd), using=DENSE_NAME, limit=pre_k, filter=filt),
            models.Prefetch(query=_sparse_model(qs), using=SPARSE_NAME, limit=pre_k, filter=filt),
        ]
        name = fusion.lower()
        if name == "rrf":
            query: Any = models.RrfQuery(rrf=models.Rrf(k=int(fusion_k)))
        elif name == "dbsf":
            query = models.FusionQuery(fusion=models.Fusion.DBSF)
        else:
            raise ValueError(f"qdrant_native fusion {fusion!r} is not rrf or dbsf")
        result = self._client.query_points(
            collection_name=self._name,
            prefetch=prefetch,
            query=query,
            query_filter=filt,
            limit=k,
            with_payload=True,
        )
        return self._to_retrieved(result.points, "fused")

    def scroll_chunks(self) -> list[RetrievedChunk]:
        offset: Any = None
        out: list[RetrievedChunk] = []
        while True:
            records, offset = self._client.scroll(
                collection_name=self._name,
                scroll_filter=self._filter(None),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            out.extend(self._to_retrieved(records, "sparse"))
            if offset is None:
                break
        return out
