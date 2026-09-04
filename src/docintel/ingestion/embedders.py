from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from docintel.core.device import resolve_device
from docintel.core.errors import MissingSecretError
from docintel.core.interfaces import BaseDenseEmbedder, BaseSparseEncoder
from docintel.core.types import SparseVector
from docintel.settings import hf_token, load_settings


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class HashDenseEmbedder(BaseDenseEmbedder):
    """Deterministic stand-in for tests and doctor store checks. Not for scored runs."""

    def __init__(self, dim: int = 8, normalize: bool = True, **_: object) -> None:
        self._dim = dim
        self.normalize = normalize

    @property
    def dim(self) -> int:
        return self._dim

    def alias(self) -> str:
        return "hash"

    def _one(self, text: str) -> list[float]:
        out: list[float] = []
        for i in range(self._dim):
            digest = hashlib.sha256(f"{i}:{text}".encode()).digest()
            out.append(int.from_bytes(digest[:4], "big") / 2**32 * 2 - 1)
        return _l2_normalize(out) if self.normalize else out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


class HashSparseEncoder(BaseSparseEncoder):
    def __init__(self, modulus: int = 10_000, **_: object) -> None:
        self.modulus = modulus

    def alias(self) -> str:
        return "hash_sparse"

    def _one(self, text: str) -> SparseVector:
        seen: dict[int, float] = {}
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.modulus
            seen[idx] = seen.get(idx, 0.0) + 1.0
        indices = sorted(seen)
        return SparseVector(indices=indices, values=[seen[i] for i in indices])

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._one(t) for t in texts]

    def encode_query(self, text: str) -> SparseVector:
        return self._one(text)


class SentenceDenseEmbedder(BaseDenseEmbedder):
    """sentence-transformers wrapper. Latest nomic docs: no trust_remote_code."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
        doc_prefix: str = "",
        query_prefix: str = "",
        max_seq_len: int | None = None,
        **_: object,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        token = hf_token()
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        self.normalize = normalize
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        kwargs: dict[str, object] = {"device": resolve_device(device)}
        if revision:
            kwargs["revision"] = revision
        if token:
            kwargs["token"] = token
        try:
            self._model = SentenceTransformer(model_id, **kwargs)
        except ValueError as exc:
            # nomic ships native in transformers>=5; older stacks still need remote code
            if "trust_remote_code" not in str(exc):
                raise
            self._model = SentenceTransformer(model_id, trust_remote_code=True, **kwargs)
        if max_seq_len is not None:
            self._model.max_seq_length = max_seq_len
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"embedder {model_id} did not report a dimension")
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def alias(self) -> str:
        return "st_dense"

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([f"{self.doc_prefix}{t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self.query_prefix}{text}"])[0]


class NomicDenseEmbedder(SentenceDenseEmbedder):
    def alias(self) -> str:
        return "nomic_v15"


class OpenAIEmbedder(BaseDenseEmbedder):
    """openai==2.x embeddings.create. dimensions is first-class for embedding-3."""

    def __init__(
        self,
        model_id: str = "text-embedding-3-small",
        dimensions: int = 1536,
        normalize: bool = True,
        batch_size: int = 64,
        **_: object,
    ) -> None:
        from openai import OpenAI

        key = load_settings().openai_api_key
        if not key:
            raise MissingSecretError("OPENAI_API_KEY")
        self.model_id = model_id
        self.dimensions = dimensions
        self.normalize = normalize
        self.batch_size = batch_size
        self._client = OpenAI(api_key=key)

    @property
    def dim(self) -> int:
        return self.dimensions

    def alias(self) -> str:
        return "openai"

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        batch: list[str] = []
        for text in texts:
            batch.append(text if text.strip() else " ")
            if len(batch) >= self.batch_size:
                out.extend(self._call(batch))
                batch = []
        if batch:
            out.extend(self._call(batch))
        return out

    def _call(self, batch: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=self.model_id,
            input=batch,
            dimensions=self.dimensions,
        )
        rows = sorted(resp.data, key=lambda item: item.index)
        vectors = [list(map(float, item.embedding)) for item in rows]
        if self.normalize:
            return [_l2_normalize(v) for v in vectors]
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


class FastEmbedBM25(BaseSparseEncoder):
    """Qdrant/bm25 via fastembed. Collection sparse vector MUST use Modifier.IDF."""

    def __init__(self, model_name: str = "Qdrant/bm25", **_: object) -> None:
        from fastembed import SparseTextEmbedding

        hf_token()
        self._model = SparseTextEmbedding(model_name=model_name)

    def alias(self) -> str:
        return "fastembed_bm25"

    def _convert(self, embedding: object) -> SparseVector:
        indices = [int(i) for i in getattr(embedding, "indices")]
        values = [float(v) for v in getattr(embedding, "values")]
        return SparseVector(indices=indices, values=values)

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._convert(item) for item in self._model.embed(list(texts))]

    def encode_query(self, text: str) -> SparseVector:
        items = list(self._model.query_embed([text]))
        return self._convert(items[0])
