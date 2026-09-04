from __future__ import annotations

import math
import tempfile

from docintel.config import AppConfig, config_hash, index_sig
from docintel.core.device import resolve_device
from docintel.core.errors import MissingSecretError
from docintel.core.interfaces import BaseDenseEmbedder
from docintel.ingestion.embedders import HashDenseEmbedder, HashSparseEncoder
from docintel.ingestion.factory import DENSE
from docintel.ingestion.qdrant_indexer import QdrantIndexer, collection_name


def _ok(name: str, detail: str) -> str:
    return f"PASS  {name}: {detail}"


def _skip(name: str, detail: str) -> str:
    return f"SKIP  {name}: {detail}"


def _fail(name: str, detail: str) -> str:
    return f"FAIL  {name}: {detail}"


def run_doctor(config: AppConfig) -> tuple[list[str], int]:
    lines: list[str] = [
        f"profile={config.profile} config_hash={config_hash(config)[:12]} "
        f"index_sig={index_sig(config)[:12]}"
    ]
    failed = 0

    device = resolve_device(str(config.ingestion.dense_embedder.params.get("device", "auto")))
    torch_detail = _torch_detail(device)
    if "error" in torch_detail:
        lines.append(_fail("device", torch_detail))
        failed += 1
    else:
        lines.append(_ok("device", torch_detail))

    qdrant_line, qdrant_ok = _qdrant_roundtrip(config)
    lines.append(qdrant_line)
    if not qdrant_ok:
        failed += 1

    embed_line, embed_ok = _embedder_check(config)
    lines.append(embed_line)
    if not embed_ok:
        failed += 1

    lines.append(_skip("llm", "WS4 chat / structured-output preflight"))
    return lines, failed


def _torch_detail(device: str) -> str:
    try:
        import torch
    except ImportError:
        return f"{device} (torch not installed; Mac destage ok)"
    cuda = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if cuda else "n/a"
    return f"{device} torch={torch.__version__} cuda={cuda} gpu={name}"


def _qdrant_roundtrip(config: AppConfig) -> tuple[str, bool]:
    dense = HashDenseEmbedder(dim=8)
    sparse = HashSparseEncoder()
    sig = "doctor" + "0" * 60
    with tempfile.TemporaryDirectory() as tmp:
        store = QdrantIndexer(path=tmp, mode="embedded", on_disk=False)
        try:
            name = collection_name("doctor", sig)
            store.ensure_collection(name, dense.dim, has_sparse=True, index_sig=sig)
            store.upsert(
                [
                    {
                        "id": "00000000-0000-4000-8000-000000000001",
                        "dense": dense.embed_query("termination clause"),
                        "sparse": sparse.encode_query("termination clause"),
                        "payload": {
                            "doc_id": "doctor",
                            "text": "termination clause",
                            "chunk_id": "doctor:0",
                            "chunk_idx": 0,
                            "page_no": 1,
                            "sha256": "abc",
                        },
                    }
                ]
            )
            hits = store.search_dense(dense.embed_query("termination clause"), k=1)
            if not hits or hits[0].chunk.doc_id != "doctor":
                return _fail("qdrant", "round-trip returned no point"), False
            return _ok("qdrant", f"embedded round-trip collection={name}"), True
        except Exception as exc:
            return _fail("qdrant", str(exc)), False
        finally:
            store.close()


def _embedder_check(config: AppConfig) -> tuple[str, bool]:
    name = config.ingestion.dense_embedder.name
    params = dict(config.ingestion.dense_embedder.params)
    if name in {"st_dense", "nomic_v15"}:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return _skip(name, "sentence-transformers not installed (uv sync --group gpu)"), True
    elif name != "openai":
        return _skip(name, "no semantic check for this embedder"), True
    try:
        embedder: BaseDenseEmbedder = DENSE.create(name, **params)
        vecs = embedder.embed_documents(["termination for convenience", "potato salad"])
        query = embedder.embed_query("termination for convenience")
        return _score_pair(name, embedder.dim, query, vecs)
    except MissingSecretError as exc:
        return _skip(name, f"{exc.env_var} not set"), True
    except Exception as exc:
        return _fail(name, str(exc)), False


def _score_pair(
    name: str, dim: int, query: list[float], docs: list[list[float]]
) -> tuple[str, bool]:
    if len(query) != dim or any(len(v) != dim for v in docs):
        return _fail(name, f"dimension mismatch dim={dim}"), False
    if any(not math.isfinite(x) for x in query):
        return _fail(name, "non-finite vector"), False
    pos = sum(a * b for a, b in zip(query, docs[0], strict=True))
    neg = sum(a * b for a, b in zip(query, docs[1], strict=True))
    if pos <= neg:
        msg = f"positive pair did not rank above negative ({pos:.3f}<={neg:.3f})"
        return _fail(name, msg), False
    return _ok(name, f"dim={dim} pos={pos:.3f} > neg={neg:.3f}"), True
