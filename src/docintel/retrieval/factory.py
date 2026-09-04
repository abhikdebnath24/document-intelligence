from __future__ import annotations

from pathlib import Path

from docintel.config import AppConfig, index_sig
from docintel.core.errors import CollectionMismatchError
from docintel.core.interfaces import BaseDenseEmbedder, BaseFusion, BaseSparseEncoder
from docintel.core.registry import Registry
from docintel.data.corpus import load_manifest, normalize_stem
from docintel.ingestion.factory import DENSE, SPARSE
from docintel.ingestion.qdrant_indexer import QdrantIndexer, collection_name
from docintel.retrieval.fusion import DBSFFusion, RRFFusion, WeightedFusion
from docintel.retrieval.pipeline import RetrievalPipeline
from docintel.retrieval.rerankers import CrossEncoderReranker, NoOpReranker
from docintel.retrieval.retrievers import (
    ClientHybridRetriever,
    DenseRetriever,
    QdrantHybridRetriever,
    SparseRetriever,
)
from docintel.retrieval.transforms import FilterExtractor, IdentityTransform

_KNOWN_TRANSFORMS = frozenset({"filter_extractor", "multi_query", "hyde"})
FUSIONS: Registry[BaseFusion] = Registry("fusion")
FUSIONS.register("rrf")(RRFFusion)
FUSIONS.register("dbsf")(DBSFFusion)
FUSIONS.register("weighted")(WeightedFusion)


def _catalog(config: AppConfig, repo_root: Path) -> list[dict[str, str]]:
    path = repo_root / config.corpus.manifest
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for item in load_manifest(path).get("documents") or []:
        stem = str(item.get("doc_stem") or "")
        rows.append(
            {
                "doc_id": normalize_stem(stem),
                "doc_stem": stem,
                "agreement_type": str(item.get("agreement_type") or ""),
            }
        )
    return rows


def build_retrieval_pipeline(
    config: AppConfig,
    *,
    repo_root: Path,
    dense: BaseDenseEmbedder | None = None,
    sparse: BaseSparseEncoder | None = None,
    store: QdrantIndexer | None = None,
) -> RetrievalPipeline:
    ing = config.ingestion
    ret = config.retrieval
    if dense is None:
        dense = DENSE.create(ing.dense_embedder.name, **ing.dense_embedder.params)
    if sparse is None:
        sparse = SPARSE.create(ing.sparse_encoder.name, **ing.sparse_encoder.params)
    if store is None:
        store = QdrantIndexer(**ing.vectorstore.params)
    sig = index_sig(config)
    name = collection_name(ing.collection_prefix, sig)
    if not store.has_collection(name):
        raise CollectionMismatchError(
            f"collection {name} not found for profile {config.profile!r}; "
            "run `docintel ingest` with a profile that shares this index_sig first"
        )
    store.ensure_collection(name, dense.dim, True, sig)
    fusion = FUSIONS.create(ret.fusion.name, **ret.fusion.params)
    fusion_k = int(ret.fusion.params.get("k", 60))
    retriever: DenseRetriever | SparseRetriever | ClientHybridRetriever | QdrantHybridRetriever
    if ret.mode == "dense":
        retriever = DenseRetriever(dense, store)
    elif ret.mode == "sparse":
        retriever = SparseRetriever(sparse, store)
    elif ret.hybrid_impl == "qdrant_native":
        retriever = QdrantHybridRetriever(
            dense, sparse, store, fusion=ret.fusion.name, fusion_k=fusion_k
        )
    else:
        retriever = ClientHybridRetriever(dense, sparse, store, fusion)
    if ret.reranker.name == "none":
        reranker: NoOpReranker | CrossEncoderReranker = NoOpReranker()
    elif ret.reranker.name == "cross_encoder":
        reranker = CrossEncoderReranker(**ret.reranker.params)
    else:
        raise ValueError(
            f"reranker {ret.reranker.name!r} is not implemented (use none or cross_encoder)"
        )
    unknown = set(ret.query_transforms) - _KNOWN_TRANSFORMS
    if unknown:
        raise ValueError(f"unknown query_transforms {sorted(unknown)}")
    # filter_extractor always runs (gated by retrieval.filters). multi_query / hyde need
    # the WS4 LLM factory and are identity until then.
    transforms: list[IdentityTransform] = []
    return RetrievalPipeline(
        config,
        retriever,
        fusion,
        reranker,
        transforms,
        FilterExtractor(_catalog(config, repo_root)),
    )
