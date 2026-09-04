from __future__ import annotations

from pathlib import Path

from docintel.config import AppConfig
from docintel.core.interfaces import BaseChunker, BaseDenseEmbedder, BaseLoader, BaseSparseEncoder
from docintel.core.registry import Registry
from docintel.ingestion.chunkers import FixedTokenChunker, RecursiveChunker
from docintel.ingestion.embedders import (
    FastEmbedBM25,
    HashDenseEmbedder,
    HashSparseEncoder,
    NomicDenseEmbedder,
    OpenAIEmbedder,
    SentenceDenseEmbedder,
)
from docintel.ingestion.loaders import PyMuPDFLoader, TxtLoader
from docintel.ingestion.pipeline import IngestionPipeline
from docintel.ingestion.qdrant_indexer import QdrantIndexer
from docintel.ingestion.registry_store import DocumentRegistry

LOADERS: Registry[BaseLoader] = Registry("loader")
CHUNKERS: Registry[BaseChunker] = Registry("chunker")
DENSE: Registry[BaseDenseEmbedder] = Registry("dense_embedder")
SPARSE: Registry[BaseSparseEncoder] = Registry("sparse_encoder")

LOADERS.register("pymupdf")(PyMuPDFLoader)
LOADERS.register("txt")(TxtLoader)
CHUNKERS.register("recursive")(RecursiveChunker)
CHUNKERS.register("fixed_token")(FixedTokenChunker)
DENSE.register("st_dense")(SentenceDenseEmbedder)
DENSE.register("nomic_v15")(NomicDenseEmbedder)
DENSE.register("openai")(OpenAIEmbedder)
DENSE.register("hash")(HashDenseEmbedder)
SPARSE.register("fastembed_bm25")(FastEmbedBM25)
SPARSE.register("hash_sparse")(HashSparseEncoder)


def build_ingest_components(
    config: AppConfig,
    *,
    repo_root: Path | None = None,
    dense: BaseDenseEmbedder | None = None,
    sparse: BaseSparseEncoder | None = None,
    store: QdrantIndexer | None = None,
    registry: DocumentRegistry | None = None,
) -> IngestionPipeline:
    ing = config.ingestion
    loader = LOADERS.create(ing.loader.name, **ing.loader.params)
    chunker = CHUNKERS.create(ing.chunker.name, **ing.chunker.params)
    if dense is None:
        dense = DENSE.create(ing.dense_embedder.name, **ing.dense_embedder.params)
    if sparse is None:
        sparse = SPARSE.create(ing.sparse_encoder.name, **ing.sparse_encoder.params)
    if store is None:
        store = QdrantIndexer(**ing.vectorstore.params)
    if registry is None:
        db_url = config.feedback.db_url
        registry = DocumentRegistry(db_url)
    return IngestionPipeline(
        config,
        loader,
        chunker,
        dense,
        sparse,
        store,
        registry,
        repo_root=repo_root,
    )
