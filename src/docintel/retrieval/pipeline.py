from __future__ import annotations

from collections.abc import Callable, Sequence

from docintel.config import AppConfig
from docintel.core.interfaces import BaseFusion, BaseQueryTransform, BaseReranker, BaseRetriever
from docintel.core.types import RetrievalQuery, RetrievedChunk
from docintel.retrieval.transforms import FilterExtractor


def _with_unknown_type(filters: dict) -> dict:
    """Uploads are agreement_type=Unknown. A hard type must would drop them."""
    at = filters.get("agreement_type")
    if isinstance(at, str) and at != "Unknown":
        return {**filters, "agreement_type": [at, "Unknown"]}
    if isinstance(at, list) and "Unknown" not in at:
        return {**filters, "agreement_type": [*at, "Unknown"]}
    return filters


class RetrievalPipeline:
    def __init__(
        self,
        config: AppConfig,
        retriever: BaseRetriever,
        fusion: BaseFusion,
        reranker: BaseReranker,
        transforms: Sequence[BaseQueryTransform],
        extractor: FilterExtractor,
        catalog_fn: Callable[[], list[dict[str, str]]] | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.fusion = fusion
        self.reranker = reranker
        self.transforms = list(transforms)
        self.extractor = extractor
        self.catalog_fn = catalog_fn

    def _extractor(self) -> FilterExtractor:
        # Re-read so a PDF uploaded after boot gets a doc hint without a restart.
        if self.catalog_fn is None:
            return self.extractor
        return FilterExtractor(self.catalog_fn())

    def search(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """Fuse candidates. Rerank is a separate graph node / retrieve() step.

        Uploads (agreement_type=Unknown) ride on _with_unknown_type. Do not add a
        second per-upload retrieve here: it doubled grade_documents input and
        pushed multi-rewrite queries past query_deadline_s.
        """
        flags = self.config.retrieval.filters
        extracted = self._extractor().extract(
            query.text,
            use_agreement_type=flags.use_agreement_type,
            use_doc_hint=flags.use_doc_hint,
        )
        filters = _with_unknown_type({**extracted, **query.filters})
        texts = [query.text]
        for transform in self.transforms:
            texts = [t for src in texts for t in transform.transform(src)]
        texts = list(dict.fromkeys(texts)) or [query.text]
        retrieve_k = max(query.k, self.config.retrieval.k_candidates)
        lists = [
            self.retriever.retrieve(
                RetrievalQuery(text=text, k=retrieve_k, filters=filters, doc_id=query.doc_id)
            )
            for text in texts
        ]
        return lists[0] if len(lists) == 1 else self.fusion.fuse(lists)

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        return self.reranker.rerank(query.text, self.search(query), query.k)

    def close(self) -> None:
        store = getattr(self.retriever, "store", None)
        if store is None:
            store = getattr(getattr(self.retriever, "dense", None), "store", None)
        closer = getattr(store, "close", None)
        if closer:
            closer()
