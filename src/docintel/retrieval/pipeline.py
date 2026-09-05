from __future__ import annotations

from collections.abc import Sequence

from docintel.config import AppConfig
from docintel.core.interfaces import BaseFusion, BaseQueryTransform, BaseReranker, BaseRetriever
from docintel.core.types import RetrievalQuery, RetrievedChunk
from docintel.retrieval.transforms import FilterExtractor


class RetrievalPipeline:
    def __init__(
        self,
        config: AppConfig,
        retriever: BaseRetriever,
        fusion: BaseFusion,
        reranker: BaseReranker,
        transforms: Sequence[BaseQueryTransform],
        extractor: FilterExtractor,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.fusion = fusion
        self.reranker = reranker
        self.transforms = list(transforms)
        self.extractor = extractor

    def search(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """Fuse candidates. Rerank is a separate graph node / retrieve() step."""
        flags = self.config.retrieval.filters
        extracted = self.extractor.extract(
            query.text,
            use_agreement_type=flags.use_agreement_type,
            use_doc_hint=flags.use_doc_hint,
        )
        filters = {**extracted, **query.filters}
        texts = [query.text]
        for transform in self.transforms:
            texts = [t for src in texts for t in transform.transform(src)]
        texts = list(dict.fromkeys(texts)) or [query.text]
        retrieve_k = max(query.k, self.config.retrieval.k_candidates)
        lists: list[list[RetrievedChunk]] = []
        for text in texts:
            lists.append(
                self.retriever.retrieve(
                    RetrievalQuery(text=text, k=retrieve_k, filters=filters, doc_id=query.doc_id)
                )
            )
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
