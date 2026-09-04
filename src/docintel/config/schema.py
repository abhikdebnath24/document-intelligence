from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvalFramework(StrEnum):
    RAGAS = "ragas"
    DEEPEVAL = "deepeval"


class TraceSink(StrEnum):
    JSONL = "jsonl"
    MLFLOW = "mlflow"
    PHOENIX = "phoenix"


class NamedStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: str = "data_manifest/corpus_manifest.json"
    pdf_root: str = "data/CUAD_v1/full_contract_pdf"
    txt_root: str = "data/CUAD_v1/full_contract_txt"
    limit_docs: int | None = None


class IngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loader: NamedStrategy
    chunker: NamedStrategy
    dense_embedder: NamedStrategy
    sparse_encoder: NamedStrategy
    vectorstore: NamedStrategy
    collection_prefix: str = "cuad"
    pipeline_version: str = "v1"


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_agreement_type: bool = True
    use_doc_hint: bool = True


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    hybrid_impl: Literal["qdrant_native", "client_side"] = "client_side"
    k_candidates: int = 20
    fusion: NamedStrategy
    reranker: NamedStrategy
    query_transforms: list[str] = Field(default_factory=list)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rewrites: int = 2
    min_relevant_chunks: int = 2
    grader: NamedStrategy
    verifier: NamedStrategy
    citation_quote_fuzzy_threshold: int = 90
    general_knowledge_disclaimer: bool = True
    abstain_show_nearest: bool = True
    abstain_message_style: str = "explicit"


class LlmRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = 0.0
    max_tokens: int | None = None


class LlmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_provider: Literal["anthropic", "openai", "google_genai"] = "anthropic"
    roles: dict[str, LlmRole]
    timeout_s: int = 60
    max_retries: int = 3
    query_deadline_s: int = 90


class RagasConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(default_factory=list)
    batch_size: int = 8


class DeepevalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(default_factory=list)
    threshold: float = 0.7
    geval_rubrics: list[str] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qa_dev: str = "evals/qa_dev.json"
    qa_test: str = "evals/qa_test.json"
    split: Literal["dev", "test"] = "dev"
    finalists_file: str = "evals/finalists.txt"
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    span_match_threshold: int = 90
    frameworks: list[EvalFramework] = Field(default_factory=lambda: [EvalFramework.RAGAS])
    judge_role: str = "judge"
    ragas: RagasConfig = Field(default_factory=RagasConfig)
    deepeval: DeepevalConfig = Field(default_factory=DeepevalConfig)
    results_root: str = "results"


class MLflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    tracking_uri: str = "file:./mlruns"
    experiment: str = "docintel"
    log_traces: bool = True


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


class FeedbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_url: str = "sqlite:///./docintel.db"


class PhoenixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = "http://127.0.0.1:6006"


class TracingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sinks: list[TraceSink] = Field(default_factory=lambda: [TraceSink.JSONL, TraceSink.MLFLOW])
    jsonl_dir: str = "traces/"
    phoenix: PhoenixConfig = Field(default_factory=PhoenixConfig)


class FrontendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["inprocess", "http"] = "inprocess"
    api_url: str = "http://127.0.0.1:8000"
    pdf_render_dpi: int = 120


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus: CorpusConfig
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    agent: AgentConfig
    llm: LlmConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    frontend: FrontendConfig = Field(default_factory=FrontendConfig)
    profile: str = "base"
