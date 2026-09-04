from __future__ import annotations

from pathlib import Path

import pytest

from docintel.config.loader import (
    apply_env_overrides,
    config_hash,
    deep_merge,
    index_sig,
    load_config,
)
from docintel.core.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]


def test_l1_retrieval_profiles_share_index_sig() -> None:
    base = load_config("base", repo_root=ROOT)
    dense = load_config("exp_dense_only", repo_root=ROOT)
    sparse = load_config("exp_sparse_only", repo_root=ROOT)
    hybrid = load_config("exp_hybrid_rrf", repo_root=ROOT)
    bge = load_config("exp_hybrid_rerank_bge", repo_root=ROOT)
    bge_base = load_config("exp_hybrid_rerank_bge_base", repo_root=ROOT)
    assert dense.retrieval.mode == "dense"
    assert sparse.retrieval.mode == "sparse"
    assert hybrid.retrieval.mode == "hybrid"
    assert hybrid.retrieval.fusion.name == "rrf"
    assert bge.retrieval.reranker.name == "cross_encoder"
    assert bge.retrieval.reranker.params["model_id"] == "BAAI/bge-reranker-v2-m3"
    assert bge_base.retrieval.reranker.params["model_id"] == "BAAI/bge-reranker-base"
    assert index_sig(dense) == index_sig(base)
    assert index_sig(sparse) == index_sig(base)
    assert index_sig(hybrid) == index_sig(base)
    assert index_sig(bge) == index_sig(base)
    assert index_sig(bge_base) == index_sig(base)
    assert config_hash(dense) != config_hash(base)


def test_openai_and_fixed_chunk_profiles_load() -> None:
    openai = load_config("openai_embed", repo_root=ROOT)
    assert openai.ingestion.dense_embedder.name == "openai"
    assert openai.ingestion.dense_embedder.params["model_id"] == "text-embedding-3-small"
    fixed = load_config("exp_chunk_fixed", repo_root=ROOT)
    assert fixed.ingestion.chunker.name == "fixed_token"
    assert index_sig(fixed) != index_sig(load_config("base", repo_root=ROOT))


def test_load_base_and_dev_cpu_profile() -> None:
    base = load_config("base", repo_root=ROOT)
    assert base.ingestion.dense_embedder.name == "nomic_v15"
    assert base.ingestion.chunker.params["chunk_tokens"] == 512
    assert base.llm.roles["judge"].model.startswith("anthropic:")

    dev = load_config("dev_cpu", repo_root=ROOT)
    assert dev.profile == "dev_cpu"
    assert dev.corpus.limit_docs == 20
    assert dev.ingestion.dense_embedder.name == "st_dense"
    assert dev.ingestion.dense_embedder.params["model_id"] == "BAAI/bge-small-en-v1.5"
    assert dev.ingestion.chunker.params["chunk_tokens"] == 512
    assert dev.retrieval.reranker.name == "none"


def test_profile_overrides_do_not_mutate_base_hash_identity() -> None:
    base = load_config("base", repo_root=ROOT)
    gpu = load_config("gpu_default", repo_root=ROOT)
    assert config_hash(base) == config_hash(gpu)
    assert index_sig(base) == index_sig(gpu)


def test_env_override_nested_leaf() -> None:
    cfg = load_config(
        "dev_cpu",
        repo_root=ROOT,
        environ={"DOCINTEL__INGESTION__CHUNKER__PARAMS__CHUNK_TOKENS": "256"},
    )
    assert cfg.ingestion.chunker.params["chunk_tokens"] == 256


def test_env_override_unknown_path_raises() -> None:
    with pytest.raises(ConfigError, match="does not match"):
        load_config(
            "dev_cpu",
            repo_root=ROOT,
            environ={"DOCINTEL__DOES_NOT_EXIST": "1"},
        )


def test_config_hash_stable_and_ignores_paths() -> None:
    a = load_config("dev_cpu", repo_root=ROOT)
    b = load_config("dev_cpu", repo_root=ROOT)
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 64

    mutated = a.model_copy(deep=True)
    mutated.corpus.pdf_root = "/tmp/other"
    mutated.feedback.db_url = "sqlite:///./other.db"
    assert config_hash(mutated) == config_hash(a)

    changed = load_config(
        "dev_cpu",
        repo_root=ROOT,
        environ={"DOCINTEL__INGESTION__CHUNKER__PARAMS__CHUNK_TOKENS": "128"},
    )
    assert changed.ingestion.chunker.params["chunk_tokens"] == 128
    assert a.ingestion.chunker.params["chunk_tokens"] == 512
    assert config_hash(changed) != config_hash(a)
    assert index_sig(changed) != index_sig(a)


def test_index_sig_ignores_runtime_knobs_but_not_model() -> None:
    a = load_config("dev_cpu", repo_root=ROOT)
    bigger_batch = load_config(
        "dev_cpu",
        repo_root=ROOT,
        environ={"DOCINTEL__INGESTION__DENSE_EMBEDDER__PARAMS__BATCH_SIZE": "64"},
    )
    assert index_sig(bigger_batch) == index_sig(a)

    other_model = load_config(
        "dev_cpu",
        repo_root=ROOT,
        environ={"DOCINTEL__INGESTION__DENSE_EMBEDDER__PARAMS__MODEL_ID": "x/other"},
    )
    assert index_sig(other_model) != index_sig(a)


def test_unknown_strategy_field_rejected(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "configs" / "profiles" / "typo.yaml").write_text(
        "ingestion:\n  chunker:\n    name: recursive\n    parms: {}\n"
    )
    with pytest.raises(ConfigError, match="invalid config"):
        load_config("typo", repo_root=tmp_path)


def test_deep_merge_nested() -> None:
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 9}})
    assert merged == {"a": {"b": 1, "c": 9}}


def test_apply_env_bool_and_int() -> None:
    data = {"retrieval": {"k_candidates": 20, "filters": {"use_agreement_type": True}}}
    out = apply_env_overrides(
        data,
        {
            "DOCINTEL__RETRIEVAL__K_CANDIDATES": "8",
            "DOCINTEL__RETRIEVAL__FILTERS__USE_AGREEMENT_TYPE": "false",
        },
    )
    assert out["retrieval"]["k_candidates"] == 8
    assert out["retrieval"]["filters"]["use_agreement_type"] is False
