from __future__ import annotations

import os


def test_hf_token_from_env_file_exported(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_unit_test\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    from docintel.settings import load_settings

    settings = load_settings()
    assert settings.hf_token == "hf_unit_test"
    assert os.environ["HF_TOKEN"] == "hf_unit_test"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_unit_test"


def test_existing_hf_token_not_overwritten(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_already_set")
    from docintel.settings import load_settings

    load_settings()
    assert os.environ["HF_TOKEN"] == "hf_already_set"
