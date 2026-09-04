# Convenience wrappers. Canonical commands are `uv run ...` (Windows-safe).

.PHONY: setup lint test ingest eval-retrieval eval-ragas app api doctor

setup:
	uv sync --group dev

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

test:
	uv run pytest tests/unit

ingest:
	uv run docintel ingest --profile $(or $(PROFILE),dev_cpu)

eval-retrieval:
	uv run docintel eval --profile $(or $(PROFILE),exp_hybrid_rrf) --split dev

eval-ragas:
	uv run docintel eval --profile $(or $(PROFILE),gpu_default) --split test

app:
	uv run streamlit run frontend/streamlit_app/app.py --server.address 127.0.0.1

api:
	uv run uvicorn docintel.api.app:app --reload

doctor:
	uv run docintel doctor --profile $(or $(PROFILE),dev_cpu)
