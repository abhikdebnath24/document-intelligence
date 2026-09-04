# document-intelligence

Agentic hybrid RAG over CUAD commercial contracts (Avathon Track D x S2).

## Setup (WS0)

```bash
uv sync --group dev
cp .env.example .env   # add keys later; not required for config/registry tests
uv run docintel --help
uv run pytest tests/unit
```

Do not commit `.env` or `data/`. Full reproduction steps land in WS9.
