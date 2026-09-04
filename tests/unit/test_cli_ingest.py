from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from docintel.cli import app
from docintel.ingestion.pipeline import IngestReport

runner = CliRunner()


def test_ingest_run_error_closes_and_does_not_mask() -> None:
    pipeline = MagicMock()
    pipeline.run.side_effect = RuntimeError("embed fail")
    with (
        patch("docintel.cli.load_config"),
        patch("docintel.cli.find_repo_root", return_value=Path("/tmp")),
        patch(
            "docintel.ingestion.factory.build_ingest_components",
            return_value=pipeline,
        ),
    ):
        result = runner.invoke(app, ["ingest"])
    assert isinstance(result.exception, RuntimeError)
    assert "embed fail" in str(result.exception)
    pipeline.store.close.assert_called_once()
    pipeline.registry.close.assert_called_once()


def test_ingest_success_echoes_and_closes() -> None:
    pipeline = MagicMock()
    pipeline.run.return_value = IngestReport(
        profile="dev_cpu", index_sig="sig", collection="cuad__abc", indexed=1
    )
    with (
        patch("docintel.cli.load_config"),
        patch("docintel.cli.find_repo_root", return_value=Path("/tmp")),
        patch(
            "docintel.ingestion.factory.build_ingest_components",
            return_value=pipeline,
        ),
    ):
        result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0
    assert "collection=cuad__abc" in result.stdout
    pipeline.store.close.assert_called_once()
    pipeline.registry.close.assert_called_once()
