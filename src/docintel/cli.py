from __future__ import annotations

from pathlib import Path

import typer

from docintel import __version__
from docintel.config import config_hash, index_sig, load_config
from docintel.config.loader import find_repo_root
from docintel.core.logging import configure_logging, get_logger
from docintel.settings import load_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Document intelligence RAG CLI")
log = get_logger(__name__)

# Default profile comes from DOCINTEL_PROFILE (.env or environment), else dev_cpu.
_PROFILE_OPT = typer.Option(load_settings().docintel_profile, "--profile", "-p")


def _not_implemented(what: str, workstream: str) -> None:
    typer.echo(f"{what} is not implemented yet ({workstream})", err=True)
    raise typer.Exit(code=2)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command("config-hash")
def config_hash_cmd(profile: str = _PROFILE_OPT) -> None:
    cfg = load_config(profile)
    typer.echo(f"profile={cfg.profile}")
    typer.echo(f"config_hash={config_hash(cfg)}")
    typer.echo(f"index_sig={index_sig(cfg)}")


@app.command()
def ingest(
    profile: str = _PROFILE_OPT,
    only_changed: bool = typer.Option(
        True,
        "--only-changed/--full",
        help="Skip docs already indexed with the same sha256 + index_sig (default). "
        "--full re-embeds everything.",
    ),
    path: list[Path] | None = typer.Option(
        None, "--path", help="PDF paths; default is the manifest"
    ),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    cfg = load_config(profile)
    from docintel.ingestion.factory import build_ingest_components

    root = find_repo_root()
    pipeline = build_ingest_components(cfg, repo_root=root)
    out = report or (root / ".cache" / "ingestion_report.json")
    try:
        result = pipeline.run(
            paths=list(path) if path else None,
            only_changed=only_changed,
            report_path=out,
        )
        typer.echo(
            f"collection={result.collection} indexed={result.indexed} "
            f"skipped={result.skipped} failed={len(result.failed)} chunks={result.chunks}"
        )
        if result.eval_failures:
            typer.echo(f"eval-split failures: {', '.join(result.eval_failures)}", err=True)
        typer.echo(f"report={out}")
        if result.failed:
            raise typer.Exit(code=1)
    finally:
        # Embedded Qdrant must be closed before interpreter shutdown; otherwise its
        # finalizer can run after Python has torn down its import machinery.
        pipeline.store.close()
        pipeline.registry.close()


@app.command()
def query(
    question: str = typer.Argument(...),
    profile: str = _PROFILE_OPT,
) -> None:
    cfg = load_config(profile)
    from docintel.core.errors import DeadlineExceeded, MissingSecretError
    from docintel.service.container import Container
    from docintel.service.query_service import QueryService

    root = find_repo_root()
    svc: QueryService | None = None
    try:
        # Container builds chat models eagerly; a missing key raises here, not in ask()
        svc = QueryService(Container(cfg, repo_root=root))
        answer, log = svc.ask(question)
    except MissingSecretError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except DeadlineExceeded as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if svc is not None:
            svc.close()
    typer.echo(answer.text)
    if answer.abstained:
        typer.echo("abstained=true")
    typer.echo(f"route={answer.route} groundedness={answer.groundedness}")
    for cite in answer.citations:
        typer.echo(f"  [{cite.chunk_id}] p.{cite.page_no} {cite.quote[:160]}")
    if log.trace_path:
        typer.echo(f"trace={log.trace_path}")


@app.command("eval")
def eval_cmd(
    profile: str = _PROFILE_OPT,
    split: str = typer.Option("dev", "--split"),
    layer: str = typer.Option("L1", "--layer", help="L1 retrieval (WS3). L2 generation is WS5."),
    scoped: bool = typer.Option(False, "--scoped", help="Force gold doc_id filter (diagnostic)."),
) -> None:
    cfg = load_config(profile)
    if layer.upper() != "L1":
        _not_implemented("eval L2", "WS5")
    from docintel.evaluation.experiment import FinalistGateError, run_retrieval_eval

    root = find_repo_root()
    try:
        path = run_retrieval_eval(cfg, split=split, repo_root=root, scoped=scoped)
    except FinalistGateError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(str(path))


@app.command()
def serve(profile: str = _PROFILE_OPT) -> None:
    load_config(profile)
    _not_implemented("serve", "WS7/WS8")


@app.command()
def doctor(profile: str = _PROFILE_OPT) -> None:
    cfg = load_config(profile)
    from docintel.ingestion.doctor import run_doctor

    lines, failed = run_doctor(cfg)
    for line in lines:
        typer.echo(line)
    if failed:
        raise typer.Exit(code=1)
