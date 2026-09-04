from __future__ import annotations

import typer

from docintel import __version__
from docintel.config import config_hash, index_sig, load_config
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
    only_changed: bool = typer.Option(False, "--only-changed"),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    load_config(profile)
    _ = (only_changed, resume)
    _not_implemented("ingest", "WS2")


@app.command()
def query(
    question: str = typer.Argument(...),
    profile: str = _PROFILE_OPT,
) -> None:
    load_config(profile)
    _ = question
    _not_implemented("query", "WS4")


@app.command("eval")
def eval_cmd(
    profile: str = _PROFILE_OPT,
    split: str = typer.Option("dev", "--split"),
) -> None:
    load_config(profile)
    _ = split
    _not_implemented("eval", "WS3/WS5")


@app.command()
def serve(profile: str = _PROFILE_OPT) -> None:
    load_config(profile)
    _not_implemented("serve", "WS7/WS8")


@app.command()
def doctor(profile: str = _PROFILE_OPT) -> None:
    cfg = load_config(profile)
    typer.echo(f"profile={cfg.profile} config_hash={config_hash(cfg)[:12]}")
    _not_implemented("doctor preflight", "WS2")
