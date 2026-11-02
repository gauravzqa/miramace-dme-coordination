"""B7 Scenarios + CLI — `mira` console entry point (WP0 provides the app and `version`)."""

import typer

from miramace import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Mira Mace DME coordination prototype.")


@app.callback()
def main() -> None:
    """Mira Mace CLI."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"miramace {__version__}")


if __name__ == "__main__":
    app()
