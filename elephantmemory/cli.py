from __future__ import annotations

import time
from pathlib import Path

import click
from dotenv import load_dotenv

from .adapters.base import build_adapter
from .report import render_report, write_run
from .runner import run_scenario
from .scenarios import load_all

load_dotenv()


@click.group()
def cli() -> None:
    """elephantmemory benchmark harness."""


@cli.command()
@click.option("--adapter", "adapter_names", multiple=True, required=True,
              help="Adapter to run. Repeat for multiple.")
@click.option("--scenarios", "scenarios_dir", type=click.Path(exists=True, path_type=Path),
              default=Path("scenarios"))
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("results/runs"))
def run(adapter_names: tuple[str, ...], scenarios_dir: Path, out_dir: Path) -> None:
    """Run scenarios against one or more adapters."""
    scenarios = load_all(scenarios_dir)
    click.echo(f"loaded {len(scenarios)} scenarios from {scenarios_dir}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_path = out_dir / run_id

    all_results = []
    for name in adapter_names:
        click.echo(f"\n=== {name} ===")
        adapter = build_adapter(name)
        adapter.setup()
        try:
            for sc in scenarios:
                click.echo(f"  {sc.scenario_id} ({sc.category})... ", nl=False)
                r = run_scenario(adapter, sc)
                if r.error:
                    click.secho("ERROR", fg="red")
                    click.echo(f"    {r.error.splitlines()[0]}")
                else:
                    passed = sum(1 for o in r.outcomes if o.passed)
                    click.secho(f"{passed}/{len(r.outcomes)} probes passed", fg="green")
                all_results.append(r)
        finally:
            adapter.teardown()

    write_run(run_path, all_results)
    click.echo(f"\nresults → {run_path}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
def report(run_dir: Path) -> None:
    """Render a markdown report for a previous run."""
    md = render_report(run_dir)
    out = run_dir / "report.md"
    out.write_text(md)
    click.echo(md)
    click.echo(f"\nreport → {out}")


if __name__ == "__main__":
    cli()
