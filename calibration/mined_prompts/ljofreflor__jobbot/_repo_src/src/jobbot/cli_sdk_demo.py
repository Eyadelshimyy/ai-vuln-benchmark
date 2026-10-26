"""CLI SDK Integration Demo.

This module demonstrates how CLI commands can be refactored to use the SDK
as a thin wrapper, separating presentation logic from business logic.

Compare this implementation with the traditional CLI commands to see
the architectural improvements.
"""

from __future__ import annotations

from collections.abc import Sequence

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from jobbot.exit_codes import GENERIC_FAILURE, SUCCESS
from jobbot.matching.models import JobMatch  # type: ignore[import-untyped]
from jobbot.models.job import JobPosting
from jobbot.sdk import JobBotClient

console = Console()
app = typer.Typer(
    name="sdk-demo",
    help="SDK-based CLI commands (demonstration)",
    no_args_is_help=True,
)


@app.command("search")
def search_command(
    query: str = typer.Argument(..., help="Search query"),
    source: str = typer.Option("torre", help="Job source (torre, etc.)"),
    remote: bool = typer.Option(False, "--remote", help="Remote jobs only"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
) -> int:
    """Search for jobs using SDK.

    This command demonstrates the SDK wrapper pattern:
    - Business logic delegated to SDK
    - CLI handles only presentation
    - Clean separation of concerns
    """
    try:
        console.print(f"[cyan]Searching for '{query}' (source={source}, remote={remote})...[/cyan]")

        with JobBotClient() as client:
            # Business logic in SDK
            jobs = client.jobs.search(query, source=source, remote=remote, limit=limit)

            if not jobs:
                console.print("[yellow]No jobs found.[/yellow]")
                return SUCCESS

            matches = client.jobs.match(jobs)
            # match() with list returns list
            assert isinstance(matches, list)

            # Presentation logic in CLI
            _display_job_matches(jobs, matches)

            console.print(f"\n[green]Found {len(jobs)} jobs[/green]")
            return SUCCESS

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return GENERIC_FAILURE
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return GENERIC_FAILURE


@app.command("match")
def match_command(
    job_id: str = typer.Argument(..., help="Job ID to match"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed analysis"),
) -> int:
    """Match a job against profile using SDK.

    Demonstrates SDK match API usage in CLI context.
    """
    try:
        with JobBotClient() as client:
            # Business logic
            match = client.jobs.match_by_id(job_id)

            if match is None:
                console.print(f"[red]Job {job_id} not found[/red]")
                return GENERIC_FAILURE

            job = client.jobs.get(job_id)
            if job is None:
                console.print(f"[red]Job {job_id} not found[/red]")
                return GENERIC_FAILURE
            
            # Type narrowing: job is guaranteed not None after the check above
            assert job is not None

            # Presentation
            if detailed:
                _display_detailed_match(job, match)
            else:
                _display_simple_match(job, match)

            return SUCCESS

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return GENERIC_FAILURE


@app.command("cv")
def cv_command(
    job_id: str | None = typer.Argument(None, help="Job ID to tailor CV for"),
    style: str = typer.Option("modern", "--style", "-s", help="CV style"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output path"),
) -> int:
    """Build CV using SDK.

    Demonstrates SDK CV building in CLI context.
    """
    try:
        with JobBotClient() as client:
            # Business logic
            if job_id:
                cv_path = client.cv.build_for_job_id(
                    job_id,
                    style=style,
                    output_path=output,
                )
                if cv_path is None:
                    console.print(f"[red]Job {job_id} not found[/red]")
                    return GENERIC_FAILURE
            else:
                cv_path = client.cv.build(style=style, output_path=output)

            # Presentation
            console.print(Panel(
                f"[green]✓[/green] CV built successfully\n\n"
                f"Path: [cyan]{cv_path}[/cyan]\n"
                f"Style: {style}\n"
                f"Job: {job_id or 'Generic'}",
                title="CV Builder",
            ))

            return SUCCESS

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return GENERIC_FAILURE
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return GENERIC_FAILURE


# Presentation Helpers
# These could be extracted to a shared cli_helpers module


def _display_job_matches(jobs: Sequence[JobPosting], matches: Sequence[JobMatch]) -> None:
    """Display jobs with match scores in a table."""
    table = Table(title="Job Search Results")
    table.add_column("ID", style="cyan")
    table.add_column("Company", style="bold")
    table.add_column("Title")
    table.add_column("Location")
    table.add_column("Match", justify="right", style="green")

    for job, match in zip(jobs, matches, strict=True):
        table.add_row(
            job.id,
            job.company,
            job.title,
            job.location or "-",
            f"{match.score}%",
        )

    console.print(table)


def _display_simple_match(job: JobPosting, match: JobMatch) -> None:
    """Display simple match summary."""
    color = "green" if match.score >= 70 else "yellow" if match.score >= 50 else "red"

    console.print(Panel(
        f"[bold]{job.company} - {job.title}[/bold]\n\n"
        f"Match Score: [{color}]{match.score}%[/{color}]\n"
        f"Location: {job.location or 'Not specified'}\n"
        f"Remote: {job.remote_type or 'Not specified'}",
        title="Match Summary",
    ))


def _display_detailed_match(job: JobPosting, match: JobMatch) -> None:
    """Display detailed match analysis."""
    color = "green" if match.score >= 70 else "yellow" if match.score >= 50 else "red"

    highlights = "\n".join(f"  ✓ {h}" for h in (match.highlights or []))
    gaps = "\n".join(f"  ✗ {g}" for g in (match.gaps or []))

    console.print(Panel(
        f"[bold]{job.company} - {job.title}[/bold]\n\n"
        f"Match Score: [{color}]{match.score}%[/{color}]\n\n"
        f"[green]Strengths:[/green]\n{highlights or '  None identified'}\n\n"
        f"[yellow]Gaps:[/yellow]\n{gaps or '  None identified'}\n\n"
        f"Location: {job.location or 'Not specified'}\n"
        f"Remote: {job.remote_type or 'Not specified'}\n"
        f"URL: {job.url or 'Not specified'}",
        title="Detailed Match Analysis",
    ))


# This app would be registered in main CLI:
# In cli.py:
#   from jobbot.cli_sdk_demo import app as sdk_demo_app
#   app.add_typer(sdk_demo_app, name="sdk-demo")
