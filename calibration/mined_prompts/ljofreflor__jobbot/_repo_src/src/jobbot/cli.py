"""CLI entrypoint for JobBot."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy.orm import Session

from jobbot import __version__
from jobbot.applications.manager import (
    ApplicationRepository,
    prepare_application_package,
)
from jobbot.browser.background import background_notice
from jobbot.companies.models import DiscoverySource
from jobbot.companies.registry import CompanyRegistry
from jobbot.config import JobbotConfig, load_config
from jobbot.cv.build import BuildTarget, build_cv
from jobbot.cv.renderer import CvStyle
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.exit_codes import (
    AUTH_REQUIRED,
    GENERIC_FAILURE,
    MANUAL_CHALLENGE,
    SUCCESS,
    UPDATE_AVAILABLE,
    USER_CANCEL,
    VALIDATION_FAILURE,
)
from jobbot.jobs.freshness import age_label, is_fresh
from jobbot.jobs.geo import detect_country, resolve_countries
from jobbot.jobs.repository import JobRepository, write_job_json
from jobbot.matching.analyzer import RuleBasedJobAnalyzer
from jobbot.matching.scoring import format_match_report
from jobbot.models.application import ApplicationStatus
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.ops.narrate import ExplorationOutcome, Narrator, Phase
from jobbot.portals.detect import AtsKind
from jobbot.profile.diff import compare_summaries, summarize_profile
from jobbot.profile.importer_common import write_generated_profile
from jobbot.profile.importer_latex import LatexImportError, import_latex_cv
from jobbot.profile.importer_pdf import PdfImportError, import_pdf_cv
from jobbot.profile.loader import ProfileLoadError, load_profile, load_profile_raw
from jobbot.profile.validator import unknown_profile_keys, validate_candidate
from jobbot.self_update import update_jobbot
from jobbot.self_update_schedule import ScheduleAction
from jobbot.workspace import (
    DEFAULT_LABEL,
    STAMP_NAME,
    WorkspaceExistsError,
    WorkspaceOwnerError,
    active_workspace,
    init_workspace,
    list_workspaces,
    read_stamp,
    resolve_root,
    sandboxes_dir,
    set_active_workspace,
    workspace_root,
)
from jobbot.workspace import adopt as adopt_workspace

# Profile searches a board command runs when no query is typed.
_DEFAULT_MAX_QUERIES = 3


def _restamp_after_profile_promote(config: JobbotConfig) -> None:
    """Promoting a CV is a deliberate identity change: refresh the owner stamp."""
    label = active_workspace() or DEFAULT_LABEL
    adopt_workspace(config.profile_path, config.output_dir, label=label)


app = typer.Typer(
    name="jobbot",
    help="Local terminal tool for job search and structured CV management.",
    no_args_is_help=True,
)
profile_app = typer.Typer(help="Candidate profile operations", no_args_is_help=True)
cv_app = typer.Typer(help="Build CV variants", no_args_is_help=True)
jobs_app = typer.Typer(help="Search and inspect jobs", no_args_is_help=True)
application_app = typer.Typer(help="Prepare or open one application", no_args_is_help=True)
applications_app = typer.Typer(help="Track applications", no_args_is_help=True)
indeed_app = typer.Typer(help="Indeed profile operations", no_args_is_help=True)
linkedin_app = typer.Typer(
    help="LinkedIn: audit, publications sync, recruiter-post sweep",
    no_args_is_help=True,
)
browser_app = typer.Typer(
    help="Browser helpers (HITL Chrome/Edge/Brave via CDP — no CAPTCHA bypass)",
    no_args_is_help=True,
)
getonboard_app = typer.Typer(
    help="Get on Board: perfil/CVs HITL + job discovery (LATAM)",
    no_args_is_help=True,
)
torre_app = typer.Typer(help="Torre: job discovery (LATAM / remoto)", no_args_is_help=True)
empleospublicos_app = typer.Typer(
    help="Empleos Públicos (Chile): concursos — fixture-first (#118)",
    no_args_is_help=True,
)
portals_app = typer.Typer(help="Recruitment portal registry (ATS)", no_args_is_help=True)
companies_app = typer.Typer(
    help="Company ↔ career platform knowledge (candidate → promote; public data only)",
    no_args_is_help=True,
)
recruiters_app = typer.Typer(
    help="Public hiring practice that feeds cv advise (practices only, never people)",
    no_args_is_help=True,
)
ops_app = typer.Typer(
    help="Local ops: failures, symptoms (appearances), loops (no telemetry)",
    no_args_is_help=True,
)
ops_failure_app = typer.Typer(help="Inspect / triage stored failures", no_args_is_help=True)
ops_symptom_app = typer.Typer(
    help="Appearances that return (symptoms) → System 1 compression (local, redacted)",
    no_args_is_help=True,
)
workspace_app = typer.Typer(
    help="Isolated homes for extra candidates (test CVs) in this checkout",
    no_args_is_help=True,
)
secrets_app = typer.Typer(
    help="Opt-in local password vault (0600; never invents; never prints)",
    no_args_is_help=True,
)

app.add_typer(workspace_app, name="workspace")
app.add_typer(secrets_app, name="secrets")
app.add_typer(profile_app, name="profile")
app.add_typer(cv_app, name="cv")
app.add_typer(jobs_app, name="jobs")
app.add_typer(application_app, name="application")
app.add_typer(applications_app, name="applications")
app.add_typer(indeed_app, name="indeed")
app.add_typer(linkedin_app, name="linkedin")
app.add_typer(getonboard_app, name="getonboard")
app.add_typer(torre_app, name="torre")
app.add_typer(empleospublicos_app, name="empleospublicos")
app.add_typer(browser_app, name="browser")
app.add_typer(portals_app, name="portals")
app.add_typer(companies_app, name="companies")
app.add_typer(recruiters_app, name="recruiters")
app.add_typer(ops_app, name="ops")
ops_app.add_typer(ops_failure_app, name="failure")
ops_app.add_typer(ops_symptom_app, name="symptom")

console = Console()
err_console = Console(stderr=True)


class _QuietExit(Exception):
    """The command refused on purpose (closed vacancy, and the like). Not a defect."""

    def __init__(self, code: int) -> None:
        self.code = code


class _RecordedExit(Exception):
    """Non-zero exit that must land in ops_failures with a real fingerprint.

    Autopoietic lane for product gaps (novel URL shapes, missing fetchers):
    exception → observability → ``ops failure work`` → issue → PR. A bare
    ``typer.Exit`` drops the message and collapses every gap into one useless
    ``Exit:`` fingerprint.
    """

    def __init__(self, code: int, *, error_class: str, message: str) -> None:
        self.code = code
        self.error_class = error_class
        self.message = message


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    prog_name: str = "jobbot",
    standalone_mode: bool = True,
) -> int:
    """Invoke the Typer app and persist non-zero exits as local ops failures."""
    args = list(sys.argv[1:] if argv is None else argv)
    exit_code = SUCCESS
    caught: BaseException | None = None
    aborted = False
    recorded_class: str | None = None
    recorded_message: str | None = None
    try:
        result = app(args, prog_name=prog_name, standalone_mode=False)
        if isinstance(result, int):
            exit_code = result
    except typer.Exit as exc:
        exit_code = int(exc.exit_code)
        if exit_code == USER_CANCEL:
            # Ctrl-C. typer rewrites KeyboardInterrupt as Exit(130); that is the
            # user stopping, not a defect, and it must not become an ops failure.
            err_console.print("Cancelled.")
            if standalone_mode:
                raise SystemExit(exit_code) from exc
            return exit_code
        caught = exc
    except KeyboardInterrupt:
        err_console.print("Cancelled.")
        if standalone_mode:
            raise SystemExit(USER_CANCEL) from None
        return USER_CANCEL
    except typer.Abort:
        exit_code = GENERIC_FAILURE
        caught = None
        aborted = True
    except WorkspaceOwnerError as exc:
        # Pointing a profile at somebody else's data is a wrong invocation, not a bug:
        # report it and stop, without recording an ops failure or writing anything.
        err_console.print(f"[bold red]Wrong workspace[/bold red]\n{exc}")
        if standalone_mode:
            raise SystemExit(VALIDATION_FAILURE) from exc
        return VALIDATION_FAILURE
    except _QuietExit as exc:
        if standalone_mode:
            raise SystemExit(exc.code) from exc
        return exc.code
    except _RecordedExit as exc:
        exit_code = exc.code
        recorded_class = exc.error_class
        recorded_message = exc.message
        caught = exc.__cause__ if isinstance(exc.__cause__, BaseException) else None
    except Exception as exc:  # noqa: BLE001 — CLI boundary capture
        exit_code = GENERIC_FAILURE
        caught = exc
        err_console.print(f"[red]Unhandled error:[/red] {exc}")

    if exit_code != SUCCESS:
        from jobbot.ops.failures import ABORT_MESSAGE, capture_cli_failure, runtime_context

        record = capture_cli_failure(
            exit_code,
            argv=["jobbot", *args],
            exc=caught if not isinstance(caught, typer.Exit) else None,
            error_class=(
                recorded_class if recorded_class is not None else ("Abort" if aborted else None)
            ),
            message=(
                recorded_message
                if recorded_message is not None
                else (ABORT_MESSAGE if aborted else None)
            ),
            context=runtime_context(),
        )
        if record is not None:
            err_console.print(
                f"[yellow]Recorded failure[/yellow] {record.id} (fingerprint={record.fingerprint})"
            )

    if standalone_mode:
        raise SystemExit(exit_code)
    return exit_code


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _session() -> tuple[Session, JobbotConfig]:
    config = load_config()
    engine = make_engine(config.database_path)
    return make_session_factory(engine)(), config


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging"),
    ] = False,
    workspace: Annotated[
        str | None,
        typer.Option(
            "--workspace",
            "-w",
            help="Run against another candidate's workspace (see `jobbot workspace list`)",
        ),
    ] = None,
) -> None:
    """JobBot CLI."""
    _setup_logging(verbose)
    set_active_workspace(workspace)
    if workspace is not None:
        console.print(f"[yellow]workspace:[/yellow] {workspace} ({workspace_root(workspace)})")
        from jobbot.advisor import retention_warning

        notice = retention_warning(workspace_root(workspace))
        if notice:
            console.print(f"[yellow]{notice}[/yellow]")


@app.command("version")
def version_cmd() -> None:
    """Show JobBot version."""
    console.print(__version__)


@app.command("init")
def workspace_init(
    directory: Annotated[
        Path,
        typer.Argument(
            help=(
                "Folder that becomes the workspace (default: .). "
                ".jobbot.toml and .local/ are created inside this folder"
            ),
        ),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing .jobbot.toml / seeds"),
    ] = False,
) -> None:
    """Create a portable workspace with ``.local/`` inside DIR (cold install; no git clone)."""
    try:
        result = init_workspace(directory, force=force)
    except WorkspaceExistsError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc
    except OSError as exc:
        err_console.print(f"[red]Could not create workspace: {exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc

    console.print("[bold green]Workspace ready[/bold green]")
    console.print(f"Root: {result.root}")
    console.print("JobBot is installed globally; this folder is your syncable workspace.")
    console.print(
        Panel(
            "Next:\n"
            "  1. Edit .local/profile.yaml  (or: jobbot profile import-pdf CV.pdf)\n"
            "  2. jobbot profile validate\n"
            "  3. jobbot getonboard search\n"
            "Sync this whole folder (including .local/) with Drive/Dropbox/Syncthing/…\n"
            "If you later run jobbot secrets init, exclude .local/.vault.yaml from sync.",
            title="Cold start",
        )
    )


@app.command("update")
def update_cmd(
    ref: Annotated[
        str | None,
        typer.Option("--ref", help="Git ref to install (default: JOBBOT_REF or main)"),
    ] = None,
    schedule: Annotated[
        ScheduleAction | None,
        typer.Option(
            "--schedule",
            help=(
                "daily: run `jobbot update` every day (launchd on macOS, crontab on Linux); "
                "off: remove it; status: show it"
            ),
        ),
    ] = None,
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help=f"Only compare installed vs latest commit; exit {UPDATE_AVAILABLE} if newer",
        ),
    ] = False,
    scheduled: Annotated[
        bool,
        typer.Option("--scheduled", hidden=True, help="Run by the daily scheduled job"),
    ] = False,
) -> None:
    """Upgrade the ``jobbot`` executable on PATH (uv tool reinstall from GitHub)."""
    if sum((schedule is not None, check, scheduled)) > 1 or (
        ref is not None and (schedule is not None or scheduled)
    ):
        err_console.print(
            "[red]--schedule, --check and --scheduled are exclusive; "
            "the scheduled job follows main (or JOBBOT_REF), not --ref.[/red]"
        )
        raise _QuietExit(VALIDATION_FAILURE)
    if scheduled:
        _run_scheduled_update()
        return
    if check:
        _check_update(ref)
        return
    if schedule is not None:
        _schedule_update(schedule)
        return
    console.print(f"Current: {__version__}")
    result = update_jobbot(ref=ref)
    if not result.ok:
        err_console.print(f"[red]{result.message}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    console.print(f"[green]{result.message}[/green]")
    if result.version_line:
        console.print(f"Now: {result.version_line}")
    else:
        console.print("Run [bold]jobbot version[/bold] in a new shell if PATH changed.")


def _check_update(ref: str | None) -> None:
    from jobbot.self_update import _in_docker, check_for_update, docker_hint

    if _in_docker():
        err_console.print(docker_hint(), markup=False)
        raise _QuietExit(GENERIC_FAILURE)
    result = check_for_update(ref=ref)
    if result.status == "up_to_date":
        console.print(f"[green]{result.message}[/green]")
        return
    if result.status == "available":
        console.print(f"[yellow]{result.message}[/yellow]")
        console.print("Run [bold]jobbot update[/bold] to install it.")
        raise _QuietExit(UPDATE_AVAILABLE)
    err_console.print(result.message, style="red", markup=False)
    raise _QuietExit(GENERIC_FAILURE)


def _schedule_update(action: ScheduleAction) -> None:
    from jobbot.self_update import _in_docker, docker_hint, installed_info
    from jobbot.self_update_schedule import (
        Scheduler,
        auto_update_opt_out,
        resolve_binary,
        user_config_path,
    )

    if _in_docker():
        err_console.print(docker_hint(), markup=False)
        raise _QuietExit(GENERIC_FAILURE)
    scheduler = Scheduler()
    if action is ScheduleAction.status:
        status = scheduler.status()
        for line in status.lines():
            console.print(line, markup=False, highlight=False)
        if status.backend is None:
            raise _QuietExit(GENERIC_FAILURE)
        return
    if action is ScheduleAction.off:
        outcome = scheduler.remove()
    else:
        if installed_info().editable:
            err_console.print(
                "[red]Editable install (development checkout): not scheduling. "
                "Use git pull.[/red]"
            )
            raise _QuietExit(VALIDATION_FAILURE)
        binary = resolve_binary()
        if binary is None:
            err_console.print("[red]jobbot not found on PATH; run scripts/install.sh first.[/red]")
            raise _QuietExit(GENERIC_FAILURE)
        uv = shutil.which("uv")
        if uv is None:
            err_console.print(
                "[yellow]uv not found on PATH; the job will fail until it is.[/yellow]"
            )
        outcome = scheduler.install(
            scheduler.spec(binary, extra_path=[Path(uv).parent if uv else None])
        )
    if not outcome.ok:
        err_console.print(outcome.message, style="red", markup=False)
        raise typer.Exit(GENERIC_FAILURE)
    console.print(outcome.message, markup=False, highlight=False)
    reason = auto_update_opt_out(scheduler.env, user_config_path(scheduler.home))
    if action is ScheduleAction.daily and reason:
        console.print(f"[yellow]Opt-out active ({reason}): the job will skip.[/yellow]")


def _run_scheduled_update() -> None:
    import os

    from jobbot.self_update import _in_docker
    from jobbot.self_update_schedule import run_scheduled_update, user_config_path

    run = run_scheduled_update(
        env=os.environ,
        config_path=user_config_path(Path.home()),
        in_docker=_in_docker,
    )
    for line in run.lines:
        typer.echo(line)
    if run.exit_code != SUCCESS:
        raise _QuietExit(run.exit_code)


@app.command("status")
def status_cmd(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable rows")] = False,
) -> None:
    """Show which permanent CVs / profiles and active company portals are up."""
    _print_cv_status(as_json=as_json)


def _print_cv_status(*, as_json: bool) -> None:
    from jobbot.cv.status import build_cv_status
    from jobbot.profile.loader import load_profile

    config = load_config()
    report = build_cv_status(config, load_profile(config.profile_path))
    if as_json:
        console.print_json(data=report.to_dict())
        return
    table = Table(title="CV / profile presence (evidence only)")
    table.add_column("Destination")
    table.add_column("Artifact")
    table.add_column("State")
    table.add_column("Evidence")
    table.add_column("Hint")
    for row in report.rows:
        table.add_row(
            row.destination,
            row.artifact,
            row.state.value,
            row.evidence,
            row.hint or "",
        )
    console.print(table)
    console.print(
        "[dim]Evidence only — missing receipt means unknown, not absent on the portal. "
        "Active company portals need page evidence; a yes is never a receipt. "
        "Per-job apply stages: jobbot applications list[/dim]"
    )


def _ingest_with_progress(
    config: JobbotConfig,
    session: Session,
    portal: Any,
    *,
    candidate: Candidate,
    html: str | None,
    ingest_hard_link: Any,
    cdp_url: str | None = None,
) -> Any:
    """Download a Get on Board page with a progress bar. Fixtures and other ATS skip it."""
    from jobbot.portals.detect import AtsKind

    downloading = html is None and portal.ats_kind == AtsKind.GETONBOARD
    if not downloading:
        return ingest_hard_link(
            config,
            session,
            portal.url,
            candidate=candidate,
            html=html,
            cdp_url=cdp_url,
        )
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("descargando la oferta", total=None)

        def on_chunk(done: int, total: int | None) -> None:
            progress.update(task, completed=done, total=total)

        return ingest_hard_link(
            config,
            session,
            portal.url,
            candidate=candidate,
            html=html,
            cdp_url=cdp_url,
            on_chunk=on_chunk,
        )


@app.command("get")
def get_hard_link(
    url: Annotated[
        str | None,
        typer.Argument(
            help="Hard job URL. Live download: Get on Board / Indeed. "
            "With --fixture, also reads saved career-page or Indeed viewjob HTML. "
            "With --park, only queues the URL (phone-friendly; no fetch).",
        ),
    ] = None,
    apply_changes: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="After preparing the package, open the ATS assist (you submit)",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to your logged-in Chrome (see browser chrome-debug)"),
    ] = None,
    fixture: Annotated[
        Path | None,
        typer.Option(
            "--fixture",
            help=(
                "Saved HTML instead of a live download. "
                "Get on Board, Indeed viewjob, or generic career pages (e.g. Phenom)."
            ),
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    park: Annotated[
        bool,
        typer.Option(
            "--park",
            help=(
                "Queue the URL under data/hard-link-inbox.txt without fetching "
                "(use from a phone / when CAPTCHA would block). Drain later with --parked."
            ),
        ),
    ] = False,
    parked: Annotated[
        bool,
        typer.Option(
            "--parked",
            help=(
                "Ingest every URL parked in data/hard-link-inbox.txt "
                "(desktop + Chrome CDP for Indeed challenges)."
            ),
        ),
    ] = False,
) -> None:
    """Ingest a hard job link: know the portal → JD → CV → package (HITL apply)."""
    if park and parked:
        err_console.print("[red]Use either --park or --parked, not both.[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    if parked:
        if url is not None:
            err_console.print("[red]--parked does not take a URL argument.[/red]")
            raise typer.Exit(VALIDATION_FAILURE)
        if fixture is not None:
            err_console.print("[red]--parked cannot combine with --fixture.[/red]")
            raise typer.Exit(VALIDATION_FAILURE)
        _drain_parked_hard_links(apply_changes=apply_changes, yes=yes, cdp=cdp)
        return
    if url is None:
        err_console.print(
            "[red]Provide a hard job URL[/red], or pass [bold]--parked[/bold] "
            "to drain data/hard-link-inbox.txt."
        )
        raise typer.Exit(VALIDATION_FAILURE)
    if park:
        _park_hard_link(url)
        return
    _ingest_one_hard_link(
        url,
        apply_changes=apply_changes,
        yes=yes,
        cdp=cdp,
        fixture=fixture,
    )


def _park_hard_link(url: str) -> None:
    """Save a share link for later (no Indeed fetch, no CAPTCHA)."""
    from jobbot.jobs.inbox import inbox_path, park_url

    _, config = _session()
    try:
        stored, existed = park_url(config, url)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc
    path = inbox_path(config)
    if existed:
        console.print(f"[dim]Already parked[/dim] {stored}")
    else:
        console.print(f"[green]Parked[/green] {stored}")
    console.print(f"Inbox: {path}")
    console.print(
        "On a desktop with Chrome: "
        "[bold]jobbot browser chrome-debug --site indeed[/bold], then "
        "[bold]jobbot get --parked --cdp http://127.0.0.1:9222[/bold]"
    )


def _drain_parked_hard_links(
    *,
    apply_changes: bool,
    yes: bool,
    cdp: str | None,
) -> None:
    from jobbot.jobs.inbox import inbox_path, list_parked, remove_parked

    _, config = _session()
    urls = list_parked(config)
    if not urls:
        console.print(f"Inbox empty ({inbox_path(config)}).")
        return
    console.print(f"Draining {len(urls)} parked URL(s)…")
    failures = 0
    for item in urls:
        console.print(f"\n[bold]→[/bold] {item}")
        try:
            _ingest_one_hard_link(
                item,
                apply_changes=apply_changes,
                yes=yes,
                cdp=cdp,
                fixture=None,
            )
        except _RecordedExit as exc:
            failures += 1
            from jobbot.ops.failures import capture_cli_failure, runtime_context

            record = capture_cli_failure(
                exc.code,
                argv=["jobbot", "get", item],
                exc=exc.__cause__ if isinstance(exc.__cause__, BaseException) else None,
                error_class=exc.error_class,
                message=exc.message,
                context=runtime_context(),
            )
            if record is not None:
                err_console.print(
                    f"[yellow]Recorded failure[/yellow] {record.id} "
                    f"(fingerprint={record.fingerprint})"
                )
            err_console.print(f"[yellow]Left in inbox[/yellow] (exit {exc.code}): {item}")
            continue
        except (_QuietExit, typer.Exit) as exc:
            failures += 1
            if isinstance(exc, _QuietExit):
                code = exc.code
            else:
                code = exc.exit_code if isinstance(exc.exit_code, int) else 1
            err_console.print(f"[yellow]Left in inbox[/yellow] (exit {code}): {item}")
            continue
        remove_parked(config, item)
        console.print(f"[dim]Removed from inbox:[/dim] {item}")
    if failures:
        # Per-URL gaps already landed in ops_failures; the drain summary is not a
        # new defect (avoid a second empty Exit fingerprint).
        raise _QuietExit(GENERIC_FAILURE if failures == len(urls) else SUCCESS)


def _ingest_one_hard_link(
    url: str,
    *,
    apply_changes: bool,
    yes: bool,
    cdp: str | None,
    fixture: Path | None,
) -> None:
    from jobbot.browser.cdp import resolve_cdp_url
    from jobbot.jobs.from_url import (
        ClosedPostingError,
        UnknownPortalError,
        UnsupportedPortalFetchError,
        ingest_hard_link,
    )
    from jobbot.portals.knowledge import lookup_portal

    session, config = _session()
    portal = lookup_portal(config, url)
    console.print(
        f"portal: [bold]{portal.domain or '(none)'}[/bold]  "
        f"ats={portal.ats_kind.value}  knowledge={portal.source.value}"
    )
    narrator = _narrator()
    narrator.phase(Phase.RECEIVING_WORLD, portal.domain or url)
    html = fixture.read_text(encoding="utf-8") if fixture is not None else None
    if not portal.known and html is None:
        err_console.print(
            f"[red]Unknown employment domain:[/red] {portal.domain or url}\n"
            "Pass [bold]--fixture[/bold] with saved career HTML, or add the host "
            "with [bold]jobbot portals add[/bold] / "
            "[bold]jobbot companies learn[/bold]."
        )
        raise typer.Exit(VALIDATION_FAILURE)

    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    cdp_url = resolve_cdp_url(cdp)
    try:
        result = _ingest_with_progress(
            config,
            session,
            portal,
            candidate=candidate,
            html=html,
            ingest_hard_link=ingest_hard_link,
            cdp_url=cdp_url,
        )
    except ClosedPostingError as exc:
        err_console.print(f"[red]Vacancy looks filled[/red] ({exc.evidence}). Not stored.")
        raise _QuietExit(VALIDATION_FAILURE) from exc
    except UnknownPortalError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc
    except UnsupportedPortalFetchError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise _RecordedExit(
            GENERIC_FAILURE,
            error_class=type(exc).__name__,
            message=str(exc),
        ) from exc
    except OSError as exc:
        err_console.print(f"[red]Could not fetch job page:[/red] {exc}")
        raise _RecordedExit(
            GENERIC_FAILURE,
            error_class=type(exc).__name__,
            message=f"Could not fetch job page: {exc}",
        ) from exc

    job = result.job
    _learn_company_knowledge(config, job, narrator=narrator)
    console.print(
        f"[green]Stored[/green] {job.id}  {job.company}  {job.title}  "
        f"(match {job.match_score if job.match_score is not None else '—'})"
    )
    console.print(f"Wrote {result.job_json}")
    if result.match is not None:
        console.print(Panel(format_match_report(result.match), title="match"))
    if result.package_dir is not None:
        console.print(f"[green]Prepared[/green] {result.package_dir}")
        gob = result.package_dir / "getonboard_es.md"
        if gob.is_file():
            console.print(f"Get on Board Spanish drafts: {gob}")

    if not apply_changes:
        console.print(
            f"Package ready. Re-run with "
            f"[bold]jobbot application apply {job.id} --apply[/bold] "
            "to open the portal (you submit), or pass [bold]--apply[/bold] here."
        )
        return

    application_apply(job.id, apply_changes=True, yes=yes, cdp=cdp)


@app.command("capture")
def capture_share_url(
    url: Annotated[
        str | None,
        typer.Argument(
            help="Share URL from phone (Indeed vacancy, company career portal, or unknown).",
        ),
    ] = None,
    list_all: Annotated[
        bool,
        typer.Option(
            "--list",
            help="Show unfinished candidates (parked hard links, company portals, unrecognized)",
        ),
    ] = False,
    company: Annotated[
        str | None,
        typer.Option("--company", help="Company name when capturing a career portal"),
    ] = None,
    country: Annotated[
        str | None,
        typer.Option("--country", help="ISO country, e.g. CL"),
    ] = None,
) -> None:
    """Keep a share URL as a candidate — no fetch, no CAPTCHA (phone-friendly)."""
    from jobbot.jobs.capture import (
        CaptureKind,
        capture_paths,
        capture_url,
        list_candidates,
    )

    _, config = _session()
    if list_all:
        inv = list_candidates(config)
        paths = capture_paths(config)
        console.print("[bold]Candidate URLs (not finished yet)[/bold]")
        console.print(f"Hard links ({paths['hard_links']}):")
        if inv.hard_links:
            for item in inv.hard_links:
                console.print(f"  · {item}")
        else:
            console.print("  (none)")
        console.print(f"Company portals ({paths['companies']}):")
        if inv.company_portals:
            for company_id, site_url, status in inv.company_portals:
                console.print(f"  · {company_id}  {site_url}  [{status}]")
        else:
            console.print("  (none)")
        console.print(f"Unrecognized ({paths['unrecognized']}):")
        if inv.unrecognized:
            for item in inv.unrecognized:
                console.print(f"  · {item}")
        else:
            console.print("  (none)")
        return

    if url is None:
        err_console.print("[red]Provide a URL[/red], or [bold]jobbot capture --list[/bold].")
        raise typer.Exit(VALIDATION_FAILURE)

    try:
        result = capture_url(config, url, company=company, country=country)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    label = {
        CaptureKind.HARD_LINK: "Hard link candidate",
        CaptureKind.COMPANY_PORTAL: "Company portal candidate",
        CaptureKind.UNRECOGNIZED: "Unrecognized candidate",
    }[result.kind]
    verb = "Already had" if result.already_present else "Captured"
    console.print(f"[green]{verb}[/green] {label}: {result.url}")
    if result.company_id:
        console.print(f"company_id={result.company_id}")
    console.print(result.detail)
    console.print(
        "Later on desktop: "
        "[bold]jobbot get --parked[/bold] (hard links) · "
        "[bold]jobbot companies recon …[/bold] (portals) · "
        "[bold]jobbot capture --list[/bold]"
    )


# ── workspace ────────────────────────────────────────────────────────────────


@workspace_app.command("list")
def workspace_list() -> None:
    """List the extra candidates living in this checkout."""
    names = list_workspaces()
    if not names:
        console.print(f"No workspaces yet in {sandboxes_dir()} — create one with `workspace new`.")
        return
    table = Table(title="Workspaces")
    table.add_column("name")
    table.add_column("profile")
    table.add_column("owner")
    for name in names:
        root = workspace_root(name)
        profile = root / "data" / "profile.yaml"
        stamp = read_stamp(root / "data" / STAMP_NAME)
        table.add_row(
            name,
            "yes" if profile.is_file() else "missing",
            stamp.fingerprint if stamp else "unstamped",
        )
    console.print(table)


@workspace_app.command("new")
def workspace_new(
    name: Annotated[str, typer.Argument(help="Workspace name, e.g. a test CV's nickname")],
) -> None:
    """Create an isolated `data/` + `output/` for another candidate."""
    root = workspace_root(name)
    if root.exists():
        err_console.print(f"Workspace already exists: {root}")
        raise typer.Exit(GENERIC_FAILURE)
    (root / "data").mkdir(parents=True)
    (root / "output").mkdir(parents=True)
    console.print(f"Created {root}")
    console.print(f"Now put the candidate's profile in {root / 'data' / 'profile.yaml'}")
    console.print(f"and run commands with [bold]--workspace {name}[/bold].")


@workspace_app.command("show")
def workspace_show() -> None:
    """Show which candidate the current run would touch."""
    name = active_workspace()
    config = load_config()
    table = Table(title="Active workspace")
    table.add_column("field")
    table.add_column("value")
    table.add_row("workspace", name or "default (current directory)")
    table.add_row("profile", str(config.profile_path))
    table.add_row("output", str(config.output_dir))
    table.add_row("database", str(config.database_path))
    stamp = read_stamp(config.profile_path.parent / STAMP_NAME)
    table.add_row("owner", stamp.fingerprint if stamp else "unstamped")
    console.print(table)


@workspace_app.command("adopt")
def workspace_adopt(
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation")] = False,
) -> None:
    """Hand this workspace's data and output over to the profile now in place."""
    name = active_workspace()
    base, _ = resolve_root(Path.cwd())
    profile_path = base / "data" / "profile.yaml"
    output_dir = base / "output"
    console.print(f"Re-stamp {profile_path.parent} and {output_dir} for {profile_path}")
    if not yes and not typer.confirm("Artifacts of the previous owner stay in place. Continue?"):
        raise typer.Abort
    fingerprint = adopt_workspace(profile_path, output_dir, label=name or DEFAULT_LABEL)
    if fingerprint is None:
        err_console.print(f"No profile with a name at {profile_path}")
        raise typer.Exit(GENERIC_FAILURE)
    console.print(f"Owner is now {fingerprint}")


@workspace_app.command("delete")
def workspace_delete(
    name: Annotated[str, typer.Argument(help="Sandbox to remove")],
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
) -> None:
    """Delete one sandbox. Never touches the checkout's own data/."""
    from jobbot.advisor import WorkspaceDeleteRefused, delete_workspace

    root = workspace_root(name)
    console.print(f"Delete {root}")
    if not yes and not typer.confirm("This removes that candidate's sandbox. Continue?"):
        raise typer.Abort
    try:
        delete_workspace(name, confirmed=True)
    except WorkspaceDeleteRefused as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc
    console.print(f"Deleted {name}")


@workspace_app.command("consent")
def workspace_consent(
    name: Annotated[str, typer.Argument()],
    consented_at: Annotated[str, typer.Option("--at", help="ISO date the client agreed")],
    scope: Annotated[str, typer.Option("--scope", help="What the client agreed to")],
    delete_after: Annotated[
        str, typer.Option("--delete-after", help="ISO date to erase the sandbox")
    ],
) -> None:
    """Record local consent and a retention date for one sandbox."""
    from jobbot.advisor import Consent, write_consent

    root = workspace_root(name)
    if not root.is_dir():
        err_console.print(f"[red]No workspace {name}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    path = write_consent(
        root,
        Consent(consented_at=consented_at, scope=scope, delete_after=delete_after),
    )
    console.print(f"Wrote {path}")


def _vault_file() -> Path:
    from jobbot.vault import vault_path

    return vault_path(load_config().profile_path)


def _die_vault(exc: BaseException) -> None:
    err_console.print(f"[red]{exc}[/red]")
    raise typer.Exit(VALIDATION_FAILURE) from exc


@secrets_app.command("init")
def secrets_init() -> None:
    """Create an empty 0600 vault next to profile.yaml. Does not invent passwords."""
    from jobbot.vault import VaultError, init_vault

    path = _vault_file()
    try:
        init_vault(path)
    except VaultError as exc:
        _die_vault(exc)
    console.print(f"Created {path} (mode 0600). JobBot never invents entries.")
    console.print("[dim]Do not sync this file to Drive/Dropbox. Hidden is not encryption.[/dim]")


@secrets_app.command("list")
def secrets_list() -> None:
    """List stored site keys. Never prints a password."""
    from jobbot.vault import VaultError, load_vault

    path = _vault_file()
    try:
        vault = load_vault(path, missing_ok=True)
    except VaultError as exc:
        _die_vault(exc)
    if vault is None:
        console.print(f"No vault at {path} — run [bold]jobbot secrets init[/bold].")
        return
    sites = vault.sites()
    fill = "on" if vault.fill_login else "off"
    console.print(f"Vault {path}  fill_login={fill}  entries={len(sites)}")
    if not sites:
        console.print("[dim]Empty. jobbot secrets set SITE stores a password you type.[/dim]")
        return
    for site in sites:
        entry = vault.get(site)
        user = entry.username if entry and entry.username else "(profile email)"
        console.print(f"  {site}  user={user}  password=***")


@secrets_app.command("set")
def secrets_set(
    site: Annotated[str, typer.Argument(help="Portal key, e.g. indeed / getonboard / acme")],
    username: Annotated[
        str | None,
        typer.Option("--username", help="Login id if not the profile email"),
    ] = None,
    password_stdin: Annotated[
        bool,
        typer.Option(
            "--password-stdin",
            help="Read the password from stdin (never from argv / ps)",
        ),
    ] = False,
) -> None:
    """Store a password the human types. Empty values are refused, never invented."""
    import getpass

    from jobbot.vault import VaultError, put_entry

    if password_stdin:
        password = sys.stdin.read().rstrip("\r\n")
    else:
        password = getpass.getpass(f"Password for {site}: ")
        again = getpass.getpass("Repeat password: ")
        if password != again:
            _die_vault(VaultError("passwords do not match"))
    path = _vault_file()
    try:
        put_entry(path, site, password, username=username)
    except VaultError as exc:
        _die_vault(exc)
    console.print(f"Stored {site} in the vault (password not shown).")


@secrets_app.command("delete")
def secrets_delete(
    site: Annotated[str, typer.Argument(help="Portal key to remove")],
) -> None:
    """Drop one stored login. Does not delete the vault file."""
    from jobbot.vault import VaultError, delete_entry

    try:
        delete_entry(_vault_file(), site)
    except VaultError as exc:
        _die_vault(exc)
    console.print(f"Removed {site}.")


@secrets_app.command("allow-fill")
def secrets_allow_fill(
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt")] = False,
) -> None:
    """Consent flag only: a later fill driver may type the password. Submit stays HITL."""
    from jobbot.vault import VaultError, set_fill_login

    if not yes and not typer.confirm(
        "Allow JobBot to type stored passwords into login fields later? "
        "CAPTCHA, 2FA, terms and submit stay yours.",
        default=False,
    ):
        console.print("fill_login left off.")
        return
    try:
        set_fill_login(_vault_file(), True)
    except VaultError as exc:
        _die_vault(exc)
    console.print(
        "fill_login=on. Typing into the browser is not shipped yet; "
        "CAPTCHA/2FA/submit stay human."
    )


@secrets_app.command("deny-fill")
def secrets_deny_fill() -> None:
    """Turn off the fill_login flag. Stored entries stay."""
    from jobbot.vault import VaultError, set_fill_login

    try:
        set_fill_login(_vault_file(), False)
    except VaultError as exc:
        _die_vault(exc)
    console.print("fill_login=off.")


advisor_app = typer.Typer(
    help="Cross-workspace status and a per-client report (no other candidate's data)",
    no_args_is_help=True,
)
app.add_typer(advisor_app, name="advisor")


@advisor_app.command("status")
def advisor_status() -> None:
    """One line per sandbox: fingerprint, counts, proposal, retention. No profile PII."""
    from datetime import UTC, datetime

    from jobbot.advisor import status_lines

    names = list_workspaces()
    lines = status_lines([(name, workspace_root(name)) for name in names], today=datetime.now(UTC))
    if not lines:
        console.print(f"No workspaces yet in {sandboxes_dir()}.")
        return
    for line in lines:
        console.print(line)


@advisor_app.command("report")
def advisor_report(
    since: Annotated[str, typer.Option("--since", help="Window, e.g. 7d")] = "7d",
) -> None:
    """Write a redacted report into the selected workspace's output/."""
    from datetime import UTC, datetime

    from jobbot.advisor import parse_since, write_report

    name = active_workspace()
    if not name:
        err_console.print("[red]Pass --workspace NAME[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    now = datetime.now(UTC)
    path = write_report(workspace_root(name), since=parse_since(since, now=now), now=now)
    console.print(f"Wrote {path}")


# ── profile ──────────────────────────────────────────────────────────────────


@profile_app.command("validate")
def profile_validate(
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Treat unknown YAML keys as errors (default: warn and exit 0)",
        ),
    ] = False,
) -> None:
    """Validate data/profile.yaml."""
    config = load_config()
    try:
        candidate = load_profile(config.profile_path)
        raw = load_profile_raw(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print("[bold red]PROFILE INVALID[/bold red]")
        err_console.print(str(exc))
        raise typer.Exit(VALIDATION_FAILURE) from exc

    result = validate_candidate(candidate)
    unknowns = unknown_profile_keys(raw)
    if not result.ok or (strict and unknowns):
        err_console.print("[bold red]PROFILE INVALID[/bold red]")
        for issue in result.issues:
            err_console.print(str(issue))
        if strict:
            for issue in unknowns:
                err_console.print(f"{issue.path}: {issue.message}")
        raise typer.Exit(VALIDATION_FAILURE)

    console.print("[bold green]PROFILE VALID[/bold green]")
    for issue in unknowns:
        err_console.print(f"[yellow]{issue.path}: {issue.message}[/yellow]")
    console.print(f"Experiences: {len(candidate.experience)}")
    console.print(f"Achievements: {candidate.achievement_count()}")
    console.print(f"Skills: {candidate.skills.count()}")
    console.print(f"Publications: {len(candidate.publications)}")
    raise typer.Exit(SUCCESS)


@profile_app.command("show")
def profile_show() -> None:
    """Show a summary of the local profile."""
    config = load_config()
    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    personal = candidate.personal
    table = Table(title="Candidate profile", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Name", personal.name)
    table.add_row("Headline", personal.headline)
    table.add_row("Location", personal.location_line() or "—")
    table.add_row("Email", str(personal.email) if personal.email else "—")
    table.add_row("Experiences", str(len(candidate.experience)))
    table.add_row("Skills", str(candidate.skills.count()))
    console.print(table)
    if candidate.summary:
        console.print(Panel(candidate.summary.strip(), title="Summary"))


@profile_app.command("import-latex")
def profile_import_latex(
    tex_path: Annotated[
        Path | None,
        typer.Argument(help="Path to legacy moderncv .tex"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination YAML"),
    ] = None,
) -> None:
    """Import a legacy LaTeX CV into profile.generated.yaml."""
    config = load_config()
    source = tex_path or config.legacy_cv_path
    if source is None:
        err_console.print("[red]Provide a .tex path or set paths.legacy_cv in .jobbot.toml[/red]")
        raise typer.Exit(GENERIC_FAILURE)

    source = source.expanduser().resolve()
    destination = (output or config.generated_profile_path).expanduser().resolve()

    try:
        result = import_latex_cv(source)
        write_generated_profile(result, destination)
    except LatexImportError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc

    console.print("[bold green]LEGACY CV IMPORTED[/bold green]")
    console.print(f"Source:  {source}")
    console.print(f"Wrote:   {destination}")
    console.print(f"Experiences: {result.experience_count}")
    console.print(f"Achievements: {result.achievement_count}")
    console.print(f"Skills: {result.skill_count}")
    console.print(f"Publications: {len(result.data.get('publications', []))}")

    if result.warnings:
        console.print("\n[yellow]Warnings[/yellow]")
        for warning in result.warnings:
            console.print(f"- {warning}")

    if config.profile_path.is_file():
        try:
            local_raw = load_profile_raw(config.profile_path)
            lines = compare_summaries(
                summarize_profile(local_raw),
                summarize_profile(result.data),
            )
            console.print("\n[bold]Comparison vs data/profile.yaml[/bold]")
            for line in lines:
                console.print(line)
        except ProfileLoadError:
            pass

    console.print(
        Panel(
            "Review the generated YAML, then:\n"
            "  jobbot profile promote-generated\n"
            "  jobbot profile validate",
            title="Next steps",
        )
    )


@profile_app.command("import-pdf")
def profile_import_pdf(
    pdf_path: Annotated[
        Path,
        typer.Argument(help="Path to a CV exported as PDF (needs a text layer)"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination YAML"),
    ] = None,
    promote: Annotated[
        bool,
        typer.Option(
            "--promote",
            help="Also copy generated YAML over profile.yaml (after validate)",
        ),
    ] = False,
) -> None:
    """Import a PDF CV into profile.generated.yaml (and optionally profile.yaml)."""
    config = load_config()
    source = pdf_path.expanduser().resolve()
    destination = (output or config.generated_profile_path).expanduser().resolve()

    try:
        result = import_pdf_cv(source)
        write_generated_profile(result, destination, command="profile import-pdf")
    except PdfImportError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc

    console.print("[bold green]PDF CV IMPORTED[/bold green]")
    console.print(f"Source:  {source}")
    console.print(f"Wrote:   {destination}")
    console.print(f"Experiences: {result.experience_count}")
    console.print(f"Achievements: {result.achievement_count}")
    console.print(f"Skills: {result.skill_count}")
    console.print(f"Education: {len(result.data.get('education', []))}")

    if result.warnings:
        console.print("\n[yellow]Warnings[/yellow]")
        for warning in result.warnings:
            console.print(f"- {warning}")

    if promote:
        try:
            candidate = load_profile(destination)
        except ProfileLoadError as exc:
            err_console.print(f"[red]Generated profile invalid: {exc}[/red]")
            raise typer.Exit(VALIDATION_FAILURE) from exc
        check = validate_candidate(candidate)
        if not check.ok:
            for issue in check.issues:
                err_console.print(f"[red]- {issue}[/red]")
            raise typer.Exit(VALIDATION_FAILURE)
        shutil.copy2(destination, config.profile_path)
        _restamp_after_profile_promote(config)
        console.print(f"[green]Promoted[/green] → {config.profile_path}")
    else:
        console.print(
            Panel(
                "A PDF carries no structure, so every field is a guess.\n"
                "Review the generated YAML, then:\n"
                "  jobbot profile promote-generated\n"
                "  jobbot profile validate\n"
                "Or re-run with --promote after a quick review.",
                title="Next steps",
            )
        )


@profile_app.command("promote-generated")
def profile_promote_generated(
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Copy profile.generated.yaml → profile.yaml after confirmation."""
    config = load_config()
    generated = config.generated_profile_path
    target = config.profile_path
    if not generated.is_file():
        err_console.print(f"[red]Generated profile not found: {generated}[/red]")
        raise typer.Exit(GENERIC_FAILURE)

    try:
        candidate = load_profile(generated)
    except ProfileLoadError as exc:
        err_console.print("[bold red]GENERATED PROFILE INVALID[/bold red]")
        err_console.print(str(exc))
        raise typer.Exit(VALIDATION_FAILURE) from exc

    result = validate_candidate(candidate)
    if not result.ok:
        err_console.print("[bold red]GENERATED PROFILE INVALID[/bold red]")
        for issue in result.issues:
            err_console.print(str(issue))
        raise typer.Exit(VALIDATION_FAILURE)

    if target.is_file() and not yes:
        console.print(f"This will overwrite [bold]{target}[/bold] with {generated}")
        if not typer.confirm("Promote generated profile?", default=False):
            console.print("Aborted.")
            raise typer.Exit(SUCCESS)

    if target.is_file():
        backup = target.with_suffix(target.suffix + ".bak")
        shutil.copy2(target, backup)
        console.print(f"Backup: {backup}")

    shutil.copy2(generated, target)
    _restamp_after_profile_promote(config)
    console.print(f"[green]Promoted[/green] {generated} → {target}")


@profile_app.command("suggest-from-market")
def profile_suggest_from_market(
    min_count: Annotated[
        int,
        typer.Option("--min-count", help="Min JD occurrences for a market term"),
    ] = 2,
    job_id: Annotated[
        str | None,
        typer.Option("--job", help="Limit market read to one stored job (Jxxxx)"),
    ] = None,
    ask: Annotated[
        bool,
        typer.Option(
            "--ask/--no-ask",
            help="Confirm suspected gaps on stdin (off by default; also used with --promote)",
        ),
    ] = False,
    promote: Annotated[
        bool,
        typer.Option(
            "--promote",
            help="Write confirmed skills into profile.yaml (after confirmation)",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Write market suggestions. Asks on stdin only with --ask or when promoting."""
    from jobbot.profile.loader import load_profile_raw
    from jobbot.profile.market import (
        merge_confirmed_skills_into_raw,
        render_suggestion_markdown,
        suggest_from_market,
    )

    session, config = _session()
    try:
        candidate = load_profile(config.profile_path)
        raw_profile = load_profile_raw(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    repo = JobRepository(session)
    if job_id:
        job = repo.get(job_id)
        if job is None:
            err_console.print(f"[red]Job not found: {job_id}[/red]")
            raise typer.Exit(VALIDATION_FAILURE)
        jobs = [job]
        # One posting: a single occurrence is enough to surface the term.
        min_count = min(min_count, 1)
    else:
        jobs = repo.list_all()
    if not jobs:
        console.print("No stored jobs. Run jobs search or linkedin sweep first.")
        raise typer.Exit(SUCCESS)

    narrator = _narrator()
    scope = f"job {jobs[0].id}" if job_id else f"{len(jobs)} job descriptions"
    narrator.phase(Phase.RECEIVING_WORLD, scope, may_ask=ask)
    suggestion = suggest_from_market(candidate, jobs, min_count=min_count)
    out = config.output_dir / "profile_market_suggestion.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_suggestion_markdown(suggestion), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {out}")
    for gap in suggestion.missing_suspected[:15]:
        console.print(f"  gap? {gap.term} — {gap.reason}")

    confirmed: list[str] = []
    if ask and suggestion.missing_suspected:
        console.print("\nConfirm skills you [bold]actually have[/bold] (never invent):")
        for gap in suggestion.missing_suspected:
            if typer.confirm(f"Add skill to baseline: {gap.term}?", default=False):
                confirmed.append(gap.term)

    suggested_path = config.root / "data" / "profile.suggested.yaml"
    suggested_raw = merge_confirmed_skills_into_raw(raw_profile, confirmed)
    suggested_path.write_text(
        yaml.safe_dump(suggested_raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    console.print(f"Suggested profile: {suggested_path}")
    if confirmed:
        console.print(f"Confirmed new skills: {', '.join(confirmed)}")
    else:
        console.print("No new skills confirmed.")

    if not promote:
        console.print(
            "Review the markdown + suggested YAML. "
            "Re-run with [bold]--promote[/bold] to copy into profile.yaml."
        )
        return
    if not confirmed and not yes:
        console.print("Nothing confirmed to promote.")
        return
    if not yes and not typer.confirm(
        f"Promote confirmed skills into {config.profile_path}?",
        default=False,
    ):
        console.print("Aborted.")
        raise typer.Exit(SUCCESS)
    backup = config.profile_path.with_suffix(config.profile_path.suffix + ".bak")
    shutil.copy2(config.profile_path, backup)
    shutil.copy2(suggested_path, config.profile_path)
    console.print(f"Backup: {backup}")
    console.print(f"[green]Promoted[/green] confirmed market skills → {config.profile_path}")


_QUERY_ORIGIN_LABEL = {
    "competency": "skill a role mentions",
    "achievements": "repeated in achievements",
    "title": "title held",
}


@profile_app.command("queries")
def profile_queries(
    limit: Annotated[int, typer.Option("--limit", help="How many queries to derive")] = 8,
    apply_changes: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Save the derived set as search_queries in profile.yaml (confirm; backup)",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Job searches derived from your experience; --apply saves them for you to edit."""
    from jobbot.profile.search_queries import derive_search_queries

    config = load_config()
    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    if candidate.search_queries:
        console.print("[bold]Saved in profile.yaml[/bold] (searches use these):")
        for text in candidate.search_queries:
            console.print(f"  • {text}")
    derived = derive_search_queries(candidate, limit=max(1, limit))
    if not derived:
        console.print("Nothing to derive: the profile has no experience to search from.")
        return

    table = Table(title="Derived from your experience")
    table.add_column("Query")
    table.add_column("From")
    table.add_column("Backed by", justify="right")
    for query in derived:
        table.add_row(
            query.text,
            _QUERY_ORIGIN_LABEL[query.origin.value],
            f"{len(query.evidence)} role(s)",
        )
    console.print(table)

    if not apply_changes:
        console.print(
            "Dry-run. Re-run with [bold]--apply[/bold] to save them as search_queries; "
            "edit the list in profile.yaml afterwards."
        )
        return
    texts = [query.text for query in derived]
    if texts == candidate.search_queries:
        console.print("Already saved.")
        return
    question = f"Save these {len(texts)} queries as search_queries in profile.yaml?"
    if not yes and not typer.confirm(question, default=False):
        console.print("Nothing written.")
        return
    raw = load_profile_raw(config.profile_path)
    raw["search_queries"] = texts
    backup = config.profile_path.with_suffix(config.profile_path.suffix + ".bak")
    shutil.copy2(config.profile_path, backup)
    config.profile_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    console.print(f"Backup: {backup}")
    console.print(f"[green]Saved[/green] {len(texts)} search queries → {config.profile_path}")
    console.print("Next: [bold]jobbot getonboard search[/bold] (no query needed)")


@profile_app.command("status")
def profile_status(
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show local vs portal consistency (snapshots if available)."""
    from jobbot.adapters.portal_status import build_profile_status

    config = load_config()
    status = build_profile_status(config)
    if as_json:
        console.print_json(data=status)
        return
    table = Table(title="PROFILE CONSISTENCY")
    table.add_column("")
    table.add_column("Local")
    table.add_column("Indeed")
    table.add_column("LinkedIn")
    for row in status["rows"]:
        table.add_row(row["section"], row["local"], row["indeed"], row["linkedin"])
    console.print(table)


@profile_app.command("diff")
def profile_diff(
    target: Annotated[
        str | None,
        typer.Option("--target", help="indeed | linkedin"),
    ] = None,
    all_targets: Annotated[bool, typer.Option("--all")] = False,
    section: Annotated[str | None, typer.Option("--section")] = None,
) -> None:
    """Diff local profile against portal snapshots."""
    from jobbot.adapters.diff_engine import render_profile_diff

    config = load_config()
    targets: list[str]
    if all_targets:
        targets = ["indeed", "linkedin"]
    elif target:
        targets = [target]
    else:
        err_console.print("Provide --target or --all")
        raise typer.Exit(GENERIC_FAILURE)

    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    for name in targets:
        text = render_profile_diff(candidate, config.output_dir, name, section=section)
        console.print(text)


# ── cv ───────────────────────────────────────────────────────────────────────


@cv_app.command("status")
def cv_status(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable rows")] = False,
) -> None:
    """Alias for `jobbot status`: permanent + active company presence."""
    _print_cv_status(as_json=as_json)


@cv_app.command("build")
def cv_build(
    target: Annotated[
        BuildTarget,
        typer.Option("--target", help="cv (PDF) or ats"),
    ] = BuildTarget.CV,
    job_id: Annotated[
        str | None,
        typer.Option("--job", help="Build job-specific CV for Jxxxx"),
    ] = None,
    style: Annotated[
        CvStyle,
        typer.Option("--style", help="moderncv (your CV design) or plain (portable article)"),
    ] = CvStyle.MODERNCV,
) -> None:
    """Build CV from profile.yaml (base or job-specific)."""
    config = load_config()
    try:
        candidate = load_profile(config.profile_path)
        raw = load_profile_raw(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    result = validate_candidate(candidate)
    if not result.ok:
        err_console.print("[bold red]PROFILE INVALID[/bold red]")
        for issue in result.issues:
            err_console.print(str(issue))
        raise typer.Exit(VALIDATION_FAILURE)

    unknowns = unknown_profile_keys(raw)
    if unknowns:
        err_console.print(
            f"[yellow]{len(unknowns)} claves desconocidas en profile.yaml se ignoran "
            "(jobbot profile validate).[/yellow]"
        )

    job = None
    match = None
    if job_id:
        session, _ = _session()
        repo = JobRepository(session)
        job = repo.get(job_id)
        if job is None:
            err_console.print(f"[red]Job not found: {job_id}[/red]")
            raise typer.Exit(GENERIC_FAILURE)
        match = RuleBasedJobAnalyzer().analyze(candidate, job)
        repo.update_match_score(job.id, match.score)

    from jobbot.cv.build import application_log_options

    try:
        outputs = build_cv(
            candidate=candidate,
            templates_dir=config.templates_dir,
            output_dir=config.output_dir,
            target=target,
            job=job,
            match=match,
            style=style,
            application_log=application_log_options(config.cv),
            credit=config.cv.jobbot_signature,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc

    for path in outputs:
        console.print(f"[green]Wrote[/green] {path}")


@cv_app.command("advise")
def cv_advise(
    limit: Annotated[
        int,
        typer.Option("--limit", help="How many suggestions this run may make"),
    ] = 3,
    apply_changes: Annotated[
        bool,
        typer.Option("--apply", help="Confirm each suggestion and write it to profile.yaml"),
    ] = False,
    job_id: Annotated[
        str | None,
        typer.Option("--job", help="Scope suggestions to one stored job (Jxxxx)"),
    ] = None,
    axis: Annotated[
        str | None,
        typer.Option("--axis", help="machine | language | layout (default: all three)"),
    ] = None,
    llm: Annotated[
        bool,
        typer.Option("--llm", help="Let a plain LLM reword lines (validated; costs tokens)"),
    ] = False,
    llm_deep: Annotated[
        bool,
        typer.Option("--llm-deep", help="Also allow a reasoning model when --llm fails"),
    ] = False,
    max_llm_calls: Annotated[
        int,
        typer.Option("--max-llm-calls", help="Budget for this run"),
    ] = 3,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="With --llm: print what would be sent, call nobody"),
    ] = False,
) -> None:
    """Suggest how the CV presents what you already did. Adds no facts, deletes none."""
    from jobbot.cv.advisor import (
        advise,
        apply_advice,
        default_advice_log_path,
        record_decision,
        render_advice_markdown,
    )
    from jobbot.cv.tune import backup_profile, char_delta_label, prompt_advice_decision
    from jobbot.portals.form_learn import default_form_knowledge_path, load_form_knowledge
    from jobbot.recruiters.playbook import advisor_notes

    session, config = _session()
    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    repo = JobRepository(session)
    if job_id:
        job = repo.get(job_id)
        if job is None:
            err_console.print(f"[red]Job not found: {job_id}[/red]")
            raise typer.Exit(VALIDATION_FAILURE)
        jobs = [job]
    else:
        jobs = repo.list_all()
    forms = load_form_knowledge(default_form_knowledge_path(config.root))
    notes = advisor_notes(config.root)
    log_path = default_advice_log_path(config.output_dir)

    narrator = _narrator()
    scope = f"job {jobs[0].id}" if job_id and jobs else f"{len(jobs)} JD"
    narrator.phase(
        Phase.RECEIVING_WORLD,
        f"{scope} · {len(forms)} formulario(s) · {len(notes)} práctica(s)",
    )
    suggestions = advise(
        candidate,
        jobs=jobs,
        forms=forms,
        playbook=notes,
        limit=max(1, min(limit, 5)),
        log_path=log_path,
    )
    if axis:
        wanted = axis.strip().casefold()
        suggestions = [item for item in suggestions if item.axis.value == wanted]
    if not suggestions:
        console.print("Nothing to suggest this run (or you already answered what there was).")
        console.print(f"Log: {log_path}")
        raise typer.Exit(SUCCESS)

    if llm or llm_deep:
        suggestions = _reword_with_llm(
            suggestions,
            candidate,
            config,
            deep=llm_deep,
            max_calls=max_llm_calls,
            dry_run=dry_run,
        )
        if dry_run:
            return

    out = config.output_dir / "cv" / "advice.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_advice_markdown(suggestions), encoding="utf-8")

    for item in suggestions:
        body = [f"[bold]{item.what}[/bold]", f"where: {item.target.describe()}", f"why: {item.why}"]
        if item.before:
            body.append(f"\nnow:      {item.before}")
        if item.is_rewrite:
            body.append(f"proposed: {item.after}")
            body.append(f"delta:    {char_delta_label(item.before, item.after)}")
        console.print(Panel("\n".join(body), title=f"{item.axis.value} · {item.id}"))
    console.print(f"[green]Wrote[/green] {out}")

    if not apply_changes:
        console.print(
            "Dry-run. Re-run with [bold]--apply[/bold] to confirm each one; "
            "notes without proposed text are yours to act on."
        )
        return

    raw = load_profile_raw(config.profile_path)
    applied = 0
    for item in suggestions:
        if not item.is_rewrite:
            console.print(f"[dim]{item.id}: nothing to write, it is a note.[/dim]")
            continue
        console.print(f"\n[bold]{item.what}[/bold]")
        console.print(f"  now:      {item.before}")
        console.print(f"  proposed: {item.after}")
        console.print(f"  delta:    {char_delta_label(item.before, item.after)}")
        decision = prompt_advice_decision(prompt=lambda msg: typer.prompt(msg, default="N"))
        if decision == "skip_all":
            console.print("Stopped — remaining suggestions left untouched.")
            break
        if decision != "yes":
            record_decision(log_path, item, "rejected")
            continue
        if apply_advice(raw, item):
            record_decision(log_path, item, "applied")
            applied += 1
        else:
            err_console.print("[yellow]The text changed since this was proposed; skipped.[/yellow]")

    if not applied:
        console.print("Nothing written.")
        return
    backup = backup_profile(config.profile_path)
    config.profile_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    console.print(f"Backup: {backup}")
    console.print(f"[green]Applied[/green] {applied} rewording(s) → {config.profile_path}")
    console.print("Next: [bold]jobbot cv build[/bold] then [bold]jobbot cv propagate[/bold]")


@cv_app.command("tune-for")
def cv_tune_for(
    ref: Annotated[
        str,
        typer.Argument(help="Stored job id (Jxxxx) or a hard-link URL to ingest first"),
    ],
    apply_changes: Annotated[
        bool,
        typer.Option("--apply", help="Confirm each suggestion and write profile.yaml"),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Max suggestions this posting may make (cap 5)"),
    ] = 3,
) -> None:
    """Bounded baseline improvement from one posting (~5% delta; #54).

    Stores the URL if needed, then runs density/clarity advise scoped to that job.
    Never invents; never deletes facts. HITL before any write.
    """
    from jobbot.cv.tune import resolve_tune_ref

    try:
        kind, value = resolve_tune_ref(ref)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    session, config = _session()
    job_id = value
    if kind == "url":
        try:
            candidate = load_profile(config.profile_path)
        except ProfileLoadError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(VALIDATION_FAILURE) from exc
        from jobbot.jobs.from_url import ingest_hard_link

        try:
            result = ingest_hard_link(
                config,
                session,
                value,
                candidate=candidate,
                build=False,
                prepare=False,
            )
        except Exception as exc:  # noqa: BLE001 — surface ingest refusal to the user
            err_console.print(f"[red]Could not ingest URL:[/red] {exc}")
            raise typer.Exit(VALIDATION_FAILURE) from exc
        job_id = result.job.id
        console.print(f"[green]Stored[/green] {job_id} from hard link")

    # Reuse advise with the same HITL path; scope via --job.
    cv_advise(
        limit=max(1, min(limit, 5)),
        apply_changes=apply_changes,
        job_id=job_id,
        axis=None,
        llm=False,
        llm_deep=False,
        max_llm_calls=3,
        dry_run=False,
    )


def _reword_with_llm(
    suggestions: list[Any],
    candidate: Candidate,
    config: JobbotConfig,
    *,
    deep: bool,
    max_calls: int,
    dry_run: bool,
) -> list[Any]:
    """Ask a model to reword the suggestions it can, keeping tier 0 when it cannot."""
    from jobbot.cv.llm_advice import Budget, LlmRewriter, default_cache_dir

    rewriter = LlmRewriter(
        cache_dir=default_cache_dir(config.output_dir),
        budget=Budget(max_calls=max(1, max_calls)),
        deep=deep,
        dry_run=dry_run,
    )
    if not rewriter.available and not dry_run:
        console.print(
            "[yellow]No LLM reachable (needs `uv sync --extra llm` and OPENAI_API_KEY). "
            "Showing the deterministic suggestions.[/yellow]"
        )
        return suggestions

    out: list[Any] = []
    improved_count = 0
    for item in suggestions:
        better = rewriter.improve(item, candidate)
        if better is not None:
            improved_count += 1
            out.append(better)
        else:
            out.append(item)

    if dry_run:
        table = Table(title="What --llm would send (nothing was sent)")
        table.add_column("Suggestion")
        table.add_column("Tier")
        table.add_column("Chars", justify="right")
        for call in rewriter.planned:
            table.add_row(call.advice_id, call.tier.value, str(call.chars))
        console.print(table)
        console.print(
            f"{len(rewriter.planned)} call(s), {rewriter.total_planned_chars} characters, "
            f"budget {max_calls} call(s). Drop --dry-run to spend it."
        )
        return out

    console.print(
        f"[dim]LLM: {improved_count} rewording(s) accepted, "
        f"{len(rewriter.rejections)} discarded by validation, "
        f"{rewriter.budget.calls} call(s) spent.[/dim]"
    )
    for reason in rewriter.rejections[:3]:
        console.print(f"[dim]  discarded: {reason}[/dim]")
    return out


@cv_app.command("propagate")
def cv_propagate(
    targets: Annotated[
        str,
        typer.Option(
            "--targets",
            help=(
                "Permanent profiles only: all|permanent|cv,getonboard,indeed,linkedin "
                "(company portals: jobbot cv sync)"
            ),
        ),
    ] = "all",
    apply_changes: Annotated[
        bool,
        typer.Option("--apply", help="Write to the portals (default: dry-run plan)"),
    ] = False,
    section: Annotated[
        str,
        typer.Option("--section", help="Indeed section: headline|summary|skills|experience|all"),
    ] = "all",
    style: Annotated[
        CvStyle,
        typer.Option("--style", help="moderncv (your CV design) or plain"),
    ] = CvStyle.MODERNCV,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip per-destination confirmation (with --apply)"),
    ] = False,
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to your logged-in Chrome (browser chrome-debug)"),
    ] = None,
) -> None:
    """Rebuild the base CV and propagate it to your permanent portal profiles (HITL)."""
    from jobbot.cv.propagate import (
        UnknownTargetError,
        parse_targets,
        plan_propagation,
        summarize_plans,
    )

    config = load_config()
    candidate = _load_valid_candidate(config)
    try:
        wanted = parse_targets(targets)
    except UnknownTargetError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    from jobbot.browser.sessions import inspect_sessions

    sessions = inspect_sessions(config.root, sites=[t.value for t in wanted])
    plans = plan_propagation(
        config,
        candidate,
        targets=wanted,
        section=section,
        sessions=sessions,
        cdp_url=cdp,
    )
    narrator = _narrator()
    narrator.phase(Phase.PROPAGATING_CV, summarize_plans(plans))
    _print_permanent_plan_table(plans, title="CV propagation plan")

    if not apply_changes:
        console.print(
            "Dry-run. Re-run with [bold]--apply[/bold] to rebuild the CV and open/write "
            "each destination (you confirm one by one). "
            "For active company portals see [bold]jobbot cv sync[/bold]."
        )
        raise typer.Exit(SUCCESS)

    _apply_permanent_plans(
        plans,
        config=config,
        candidate=candidate,
        style=style,
        section=section,
        cdp=cdp,
        yes=yes,
        apply_changes=apply_changes,
    )


@cv_app.command("sync")
def cv_sync(
    apply_changes: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Write permanent portals and active company fill/signup (default: dry-run)",
        ),
    ] = False,
    section: Annotated[
        str,
        typer.Option("--section", help="Indeed section: headline|summary|skills|experience|all"),
    ] = "all",
    style: Annotated[
        CvStyle,
        typer.Option("--style", help="moderncv (your CV design) or plain"),
    ] = CvStyle.MODERNCV,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip per-destination confirmation (with --apply)"),
    ] = False,
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to your logged-in Chrome (browser chrome-debug)"),
    ] = None,
) -> None:
    """Standing presence: permanent profiles + active company portals (issue #43)."""
    from jobbot.browser.sessions import inspect_sessions
    from jobbot.cv.propagate import PropagationTarget, summarize_plans
    from jobbot.cv.sync import page_urls_from_cdp, plan_sync

    config = load_config()
    candidate = _load_valid_candidate(config)
    sessions = inspect_sessions(
        config.root,
        sites=[t.value for t in PropagationTarget if t != PropagationTarget.CV],
    )
    plan = plan_sync(
        config,
        candidate,
        section=section,
        sessions=sessions,
        cdp_url=cdp,
        page_urls=page_urls_from_cdp(cdp),
    )
    narrator = _narrator()
    narrator.phase(Phase.PROPAGATING_CV, summarize_plans(plan.permanent))
    _print_permanent_plan_table(plan.permanent, title="CV sync — permanent profiles")

    company_table = Table(title="CV sync — active company portals")
    company_table.add_column("Company")
    company_table.add_column("ATS")
    company_table.add_column("Action")
    company_table.add_column("Hint")
    if plan.companies:
        for row in plan.companies:
            company_table.add_row(
                row.company_name,
                row.ats.value,
                row.action,
                row.hint,
            )
        console.print(company_table)
    else:
        console.print(
            "[dim]No active company portals in data/companies.yaml — "
            "promote candidates first, or import a shared export.[/dim]"
        )

    if not apply_changes:
        console.print(
            "Dry-run. Re-run with [bold]--apply[/bold] to write permanent destinations "
            "and handle active company portals one by one (HITL). "
            "Needs-account rows show the signup sheet (#44); fill rows open the career "
            "URL and type only profile facts — you submit. A yes is not a receipt."
        )
        raise typer.Exit(SUCCESS)

    _apply_permanent_plans(
        plan.permanent,
        config=config,
        candidate=candidate,
        style=style,
        section=section,
        cdp=cdp,
        yes=yes,
        apply_changes=apply_changes,
    )
    _apply_company_sync_rows(plan.companies, config=config, candidate=candidate, cdp=cdp, yes=yes)


def _apply_company_sync_rows(
    rows: Sequence[Any],
    *,
    config: JobbotConfig,
    candidate: Candidate,
    cdp: str | None,
    yes: bool,
) -> None:
    from jobbot.adapters.getonboard.cv_upload import resolve_base_cv
    from jobbot.cv.company_apply import company_apply_intent
    from jobbot.cv.sync import CompanySyncRow

    for row in rows:
        assert isinstance(row, CompanySyncRow)
        intent = company_apply_intent(row, candidate, cv_path=resolve_base_cv(config.output_dir))
        if intent.mode == "skip":
            console.print(f"{row.company_name}: skipped — {intent.note}")
            continue
        if intent.mode == "signup_sheet":
            if not yes and not typer.confirm(
                f"Show signup sheet for {row.company_name} (no account created)?",
                default=True,
            ):
                console.print(f"{row.company_name}: skipped.")
                continue
            _print_company_signup_sheet(intent)
            continue
        if not yes and not typer.confirm(
            f"Open {row.company_name} career site and fill known profile fields "
            "(you submit; a yes is not a receipt)?",
            default=False,
        ):
            console.print(f"{row.company_name}: skipped.")
            continue
        _run_company_fill(intent, config=config, candidate=candidate, cdp=cdp)


def _print_company_signup_sheet(intent: Any) -> None:
    console.print(
        Panel(
            f"[bold]{intent.company_name}[/bold] ({intent.company_id})\n"
            f"{intent.url}\n\n{intent.note}",
            title="Registration (you complete it)",
        )
    )
    table = Table(title="Have this at hand")
    table.add_column("The portal asks")
    table.add_column("Your profile answers")
    table.add_column("From")
    for item in intent.sheet:
        value = item.value or "[yellow]you decide[/yellow]"
        table.add_row(item.label[:40], value[:46], item.source[:34])
    console.print(table)
    console.print(
        "[dim]JobBot does not create the account, does not invent a password, "
        "does not accept terms, and does not solve CAPTCHA/2FA.[/dim]"
    )


def _run_company_fill(
    intent: Any,
    *,
    config: JobbotConfig,
    candidate: Candidate,
    cdp: str | None,
) -> None:
    from jobbot.browser.cdp import resolve_cdp_url
    from jobbot.browser.session import BrowserSession
    from jobbot.cv.company_apply import perform_company_apply

    cdp_url = resolve_cdp_url(cdp)
    with BrowserSession(
        profile_dir=config.root / "browser-data" / "companies-sync-cdp",
        headless=False,
        cdp_url=cdp_url,
        debug_root=config.output_dir / "debug",
    ) as browser:
        result = perform_company_apply(
            intent,
            candidate,
            page=browser.page,
            output_dir=config.output_dir,
        )
    console.print(
        f"[green]Opened[/green] {intent.url} · filled {len(result.filled)} field(s)"
        + (" · CV attached" if result.attached else "")
    )
    if result.receipt_path is not None:
        console.print(f"Page receipt: {result.receipt_path}")
    else:
        console.print(
            "[yellow]No page receipt[/yellow] — status stays unknown until the portal "
            "shows a profile fact. Submit yourself; JobBot does not click submit."
        )


def _load_valid_candidate(config: JobbotConfig) -> Candidate:
    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc
    validation = validate_candidate(candidate)
    if not validation.ok:
        err_console.print("[bold red]PROFILE INVALID[/bold red]")
        for issue in validation.issues:
            err_console.print(str(issue))
        raise typer.Exit(VALIDATION_FAILURE)
    return candidate


def _print_permanent_plan_table(plans: Sequence[Any], *, title: str) -> None:
    from jobbot.cv.propagate import TargetPlan

    table = Table(title=title)
    table.add_column("Destination")
    table.add_column("Operations")
    table.add_column("Detail")
    for plan in plans:
        assert isinstance(plan, TargetPlan)
        detail = plan.blocked_reason or plan.note or ""
        if plan.hint:
            detail = f"{detail} → {plan.hint}"
        table.add_row(
            plan.target.value,
            "blocked" if not plan.ready else str(len(plan.operations)),
            detail,
        )
    console.print(table)
    narrator = _narrator()
    for plan in plans:
        assert isinstance(plan, TargetPlan)
        for operation in plan.operations:
            narrator.note(f"{plan.target.value}: {operation}")


def _apply_permanent_plans(
    plans: Sequence[Any],
    *,
    config: JobbotConfig,
    candidate: Candidate,
    style: CvStyle,
    section: str,
    cdp: str | None,
    yes: bool,
    apply_changes: bool,
) -> None:
    from jobbot.cv.propagate import PropagationTarget, TargetPlan

    for plan in plans:
        assert isinstance(plan, TargetPlan)
        if not plan.ready:
            err_console.print(
                f"[yellow]Skipping {plan.target.value}[/yellow]: {plan.blocked_reason}"
            )
            continue
        if not plan.actionable:
            console.print(f"{plan.target.value}: nothing to do.")
            continue
        if not yes and not typer.confirm(
            f"Propagate to {plan.target.value} ({len(plan.operations)} operation(s))?",
            default=plan.target == PropagationTarget.CV,
        ):
            console.print(f"{plan.target.value}: skipped.")
            continue
        if plan.target == PropagationTarget.CV:
            _propagate_base_cv(config, candidate, style=style)
        elif plan.target == PropagationTarget.GETONBOARD:
            _propagate_getonboard(config, apply_changes=apply_changes, cdp=cdp, yes=yes)
        elif plan.target == PropagationTarget.INDEED:
            _propagate_indeed(config, section=section, cdp=cdp, yes=yes)
        elif plan.target == PropagationTarget.LINKEDIN:
            _propagate_linkedin(config, cdp=cdp, yes=yes)


def _propagate_base_cv(config: JobbotConfig, candidate: Candidate, *, style: CvStyle) -> None:
    for target in (BuildTarget.CV, BuildTarget.ATS):
        try:
            outputs = build_cv(
                candidate=candidate,
                templates_dir=config.templates_dir,
                output_dir=config.output_dir,
                target=target,
                style=style,
                credit=config.cv.jobbot_signature,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            err_console.print(f"[red]cv ({target.value}): {exc}[/red]")
            continue
        for path in outputs:
            console.print(f"[green]Wrote[/green] {path}")


def _getonboard_session(config: JobbotConfig, cdp: str | None) -> Any:
    from jobbot.browser.cdp import resolve_cdp_url
    from jobbot.browser.session import BrowserSession

    return BrowserSession(
        profile_dir=config.root / "browser-data" / "getonboard-cdp",
        headless=False,
        cdp_url=resolve_cdp_url(cdp),
        debug_root=config.output_dir / "debug",
    )


def _propagate_getonboard(
    config: JobbotConfig,
    *,
    apply_changes: bool,
    cdp: str | None,
    yes: bool,
) -> None:
    from jobbot.adapters.getonboard.client import GetOnBoardProfileClient
    from jobbot.adapters.getonboard.draft import load_permanent_profile
    from jobbot.adapters.getonboard.profile_edit import (
        GOB_EDIT_URL,
        apply_writes,
        desired_from_fields,
        plan_writes,
        read_current,
    )

    client = GetOnBoardProfileClient.from_config(config)
    path, result = client.prepare_package()
    console.print(f"[green]Wrote[/green] {path}  (mode={result.mode})")
    if not apply_changes:
        console.print(f"Dry-run: --apply escribe el perfil en {GOB_EDIT_URL}")
        return

    fields = load_permanent_profile(config.output_dir)
    if fields is None:
        err_console.print("[red]getonboard: no permanent profile to write[/red]")
        return
    desired = desired_from_fields(
        fields.headline,
        fields.experiencia_y_perfil,
        fields.formacion_academica,
    )
    with _getonboard_session(config, cdp) as browser:
        page = browser.page
        page.goto(GOB_EDIT_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        writes = plan_writes(read_current(page), desired)
        if not writes:
            console.print("getonboard: el portal ya tiene estos textos.")
        else:
            for write in writes:
                console.print(f"  {write.describe()}")
            if not yes and not typer.confirm(
                f"Write these {len(writes)} field(s) on Get on Board?",
                default=False,
            ):
                console.print("getonboard: skipped profile fields.")
            else:
                done = apply_writes(page, writes)
                console.print(f"[green]Wrote[/green] {len(done)} field(s) on Get on Board.")
    _propagate_getonboard_cv(config, cdp=cdp, yes=yes)


def _propagate_getonboard_cv(
    config: JobbotConfig,
    *,
    cdp: str | None,
    yes: bool,
) -> None:
    """Upload output/base/cv.pdf after validating it; mark default when confirmed."""
    from jobbot.adapters.getonboard.cv_upload import (
        resolve_base_cv,
        upload_cv,
        validate_cv_for_upload,
        write_upload_receipt,
    )

    path = resolve_base_cv(config.output_dir)
    check = validate_cv_for_upload(path)
    for line in check.summary_lines():
        console.print(f"  {line}", soft_wrap=False, overflow="ignore", crop=False)
    if not check.ok:
        err_console.print("[yellow]CV not uploaded: local validation failed.[/yellow]")
        return
    if not yes and not typer.confirm(
        "Upload this PDF to Get on Board Tus CVs and mark it default?",
        default=False,
    ):
        console.print("getonboard CV: skipped.")
        return
    with _getonboard_session(config, cdp) as browser:
        result = upload_cv(
            browser.page,
            path,
            label="CV base",
            make_default=True,
            check=check,
        )
    write_upload_receipt(config.output_dir, check, result)
    console.print(
        f"[green]Uploaded[/green] {result.path.name} as {result.label!r} "
        f"(default={result.is_default}) → {result.resumes_url}"
    )


def _propagate_indeed(config: JobbotConfig, *, section: str, cdp: str | None, yes: bool) -> None:
    from jobbot.adapters.indeed.client import IndeedAdapter
    from jobbot.browser.cdp import resolve_cdp_url

    adapter = IndeedAdapter.from_config(config, cdp_url=resolve_cdp_url(cdp))
    plan = adapter.build_sync_plan(section=section)
    if not plan.actionable:
        console.print("indeed: no changes required.")
        return
    if not yes and not typer.confirm(
        f"Apply these {len(plan.actionable)} Indeed modifications?",
        default=False,
    ):
        console.print("indeed: skipped.")
        return
    console.print(adapter.apply_sync_plan(plan).message)


def _propagate_linkedin(config: JobbotConfig, *, cdp: str | None, yes: bool) -> None:
    from jobbot.adapters.linkedin.client import LinkedInAdapter
    from jobbot.adapters.linkedin.package import LinkedInPublicationItem
    from jobbot.browser.cdp import resolve_cdp_url

    adapter = LinkedInAdapter.from_config(config, cdp_url=resolve_cdp_url(cdp))
    console.print("Fetching remote publication titles…")
    remote_titles = adapter.fetch_remote_publication_titles()

    def _confirm(item: LinkedInPublicationItem) -> bool:
        if yes:
            return True
        return typer.confirm(f"Add publication: {item.title}?", default=True)

    console.print(
        adapter.apply_publications(confirm_each=_confirm, remote_titles=remote_titles).message
    )


# ── jobs ─────────────────────────────────────────────────────────────────────


@jobs_app.command("add")
def jobs_add(
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="Path to JD text file"),
    ] = None,
    url: Annotated[str | None, typer.Option("--url")] = None,
    stdin: Annotated[
        bool,
        typer.Option("--stdin", help="Read JD from stdin"),
    ] = False,
) -> None:
    """Add a job posting from a text file or stdin (manual fallback)."""
    session, config = _session()
    if file is not None:
        text = file.read_text(encoding="utf-8")
    elif stdin:
        if sys.stdin.isatty():
            err_console.print(
                "Waiting for a job description on stdin. Paste the text, then press Ctrl-D."
            )
        text = typer.get_text_stream("stdin").read()
    else:
        err_console.print("Provide --file or --stdin (prefer: jobbot jobs search)")
        raise typer.Exit(GENERIC_FAILURE)

    from jobbot.jobs.closure import closure_evidence, fetch_posting_text

    filled = closure_evidence(text)
    if filled is None and url:
        try:
            filled = closure_evidence(fetch_posting_text(url))
        except OSError:
            filled = None
    if filled:
        err_console.print(f"[red]Vacancy is filled[/red] ({filled}). Not stored.")
        raise _QuietExit(VALIDATION_FAILURE)

    repo = JobRepository(session)
    job = repo.add_from_text(text, url=url)
    path = write_job_json(job, config.output_dir)
    console.print(f"[green]Added[/green] {job.id}  {job.company}  {job.title}")
    console.print(f"Wrote {path}")
    _learn_company_knowledge(config, job)


@jobs_app.command("search")
def jobs_search(
    query: Annotated[str, typer.Argument(help="Search query, e.g. the role you want")],
    location: Annotated[
        str | None,
        typer.Option("--location", "-l", help="Location (default: Santiago)"),
    ] = "Santiago",
    remote: Annotated[bool, typer.Option("--remote", help="Prefer remote filters")] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Max results (1-50, default 20)"),
    ] = 20,
    enrich: Annotated[
        bool,
        typer.Option(
            "--enrich/--no-enrich",
            help="Open each job page for full description (slower)",
        ),
    ] = True,
    cdp: Annotated[
        str | None,
        typer.Option(
            "--cdp",
            help="Attach to user Chrome via CDP (e.g. http://127.0.0.1:9222). "
            "Also: JOBBOT_CDP_URL. Start with: jobbot browser chrome-debug",
        ),
    ] = None,
) -> None:
    """Search Indeed and store job postings locally (small volumes)."""
    from jobbot.adapters.indeed.jobs import IndeedJobSource
    from jobbot.browser.cdp import resolve_cdp_url
    from jobbot.jobs.sources import JobSearchQuery

    if limit < 1 or limit > 50:
        err_console.print("--limit must be between 1 and 50")
        raise typer.Exit(GENERIC_FAILURE)

    session, config = _session()
    cdp_url = resolve_cdp_url(cdp)
    source = IndeedJobSource(config, cdp_url=cdp_url)
    q = JobSearchQuery(query=query, location=location, remote=remote, limit=limit)

    if cdp_url:
        console.print(f"Using CDP Chrome at [bold]{cdp_url}[/bold]")
    console.print(f"Searching Indeed ({source.base}) for [bold]{query}[/bold]…")
    try:
        found = source.search_jobs(q) if enrich else source.search_cards(q)
    except Exception as exc:
        err_console.print(f"[red]Indeed search failed: {exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc

    if not found:
        console.print("No jobs found.")
        raise typer.Exit(SUCCESS)

    repo = JobRepository(session)
    table = Table(title="Indeed search results")
    table.add_column("ID")
    table.add_column("Company")
    table.add_column("Role")
    table.add_column("Match")
    stored: list[str] = []
    for raw in found:
        job = repo.upsert_external(raw)
        write_job_json(job, config.output_dir)
        _learn_company_knowledge(config, job)
        stored.append(job.id)
        score = f"{job.match_score:.0f}%" if job.match_score is not None else "-"
        table.add_row(job.id, job.company, job.title, score)
    console.print(table)
    console.print(f"Stored {len(stored)} jobs. Next: [bold]jobbot jobs match {stored[0]}[/bold]")
    console.print("Or: [bold]jobbot jobs shortlist[/bold]")


@jobs_app.command("queries")
def jobs_queries(
    region: Annotated[
        list[str] | None,
        typer.Option(
            "--region",
            help="Geo token for the query (repeatable), e.g. Spain, Europe, EMEA, Chile",
        ),
    ] = None,
    remote: Annotated[
        bool,
        typer.Option("--remote/--no-remote", help='Include "remote" in the query'),
    ] = True,
    ats: Annotated[
        list[str] | None,
        typer.Option(
            "--ats",
            help="ATS kinds to target (repeatable): ashby, greenhouse, lever",
        ),
    ] = None,
    keyword: Annotated[
        list[str] | None,
        typer.Option("--keyword", "-k", help="Extra quoted keyword (repeatable)"),
    ] = None,
) -> None:
    """Print web-search queries for ATS hosts (Ashby/Greenhouse/Lever). Does not search."""
    from jobbot.jobs.ats_queries import build_ats_search_queries, format_queries_help
    from jobbot.portals.detect import AtsKind

    kinds: tuple[AtsKind, ...] | None = None
    if ats:
        parsed: list[AtsKind] = []
        for raw in ats:
            try:
                parsed.append(AtsKind(raw.casefold().strip()))
            except ValueError:
                err_console.print(f"[red]Unknown --ats {raw!r}[/red]; use ashby, greenhouse, lever")
                raise typer.Exit(VALIDATION_FAILURE) from None
        kinds = tuple(parsed)

    queries = build_ats_search_queries(
        kinds=kinds,
        regions=tuple(region or ()),
        remote=remote,
        keywords=tuple(keyword or ()),
    )
    if not queries:
        err_console.print("[red]No queries for the selected ATS kinds[/red]")
        raise typer.Exit(VALIDATION_FAILURE)

    console.print(format_queries_help(queries))
    raise typer.Exit(SUCCESS)


@jobs_app.command("show")
def jobs_show(
    job_id: Annotated[str, typer.Argument()],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show a stored job posting."""
    session, _ = _session()
    job = JobRepository(session).get(job_id)
    if job is None:
        err_console.print(f"[red]Job not found: {job_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    if as_json:
        console.print_json(data=job.model_dump(mode="json"))
        return
    console.print(f"[bold]{job.id}[/bold]  {job.title}")
    console.print(f"Company: {job.company}")
    console.print(f"Location: {job.location or '—'}")
    console.print(f"Seniority: {job.seniority or '—'}")
    console.print(f"Remote: {job.remote_type or '—'}")
    console.print(f"URL: {job.url or '—'}")
    console.print(f"Skills: {', '.join(job.skills) or '—'}")
    if job.requirements:
        console.print("Requirements:")
        for req in job.requirements:
            console.print(f"  - {req}")
    console.print(Panel(job.description[:2000], title="Description"))


@jobs_app.command("match")
def jobs_match(
    job_id: Annotated[str, typer.Argument()],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Match a job against the local profile (decision aid)."""
    session, config = _session()
    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    repo = JobRepository(session)
    job = repo.get(job_id)
    if job is None:
        err_console.print(f"[red]Job not found: {job_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)

    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    repo.update_match_score(job.id, match.score)
    job_dir = config.output_dir / "jobs" / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "match.json").write_text(
        json.dumps(match.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if as_json:
        console.print_json(data=match.to_dict())
        return
    console.print(format_match_report(match))


@jobs_app.command("conditions")
def jobs_conditions(
    job_ids: Annotated[list[str] | None, typer.Argument(help="Job IDs (J0001 …)")] = None,
    all_prepared: Annotated[
        bool, typer.Option("--all-prepared", help="Every job with a prepared application")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Review a posting's special conditions against you before applying (exit 1: dealbreaker)."""
    from jobbot.jobs.conditions import posting_conditions
    from jobbot.jobs.eligibility import (
        assess_conditions,
        default_answers_path,
        format_conditions_report,
        has_dealbreaker,
        job_answers_path,
        load_application_answers,
    )

    session, config = _session()
    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc
    ids = list(job_ids or [])
    if all_prepared:
        for application in ApplicationRepository(session).list_all():
            if application.status == ApplicationStatus.PREPARED and application.job_id not in ids:
                ids.append(application.job_id)
    if not ids:
        err_console.print("Give one or more job IDs, or --all-prepared.")
        raise typer.Exit(VALIDATION_FAILURE)

    repo = JobRepository(session)
    answers_path = default_answers_path(config.root)
    reports: list[dict[str, object]] = []
    texts: list[str] = []
    blocked = False
    for job_id in ids:
        job = repo.get(job_id)
        if job is None:
            err_console.print(f"[red]Job not found: {job_id}[/red]")
            raise typer.Exit(GENERIC_FAILURE)
        try:
            answers = load_application_answers(
                answers_path, job_answers=job_answers_path(config.output_dir, job.id)
            )
        except ValueError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(VALIDATION_FAILURE) from exc
        verdicts = assess_conditions(posting_conditions(job), candidate, answers)
        blocked = blocked or has_dealbreaker(verdicts)
        reports.append(
            {
                "job_id": job.id,
                "title": job.title,
                "company": job.company,
                "dealbreaker": has_dealbreaker(verdicts),
                "verdicts": [v.to_dict() for v in verdicts],
            }
        )
        texts.append(format_conditions_report(job, verdicts))

    if as_json:
        console.print_json(data=reports)
    else:
        state = "found" if answers_path.is_file() else "missing; copy the .example.yaml"
        texts.append(f"Answers: {answers_path} ({state})")
        console.print("\n\n".join(texts), markup=False, highlight=False, soft_wrap=True)
    if blocked:
        raise _QuietExit(GENERIC_FAILURE)


@jobs_app.command("shortlist")
def jobs_shortlist() -> None:
    """Rank stored jobs by match score."""
    session, config = _session()
    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    repo = JobRepository(session)
    analyzer = RuleBasedJobAnalyzer()
    rows: list[tuple[float, str, str, str]] = []
    for job in repo.list_all():
        match = analyzer.analyze(candidate, job)
        repo.update_match_score(job.id, match.score)
        rows.append((match.score, job.id, job.title, job.company))
    rows.sort(key=lambda r: r[0], reverse=True)
    console.print("[bold]TOP MATCHES[/bold]")
    for score, jid, title, company in rows:
        console.print(f"{score:5.1f}%  {jid}  {title}  {company}")
    _warn_if_matcher_blind([row[0] for row in rows])


@jobs_app.command("backfill-dates")
def jobs_backfill_dates() -> None:
    """Date already-stored posts from the activity id in their URL (offline)."""
    from jobbot.jobs.backfill import backfill_posted_at

    session, _ = _session()
    repo = JobRepository(session)
    updated = backfill_posted_at(repo)
    console.print(f"Dated [bold]{updated}[/bold] stored jobs from their post URL.")
    stale = [
        job for job in repo.list_all() if job.posted_at is not None and not is_fresh(job.posted_at)
    ]
    if stale:
        console.print("Older than the freshness window:")
        for job in stale:
            console.print(f"  {job.id}  {age_label(job.posted_at):>9}  {job.title[:50]}")


@jobs_app.command("note")
def jobs_note(
    job_id: Annotated[str, typer.Argument()],
    text: Annotated[str, typer.Argument()],
) -> None:
    """Attach a free-text note to a job."""
    session, _ = _session()
    try:
        JobRepository(session).set_note(job_id, text)
    except KeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc
    console.print(f"[green]Noted[/green] {job_id}")


@jobs_app.command("audit")
def jobs_audit(
    source: Annotated[
        str,
        typer.Option("--source", help="Job source to audit (only 'indeed' today)"),
    ],
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Mark decoys/duplicates invalid (skip submitted)"),
    ] = False,
) -> None:
    """List (or mark) stored Indeed decoys and title/company duplicates (#211)."""
    from jobbot.jobs.audit import audit_indeed_jobs

    if source.casefold() != "indeed":
        err_console.print(f"[red]Unknown audit source:[/red] {source} (supported: indeed)")
        raise typer.Exit(VALIDATION_FAILURE)

    try:
        session, _ = _session()
        findings = audit_indeed_jobs(
            JobRepository(session),
            ApplicationRepository(session),
            apply=apply,
        )
    except OSError as exc:
        err_console.print(f"[red]Could not open job store:[/red] {exc}")
        raise typer.Exit(GENERIC_FAILURE) from exc

    if not findings:
        console.print("[green]No suspicious Indeed jobs.[/green]")
        return

    console.print("[bold]Indeed audit[/bold]" + (" (--apply)" if apply else " (dry-run)"))
    for finding in findings:
        flag = " [protected: submitted]" if finding.protected else ""
        action = "marked" if apply and not finding.protected else "suspect"
        console.print(f"  {finding.job_id}  {action}  {finding.reason}{flag}")


# ── application(s) ───────────────────────────────────────────────────────────


def _filled_vacancy(job: JobPosting) -> str | None:
    """Phrase that says this posting is gone, from the stored text or the live page."""
    from jobbot.jobs.closure import closure_evidence_for_job, fetch_posting_text

    return closure_evidence_for_job(job, fetch=fetch_posting_text)


def _stop_if_filled(session: Session, job: JobPosting) -> None:
    """Refuse to open or mark applied when the vacancy itself says it is filled."""
    evidence = _filled_vacancy(job)
    if evidence is None:
        return
    url = job.ats_url or job.url or job.id
    err_console.print(f"[red]Vacancy is filled[/red] ({evidence}). Not opening {url}.")
    err_console.print("No portal receipt — status stays unknown, not applied.")
    existing = ApplicationRepository(session).get_for_job(job.id)
    if existing is not None:
        ApplicationRepository(session).add_event(existing.id, "closed", f"{evidence} | {url}")
    raise _QuietExit(VALIDATION_FAILURE)


def _note_no_receipt(session: Session, job_id: str, package_dir: str, url: str) -> None:
    """Opening a portal is not a submission. Record the URL and leave status prepared."""
    repo = ApplicationRepository(session)
    app = repo.upsert_for_job(
        job_id,
        status=ApplicationStatus.PREPARED,
        package_dir=package_dir,
    )
    repo.add_event(app.id, "no_receipt", url)
    console.print(
        f"[yellow]No portal receipt[/yellow] for {url}. "
        "Status stays prepared — unknown whether it was submitted."
    )


def _build_job_cv(config: JobbotConfig, candidate: Candidate, job: JobPosting) -> None:
    """Build the CV adapted to this job, unless the one on disk is still current."""
    from jobbot.cv.build import build_adapted_cv

    job_dir = config.output_dir / "jobs" / job.id
    existed = (job_dir / "cv.pdf").is_file() or (job_dir / "cv_ats.txt").is_file()
    match = RuleBasedJobAnalyzer().analyze(candidate, job)
    written = build_adapted_cv(config, candidate, job, match)
    if not written:
        return
    if existed:
        console.print(f"Rebuilt the CV for {job.id} (profile or CV settings changed).")
    else:
        console.print(f"Built CV adapted to {job.id}.")
    if not any(path.suffix == ".pdf" for path in written):
        # PDF may fail without xelatex; ATS text is enough for the package
        err_console.print("[yellow]PDF not built (XeLaTeX unavailable); ATS text written.[/yellow]")


@application_app.command("prepare")
def application_prepare(job_id: Annotated[str, typer.Argument()]) -> None:
    """Prepare application package for a job (does not invent answers)."""
    session, config = _session()
    repo = JobRepository(session)
    job = repo.get(job_id)
    if job is None:
        err_console.print(f"[red]Job not found: {job_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)

    job_dir = config.output_dir / "jobs" / job.id
    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    narrator = _narrator()
    narrator.phase(Phase.PROPAGATING_CV, f"{job.company} · {job.title}"[:60])
    _build_job_cv(config, candidate, job)

    app_dir = prepare_application_package(
        job, config.output_dir, job_dir=job_dir, candidate=candidate
    )
    ApplicationRepository(session).upsert_for_job(
        job.id,
        status=ApplicationStatus.PREPARED,
        package_dir=str(app_dir),
    )
    console.print(f"[green]Prepared[/green] {app_dir}")
    gob = app_dir / "getonboard_es.md"
    if gob.is_file():
        console.print(f"Get on Board Spanish drafts: {gob}")


@application_app.command("open")
def application_open(job_id: Annotated[str, typer.Argument()]) -> None:
    """Open the job URL (or ATS URL) in the default browser (does not apply)."""
    session, _ = _session()
    job = JobRepository(session).get(job_id)
    if job is None:
        err_console.print(f"[red]Job not found: {job_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    from jobbot.adapters.ats.apply import open_ats_in_browser, resolve_ats_url

    _stop_if_filled(session, job)
    ats_url, _kind = resolve_ats_url(job)
    target = ats_url or job.url
    if not target:
        err_console.print(f"[red]No URL for job {job_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    console.print(open_ats_in_browser(target))


def _warn_if_login_needed(config: JobbotConfig, ats_kind: AtsKind, cdp: str | None) -> None:
    from urllib.parse import urlparse

    from jobbot.applications.login_gate import login_warning, session_site
    from jobbot.browser import sessions as browser_sessions
    from jobbot.companies.signup import AccountNeed, account_need

    if account_need(ats_kind) is not AccountNeed.NEEDED:
        return
    site = session_site(ats_kind)
    states: list[browser_sessions.SessionState] = []
    if site is not None:
        ports = list(browser_sessions.DEFAULT_PORTS)
        cdp_port = urlparse(cdp).port if cdp else None
        if cdp_port is not None and cdp_port not in ports:
            ports.append(cdp_port)
        states = browser_sessions.inspect_sessions(config.root, sites=[site], ports=ports)
    warning = login_warning(ats_kind, states)
    if warning:
        console.print(f"[bold yellow]{warning}[/bold yellow]")


def _ats_target_url(adapter: Any, job: JobPosting, ats_url: str) -> str:
    """The page `adapter.open` would show (Indeed routes to its apply step)."""
    from jobbot.adapters.ats.indeed_apply import IndeedApplyAdapter, apply_target

    if isinstance(adapter, IndeedApplyAdapter):
        return apply_target(job) or ats_url
    return ats_url


def _fill_ats_over_cdp(
    cdp: str, url: str, candidate: Candidate, config: JobbotConfig, job_id: str
) -> bool:
    """Open the ATS in a background tab of your Chrome and fill known fields. False → fall back."""
    from jobbot.adapters.ats import apply_fill
    from jobbot.adapters.ats.email_apply import resolve_cv_path

    cv_pdf = resolve_cv_path(config.output_dir, job_id)
    try:
        result = apply_fill.open_and_fill_over_cdp(cdp, url, candidate, cv_path=cv_pdf)
    except apply_fill.ApplyFillError as exc:
        err_console.print(f"[yellow]{exc}[/yellow]")
        err_console.print("[yellow]Falling back to your default browser + prefill sheet.[/yellow]")
        return False
    console.print(background_notice(result.url, where=f"una pestaña nueva de Chrome ({cdp})"))
    if result.login_required:
        console.print(
            "[yellow]The page asks you to sign in. JobBot typed nothing: sign in in that "
            "tab yourself, then re-run this command.[/yellow]"
        )
    elif not result.readable:
        console.print(f"[yellow]{result.note}. Fill it by hand with the prefill sheet.[/yellow]")
    for label in result.filled:
        console.print(f"  [green]filled[/green] {label}")
    for label in result.kept:
        console.print(f"  [cyan]kept (already had a value)[/cyan] {label}")
    if result.attached and cv_pdf is not None:
        console.print(f"  [green]attached[/green] {cv_pdf.name}  ({cv_pdf})")
    console.print("Left for you:")
    for label in result.left_for_human:
        console.print(f"  • {label}")
    console.print("Tab left open. Review everything; you submit — JobBot never clicks it.")
    return True


@application_app.command("apply")
def application_apply(
    job_id: Annotated[
        str | None,
        typer.Argument(help="Job id (omit with --all)"),
    ] = None,
    all_jobs: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Assist every eligible stored job, one at a time (HITL between jobs)",
        ),
    ] = False,
    apply_changes: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Open ATS/Gmail and record assisted apply (default: dry-run plan only)",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
    attach: Annotated[
        bool,
        typer.Option(
            "--attach/--no-attach",
            help="Email apply: attach the CV in Gmail via browser (--no-attach = compose URL only)",
        ),
    ] = True,
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to your logged-in Chrome (see browser chrome-debug)"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="With --all, open at most N jobs"),
    ] = None,
) -> None:
    """Plan or open ATS/Gmail apply for a job (HITL; no CAPTCHA bypass; no invented answers).

    ``--all`` queues eligible jobs (URL present, not already applied) and assists
    one by one. Portal Next/Submit stays human; between jobs JobBot asks before
    opening the next.
    """
    if all_jobs and job_id:
        err_console.print("[red]Pass a job id or --all, not both.[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    if all_jobs:
        _application_apply_all(
            apply_changes=apply_changes,
            yes=yes,
            attach=attach,
            cdp=cdp,
            limit=limit,
        )
        return
    if not job_id:
        err_console.print("[red]Job id required (or pass --all).[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    _application_apply_one(
        job_id,
        apply_changes=apply_changes,
        yes=yes,
        attach=attach,
        cdp=cdp,
    )


def _application_apply_all(
    *,
    apply_changes: bool,
    yes: bool,
    attach: bool,
    cdp: str | None,
    limit: int | None,
) -> None:
    """Dry-run lists the queue; --apply opens one job at a time with a pause between."""
    from jobbot.applications.batch import select_batch_apply_jobs

    session, config = _session()
    try:
        load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    items = select_batch_apply_jobs(
        JobRepository(session).list_all(),
        ApplicationRepository(session).list_all(),
        limit=limit,
    )
    if not items:
        console.print("No eligible jobs (need a URL; skip applied/rejected/…).")
        return

    console.print(f"[bold]Batch apply[/bold] — {len(items)} job(s)")
    console.print(
        "[dim]JobBot opens the portal and stops. Next/Submit are yours. "
        "Between jobs it asks before opening the next.[/dim]"
    )
    for item in items:
        console.print(f"  {item.label}")

    if not apply_changes:
        console.print(
            "Dry-run only. Re-run with [bold]--all --apply[/bold] "
            "to open one at a time (you submit each)."
        )
        return

    opened = 0
    for index, item in enumerate(items):
        console.print()
        console.print(f"[bold]({index + 1}/{len(items)})[/bold] {item.label}")
        if not yes and not typer.confirm(
            f"Open assisted apply for {item.job.id}? "
            "(finish Next/Submit yourself before continuing)",
            default=True,
        ):
            console.print("Skipped.")
            if not typer.confirm("Continue with the remaining jobs?", default=True):
                console.print("Stopped.")
                break
            continue
        try:
            _application_apply_one(
                item.job.id,
                apply_changes=True,
                yes=True,
                attach=attach,
                cdp=cdp,
            )
            opened += 1
        except _QuietExit:
            console.print(f"[yellow]Skipped[/yellow] {item.job.id} (filled or refused).")
        if index + 1 >= len(items):
            break
        if yes:
            console.print(
                "[dim]Next job — finish any open wizard first "
                "(JobBot does not click Next/Submit).[/dim]"
            )
            continue
        if not typer.confirm(
            "Ready for the next job? (only after you finished Next/Submit here)",
            default=False,
        ):
            console.print("Stopped.")
            break
    console.print(
        f"Opened assist for [bold]{opened}[/bold] job(s). "
        "Status stays prepared without a portal receipt."
    )


def _application_apply_one(
    job_id: str,
    *,
    apply_changes: bool,
    yes: bool,
    attach: bool,
    cdp: str | None,
) -> None:
    """Assist apply for a single job id (same behaviour as before --all)."""
    from jobbot.adapters.ats.apply import (
        build_apply_plan,
        prefill_field_map,
        prefill_sheet_fields,
    )
    from jobbot.adapters.ats.email_apply import build_email_draft, open_gmail_compose
    from jobbot.adapters.ats.registry import adapter_for_job
    from jobbot.adapters.base import ApplyMethod
    from jobbot.applications.manager import FilesystemApplicationPackage

    session, config = _session()
    job = JobRepository(session).get(job_id)
    if job is None:
        err_console.print(f"[red]Job not found: {job_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    _stop_if_filled(session, job)
    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    plan = build_apply_plan(candidate, job)
    adapter = adapter_for_job(job)
    adapter_name = adapter.name if adapter is not None else plan.ats_kind.value
    console.print(f"[bold]{job.id}[/bold]  {job.title} @ {job.company}")
    console.print(f"ATS: {plan.ats_kind.value}  adapter={adapter_name}  {plan.ats_url or '(none)'}")
    console.print(f"Method: {plan.method.value}")
    console.print(plan.message)
    _warn_if_login_needed(config, plan.ats_kind, cdp)

    if plan.method == ApplyMethod.LINKEDIN_MESSAGE:
        from jobbot.adapters.ats.apply import open_ats_in_browser

        if not apply_changes:
            console.print(
                "Dry-run only. Re-run with [bold]--apply[/bold] to open the LinkedIn post "
                "(you write the message; JobBot never sends)."
            )
            return
        if not yes and not typer.confirm(
            "¿Abrir la publicación de LinkedIn para que le escribas vos? "
            "JobBot no envía el mensaje.",
            default=False,
        ):
            console.print("Aborted.")
            raise typer.Exit(SUCCESS)
        job_dir = config.output_dir / "jobs" / job.id
        app_dir = prepare_application_package(
            job, config.output_dir, job_dir=job_dir, candidate=candidate
        )
        console.print(open_ats_in_browser(plan.ats_url))
        console.print(
            "[green]Opened the post.[/green] Write the message yourself. JobBot never sends InMail."
        )
        _note_no_receipt(session, job.id, str(app_dir), plan.ats_url)
        return

    if plan.method == ApplyMethod.EMAIL:
        from jobbot.adapters.ats.email_apply import is_tailored_cv, resolve_cv_path

        if apply_changes:
            _build_job_cv(config, candidate, job)
            prepare_application_package(
                job,
                config.output_dir,
                job_dir=config.output_dir / "jobs" / job.id,
                candidate=candidate,
            )
        cv_pdf = resolve_cv_path(config.output_dir, job.id)
        tailored = is_tailored_cv(cv_pdf, job.id)
        draft = build_email_draft(candidate, job, cv_path=cv_pdf)
        if cv_pdf is None:
            cv_line = "(none — run jobbot cv build --job " + job.id + ")"
        elif tailored:
            cv_line = f"{cv_pdf}  [green](adapted to {job.id})[/green]"
        else:
            cv_line = f"{cv_pdf}  [yellow](base CV — not adapted)[/yellow]"
        console.print(
            Panel(
                f"[bold]To:[/bold] {draft.to}\n"
                f"[bold]Subject:[/bold] {draft.subject}\n\n"
                f"{draft.body}\n\n"
                f"[bold]CV:[/bold] {cv_line}",
                title="email apply draft (HITL)",
            )
        )
        if draft.body_truncated_for_url:
            console.print(
                "[yellow]Body is long — Gmail URL may truncate; "
                "paste the rest from this dry-run if needed.[/yellow]"
            )
        if not apply_changes:
            if not tailored:
                console.print(
                    f"Tip: [bold]jobbot cv build --job {job.id}[/bold] first — "
                    "--apply builds the adapted CV for you."
                )
            console.print(
                "Dry-run only. Re-run with [bold]--apply[/bold] to open Gmail compose "
                "(log in if asked, attach CV, press Send yourself)."
            )
            return
        if not yes and not typer.confirm(
            "¿Abrir Gmail con el CV adjunto para que lo revises y envíes vos?",
            default=False,
        ):
            console.print("Aborted.")
            raise typer.Exit(SUCCESS)
        from jobbot.browser.sessions import (
            AccountExpectation,
            inspect_sessions,
            refuse_to_open,
            session_for,
        )

        expected = AccountExpectation(
            email=str(candidate.personal.email) if candidate.personal.email else None,
            linkedin=candidate.personal.linkedin,
        )
        gmail = session_for(
            inspect_sessions(config.root, sites=["gmail"], account=expected),
            "gmail",
        )
        if gmail is not None:
            blocked = refuse_to_open(gmail.status, gmail.evidence)
            if blocked:
                err_console.print(f"[red]{blocked}[/red]")
                raise typer.Exit(GENERIC_FAILURE)
        if attach and cv_pdf is not None:
            from jobbot.adapters.gmail.compose import GmailComposeAdapter, GmailComposeError

            try:
                GmailComposeAdapter(config, cdp_url=cdp).open_draft_for_review(
                    draft, cv_path=cv_pdf
                )
                console.print(
                    f"[green]Gmail draft ready with[/green] {cv_pdf.name} "
                    "[green]attached. Review it and press Send yourself.[/green]"
                )
            except GmailComposeError as exc:
                err_console.print(f"[yellow]Could not attach in Gmail: {exc}[/yellow]")
                open_gmail_compose(draft)
                err_console.print(
                    f"[yellow]Opened plain compose — attach {cv_pdf} by hand.[/yellow]"
                )
        else:
            url = open_gmail_compose(draft)
            console.print(background_notice(f"Gmail compose ({len(url)} chars URL)"))
            console.print(
                f"Adjuntá el CV ({cv_pdf or 'output/base/cv.pdf'}) "
                "y pulsá [bold]Enviar[/bold] en Gmail. JobBot no envía por vos."
            )
        job_dir = config.output_dir / "jobs" / job.id
        app_dir = prepare_application_package(
            job, config.output_dir, job_dir=job_dir, candidate=candidate
        )
        _note_no_receipt(session, job.id, str(app_dir), plan.ats_url or draft.to)
        return

    console.print("Known fields to inject:")
    for key, value in prefill_field_map(candidate).items():
        console.print(f"  {key}: {value}")
    prefill_review = plan.needs_review
    if adapter is not None:
        pkg_preview = FilesystemApplicationPackage(
            config.output_dir / "jobs" / job.id / "application"
        )
        prefill_review = adapter.prefill(candidate, job, pkg_preview).needs_review
    console.print("Needs human review:")
    for item in prefill_review:
        console.print(f"  • {item}")
    if not apply_changes:
        console.print(
            "Dry-run only. Prepare package then re-run with [bold]--apply[/bold] "
            "to open the ATS (you submit)."
        )
        return
    if not plan.ats_url:
        err_console.print("[red]No ATS URL — cannot open portal.[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    if not yes and not typer.confirm(
        f"Open {plan.ats_kind.value} and assist apply (you submit)?",
        default=False,
    ):
        console.print("Aborted.")
        raise typer.Exit(SUCCESS)
    # Ensure package exists
    job_dir = config.output_dir / "jobs" / job.id
    app_dir = prepare_application_package(
        job, config.output_dir, job_dir=job_dir, candidate=candidate
    )
    filled_in_cdp = bool(cdp) and _fill_ats_over_cdp(
        str(cdp), _ats_target_url(adapter, job, plan.ats_url), candidate, config, job.id
    )
    if not filled_in_cdp:
        if adapter is not None and hasattr(adapter, "open"):
            opened = adapter.open(job)
            if opened:
                console.print(background_notice(opened))
        else:
            from jobbot.adapters.ats.apply import open_ats_in_browser

            console.print(open_ats_in_browser(plan.ats_url))
    # Write prefill cheat-sheet next to package. Contact stays in profile.yaml.
    cheat = app_dir / "ats_prefill.yaml"
    import yaml

    cheat.write_text(
        yaml.safe_dump(
            {
                "ats_url": plan.ats_url,
                "ats_kind": plan.ats_kind.value,
                "fields": prefill_sheet_fields(candidate),
                "omitted_contact": ["email", "phone"],
                "needs_review": prefill_review,
                "note": (
                    "Fill only known fields; leave blank rather than inventing. "
                    "Email and phone are not copied here — read them from profile.yaml."
                ),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    where = "in your Chrome (CDP tab left open)" if filled_in_cdp else ""
    console.print(f"[green]Opened ATS[/green] {plan.ats_url} {where}".rstrip())
    console.print(f"Prefill sheet: {cheat}")
    console.print("Submit manually after reviewing HITL fields.")
    _learn_form_from_apply(config, plan.ats_url, job.company)
    _note_no_receipt(session, job.id, str(app_dir), plan.ats_url)


@application_app.command("check-answer")
def application_check_answer(
    job_id: Annotated[str, typer.Argument()],
    question: Annotated[str, typer.Option("--question", "-q", help="The form's question")],
    text: Annotated[str | None, typer.Option("--text", help="The answer to check")] = None,
    file: Annotated[
        Path | None, typer.Option("--file", help="Read the answer from this file")
    ] = None,
) -> None:
    """Check a free-text answer against profile.yaml and the stored posting (pastes nothing).

    Exit 0 when every claim is backed, 2 when a claim is unbacked or misattributed,
    4 when the question asks for motivation (only you can answer it).
    """
    from jobbot.applications.answer_check import Verdict, check_answer, render_answer_check

    if text is not None and file is None:
        answer = text
    elif file is not None and text is None:
        answer = file.read_text(encoding="utf-8")
    else:
        err_console.print("[red]Give the answer with --text or --file (one of them).[/red]")
        raise _QuietExit(VALIDATION_FAILURE)
    session, config = _session()
    job = JobRepository(session).get(job_id)
    if job is None:
        err_console.print(f"[red]Job not found: {job_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    check = check_answer(question, answer, candidate, job)
    for line in render_answer_check(check, job):
        console.print(line, markup=False, highlight=False)
    if check.verdict is Verdict.REJECTED:
        raise _QuietExit(VALIDATION_FAILURE)
    if check.verdict is Verdict.NEEDS_CANDIDATE:
        raise _QuietExit(MANUAL_CHALLENGE)


@application_app.command("show")
def application_show(job_id: Annotated[str, typer.Argument()]) -> None:
    """Show application package path and status for a job."""
    session, config = _session()
    apps = [a for a in ApplicationRepository(session).list_all() if a.job_id == job_id]
    job = JobRepository(session).get(job_id)
    if job is None:
        err_console.print(f"[red]Job not found: {job_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    console.print(f"{job.id}  {job.company}  {job.title}")
    if apps:
        app = apps[0]
        console.print(f"Application: {app.id}  status={app.status.value}")
        pkg = app.package_dir or str(config.output_dir / "jobs" / job_id / "application")
        console.print(f"Package: {pkg}")
        for event in ApplicationRepository(session).events_for(app.id):
            console.print(f"  event {event.event_type}: {event.detail or '—'}")
    else:
        console.print("No application record yet. Run: jobbot application prepare " + job_id)


@applications_app.command("list")
def applications_list() -> None:
    """List tracked applications."""
    session, _ = _session()
    jobs = {j.id: j for j in JobRepository(session).list_all()}
    table = Table(title="Applications")
    table.add_column("ID")
    table.add_column("Company")
    table.add_column("Role")
    table.add_column("Status")
    for app in ApplicationRepository(session).list_all():
        job = jobs.get(app.job_id)
        table.add_row(
            app.id,
            job.company if job else "?",
            job.title if job else app.job_id,
            app.status.value,
        )
    console.print(table)


@applications_app.command("status")
def applications_status() -> None:
    """Alias for applications list."""
    applications_list()


# ── indeed / linkedin (Vertical B) ───────────────────────────────────────────


@indeed_app.command("login")
def indeed_login(
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to Chrome CDP (e.g. http://127.0.0.1:9222)"),
    ] = None,
    continue_url: Annotated[
        str | None,
        typer.Option(
            "--continue-url",
            help=(
                "Magic / verify link from Indeed's email (paste from phone). "
                "JobBot opens it in the persistent Chrome profile; never types a password."
            ),
        ),
    ] = None,
) -> None:
    """Open Indeed login (HITL). Prefer email/magic link; optional --continue-url."""
    from jobbot.adapters.indeed.client import IndeedAdapter, normalize_indeed_continue_url
    from jobbot.browser.cdp import resolve_cdp_url

    if continue_url is not None:
        try:
            continue_url = normalize_indeed_continue_url(continue_url)
        except ValueError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(VALIDATION_FAILURE) from exc

    IndeedAdapter.from_config(load_config(), cdp_url=resolve_cdp_url(cdp)).login(
        continue_url=continue_url
    )


@indeed_app.command("status")
def indeed_status(
    cdp: Annotated[str | None, typer.Option("--cdp")] = None,
) -> None:
    """Check whether an Indeed session appears valid."""
    from jobbot.adapters.indeed.client import IndeedAdapter
    from jobbot.browser.cdp import resolve_cdp_url

    ok = IndeedAdapter.from_config(load_config(), cdp_url=resolve_cdp_url(cdp)).session_valid()
    console.print("Indeed session: " + ("valid" if ok else "not authenticated"))


@indeed_app.command("inspect")
def indeed_inspect(
    cdp: Annotated[str | None, typer.Option("--cdp")] = None,
) -> None:
    """Inspect Indeed page roles/labels for selector development."""
    from jobbot.adapters.indeed.client import IndeedAdapter
    from jobbot.browser.cdp import resolve_cdp_url

    IndeedAdapter.from_config(load_config(), cdp_url=resolve_cdp_url(cdp)).inspect()


@indeed_app.command("prepare")
def indeed_prepare() -> None:
    """Write Indeed sync package markdown from profile.yaml (no portal write)."""
    from jobbot.adapters.indeed.client import IndeedAdapter

    path = IndeedAdapter.from_config(load_config()).prepare_package()
    console.print(f"[green]Wrote[/green] {path}")
    console.print("Review, then: jobbot indeed sync --section all [--apply]")


@indeed_app.command("pull")
def indeed_pull(
    cdp: Annotated[str | None, typer.Option("--cdp")] = None,
) -> None:
    """Pull Indeed profile snapshot (read)."""
    from jobbot.adapters.indeed.client import IndeedAdapter
    from jobbot.browser.cdp import resolve_cdp_url

    path = IndeedAdapter.from_config(load_config(), cdp_url=resolve_cdp_url(cdp)).pull_and_save()
    console.print(f"[green]Wrote[/green] {path}")


@indeed_app.command("diff")
def indeed_diff(
    section: Annotated[str | None, typer.Option("--section")] = None,
) -> None:
    """Diff local profile vs Indeed snapshot."""
    from jobbot.adapters.diff_engine import render_profile_diff

    config = load_config()
    candidate = load_profile(config.profile_path)
    console.print(render_profile_diff(candidate, config.output_dir, "indeed", section=section))


@indeed_app.command("sync")
def indeed_sync(
    section: Annotated[
        str,
        typer.Option(
            "--section",
            help="headline|summary|skills|experience|all (default: all)",
        ),
    ] = "all",
    apply_changes: Annotated[
        bool,
        typer.Option("--apply", help="Apply changes (default is dry-run)"),
    ] = False,
    reconcile: Annotated[
        bool,
        typer.Option(
            "--reconcile",
            help="Full mirror: update/add/delete Indeed Resume to match profile.yaml",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Skip interactive confirmation (with --apply)"),
    ] = False,
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to Chrome CDP (e.g. http://127.0.0.1:9222)"),
    ] = None,
) -> None:
    """Dry-run (default) or apply Indeed profile sync from profile.yaml."""
    from jobbot.adapters.indeed.client import IndeedAdapter
    from jobbot.browser.cdp import resolve_cdp_url

    adapter = IndeedAdapter.from_config(load_config(), cdp_url=resolve_cdp_url(cdp))
    if reconcile:
        if not apply_changes:
            console.print(
                "Reconcile dry-run: will update/add/delete Indeed Resume to mirror "
                "profile.yaml (publications are NOT written to Recognitions)."
            )
            console.print("Re-run with [bold]--reconcile --apply[/bold] to write.")
            return
        if not yes and not typer.confirm(
            "Full mirror Indeed ← profile.yaml (may delete legacy Indeed rows)?",
            default=False,
        ):
            console.print("Aborted.")
            raise typer.Exit(SUCCESS)
        result = adapter.apply_reconcile()
        console.print(result.message)
        return

    plan = adapter.build_sync_plan(section=section)
    if not plan.actionable:
        console.print("No changes required.")
        return
    console.print("Operations:")
    for op in plan.actionable:
        console.print(f"  {op.op.value:6} {op.section}.{op.field}  {op.label}")
    if not apply_changes:
        console.print(
            "Dry-run only. Re-run with [bold]--apply[/bold] "
            "or full mirror with [bold]--reconcile --apply[/bold]."
        )
        return
    if not yes and not typer.confirm(
        f"Apply these {len(plan.actionable)} modifications?",
        default=False,
    ):
        console.print("Aborted.")
        raise typer.Exit(SUCCESS)
    result = adapter.apply_sync_plan(plan)
    console.print(result.message)


@linkedin_app.command("login")
def linkedin_login(
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to Chrome CDP (e.g. http://127.0.0.1:9222)"),
    ] = None,
) -> None:
    """Open LinkedIn login with persistent browser profile."""
    from jobbot.adapters.linkedin.client import LinkedInAdapter
    from jobbot.browser.cdp import resolve_cdp_url

    LinkedInAdapter.from_config(load_config(), cdp_url=resolve_cdp_url(cdp)).login()


@linkedin_app.command("status")
def linkedin_status(
    cdp: Annotated[str | None, typer.Option("--cdp")] = None,
) -> None:
    """Check whether a LinkedIn session appears valid."""
    from jobbot.adapters.linkedin.client import LinkedInAdapter
    from jobbot.browser.cdp import resolve_cdp_url

    ok = LinkedInAdapter.from_config(load_config(), cdp_url=resolve_cdp_url(cdp)).session_valid()
    console.print("LinkedIn session: " + ("valid" if ok else "not authenticated"))


@linkedin_app.command("inspect")
def linkedin_inspect(
    cdp: Annotated[str | None, typer.Option("--cdp")] = None,
) -> None:
    """Inspect LinkedIn page for selector development."""
    from jobbot.adapters.linkedin.client import LinkedInAdapter
    from jobbot.browser.cdp import resolve_cdp_url

    LinkedInAdapter.from_config(load_config(), cdp_url=resolve_cdp_url(cdp)).inspect()


@linkedin_app.command("pull")
def linkedin_pull(
    cdp: Annotated[str | None, typer.Option("--cdp")] = None,
) -> None:
    """Pull LinkedIn profile snapshot (read-only)."""
    from jobbot.adapters.linkedin.client import LinkedInAdapter
    from jobbot.browser.cdp import resolve_cdp_url

    path = LinkedInAdapter.from_config(load_config(), cdp_url=resolve_cdp_url(cdp)).pull_and_save()
    console.print(f"[green]Wrote[/green] {path}")


@linkedin_app.command("diff")
def linkedin_diff(
    section: Annotated[str | None, typer.Option("--section")] = None,
) -> None:
    """Diff local profile vs LinkedIn snapshot (read-only audit)."""
    from jobbot.adapters.diff_engine import render_profile_diff

    config = load_config()
    candidate = load_profile(config.profile_path)
    console.print(render_profile_diff(candidate, config.output_dir, "linkedin", section=section))


@linkedin_app.command("sync")
def linkedin_sync(
    section: Annotated[
        str,
        typer.Option("--section", help="Only 'publications' is writable"),
    ] = "publications",
    apply_changes: Annotated[
        bool,
        typer.Option("--apply", help="Apply changes (default is dry-run)"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Skip per-item confirmation (with --apply)"),
    ] = False,
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to Chrome CDP (e.g. http://127.0.0.1:9222)"),
    ] = None,
    fetch_remote: Annotated[
        bool,
        typer.Option(
            "--fetch-remote/--no-fetch-remote",
            help="Scrape LinkedIn publications list to skip already-present titles",
        ),
    ] = True,
    enrich: Annotated[
        bool,
        typer.Option(
            "--enrich/--no-enrich",
            help="Update existing LinkedIn pubs with date, DOI URL, authors description",
        ),
    ] = False,
) -> None:
    """Dry-run or apply LinkedIn Publications from profile.yaml (DOI + coauthors)."""
    from jobbot.adapters.linkedin.client import LinkedInAdapter
    from jobbot.adapters.linkedin.package import build_linkedin_publication_items
    from jobbot.browser.cdp import resolve_cdp_url

    if section != "publications":
        err_console.print(
            f"[red]LinkedIn sync only supports --section publications (got {section!r}).[/red]"
        )
        raise typer.Exit(GENERIC_FAILURE)

    config = load_config()
    adapter = LinkedInAdapter.from_config(config, cdp_url=resolve_cdp_url(cdp))
    candidate = load_profile(config.profile_path)
    items = build_linkedin_publication_items(candidate)

    remote_titles: list[str] = []
    if fetch_remote and apply_changes:
        console.print("Fetching remote publication titles…")
        remote_titles = adapter.fetch_remote_publication_titles()
    elif fetch_remote and not apply_changes:
        console.print(
            "[dim]Dry-run assumes remote empty unless you pass --apply "
            "(which fetches remote titles).[/dim]"
        )

    plan = adapter.build_publications_plan(
        candidate, remote_titles=remote_titles, enrich_existing=enrich
    )
    actionable = plan.actionable
    console.print(f"Local publications: {len(items)}")
    if remote_titles:
        console.print(f"Remote titles matched: {len(remote_titles)}")
    if not actionable:
        console.print("No publications to add/enrich (all present or profile empty).")
        return
    console.print("Operations:")
    for op in actionable:
        after = op.after if isinstance(op.after, dict) else {}
        coauthors = ", ".join(after.get("coauthors") or []) or "(solo)"
        console.print(f"  {op.op.value:6} {op.label}")
        console.print(f"      publisher={after.get('publisher')} year={after.get('year')}")
        console.print(f"      url={after.get('url')}")
        console.print(f"      coauthors={coauthors}")
    if not apply_changes:
        console.print(
            "Dry-run only. Re-run with [bold]--apply[/bold] "
            "(confirms each publication unless [bold]--yes[/bold]). "
            "Use [bold]--enrich[/bold] to update existing entries."
        )
        return

    def _confirm(item: object) -> bool:
        if yes:
            return True
        title = getattr(item, "title", str(item))
        return typer.confirm(f"Write publication to LinkedIn?\n  {title}", default=True)

    result = adapter.apply_publications(
        confirm_each=_confirm,
        remote_titles=remote_titles,
        enrich_existing=enrich,
    )
    console.print(result.message)


@linkedin_app.command("sweep")
def linkedin_sweep(
    query: Annotated[
        str | None,
        typer.Argument(help="Content search keywords (default: 'hiring' + your profile searches)"),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max posts to keep")] = 20,
    max_queries: Annotated[
        int,
        typer.Option("--max-queries", help="Profile searches to run when no query is given"),
    ] = _DEFAULT_MAX_QUERIES,
    fixture: Annotated[
        Path | None,
        typer.Option("--fixture", help="Parse posts from a text fixture (no browser)"),
    ] = None,
    register_portals: Annotated[
        bool,
        typer.Option(
            "--register-portals/--no-register-portals",
            help="Upsert detected ATS domains into data/portals.yaml",
        ),
    ] = True,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Auto-add new portals without prompting"),
    ] = False,
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to Chrome CDP"),
    ] = None,
    country: Annotated[
        list[str] | None,
        typer.Option(
            "--country",
            help="Where you want to work, repeatable (default: [search].countries, CL)",
        ),
    ] = None,
    any_country: Annotated[
        bool,
        typer.Option("--any-country", help="Keep posts from every country"),
    ] = False,
    copy_links: Annotated[
        bool,
        typer.Option(
            "--copy-links/--no-copy-links",
            help="Read each post's own URL from its '…' menu (one extra click per post)",
        ),
    ] = True,
    max_age_days: Annotated[
        int | None,
        typer.Option(
            "--max-age-days",
            help="Skip posts older than N days (default: [search].max_age_days, 30)",
        ),
    ] = None,
    any_age: Annotated[
        bool,
        typer.Option("--any-age", help="Keep posts of every age"),
    ] = False,
) -> None:
    """Sweep LinkedIn recruiter posts → store jobs + detect ATS URLs (MVP: posts)."""
    from jobbot.adapters.getonboard.jobs import remember_portal_from_url
    from jobbot.adapters.linkedin.posts_source import LinkedInPostJobSource
    from jobbot.browser.cdp import resolve_cdp_url
    from jobbot.jobs.sources import JobSearchQuery, get_job_source
    from jobbot.portals.detect import AtsKind
    from jobbot.portals.registry import (
        default_portals_path,
        domain_from_url,
        load_registry,
        save_registry,
    )

    if limit < 1 or limit > 50:
        err_console.print("--limit must be between 1 and 50")
        raise typer.Exit(GENERIC_FAILURE)

    session, config = _session()
    if query is None:
        queries = [f"hiring {text}" for text in _search_queries(config, None, max_queries)]
    else:
        queries = [query]
    countries = resolve_countries(config.search.countries, country, any_country=any_country)
    max_age = 0 if any_age else config.search.max_age_days
    if max_age_days is not None and not any_age:
        max_age = max_age_days
    source = get_job_source("linkedin_post", config, cdp_url=resolve_cdp_url(cdp))
    assert isinstance(source, LinkedInPostJobSource)
    if countries:
        console.print(
            f"Country filter: [bold]{', '.join(countries)}[/bold] (--any-country to lift)"
        )
    if max_age > 0:
        console.print(
            f"Age filter: posts newer than [bold]{max_age} days[/bold] (--any-age to lift)"
        )
    found: list[JobPosting] = []
    for text in queries:
        if fixture is not None:
            found.extend(
                source.search_from_fixture(
                    fixture.expanduser().resolve(),
                    query=text,
                    countries=countries,
                    allow_remote=config.search.allow_remote,
                    max_age_days=max_age,
                )
            )
        else:
            console.print(f"Sweeping LinkedIn content for [bold]{text}[/bold]…")
            found.extend(
                source.search_jobs(
                    JobSearchQuery(
                        query=text,
                        limit=limit,
                        countries=countries,
                        allow_remote=config.search.allow_remote,
                        copy_permalinks=copy_links,
                        max_age_days=max_age,
                    )
                )
            )
    found = _unique_postings(found)

    if not found:
        console.print("No relevant posts found.")
        raise typer.Exit(SUCCESS)

    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError:
        candidate = None

    analyzer = RuleBasedJobAnalyzer()
    repo = JobRepository(session)
    registry = load_registry(default_portals_path(config.root))
    narrator = _narrator()
    narrator.phase(Phase.UPDATING_WORLD, f"{len(found)} recruiter posts")
    table = Table(title="LinkedIn post sweep")
    table.add_column("ID")
    table.add_column("Company")
    table.add_column("Role")
    table.add_column("ATS")
    table.add_column("Age")
    table.add_column("Where")
    table.add_column("Match")
    stored: list[str] = []
    scores: list[float] = []
    for raw in found[:limit]:
        job = repo.upsert_external(raw)
        if candidate is not None:
            match = analyzer.analyze(candidate, job)
            repo.update_match_score(job.id, match.score)
            job.match_score = match.score
            scores.append(match.score)
        write_job_json(job, config.output_dir)
        stored.append(job.id)
        ats = job.ats_kind or "-"
        score = f"{job.match_score:.0f}%" if job.match_score is not None else "-"
        table.add_row(
            job.id,
            job.company[:28],
            job.title[:40],
            ats,
            age_label(job.posted_at),
            detect_country(f"{job.title}\n{job.description}") or "?",
            score,
        )
        if register_portals and job.ats_url:
            domain = domain_from_url(job.ats_url)
            kind = AtsKind(job.ats_kind) if job.ats_kind else AtsKind.UNKNOWN
            existing = registry.find(domain)
            should_ask = existing is None and not yes
            if should_ask and not typer.confirm(
                f"New ATS portal {domain} ({kind.value}). Add to registry?",
                default=True,
            ):
                continue
            registry.upsert(
                domain=domain,
                ats_kind=kind,
                example_url=job.ats_url,
                seen=True,
            )
            # Always persist newly seen employment platforms
            remember_portal_from_url(
                config,
                job.ats_url,
                notes=f"learned from LinkedIn post job {job.id}",
            )
            _learn_company_knowledge(
                config,
                job,
                source=DiscoverySource.LINKEDIN_POST,
                narrator=narrator,
            )
    if register_portals:
        save_registry(registry, default_portals_path(config.root))
        console.print(f"Portal registry updated: {default_portals_path(config.root)}")
    console.print(table)
    console.print(
        f"Stored {len(stored)} jobs. Next: [bold]jobbot jobs match {stored[0]}[/bold] "
        f"→ [bold]jobbot application apply {stored[0]}[/bold]"
    )
    if candidate is not None:
        _warn_if_matcher_blind(scores)


@getonboard_app.command("prepare")
def getonboard_prepare(
    cold: Annotated[
        bool,
        typer.Option(
            "--cold",
            help="Ignore previous draft (cold start from profile.yaml only)",
        ),
    ] = False,
    use_llm: Annotated[
        bool,
        typer.Option(
            "--llm",
            help="Refine with LangChain (needs jobbot[llm] + OPENAI_API_KEY)",
        ),
    ] = False,
    seed: Annotated[
        Path | None,
        typer.Option(
            "--seed",
            help="Seed previous text from a file (old GoB paste = accumulated capital)",
        ),
    ] = None,
) -> None:
    """Maintain permanent GoB profile: cumulative refine by default (not cold replace)."""
    from jobbot.adapters.getonboard.client import GetOnBoardProfileClient

    seed_exp = seed.read_text(encoding="utf-8") if seed is not None else None
    path, result = GetOnBoardProfileClient.from_config(load_config()).prepare_package(
        cold=cold,
        use_llm=use_llm,
        seed_experiencia=seed_exp,
    )
    console.print(f"[green]Wrote permanent profile[/green] {path}")
    console.print(
        f"mode={result.mode}  kept={result.kept_paragraphs}  "
        f"added={result.added_paragraphs}  dropped={result.dropped_paragraphs}"
    )
    console.print(
        "También: output/getonboard/profile_permanent.yaml + sync_package.md "
        "(history/ guarda versiones previas)"
    )
    console.print(
        "Orden: 1) [bold]jobbot getonboard open-profile[/bold] "
        "(reemplaza textos viejos en el portal)  "
        "2) [bold]jobbot getonboard open-cvs[/bold]  "
        "3) postular."
    )


@getonboard_app.command("show-profile")
def getonboard_show_profile() -> None:
    """Show paths / char counts for the permanent GoB profile maintainer."""
    from jobbot.adapters.getonboard.draft import (
        EDUCATION_MAX,
        EXPERIENCE_MAX,
        load_permanent_profile,
        permanent_profile_md_path,
        permanent_profile_yaml_path,
    )

    config = load_config()
    fields = load_permanent_profile(config.output_dir)
    md = permanent_profile_md_path(config.output_dir)
    yml = permanent_profile_yaml_path(config.output_dir)
    if fields is None:
        console.print("No hay perfil permanente generado. Corre: jobbot getonboard prepare")
        raise typer.Exit(SUCCESS)
    console.print(f"Markdown: {md}")
    console.print(f"YAML:     {yml}")
    console.print(f"experiencia_y_perfil: {len(fields.experiencia_y_perfil)} / {EXPERIENCE_MAX}")
    console.print(f"formacion_academica:  {len(fields.formacion_academica)} / {EDUCATION_MAX}")
    console.print(f"headline: {fields.headline}")
    console.print(f"skills: {', '.join(fields.skills)}")


@getonboard_app.command("open-profile")
def getonboard_open_profile() -> None:
    """Open Get on Board 'Editar perfil' (HITL; paste permanent profile)."""
    from jobbot.adapters.getonboard.client import GetOnBoardProfileClient

    config = load_config()
    client = GetOnBoardProfileClient.from_config(config)
    # Ensure permanent texts exist before opening (cumulative refine)
    client.prepare_package()
    url = client.open_profile_edit()
    console.print(background_notice(url))
    console.print("Reemplaza textos viejos con output/getonboard/profile_permanent.md")


@getonboard_app.command("upload-cv")
def getonboard_upload_cv(
    apply_changes: Annotated[
        bool,
        typer.Option("--apply", help="Upload after validation (default: validate only)"),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    make_default: Annotated[
        bool,
        typer.Option("--default/--no-default", help="Mark the uploaded CV as default"),
    ] = True,
    label: Annotated[
        str,
        typer.Option("--label", help="Name shown under Tus CVs"),
    ] = "CV base",
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to your logged-in Chrome (browser chrome-debug)"),
    ] = None,
    cv: Annotated[
        Path | None,
        typer.Option("--cv", help="PDF to upload (default: output/base/cv.pdf)"),
    ] = None,
) -> None:
    """Validate the local CV PDF, then upload it to Get on Board Tus CVs."""
    from jobbot.adapters.getonboard.cv_upload import (
        GobCvUploadError,
        resolve_base_cv,
        upload_cv,
        validate_cv_for_upload,
        write_upload_receipt,
    )

    config = load_config()
    path = cv.expanduser().resolve() if cv else resolve_base_cv(config.output_dir)
    check = validate_cv_for_upload(path)
    console.print("[bold]Local CV check[/bold] (before any upload)")
    for line in check.summary_lines():
        # Paths must stay one token: soft-wrap mid-filename breaks copy/paste and tests.
        console.print(f"  {line}", soft_wrap=False, overflow="ignore", crop=False)
    if not check.ok:
        err_console.print("[red]Validation failed — nothing was uploaded.[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    if not apply_changes:
        console.print(
            "Dry-run. Re-run with [bold]--apply --cdp URL[/bold] to upload "
            "(you confirm after seeing the hash)."
        )
        return
    if not yes and not typer.confirm(
        f"Upload {path.name} as {label!r}" + (" and mark default?" if make_default else "?"),
        default=False,
    ):
        console.print("Aborted.")
        raise typer.Exit(SUCCESS)
    try:
        with _getonboard_session(config, cdp) as browser:
            result = upload_cv(
                browser.page,
                path,
                label=label,
                make_default=make_default,
                check=check,
            )
    except GobCvUploadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc
    write_upload_receipt(config.output_dir, check, result)
    console.print(
        f"[green]Uploaded[/green] {result.path.name} as {result.label!r} "
        f"(default={result.is_default}) → {result.resumes_url}"
    )


@getonboard_app.command("open-cvs")
def getonboard_open_cvs() -> None:
    """Open Get on Board 'Tus CVs' (HITL fallback if upload-cv is not enough)."""
    from jobbot.adapters.getonboard.client import GetOnBoardProfileClient
    from jobbot.adapters.getonboard.cv_upload import resolve_base_cv, validate_cv_for_upload

    config = load_config()
    path = resolve_base_cv(config.output_dir)
    check = validate_cv_for_upload(path)
    url = GetOnBoardProfileClient.from_config(config).open_resumes()
    console.print(background_notice(url))
    console.print("[bold]Local CV check[/bold]")
    for line in check.summary_lines():
        console.print(f"  {line}", soft_wrap=False, overflow="ignore", crop=False)
    if check.ok:
        console.print(
            "Prefer: [bold]jobbot getonboard upload-cv --apply --cdp URL[/bold] "
            "(validates, then uploads)."
        )
    else:
        console.print("Fix the PDF first: [bold]jobbot cv build[/bold]")


@getonboard_app.command("sync")
def getonboard_sync(
    apply_changes: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Open profile + CV editors (default: dry-run checklist only)",
        ),
    ] = False,
) -> None:
    """Maintain permanent GoB profile: regenerate texts; optionally open editors."""
    from jobbot.adapters.getonboard.client import GetOnBoardProfileClient

    client = GetOnBoardProfileClient.from_config(load_config())
    path, result = client.prepare_package()
    console.print(f"[green]Wrote[/green] {path}  (mode={result.mode})")
    if not apply_changes:
        console.print(
            "Dry-run del mantenedor (computación acumulativa). Revisa "
            "profile_permanent.md, luego:\n"
            "  jobbot cv build\n"
            "  jobbot getonboard sync --apply   # abre Editar perfil + Tus CVs\n"
            "  jobbot getonboard prepare --seed old.txt --llm  # refine con LLM"
        )
        return
    console.print("Abriendo Editar perfil (pega profile_permanent.md)…")
    console.print(background_notice(client.open_profile_edit()))
    console.print("Abriendo zona profesional (navega a Tus CVs)…")
    console.print(background_notice(client.open_resumes()))
    console.print("HITL: reemplaza textos viejos, sube CV default, guarda. Luego postula.")


@getonboard_app.command("search")
def getonboard_search(
    query: Annotated[
        str | None,
        typer.Argument(help="Search query (default: the searches in your own profile)"),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max results per query (1-50)")] = 20,
    max_queries: Annotated[
        int,
        typer.Option("--max-queries", help="Profile searches to run when no query is given"),
    ] = _DEFAULT_MAX_QUERIES,
) -> None:
    """Search Get on Board (Spanish/LATAM) and store jobs locally."""
    from jobbot.adapters.getonboard.jobs import GetOnBoardJobSource
    from jobbot.jobs.sources import JobSearchQuery

    if limit < 1 or limit > 50:
        err_console.print("--limit must be between 1 and 50")
        raise typer.Exit(GENERIC_FAILURE)

    session, config = _session()
    queries = _search_queries(config, query, max_queries)
    source = GetOnBoardJobSource(config)
    found: list[JobPosting] = []
    for text in queries:
        console.print(f"Searching Get on Board for [bold]{text}[/bold]…")
        try:
            found.extend(source.search_jobs(JobSearchQuery(query=text, limit=limit)))
        except Exception as exc:
            err_console.print(f"[red]Get on Board search failed: {exc}[/red]")
            raise typer.Exit(GENERIC_FAILURE) from exc
    found = _unique_postings(found)

    if not found:
        console.print("No jobs found.")
        raise typer.Exit(SUCCESS)

    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError:
        candidate = None
    analyzer = RuleBasedJobAnalyzer()
    repo = JobRepository(session)
    table = Table(title="Get on Board results")
    table.add_column("ID")
    table.add_column("Company")
    table.add_column("Role")
    table.add_column("Where")
    table.add_column("Match")
    stored: list[str] = []
    scores: list[float] = []
    for raw in found:
        job = repo.upsert_external(raw)
        if candidate is not None:
            match = analyzer.analyze(candidate, job)
            repo.update_match_score(job.id, match.score)
            job.match_score = match.score
            scores.append(match.score)
        write_job_json(job, config.output_dir)
        stored.append(job.id)
        score = f"{job.match_score:.0f}%" if job.match_score is not None else "-"
        table.add_row(
            job.id,
            job.company[:24],
            job.title[:36],
            (job.location or "-")[:20],
            score,
        )
    console.print(table)
    console.print(
        f"Stored {len(stored)} jobs (portal getonbrd.com learned). "
        f"Next: [bold]jobbot jobs match {stored[0]}[/bold]"
    )
    if candidate is not None:
        _warn_if_matcher_blind(scores)


@torre_app.command("search")
def torre_search(
    query: Annotated[
        str | None,
        typer.Argument(help="Search query (default: the searches in your own profile)"),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max results per query (1-50)")] = 20,
    remote: Annotated[bool, typer.Option("--remote", help="Only remote opportunities")] = False,
    max_queries: Annotated[
        int,
        typer.Option("--max-queries", help="Profile searches to run when no query is given"),
    ] = _DEFAULT_MAX_QUERIES,
) -> None:
    """Search Torre (LATAM / remote) and store jobs locally."""
    from jobbot.adapters.torre.jobs import TorreJobSource
    from jobbot.jobs.sources import JobSearchQuery

    if limit < 1 or limit > 50:
        err_console.print("--limit must be between 1 and 50")
        raise typer.Exit(GENERIC_FAILURE)

    session, config = _session()
    queries = _search_queries(config, query, max_queries)

    narrator = _narrator()
    source = TorreJobSource(config)
    found: list[JobPosting] = []
    for text in queries:
        narrator.phase(Phase.RECEIVING_WORLD, f"Torre: {text}")
        try:
            found.extend(source.search_jobs(JobSearchQuery(query=text, limit=limit, remote=remote)))
        except Exception as exc:
            err_console.print(f"[red]Torre search failed: {exc}[/red]")
            raise typer.Exit(GENERIC_FAILURE) from exc
    found = _unique_postings(found)

    if not found:
        console.print("No jobs found.")
        raise typer.Exit(SUCCESS)

    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError:
        candidate = None
    analyzer = RuleBasedJobAnalyzer()
    repo = JobRepository(session)
    table = Table(title="Torre results")
    table.add_column("ID")
    table.add_column("Company")
    table.add_column("Role")
    table.add_column("Where")
    table.add_column("Match")
    stored: list[str] = []
    scores: list[float] = []
    for raw in found:
        job = repo.upsert_external(raw)
        if candidate is not None:
            match = analyzer.analyze(candidate, job)
            repo.update_match_score(job.id, match.score)
            job.match_score = match.score
            scores.append(match.score)
        write_job_json(job, config.output_dir)
        _learn_company_knowledge(config, job)
        stored.append(job.id)
        score = f"{job.match_score:.0f}%" if job.match_score is not None else "-"
        table.add_row(
            job.id,
            job.company[:24],
            job.title[:36],
            (job.location or "-")[:20],
            score,
        )
    console.print(table)
    console.print(
        f"Stored {len(stored)} jobs (portal torre.ai learned). "
        f"Next: [bold]jobbot jobs match {stored[0]}[/bold]"
    )
    if candidate is not None:
        _warn_if_matcher_blind(scores)


@empleospublicos_app.command("search")
def empleospublicos_search(
    query: Annotated[
        str | None,
        typer.Argument(help="Filter text (default: profile searches)"),
    ] = None,
    fixture: Annotated[
        Path | None,
        typer.Option(
            "--fixture",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Saved search JSON dump (required — live board is JS / often 403)",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max results (1-50)")] = 20,
    max_queries: Annotated[
        int,
        typer.Option("--max-queries", help="Profile searches when no query is given"),
    ] = _DEFAULT_MAX_QUERIES,
) -> None:
    """List Empleos Públicos concursos from a saved search dump and store leads."""
    from jobbot.adapters.empleospublicos.jobs import (
        EmpleosPublicosJobSource,
        EmpleosPublicosParseError,
    )
    from jobbot.jobs.sources import JobSearchQuery

    if fixture is None:
        err_console.print(
            "[red]--fixture PATH is required[/red] "
            "(Empleos Públicos search is fixture-first; live board is JS / often 403)."
        )
        raise typer.Exit(VALIDATION_FAILURE)
    if limit < 1 or limit > 50:
        err_console.print("--limit must be between 1 and 50")
        raise typer.Exit(GENERIC_FAILURE)

    session, config = _session()
    queries = _search_queries(config, query, max_queries)
    source = EmpleosPublicosJobSource(config, fixture=fixture.expanduser().resolve())
    found: list[JobPosting] = []
    for text in queries:
        try:
            found.extend(source.search_jobs(JobSearchQuery(query=text, limit=limit)))
        except EmpleosPublicosParseError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(GENERIC_FAILURE) from exc
    found = _unique_postings(found)
    if not found:
        console.print("No concursos matched.")
        raise typer.Exit(SUCCESS)

    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError:
        candidate = None
    analyzer = RuleBasedJobAnalyzer()
    repo = JobRepository(session)
    table = Table(title="Empleos Públicos (leads)")
    table.add_column("ID")
    table.add_column("Institución")
    table.add_column("Cargo")
    table.add_column("Región")
    table.add_column("Match")
    stored: list[str] = []
    scores: list[float] = []
    for raw in found:
        job = repo.upsert_external(raw)
        if candidate is not None:
            match = analyzer.analyze(candidate, job)
            repo.update_match_score(job.id, match.score)
            job.match_score = match.score
            scores.append(match.score)
        write_job_json(job, config.output_dir)
        _learn_company_knowledge(config, job)
        stored.append(job.id)
        score = f"{job.match_score:.0f}%" if job.match_score is not None else "-"
        table.add_row(
            job.id,
            job.company[:28],
            job.title[:36],
            (job.location or "-")[:20],
            score,
        )
    console.print(table)
    console.print(
        f"Stored {len(stored)} leads. Detail: "
        f"[bold]jobbot get URL --fixture ficha.html[/bold]. "
        f"Next: [bold]jobbot jobs match {stored[0]}[/bold]"
    )
    if candidate is not None:
        _warn_if_matcher_blind(scores)


@portals_app.command("list")
def portals_list() -> None:
    """List known recruitment portals."""
    from jobbot.portals.registry import default_portals_path, load_registry

    config = load_config()
    registry = load_registry(default_portals_path(config.root))
    if not registry.portals:
        console.print("No portals registered yet. Run linkedin sweep or portals add.")
        return
    table = Table(title="Portal registry")
    table.add_column("Domain")
    table.add_column("ATS")
    table.add_column("Registered")
    table.add_column("Last seen")
    for entry in registry.portals:
        table.add_row(
            entry.domain,
            entry.ats_kind.value,
            "yes" if entry.registered else "no",
            entry.last_seen.isoformat(timespec="seconds") if entry.last_seen else "—",
        )
    console.print(table)


@portals_app.command("detect")
def portals_detect(
    url: Annotated[str, typer.Argument(help="Job or ATS URL")],
) -> None:
    """Detect ATS kind for a URL."""
    from jobbot.portals.detect import detect_ats
    from jobbot.portals.registry import domain_from_url

    kind = detect_ats(url)
    console.print(f"domain={domain_from_url(url)}  ats_kind={kind.value}")


@portals_app.command("add")
def portals_add(
    url: Annotated[str, typer.Argument(help="ATS URL or domain")],
    registered: Annotated[
        bool,
        typer.Option("--registered/--not-registered", help="You have an account there"),
    ] = False,
    notes: Annotated[str | None, typer.Option("--notes")] = None,
) -> None:
    """Add or update a portal in data/portals.yaml."""
    from jobbot.portals.detect import detect_ats
    from jobbot.portals.registry import (
        default_portals_path,
        domain_from_url,
        load_registry,
        save_registry,
    )

    config = load_config()
    path = default_portals_path(config.root)
    registry = load_registry(path)
    domain = domain_from_url(url)
    kind = detect_ats(url if "://" in url else f"https://{url}")
    entry = registry.upsert(
        domain=domain,
        ats_kind=kind,
        registered=registered,
        notes=notes,
        example_url=url if "://" in url else None,
        seen=True,
    )
    save_registry(registry, path)
    console.print(
        f"[green]Upserted[/green] {entry.domain} ({entry.ats_kind.value}) "
        f"registered={entry.registered}"
    )


def _learn_form_from_apply(config: JobbotConfig, url: str, company: str | None) -> None:
    """While you fill the form, record what it asks. Best effort: never blocks the apply."""
    from jobbot.portals.form_learn import learn_form_html

    try:
        html = _fetch_public_html(url)
        if html is None:
            return
        form = learn_form_html(html, url=url, company=company)
        if not form.readable:
            console.print(f"[dim]Form questions: {form.evidence}[/dim]")
            return
        _store_form_knowledge(config, form)
    except Exception:  # noqa: BLE001 — learning is a bonus, applying is the task
        console.print("[dim]Form questions: could not be read this time.[/dim]")
        return
    questions = form.screening_questions()
    narrator = _narrator()
    narrator.phase(Phase.RECEIVING_WORLD, f"{len(form.fields)} campos del formulario")
    if questions:
        console.print(f"[dim]This form asks {len(questions)} question(s) beyond your data:[/dim]")
        for question in questions[:6]:
            console.print(f"[dim]  • {question}[/dim]")
        console.print("[dim]Feeds jobbot cv advise.[/dim]")


@portals_app.command("form-learn")
def portals_form_learn(
    url: Annotated[str, typer.Argument(help="Application form URL (public page)")],
    fixture: Annotated[
        Path | None,
        typer.Option("--fixture", help="Read a saved HTML instead of fetching the URL"),
    ] = None,
    company: Annotated[
        str | None,
        typer.Option("--company", help="Who the form belongs to"),
    ] = None,
    fetch: Annotated[
        bool,
        typer.Option("--fetch", help="Allow one polite GET of the URL (obeys robots.txt)"),
    ] = False,
    save: Annotated[bool, typer.Option("--save/--no-save", help="Store what it learned")] = True,
) -> None:
    """Learn what an application form asks. Reads only; never fills or submits."""
    from jobbot.portals.form_learn import learn_form_html

    config = load_config()
    if fixture is not None:
        path = fixture.expanduser().resolve()
        if not path.is_file():
            err_console.print(f"[red]Fixture not found: {path}[/red]")
            raise typer.Exit(VALIDATION_FAILURE)
        html = path.read_text(encoding="utf-8")
    elif fetch:
        fetched = _fetch_public_html(url)
        if fetched is None:
            raise typer.Exit(GENERIC_FAILURE)
        html = fetched
    else:
        err_console.print(
            "Reading a live form needs [bold]--fetch[/bold] (one GET, obeys robots.txt), "
            "or pass [bold]--fixture PATH[/bold] with saved HTML."
        )
        raise typer.Exit(VALIDATION_FAILURE)

    form = learn_form_html(html, url=url, company=company)
    _print_form_knowledge(form)
    if save and (form.readable or form.sso_providers):
        stored = _store_form_knowledge(config, form)
        console.print(f"Learned (candidate knowledge): {stored}")
    elif save:
        console.print("[yellow]Nothing stored: the form could not be read.[/yellow]")


def _fetch_public_html(url: str) -> str | None:
    """One polite GET, robots.txt respected. Returns None when we must not look."""
    from jobbot.companies.oneshot import RobotsPolicy, RobotsVerdict, UrllibFetcher

    fetcher = UrllibFetcher()
    verdict = RobotsPolicy(fetcher).verdict(url)
    if verdict is RobotsVerdict.DISALLOWED:
        err_console.print(f"[yellow]robots.txt disallows {url} — not fetched.[/yellow]")
        return None
    if verdict is RobotsVerdict.HOST_REFUSED:
        err_console.print(
            f"[yellow]{url} would not serve its robots.txt — staying out "
            "(unknown, not empty).[/yellow]"
        )
        return None
    result = fetcher.fetch(url)
    if not result.ok:
        err_console.print(f"[yellow]{url} answered HTTP {result.status}.[/yellow]")
        return None
    return result.html


def _print_form_knowledge(form: Any) -> None:
    from jobbot.portals.form_learn import FieldKind

    console.print(f"[bold]{form.url}[/bold]  ats={form.ats.value}")
    if form.sso_providers:
        names = ", ".join(form.sso_providers)
        console.print(f"Sign in with: {names}  [dim](you click; JobBot never starts OAuth)[/dim]")
    if not form.readable:
        console.print(f"[yellow]{form.evidence}[/yellow]")
        console.print("Tip: open the page, save the HTML, and pass it with --fixture.")
        return
    table = Table(title="What this form asks")
    table.add_column("Question")
    table.add_column("Kind")
    table.add_column("Req", justify="center")
    table.add_column("Choices")
    for field in form.fields:
        mark = "yes" if field.required else ""
        choices = ", ".join(field.options[:4]) if field.options else ""
        if field.kind is FieldKind.FILE and field.accepts:
            choices = " ".join(field.accepts)
        table.add_row(field.label[:52], field.kind.value, mark, choices[:34])
    console.print(table)
    questions = form.screening_questions()
    if questions:
        console.print(f"Questions your CV has to answer ({len(questions)}):")
        for question in questions:
            console.print(f"  • {question}")


def _store_form_knowledge(config: JobbotConfig, form: Any) -> Path:
    from jobbot.portals.form_learn import (
        default_form_knowledge_path,
        load_form_knowledge,
        save_form_knowledge,
        upsert_form,
    )

    path = default_form_knowledge_path(config.root)
    forms = upsert_form(load_form_knowledge(path), form)
    return save_form_knowledge(forms, path)


# ── companies (collaborative career-platform knowledge) ──────────────────────


def _narrator(quiet: bool = False) -> Narrator:
    """Phase narration for the long loops (dim lines, local stdout only)."""
    return Narrator(sink=lambda line: console.print(f"[dim]{line}[/dim]"), quiet=quiet)


def _companies_registry(config: JobbotConfig) -> tuple[CompanyRegistry, Path]:
    from jobbot.companies.registry import default_companies_path, load_companies

    path = default_companies_path(config.root)
    return load_companies(path), path


def _search_queries(config: JobbotConfig, query: str | None, max_queries: int) -> list[str]:
    """An explicit query runs alone; otherwise the profile's own searches.

    A default query hardcodes one career into the tool, and a headline can be a
    degree. The saved ``search_queries`` win; without them the set is derived from
    the experience, and only a profile with no experience falls back to its headline.
    """
    from jobbot.profile.search_queries import search_queries_for

    if query is not None:
        return [query]
    try:
        candidate = load_profile(config.profile_path)
    except Exception:  # noqa: BLE001 — no profile yet is a normal first run
        candidate = None
    queries: list[str] = []
    where = ""
    if candidate is not None:
        chosen = search_queries_for(candidate)
        queries = chosen.queries
        where = "saved in profile.yaml" if chosen.saved else "derived from your experience"
        if not queries:
            role = _headline_role(candidate)
            queries = [role] if role else []
            where = "from your headline"
    if not queries:
        err_console.print(
            "No query given and nothing in the profile to search for: pass the search terms, "
            "or add experience (or search_queries) to profile.yaml."
        )
        raise typer.Exit(GENERIC_FAILURE)
    picked = queries[: max(1, max_queries)]
    console.print(f"[dim]Queries {where}: {', '.join(picked)}[/dim]")
    if len(queries) > len(picked):
        console.print(f"[dim]{len(queries) - len(picked)} more left out (--max-queries).[/dim]")
    if where == "derived from your experience":
        console.print("[dim]Review and save them: jobbot profile queries --apply[/dim]")
    return picked


def _headline_role(candidate: Candidate) -> str | None:
    headline = (candidate.personal.headline or "").strip()
    if headline:
        return re.split(r"\s*[|/·–—]\s*", headline)[0].strip() or None
    return None


def _unique_postings(found: Sequence[JobPosting]) -> list[JobPosting]:
    """Several queries return the same vacancy; keep the first sighting."""
    seen: set[tuple[str, ...]] = set()
    unique: list[JobPosting] = []
    for job in found:
        if job.source_job_id:
            key: tuple[str, ...] = (job.source, job.source_job_id)
        elif job.url:
            key = (job.url,)
        else:
            key = (job.source, job.title.casefold(), job.company.casefold())
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def _warn_if_matcher_blind(scores: Sequence[float]) -> None:
    from jobbot.matching.scoring import blind_matcher_warning

    warning = blind_matcher_warning(scores)
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")


def _learn_company_knowledge(
    config: JobbotConfig,
    job: JobPosting,
    *,
    source: DiscoverySource = DiscoverySource.JOB_SOURCE,
    narrator: Narrator | None = None,
) -> None:
    """Record the company↔portal relation behind a stored job (candidate knowledge)."""
    from jobbot.companies.learn import learn_from_job

    result = learn_from_job(config, job, source=source)
    if result is None:
        return
    voice = narrator or _narrator()
    verb = "learned" if result.created else "confirmed"
    voice.note(
        f"{verb} {result.company_id}: {result.url} "
        f"({result.site_type.value}, ats={result.ats}) — candidate"
    )
    if result.conflict:
        err_console.print(f"[yellow]Contradiction:[/yellow] {result.conflict}")


@companies_app.command("detect")
def companies_detect(
    url: Annotated[str, typer.Argument(help="Career page, ATS or job URL")],
    resolve: Annotated[
        bool,
        typer.Option("--resolve/--no-resolve", help="Follow redirects before classifying"),
    ] = False,
) -> None:
    """Classify a URL (posting / career portal / ATS / redirect). Writes nothing."""
    from jobbot.companies.discovery import career_root_url, classify_url
    from jobbot.companies.urls import PrivateRouteRejected, company_hint_from_url

    try:
        found = classify_url(url, resolve=resolve)
    except PrivateRouteRejected as exc:
        err_console.print(f"[red]Not shareable company knowledge:[/red] {exc}")
        raise typer.Exit(VALIDATION_FAILURE) from exc
    table = Table(title="URL classification")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("url", found.url)
    table.add_row("domain", found.domain)
    table.add_row("site_type", found.site_type.value)
    table.add_row("ats", found.ats.value)
    table.add_row("career_root", career_root_url(found.url, found.ats))
    table.add_row("redirects_to", found.redirects_to or "—")
    table.add_row("evidence", found.evidence or "(none — ats stays unknown)")
    table.add_row("company_hint", company_hint_from_url(found.url))
    console.print(table)


@companies_app.command("learn")
def companies_learn(
    url: Annotated[str, typer.Argument(help="Career portal / ATS URL")],
    company: Annotated[
        str | None,
        typer.Option("--company", help="Company name (defaults to a hint from the URL)"),
    ] = None,
    country: Annotated[str | None, typer.Option("--country", help="ISO code, e.g. CL")] = None,
    resolve: Annotated[
        bool,
        typer.Option("--resolve/--no-resolve", help="Follow redirects before classifying"),
    ] = False,
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Promote straight to active (you confirm it is right)"),
    ] = False,
) -> None:
    """Register a company/portal relation you found yourself (candidate by default)."""
    from jobbot.companies.learn import learn_from_url
    from jobbot.companies.models import DiscoverySource
    from jobbot.companies.registry import save_companies
    from jobbot.companies.urls import PrivateRouteRejected, company_hint_from_url

    config = load_config()
    registry, path = _companies_registry(config)
    narrator = _narrator()
    narrator.phase(Phase.UPDATING_WORLD, "company career platforms")
    try:
        outcome = learn_from_url(
            registry,
            company=company or company_hint_from_url(url),
            url=url,
            source=DiscoverySource.USER_OBSERVATION,
            country=country,
            resolve=resolve,
        )
    except PrivateRouteRejected as exc:
        err_console.print(f"[red]Not shareable company knowledge:[/red] {exc}")
        raise typer.Exit(VALIDATION_FAILURE) from exc
    if outcome is None:
        console.print(
            "[yellow]Not company-specific[/yellow] (job board or unclear URL); nothing stored."
        )
        raise typer.Exit(SUCCESS)
    if outcome.conflict and outcome.site is None:
        err_console.print(f"[yellow]Duplicate:[/yellow] {outcome.conflict}")
        raise typer.Exit(SUCCESS)
    assert outcome.site is not None
    if verify:
        registry.promote(outcome.company.id, url=outcome.site.url)
    save_companies(registry, path)
    verb = "Added" if outcome.created_site else "Merged observation into"
    narrator.note(
        f"{verb} {outcome.site.url} → company={outcome.company.id} "
        f"type={outcome.site.site_type.value} ats={outcome.site.ats.value} "
        f"status={outcome.site.status.value} confidence={outcome.site.confidence}"
    )
    if outcome.conflict:
        err_console.print(f"[yellow]Contradiction:[/yellow] {outcome.conflict}")
    if not verify:
        console.print(
            f"Candidate stored. Confirm with: [bold]jobbot companies promote "
            f"{outcome.company.id} --site {outcome.site.url}[/bold]"
        )


@companies_app.command("list")
def companies_list(
    status: Annotated[
        str,
        typer.Option("--status", help="candidate | active | stale | rejected | all"),
    ] = "all",
) -> None:
    """List known companies and their career platforms."""
    from jobbot.companies.models import KnowledgeStatus
    from jobbot.companies.urls import display_url

    config = load_config()
    registry, path = _companies_registry(config)
    if not registry.companies:
        console.print(
            "No company knowledge yet. Try [bold]jobbot companies learn URL --company NAME[/bold] "
            "or [bold]jobbot companies discover data/companies-cl.example.yaml[/bold]."
        )
        return
    wanted: set[str] | None = None
    if status != "all":
        if status not in {s.value for s in KnowledgeStatus}:
            err_console.print(f"[red]Invalid --status {status!r}[/red]")
            raise typer.Exit(GENERIC_FAILURE)
        wanted = {status}
    table = Table(title=f"Company career platforms ({path.name})")
    table.add_column("Company", overflow="fold")
    table.add_column("Country")
    table.add_column("Career site", overflow="fold", min_width=24, ratio=3)
    table.add_column("Type", overflow="fold")
    table.add_column("ATS")
    table.add_column("Status")
    table.add_column("Conf.")
    rows = 0
    for record in registry.companies:
        for site in record.career_sites:
            if wanted is not None and site.status.value not in wanted:
                continue
            table.add_row(
                record.id,
                record.country or "—",
                display_url(site.url),
                site.site_type.value,
                site.ats.value,
                site.status.value,
                str(site.confidence),
            )
            rows += 1
    if rows == 0:
        console.print(f"No career sites with status {status!r}.")
        return
    console.print(table)


@companies_app.command("show")
def companies_show(
    company: Annotated[str, typer.Argument(help="Company id or name")],
) -> None:
    """Show one company with every portal, observation and contradiction."""
    config = load_config()
    registry, _ = _companies_registry(config)
    record = registry.find_company(company)
    if record is None:
        err_console.print(f"[red]Unknown company {company!r}[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    console.print(
        Panel(
            f"[bold]{record.name}[/bold] ({record.id})\n"
            f"country: {record.country or '—'}  sector: {record.sector or '—'}\n"
            f"domains: {', '.join(record.domains) or '—'}",
            title="Company",
        )
    )
    for site in record.career_sites:
        verified = site.last_verified.date().isoformat() if site.last_verified else "—"
        lines = [
            f"type: {site.site_type.value}   ats: {site.ats.value}   status: {site.status.value}",
            f"first_seen: {site.first_seen.date().isoformat()}   last_verified: {verified}",
        ]
        if site.reached_from:
            lines.append(f"reached_from: {site.reached_from}")
        lines.append(f"confidence: {site.confidence} independent source(s)")
        for obs in site.observations:
            lines.append(
                f"  · {obs.checked_at.date().isoformat()} {obs.source.value}: "
                f"{obs.evidence or '(no technical evidence)'}"
            )
        for conflict in site.conflicts:
            lines.append(f"  ! {conflict}")
        console.print(Panel("\n".join(lines), title=site.url))


@companies_app.command("recon")
def companies_recon(
    company: Annotated[str, typer.Argument(help="Company id or name already in the registry")],
    apply_changes: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Write ATS observation + form questions (default: dry-run report)",
        ),
    ] = False,
    fixture: Annotated[
        Path | None,
        typer.Option(
            "--fixture",
            help="Local HTML captured after entering (tests / offline)",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to Chrome; read the open page (HITL login first)"),
    ] = None,
    site_url: Annotated[
        str | None,
        typer.Option("--site", help="Which career site, when the company has several"),
    ] = None,
    navigate: Annotated[
        bool,
        typer.Option(
            "--navigate/--no-navigate",
            help="With --cdp, open the career URL before capturing HTML",
        ),
    ] = True,
) -> None:
    """Learn ATS markers and form questions from a page you entered (issue #45)."""
    from jobbot.companies.recon import ReconError, plan_recon

    config = load_config()
    html: str | None = None
    if fixture is not None:
        html = fixture.read_text(encoding="utf-8")
    elif cdp is not None:
        html = _capture_recon_html(config, company, cdp=cdp, site_url=site_url, navigate=navigate)
    elif apply_changes:
        err_console.print(
            "[red]--apply needs HTML: pass --fixture PATH or --cdp URL "
            "(after you are signed in).[/red]"
        )
        raise typer.Exit(VALIDATION_FAILURE)

    try:
        report = plan_recon(
            config,
            company,
            html=html,
            site_url=site_url,
            apply=apply_changes,
        )
    except ReconError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    table = Table(title=f"Portal recon — {report.company_name}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("url", report.url)
    table.add_row("ats before", report.previous_ats.value)
    table.add_row("ats detected", report.detected_ats.value)
    table.add_row("evidence", report.evidence or "(none)")
    table.add_row("account", report.account_need.value)
    table.add_row(
        "form",
        (
            f"{report.form_field_count} field(s)"
            if report.form.readable
            else f"unreadable: {report.form.evidence}"
        ),
    )
    table.add_row(
        "wrote",
        f"companies={report.wrote_companies} forms={report.wrote_forms}",
    )
    console.print(table)
    if report.field_labels:
        console.print("Fields: " + ", ".join(report.field_labels[:12]))
        if len(report.field_labels) > 12:
            console.print(f"  … +{len(report.field_labels) - 12} more")
    if report.conflict:
        err_console.print(f"[yellow]Contradiction:[/yellow] {report.conflict}")
    if report.refused:
        err_console.print(f"[yellow]Not stored:[/yellow] {report.refused}")
    console.print(report.hint)
    if not apply_changes:
        console.print(
            "Dry-run. Re-run with [bold]--apply[/bold] plus [bold]--fixture[/bold] or "
            "[bold]--cdp[/bold] to store evidence (#45)."
        )


def _capture_recon_html(
    config: JobbotConfig,
    company: str,
    *,
    cdp: str,
    site_url: str | None,
    navigate: bool,
) -> str:
    """Attach to the user's Chrome and read page HTML (no login automation)."""
    from jobbot.browser.cdp import resolve_cdp_url
    from jobbot.browser.session import BrowserSession
    from jobbot.companies.recon import resolve_recon_site
    from jobbot.companies.registry import default_companies_path, load_companies

    registry = load_companies(default_companies_path(config.root))
    _record, site = resolve_recon_site(registry, company, site_url=site_url)
    cdp_url = resolve_cdp_url(cdp)
    if not cdp_url:
        err_console.print("[red]No CDP URL — pass --cdp or set JOBBOT_CDP_URL[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    with BrowserSession(
        profile_dir=config.root / "browser-data" / "companies-recon-cdp",
        headless=False,
        cdp_url=cdp_url,
        debug_root=config.output_dir / "debug",
    ) as browser:
        page = browser.page
        if navigate:
            page.goto(site.url, wait_until="domcontentloaded")
            console.print(f"Opened {site.url} — sign in / enter if needed, then recon reads HTML.")
        return str(page.content())


@companies_app.command("signup")
def companies_signup(
    company: Annotated[str, typer.Argument(help="Company id or name (already in the registry)")],
    open_page: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open the portal in your browser"),
    ] = True,
    site_url: Annotated[
        str | None,
        typer.Option("--site", help="Which career site, when the company has several"),
    ] = None,
    apply_fill: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Fill known fields from profile.yaml (stops before password/terms/submit)",
        ),
    ] = False,
    cdp: Annotated[
        str | None,
        typer.Option("--cdp", help="Attach to Chrome via CDP (e.g. http://127.0.0.1:9222)"),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the fill confirmation (with --apply)"),
    ] = False,
) -> None:
    """Sheet of what registering asks; --apply fills known fields (HITL create).

    With --apply, fills fields profile.yaml already answers and may attach the built
    CV. Stops before password, terms, CAPTCHA/2FA, and final create/submit.
    Never creates the account.
    """
    from jobbot.companies.signup import (
        AccountNeed,
        screening_to_prepare,
        signup_sheet,
        signup_target,
    )
    from jobbot.portals.form_learn import default_form_knowledge_path, load_form_knowledge

    config = load_config()
    registry, _ = _companies_registry(config)
    record = registry.find_company(company)
    if record is None:
        err_console.print(f"[red]Unknown company {company!r}[/red]")
        err_console.print("Learn it first: [bold]jobbot companies learn URL --company NAME[/bold]")
        raise typer.Exit(VALIDATION_FAILURE)
    sites = record.career_sites
    if site_url:
        sites = [site for site in sites if site.url == site_url]
    if not sites:
        err_console.print(f"[red]No career site stored for {record.name}[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    site = sites[0]
    target = signup_target(site)

    try:
        candidate = load_profile(config.profile_path)
    except ProfileLoadError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc

    forms = load_form_knowledge(default_form_knowledge_path(config.root))
    form = next(
        (f for f in forms if f.url.startswith(site.url) or site.url.startswith(f.url)),
        None,
    )

    console.print(
        Panel(
            f"[bold]{record.name}[/bold] ({record.id})\n"
            f"{target.url}\n"
            f"ats: {site.ats.value}   account: {target.need.value}\n\n{target.note}",
            title="Registration (you complete it)",
        )
    )
    table = Table(title="Have this at hand" + ("  (from the form we saw)" if form else ""))
    table.add_column("The portal asks")
    table.add_column("Your profile answers")
    table.add_column("From")
    for item in signup_sheet(candidate, form=form):
        value = item.value or "[yellow]you decide[/yellow]"
        table.add_row(item.label[:40], value[:46], item.source[:34])
    console.print(table)
    for question in screening_to_prepare(form):
        console.print(f"  • decide beforehand: {question}")
    console.print(
        "[dim]JobBot does not create the account, does not set a password and does not "
        "accept terms for you.[/dim]"
    )
    if target.need is AccountNeed.NOT_NEEDED:
        console.print("You may not need an account at all — check before registering.")

    if apply_fill:
        _companies_signup_apply(
            config=config,
            candidate=candidate,
            target_url=target.url,
            need=target.need,
            form=form,
            cdp=cdp,
            yes=yes,
        )
        return

    if open_page:
        from jobbot.adapters.ats.apply import open_ats_in_browser

        console.print(open_ats_in_browser(target.url))


def _companies_signup_apply(
    *,
    config: JobbotConfig,
    candidate: Candidate,
    target_url: str,
    need: Any,
    form: Any,
    cdp: str | None,
    yes: bool,
) -> None:
    from jobbot.adapters.ats.signup_fill import build_fill_plan, fill_signup_with_session
    from jobbot.adapters.getonboard.cv_upload import resolve_base_cv
    from jobbot.browser.cdp import resolve_cdp_url
    from jobbot.browser.session import BrowserSession

    cv_path = resolve_base_cv(config.output_dir)
    plan = build_fill_plan(candidate, target_url, need, form=form, cv_path=cv_path)

    console.print("\n[bold]Fill plan[/bold] (will not create/submit):")
    fillable = ", ".join(plan.fields_to_fill) if plan.fields_to_fill else "none"
    console.print(f"  • Will fill: {fillable}")
    manual = ", ".join(plan.needs_manual[:4])
    if len(plan.needs_manual) > 4:
        manual += "…"
    console.print(f"  • Requires you: {manual}")
    if plan.can_attach_cv:
        console.print(f"  • Will attach CV: {plan.cv_path}")
    else:
        console.print("  • CV: not attached (build one with [bold]jobbot cv build[/bold])")

    if not yes and not typer.confirm(
        "Fill known profile fields now? (password, terms, CAPTCHA/2FA and create "
        "stay with you — JobBot does not submit)",
        default=False,
    ):
        console.print("Skipped fill. Sheet above still stands.")
        raise typer.Exit(SUCCESS)

    cdp_url = resolve_cdp_url(cdp)
    with BrowserSession(
        profile_dir=config.root / "browser-data" / "signup",
        headless=False,
        cdp_url=cdp_url,
        debug_root=config.output_dir / "debug",
    ) as session:
        result = fill_signup_with_session(
            session,
            target_url,
            candidate,
            form=form,
            cv_path=cv_path,
            need=need,
            confirm_submit=False,
        )

    console.print(
        f"\n[green]Filled[/green] {len(result.filled_fields)} field(s)"
        + (" · CV attached" if result.attached else "")
    )
    for field in result.filled_fields:
        console.print(f"  • {field}")
    console.print(f"[bold]Stopped at:[/bold] {result.stopped_at}")
    console.print(
        "[yellow]Complete password, terms, CAPTCHA/2FA and create/submit yourself.[/yellow]"
    )
    assert result.submitted is False


@companies_app.command("promote")
def companies_promote(
    company: Annotated[str, typer.Argument(help="Company id or name")],
    site: Annotated[
        str | None,
        typer.Option("--site", help="Promote only this career URL"),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Promote candidate knowledge to active (the only way it becomes truth)."""
    from jobbot.companies.models import KnowledgeStatus
    from jobbot.companies.registry import save_companies
    from jobbot.companies.urls import canonical_key

    config = load_config()
    registry, path = _companies_registry(config)
    record = registry.find_company(company)
    if record is None:
        err_console.print(f"[red]Unknown company {company!r}[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    targets = [
        s
        for s in record.career_sites
        if s.status != KnowledgeStatus.REJECTED and (site is None or s.key == canonical_key(site))
    ]
    if not targets:
        console.print("Nothing to promote.")
        raise typer.Exit(SUCCESS)
    narrator = _narrator()
    narrator.phase(Phase.RECEIVING_WORLD, f"company {record.id}", may_ask=not yes)
    promoted = 0
    for target in targets:
        narrator.note(
            f"{target.url} type={target.site_type.value} ats={target.ats.value} "
            f"evidence={target.observations[-1].evidence if target.observations else '—'}"
        )
        if not yes and not typer.confirm(f"Promote {target.url} to active?", default=True):
            continue
        registry.promote(record.id, url=target.url)
        promoted += 1
    if promoted:
        save_companies(registry, path)
    console.print(f"[green]Promoted[/green] {promoted} career site(s) → {path}")


@companies_app.command("reject")
def companies_reject(
    company: Annotated[str, typer.Argument(help="Company id or name")],
    site: Annotated[str, typer.Option("--site", help="Career URL to reject")],
) -> None:
    """Mark a discovered portal as wrong so it stops coming back."""
    from jobbot.companies.registry import save_companies

    config = load_config()
    registry, path = _companies_registry(config)
    rejected = registry.reject(company, url=site)
    if rejected is None:
        err_console.print(f"[red]No stored site {site} for {company!r}[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    save_companies(registry, path)
    console.print(f"[yellow]Rejected[/yellow] {rejected.url}")


@companies_app.command("discover")
def companies_discover(
    seeds_file: Annotated[Path, typer.Argument(help="YAML list of companies to seed")],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Where to write candidates (default: output/discovery/…)"),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Max companies to probe")] = None,
    search_results: Annotated[
        Path | None,
        typer.Option("--search-results", help="YAML with already-obtained web search hits"),
    ] = None,
    delay: Annotated[
        float,
        typer.Option("--delay", help="Seconds between HTTP requests (be polite)"),
    ] = 1.0,
) -> None:
    """One-shot: seed candidate career portals for a company list (never canonical)."""
    from jobbot.companies.oneshot import (
        CompanyPortalCandidate,
        CompanySeed,
        UrllibFetcher,
        group_candidates,
        load_search_hits,
        load_seeds,
        run_oneshot,
        write_candidates,
    )
    from jobbot.companies.registry import generated_candidates_path

    config = load_config()
    path = seeds_file.expanduser().resolve()
    if not path.is_file():
        err_console.print(f"[red]Seed file not found: {path}[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    try:
        seeds = load_seeds(path)
    except (ValueError, yaml.YAMLError) as exc:
        err_console.print(f"[red]Invalid seed file:[/red] {exc}")
        raise typer.Exit(VALIDATION_FAILURE) from exc
    hits = load_search_hits(search_results.expanduser().resolve()) if search_results else None

    narrator = _narrator()
    narrator.phase(Phase.EXPLORING, f"{len(seeds)} companies")

    def on_company(seed: CompanySeed, found: list[CompanyPortalCandidate]) -> None:
        narrator.note(f"{seed.name}: {len(found)} candidate portal(s)")

    report = run_oneshot(
        seeds,
        UrllibFetcher(delay=delay),
        search_hits=hits,
        limit=limit,
        on_company=on_company,
    )
    target = out.expanduser().resolve() if out else generated_candidates_path(config.output_dir)
    write_candidates(report.candidates, target)
    narrator.phase(Phase.SURFACING, f"{len(report.candidates)} portal(s)")
    narrator.outcome(
        ExplorationOutcome(
            found=len({candidate.company_id for candidate in report.candidates}),
            refused=len(report.companies_blocked),
            unknown=len(report.companies_without_portal),
        )
    )
    console.print(
        f"Probed {report.companies_seen} companies with {report.requests_made} requests → "
        f"{len(report.candidates)} candidate portal(s)."
    )
    groups = group_candidates(report.candidates)
    if rows := groups.rows():
        table = Table(title="Portals found, grouped")
        table.add_column("Axis")
        table.add_column("Group")
        table.add_column("Portals", justify="right")
        table.add_column("Companies", justify="right")
        for axis, label, portals, companies in rows:
            table.add_row(axis, label, str(portals), str(companies))
        console.print(table)
    if report.companies_robots_skipped:
        console.print(
            "[yellow]Skipped by robots.txt (omitted, not absent):[/yellow] "
            + ", ".join(report.companies_robots_skipped[:10])
        )
    if report.companies_without_portal:
        console.print(
            "No public portal found for: " + ", ".join(report.companies_without_portal[:10])
        )
    if report.companies_blocked:
        console.print(
            "[yellow]Refused our requests (unknown, not absent):[/yellow] "
            + ", ".join(report.companies_blocked[:10])
        )
        console.print("Give those a hand: [bold]companies learn URL --company NAME[/bold]")
    console.print(f"Candidates (not truth yet): {target}")
    console.print(f"Next: [bold]jobbot companies import {target}[/bold]")


@companies_app.command("import")
def companies_import(
    candidates_file: Annotated[Path, typer.Argument(help="Generated candidates YAML")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Import without asking")] = False,
) -> None:
    """Merge oneshot candidates into the registry (still candidate, never active)."""
    from jobbot.companies.oneshot import CompanyPortalCandidate, import_candidates, load_candidates
    from jobbot.companies.registry import save_companies

    config = load_config()
    path = candidates_file.expanduser().resolve()
    if not path.is_file():
        err_console.print(f"[red]Candidates file not found: {path}[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    candidates = load_candidates(path)
    registry, registry_path = _companies_registry(config)
    narrator = _narrator()
    narrator.phase(Phase.UPDATING_WORLD, f"{len(candidates)} candidates", may_ask=not yes)

    def confirm(candidate: CompanyPortalCandidate) -> bool:
        narrator.note(
            f"{candidate.company}: {candidate.career_url} "
            f"type={candidate.site_type.value} ats={candidate.ats.value} "
            f"evidence={candidate.evidence or '—'}"
        )
        if yes:
            return True
        return typer.confirm("Keep as candidate knowledge?", default=True)

    outcomes = import_candidates(registry, candidates, confirm=confirm)
    save_companies(registry, registry_path)
    created = sum(1 for o in outcomes if o.created_site)
    merged = sum(1 for o in outcomes if o.merged)
    conflicts = [o.conflict for o in outcomes if o.conflict]
    console.print(
        f"[green]Imported[/green] {created} new, merged {merged} into existing → {registry_path}"
    )
    for conflict in conflicts:
        err_console.print(f"[yellow]Conflict:[/yellow] {conflict}")
    console.print("Promote what you verified: [bold]jobbot companies promote COMPANY[/bold]")


@companies_app.command("export")
def companies_export(
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Destination file (default: output/discovery/…)"),
    ] = None,
) -> None:
    """Write a shareable snapshot: active entries only, no candidate PII."""
    from jobbot.companies.registry import shareable_payload, shared_export_path

    config = load_config()
    registry, _ = _companies_registry(config)
    payload = shareable_payload(registry)
    target = out.expanduser().resolve() if out else shared_export_path(config.output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    companies = payload.get("companies")
    count = len(companies) if isinstance(companies, list) else 0
    console.print(f"[green]Exported[/green] {count} company(ies) with active portals → {target}")


@companies_app.command("sites")
def companies_sites(
    include_candidates: Annotated[
        bool,
        typer.Option("--include-candidates", help="Also list unverified knowledge"),
    ] = False,
) -> None:
    """Career portals reusable by later job discovery."""
    from jobbot.companies.registry import active_career_sites

    config = load_config()
    registry, _ = _companies_registry(config)
    pairs = active_career_sites(registry, include_candidates=include_candidates)
    if not pairs:
        console.print("No reusable career sites yet (promote some candidates first).")
        return
    for record, site in pairs:
        console.print(f"{record.id}\t{site.ats.value}\t{site.url}")


@browser_app.command("chrome-debug")
def browser_chrome_debug(
    port: Annotated[int, typer.Option("--port", help="Remote debugging port")] = 9222,
    site: Annotated[
        str,
        typer.Option("--site", help="indeed | linkedin | gmail | getonboard (profile + start URL)"),
    ] = "indeed",
    launch: Annotated[
        bool,
        typer.Option("--launch/--print-only", help="Launch Chrome (default) or only print argv"),
    ] = True,
) -> None:
    """Open a Chromium browser (Chrome/Edge/Brave) with CDP for manual challenges.

    Automatically detects and launches any available Chromium-based browser:
    Chrome, Microsoft Edge, Brave, or Chromium.
    """
    from jobbot.browser.background import launch_detached
    from jobbot.browser.cdp import cdp_http_url, chrome_debug_argv
    from jobbot.browser.sessions import (
        KNOWN_SITES,
        ProfileBusyError,
        ensure_profile_free,
        site_spec,
    )

    config = load_config()
    spec = site_spec(site)
    if spec is None:
        known = "|".join(KNOWN_SITES)
        err_console.print(f"[red]Unknown --site {site!r} (use {known})[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    site_key = spec.site
    profile_dir = config.root / "browser-data" / spec.cdp_profile
    start_url = spec.start_url
    tip = spec.next_command.format(cdp=cdp_http_url(port))
    profile_dir.mkdir(parents=True, exist_ok=True)
    argv = chrome_debug_argv(profile_dir=profile_dir, port=port, start_url=start_url)
    url = cdp_http_url(port)
    console.print("[bold]HITL Chrome (no CAPTCHA bypass)[/bold]")
    console.print("1. Close any JobBot-automated Chrome/Chromium windows for this site.")
    console.print(f"2. In this Chrome window, log in / complete challenges for {site_key}.")
    console.print(f"3. Then: [bold]{tip}[/bold]")
    console.print("Command:")
    console.print(" ".join(argv))
    if not launch:
        return
    try:
        ensure_profile_free(profile_dir)
    except ProfileBusyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc
    launch_detached(argv)
    console.print(f"[green]Launched[/green] Chrome with CDP at {url}")
    console.print(background_notice(start_url, where="una ventana nueva de Chrome"))


@browser_app.command("sessions")
def browser_sessions(
    site: Annotated[
        list[str] | None,
        typer.Option("--site", help="Limit to these sites (repeatable)"),
    ] = None,
    port: Annotated[
        list[int] | None,
        typer.Option("--port", help="Debugging ports to probe (repeatable)"),
    ] = None,
) -> None:
    """Report which browser sessions JobBot can reach (read-only; never attaches)."""
    from jobbot.browser.sessions import DEFAULT_PORTS, SessionStatus, inspect_sessions

    config = load_config()
    states = inspect_sessions(
        config.root,
        sites=site or None,
        ports=tuple(port) if port else DEFAULT_PORTS,
    )
    table = Table(title="Browser sessions")
    table.add_column("Site")
    table.add_column("Status")
    table.add_column("Endpoint")
    table.add_column("Evidence")
    for state in states:
        colour = {
            SessionStatus.READY: "green",
            SessionStatus.NEEDS_LOGIN: "yellow",
            SessionStatus.PROFILE_BUSY: "red",
            SessionStatus.WRONG_ACCOUNT: "red",
        }.get(state.status, "white")
        table.add_row(
            state.site,
            f"[{colour}]{state.status.value}[/{colour}]",
            state.cdp_url or "-",
            state.evidence,
        )
    console.print(table)
    for state in states:
        if state.hint:
            console.print(f"  {state.site}: {state.hint}")


@browser_app.command("login")
def browser_login(
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Open the next portal that still needs you (you sign in)"),
    ] = False,
    port: Annotated[
        int | None,
        typer.Option("--port", help="Debugging port when opening the next portal"),
    ] = None,
) -> None:
    """List permanent, active and candidate portals that still need you to sign in."""
    from jobbot.browser.background import launch_detached
    from jobbot.browser.cdp import chrome_debug_argv
    from jobbot.browser.login_plan import plan_logins
    from jobbot.browser.sessions import ProfileBusyError, SessionStatus, ensure_profile_free

    config = load_config()
    plan = plan_logins(config.root)
    table = Table(title="Login tour")
    table.add_column("Bucket")
    table.add_column("Portal")
    table.add_column("Status")
    table.add_column("Account")
    for row in plan.rows:
        colour = {
            SessionStatus.READY: "green",
            SessionStatus.NEEDS_LOGIN: "yellow",
            SessionStatus.PROFILE_BUSY: "red",
            SessionStatus.WRONG_ACCOUNT: "red",
        }.get(row.status, "white")
        table.add_row(
            row.bucket.value,
            row.name,
            f"[{colour}]{row.status.value}[/{colour}]",
            row.need.value,
        )
    console.print(table)
    for row in plan.rows:
        if row.status is not SessionStatus.READY and row.hint:
            console.print(f"  {row.name}: {row.hint}")

    gap = plan.next_to_open()
    if not apply:
        if gap is None:
            console.print(
                "Nothing left to open. A company host is still unknown until you are signed in."
            )
        else:
            console.print(
                "Dry-run. Re-run with [bold]--apply[/bold] to open the next portal. "
                "You type the password; JobBot never does."
            )
        return
    if gap is None:
        console.print("Nothing left to open.")
        return
    from jobbot.browser.sessions import refuse_to_open

    reason = refuse_to_open(gap.status, gap.evidence)
    if reason:
        err_console.print(f"[red]{reason}[/red]")
        if gap.hint and gap.status is not SessionStatus.WRONG_ACCOUNT:
            err_console.print(gap.hint)
        raise typer.Exit(GENERIC_FAILURE)

    profile_dir = config.root / "browser-data" / gap.profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    chosen = port if port is not None else plan.port
    try:
        argv = chrome_debug_argv(profile_dir=profile_dir, port=chosen, start_url=gap.url)
    except FileNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc
    try:
        ensure_profile_free(profile_dir)
    except ProfileBusyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(GENERIC_FAILURE) from exc
    launch_detached(argv)
    console.print(background_notice(gap.url, where="una ventana nueva de Chrome"))
    console.print(
        f"[green]Opened[/green] {gap.name}. Sign in yourself (password, CAPTCHA, 2FA). "
        "Then re-run [bold]jobbot browser login[/bold]."
    )


# ── ops (local failure observability) ────────────────────────────────────────


@ops_app.command("failures")
def ops_failures(
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter: new|triaged|fixed|wontfix (default: new)"),
    ] = "new",
    all_statuses: Annotated[
        bool,
        typer.Option("--all", help="List all statuses"),
    ] = False,
    limit: Annotated[int, typer.Option("--limit", help="Max rows")] = 50,
) -> None:
    """List stored failures (grouped by fingerprint)."""
    from jobbot.ops.failures import STATUSES, group_by_fingerprint, list_failures

    if all_statuses:
        status_filter: str | None = None
    else:
        status_filter = status
        if status_filter is not None and status_filter not in STATUSES:
            err_console.print(f"[red]Invalid --status {status_filter!r}[/red]")
            raise typer.Exit(GENERIC_FAILURE)

    session, _ = _session()
    records = list_failures(session, status=status_filter, limit=limit)
    if not records:
        console.print("No failures stored.")
        raise typer.Exit(SUCCESS)

    table = Table(title="ops failures")
    table.add_column("id")
    table.add_column("fp")
    table.add_column("n")
    table.add_column("component")
    table.add_column("exit")
    table.add_column("status")
    table.add_column("message")
    for fingerprint, group in group_by_fingerprint(records):
        head = group[0]
        msg = head.message.splitlines()[0] if head.message else head.error_class
        if len(msg) > 60:
            msg = msg[:57] + "…"
        table.add_row(
            head.id,
            fingerprint,
            str(len(group)),
            head.component,
            str(head.exit_code),
            head.status,
            msg,
        )
    console.print(table)
    console.print("Detail: [bold]jobbot ops failure show Fxxxx[/bold]")


@ops_failure_app.command("show")
def ops_failure_show(
    failure_id: Annotated[str, typer.Argument(help="Failure id, e.g. F0001")],
) -> None:
    """Show one failure record."""
    from jobbot.ops.failures import get_failure

    session, config = _session()
    record = get_failure(session, failure_id)
    if record is None:
        err_console.print(f"[red]Unknown failure {failure_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    console.print(
        Panel(
            f"[bold]{record.id}[/bold]  fingerprint={record.fingerprint}\n"
            f"component={record.component}  exit={record.exit_code}  "
            f"status={record.status}\n"
            f"command: {record.command}\n"
            f"error: {record.error_class}: {record.message}\n"
            f"ts: {record.ts.isoformat()}\n"
            f"issue: {record.issue_url or '(none)'}\n\n"
            f"context:\n{json.dumps(record.context, indent=2, ensure_ascii=False)}\n\n"
            f"traceback:\n{record.traceback or '(none)'}",
            title="failure",
        )
    )
    mirror = config.output_dir / "ops" / "failures" / f"{record.id}.json"
    if mirror.is_file():
        console.print(f"Mirror: {mirror}")


@ops_failure_app.command("triage")
def ops_failure_triage(
    failure_id: Annotated[str, typer.Argument(help="Failure id, e.g. F0001")],
    status: Annotated[
        str,
        typer.Option("--status", help="new|triaged|fixed|wontfix"),
    ] = "triaged",
) -> None:
    """Update failure status after review / fix."""
    from jobbot.ops.failures import STATUSES, mark_status

    if status not in STATUSES:
        err_console.print(f"[red]Invalid --status {status!r}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    session, _ = _session()
    record = mark_status(session, failure_id, status)
    if record is None:
        err_console.print(f"[red]Unknown failure {failure_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    console.print(f"[green]{record.id}[/green] → status={record.status}")


@ops_failure_app.command("issue")
def ops_failure_issue(
    failure_id: Annotated[str, typer.Argument(help="Failure id, e.g. F0001")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation"),
    ] = False,
) -> None:
    """Create a GitHub issue from a failure (HITL; uses gh). Never auto-fires on crash."""
    import subprocess

    from jobbot.ops.failures import get_failure, issue_body, issue_title, mark_status

    session, _ = _session()
    record = get_failure(session, failure_id)
    if record is None:
        err_console.print(f"[red]Unknown failure {failure_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    if record.issue_url:
        console.print(f"Already linked: {record.issue_url}")
        raise typer.Exit(SUCCESS)

    title = issue_title(record)
    body = issue_body(record)
    console.print(Panel(f"{title}\n\n{body}", title="proposed GitHub issue"))
    if not yes and not typer.confirm("Create GitHub issue with gh?"):
        console.print("Aborted.")
        raise typer.Exit(SUCCESS)

    proc = subprocess.run(  # noqa: S603
        ["gh", "issue", "create", "--title", title, "--body", body],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err_console.print(f"[red]gh failed:[/red] {proc.stderr or proc.stdout}")
        raise typer.Exit(GENERIC_FAILURE)
    url = (proc.stdout or "").strip()
    mark_status(session, failure_id, "triaged", issue_url=url or None)
    console.print(f"[green]Created[/green] {url or '(see gh output)'}")


@ops_failure_app.command("work")
def ops_failure_work(
    failure_id: Annotated[str, typer.Argument(help="Failure id, e.g. F0001")],
    apply_changes: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="HITL: create issue if needed, git fetch + checkout -b; never commit/push/PR",
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    open_issue: Annotated[
        bool,
        typer.Option(
            "--issue/--no-issue",
            help="With --apply, also run ops failure issue when none is linked",
        ),
    ] = True,
) -> None:
    """Print (or open) the bash maintainer lane: issue → branch → PR → retry (#46)."""
    import subprocess

    from jobbot.ops.failure_work import branch_name, work_script
    from jobbot.ops.failures import get_failure

    session, _ = _session()
    record = get_failure(session, failure_id)
    if record is None:
        err_console.print(f"[red]Unknown failure {failure_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)

    script = work_script(record)
    console.print(Panel(script, title=f"bash lane · {record.id} → {branch_name(record)}"))

    if not apply_changes:
        console.print(
            "Dry-run (script only). Re-run with [bold]--apply[/bold] to open the lane: "
            "issue (HITL) + [bold]git fetch --all[/bold] + [bold]checkout -b[/bold]. "
            "Commit / push / PR stay bash you run yourself."
        )
        raise typer.Exit(SUCCESS)

    if not yes and not typer.confirm(
        f"Open lane for {record.id} (issue if needed + branch {branch_name(record)})?",
        default=False,
    ):
        console.print("Aborted.")
        raise typer.Exit(SUCCESS)

    if open_issue and not record.issue_url:
        # Reuse the same HITL path as `ops failure issue` (reviews body).
        ops_failure_issue(failure_id, yes=yes)
        record = get_failure(session, failure_id) or record

    def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        console.print(f"$ {' '.join(argv)}")
        return subprocess.run(argv, check=False, capture_output=True, text=True)  # noqa: S603

    fetch = _run(["git", "fetch", "--all"])
    if fetch.returncode != 0:
        err_console.print(f"[red]git fetch failed:[/red] {fetch.stderr or fetch.stdout}")
        raise typer.Exit(GENERIC_FAILURE)

    base = _run(["git", "checkout", "develop"])
    if base.returncode != 0:
        console.print("[yellow]develop missing — trying main[/yellow]")
        base = _run(["git", "checkout", "main"])
    if base.returncode != 0:
        err_console.print(f"[red]git checkout base failed:[/red] {base.stderr or base.stdout}")
        raise typer.Exit(GENERIC_FAILURE)

    pull = _run(["git", "pull", "--ff-only"])
    if pull.returncode != 0:
        err_console.print(f"[yellow]git pull:[/yellow] {pull.stderr or pull.stdout}")

    name = branch_name(record)
    created = _run(["git", "checkout", "-b", name])
    if created.returncode != 0:
        switched = _run(["git", "checkout", name])
        if switched.returncode != 0:
            err_console.print(f"[red]branch failed:[/red] {created.stderr or created.stdout}")
            raise typer.Exit(GENERIC_FAILURE)
        console.print(f"[dim]branch {name} already existed — checked out[/dim]")

    from jobbot.ops.failure_work import original_command

    console.print(
        f"[green]Lane open[/green] on {name}. Next (bash):\n"
        "  # regression test that fails first, then fix\n"
        "  uv run ruff check . && uv run mypy src && uv run pytest\n"
        '  git add -A && git commit -m "fix: …" && git push -u origin HEAD\n'
        "  gh pr create --fill\n"
        f"  uv run jobbot ops failure triage {record.id} --status fixed\n"
        "  git fetch --all\n"
        f"  {original_command(record)}"
    )


@ops_symptom_app.command("note")
def ops_symptom_note(
    intent: Annotated[
        str,
        typer.Argument(help="What keeps returning in vibecode (will be redacted)"),
    ],
    area: Annotated[
        str,
        typer.Option("--area", help="cv|jobs|portals|matching|ops|nlp|companies|profile|other"),
    ] = "other",
    title: Annotated[
        str | None,
        typer.Option("--title", help="Short label (redacted)"),
    ] = None,
    rule: Annotated[
        str,
        typer.Option("--rule", help="Falsifiable rule hypothesis (redacted)"),
    ] = "",
) -> None:
    """Capture a latent requirement. Same fingerprint increments sightings."""
    from jobbot.ops.symptoms import note_symptom

    session, config = _session()
    try:
        record = note_symptom(
            session,
            intent=intent,
            area=area,
            title=title,
            rule_hypothesis=rule,
            output_dir=config.output_dir,
        )
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc
    console.print(
        f"[green]{record.id}[/green] area={record.area} "
        f"sightings={record.sightings} status={record.status} fp={record.fingerprint}"
    )
    console.print(f"Plan: [bold]jobbot ops symptom plan {record.id}[/bold]")


@ops_symptom_app.command("list")
def ops_symptom_list(
    status: Annotated[
        str | None,
        typer.Option("--status", help="latent|acknowledged|compressing|resolved|wontfix"),
    ] = None,
    area: Annotated[
        str | None,
        typer.Option("--area", help="Filter by area"),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=200)] = 50,
) -> None:
    """List latent requirements (symptoms), newest first."""
    from jobbot.ops.symptoms import list_symptoms

    session, _ = _session()
    rows = list_symptoms(session, status=status, area=area, limit=limit)
    if not rows:
        console.print("No symptoms stored.")
        raise typer.Exit(SUCCESS)
    table = Table(title="ops symptoms (latent requirements)")
    table.add_column("id")
    table.add_column("area")
    table.add_column("n")
    table.add_column("status")
    table.add_column("title")
    for r in rows:
        table.add_row(r.id, r.area, str(r.sightings), r.status, r.title[:60])
    console.print(table)


@ops_symptom_app.command("show")
def ops_symptom_show(
    symptom_id: Annotated[str, typer.Argument(help="Symptom id, e.g. S0001")],
) -> None:
    """Show one symptom (already redacted at write time)."""
    from jobbot.ops.symptoms import get_symptom

    session, config = _session()
    record = get_symptom(session, symptom_id)
    if record is None:
        err_console.print(f"[red]Unknown symptom {symptom_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    console.print(
        Panel(
            f"area={record.area}\nsightings={record.sightings}\nstatus={record.status}\n"
            f"fingerprint={record.fingerprint}\n\n"
            f"title: {record.title}\n\nintent:\n{record.intent}\n\n"
            f"rule:\n{record.rule_hypothesis or '(none yet)'}\n\n"
            f"feature: {record.feature_path or '(none)'}\n"
            f"issue: {record.issue_url or '(none)'}",
            title=record.id,
        )
    )
    mirror = config.output_dir / "ops" / "symptoms" / f"{record.id}.json"
    if mirror.is_file():
        console.print(f"Mirror: {mirror}")


@ops_symptom_app.command("plan")
def ops_symptom_plan(
    symptom_id: Annotated[str, typer.Argument(help="Symptom id, e.g. S0001")],
) -> None:
    """Print the System 2 → System 1 compression plan (does not write code)."""
    from jobbot.ops.symptoms import get_symptom, promote_plan

    session, _ = _session()
    record = get_symptom(session, symptom_id)
    if record is None:
        err_console.print(f"[red]Unknown symptom {symptom_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    console.print(Panel(promote_plan(record), title="compress to System 1"))


@ops_symptom_app.command("triage")
def ops_symptom_triage(
    symptom_id: Annotated[str, typer.Argument(help="Symptom id, e.g. S0001")],
    status: Annotated[
        str,
        typer.Option("--status", help="latent|acknowledged|compressing|resolved|wontfix"),
    ],
    feature: Annotated[
        str | None,
        typer.Option("--feature", help="Path of the compressed feature module"),
    ] = None,
) -> None:
    """Update symptom status after review / compression."""
    from jobbot.ops.symptoms import mark_symptom_status

    session, _ = _session()
    try:
        record = mark_symptom_status(session, symptom_id, status, feature_path=feature)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(VALIDATION_FAILURE) from exc
    if record is None:
        err_console.print(f"[red]Unknown symptom {symptom_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    console.print(
        f"[green]{record.id}[/green] → status={record.status}"
        + (f" feature={record.feature_path}" if record.feature_path else "")
    )


@ops_symptom_app.command("issue")
def ops_symptom_issue(
    symptom_id: Annotated[str, typer.Argument(help="Symptom id, e.g. S0001")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation"),
    ] = False,
) -> None:
    """Create a GitHub issue from a redacted symptom (HITL; uses gh). Never auto."""
    import subprocess

    from jobbot.ops.symptoms import get_symptom, issue_body, issue_title, mark_symptom_status

    session, _ = _session()
    record = get_symptom(session, symptom_id)
    if record is None:
        err_console.print(f"[red]Unknown symptom {symptom_id}[/red]")
        raise typer.Exit(GENERIC_FAILURE)
    if record.issue_url:
        console.print(f"Already linked: {record.issue_url}")
        raise typer.Exit(SUCCESS)

    title = issue_title(record)
    body = issue_body(record)
    console.print(Panel(f"{title}\n\n{body}", title="proposed GitHub issue (redacted)"))
    if not yes and not typer.confirm("Create GitHub issue with gh?"):
        console.print("Aborted.")
        raise typer.Exit(SUCCESS)

    proc = subprocess.run(  # noqa: S603
        [
            "gh",
            "issue",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--assignee",
            "cursoragent",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err_console.print(f"[red]gh failed:[/red] {proc.stderr or proc.stdout}")
        raise typer.Exit(GENERIC_FAILURE)
    url = (proc.stdout or "").strip()
    mark_symptom_status(session, symptom_id, "acknowledged", issue_url=url or None)
    console.print(f"[green]Created[/green] {url or '(see gh output)'}")


@ops_app.command("loop")
def ops_loop(
    cmd: Annotated[
        str,
        typer.Option("--cmd", help="Loop step: getonboard-prepare | getonboard-sync"),
    ] = "getonboard-prepare",
    interval: Annotated[
        float,
        typer.Option("--interval", help="Seconds between ticks"),
    ] = 300.0,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop on first failure"),
    ] = False,
    max_ticks: Annotated[
        int | None,
        typer.Option("--max-ticks", help="Stop after N ticks (default: forever)"),
    ] = None,
) -> None:
    """Run a maintainer step continuously; persist Fxxxx on crash and continue."""
    from jobbot.ops.loop import LOOP_COMMANDS, run_loop

    if cmd not in LOOP_COMMANDS:
        err_console.print(
            f"[red]Unknown --cmd {cmd!r}[/red]; expected one of {sorted(LOOP_COMMANDS)}"
        )
        raise typer.Exit(GENERIC_FAILURE)

    def _on_tick(result: object) -> None:
        from jobbot.ops.loop import LoopTickResult

        assert isinstance(result, LoopTickResult)
        if result.ok:
            console.print("[green]tick ok[/green]")
        else:
            fid = result.failure.id if result.failure else "?"
            console.print(
                f"[yellow]tick failed[/yellow] exit={result.exit_code} "
                f"failure={fid}  {result.detail}"
            )

    def _hitl(reason: str) -> None:
        err_console.print(f"[bold red]{reason}[/bold red]")
        err_console.print("Auth/challenge — fix manually, then restart loop. No CAPTCHA bypass.")
        raise typer.Exit(AUTH_REQUIRED if "exit 3" in reason else MANUAL_CHALLENGE)

    console.print(
        f"Loop [bold]{cmd}[/bold] every {interval}s (fail_fast={fail_fast}, max_ticks={max_ticks})"
    )
    code = run_loop(
        cmd,
        interval_sec=interval,
        fail_fast=fail_fast,
        max_ticks=max_ticks,
        on_tick=_on_tick,
        hitl_pause=_hitl,
    )
    raise typer.Exit(code)


@ops_app.command("capabilities")
def ops_capabilities(
    write: Annotated[
        bool,
        typer.Option("--write", help="Write docs/capabilities.md (default: print)"),
    ] = False,
) -> None:
    """Index of commands and modules, so work already done is not re-derived."""
    from jobbot.ops.capabilities import build_index, write_index
    from jobbot.workspace import repo_root

    root = repo_root()
    if not write:
        # Raw echo: rich would wrap the markdown at the console width.
        typer.echo(build_index(root, app), nl=False)
        return
    target = write_index(root, app)
    console.print(f"Wrote [bold]{target}[/bold]")


@app.command("probe-exit", hidden=True)
def probe_exit(
    code: Annotated[
        int,
        typer.Argument(help="Exit code to raise (e.g. 5 = UI_CHANGED)"),
    ] = 5,
) -> None:
    """Hidden helper for failure-capture drills and unit tests."""
    raise typer.Exit(code)


# ── recruiters (public hiring practice; practices only, never people) ─────────


def _recruiters_state(config: JobbotConfig) -> tuple[list[Any], Path]:
    from jobbot.recruiters.sources import default_recruiters_path, load_sources

    path = default_recruiters_path(config.root)
    return load_sources(path), path


@recruiters_app.command("discover")
def recruiters_discover(
    urls: Annotated[
        list[str] | None,
        typer.Argument(help="Public URLs about hiring practice"),
    ] = None,
    from_file: Annotated[
        Path | None,
        typer.Option("--from-file", help="Text/YAML file with one URL per line"),
    ] = None,
    fixture: Annotated[
        Path | None,
        typer.Option("--fixture", help="Read a saved HTML instead of fetching (single URL)"),
    ] = None,
    delay: Annotated[
        float,
        typer.Option("--delay", help="Seconds between requests (be polite)"),
    ] = 1.0,
) -> None:
    """Read public pages about hiring and store what they teach as candidates."""
    from jobbot.companies.oneshot import FetchResult, UrllibFetcher
    from jobbot.recruiters.discover import discover_sources
    from jobbot.recruiters.sources import save_sources, upsert_source

    config = load_config()
    targets = list(urls or [])
    if from_file is not None:
        path = from_file.expanduser().resolve()
        if not path.is_file():
            err_console.print(f"[red]File not found: {path}[/red]")
            raise typer.Exit(VALIDATION_FAILURE)
        targets.extend(
            line.strip().lstrip("- ").strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not targets:
        err_console.print("Give at least one URL, or --from-file with a list.")
        raise typer.Exit(VALIDATION_FAILURE)

    narrator = _narrator()
    narrator.phase(Phase.EXPLORING, f"{len(targets)} public page(s)")

    if fixture is not None:
        saved = fixture.expanduser().resolve()
        if not saved.is_file():
            err_console.print(f"[red]Fixture not found: {saved}[/red]")
            raise typer.Exit(VALIDATION_FAILURE)
        html = saved.read_text(encoding="utf-8")

        class _FixtureFetcher:
            def fetch(self, url: str) -> FetchResult:
                return FetchResult(url=url, status=200, html=html)

        report = discover_sources(targets[:1], _FixtureFetcher(), respect_robots=False)
    else:
        report = discover_sources(targets, UrllibFetcher(delay=delay))

    narrator.phase(Phase.SURFACING, f"{len(report.sources)} source(s)")
    narrator.outcome(
        ExplorationOutcome(
            found=len(report.sources),
            refused=len(report.refused),
            unknown=len(report.skipped_robots)
            + len(report.skipped_login)
            + len(report.nothing_learned),
        )
    )
    if report.skipped_robots:
        console.print(
            "[yellow]robots.txt asked us not to read (omitted, not absent):[/yellow] "
            + ", ".join(report.skipped_robots[:5])
        )
    if report.skipped_login:
        console.print(
            "[yellow]Behind a login (not public knowledge):[/yellow] "
            + ", ".join(report.skipped_login[:5])
        )
    if report.refused:
        console.print(
            "[yellow]Refused our request (unknown, not empty):[/yellow] "
            + ", ".join(report.refused[:5])
        )
    if not report.sources:
        console.print("Nothing learned this run.")
        raise typer.Exit(SUCCESS)

    sources, path = _recruiters_state(config)
    for source in report.sources:
        sources = upsert_source(sources, source)
        console.print(
            f"[green]candidate[/green] {source.url}  ({len(source.practices)} practice(s))"
        )
    save_sources(sources, path)
    console.print(f"Stored (candidates): {path}")
    console.print("Only [bold]jobbot recruiters promote URL[/bold] lets these advise your CV.")


@recruiters_app.command("list")
def recruiters_list(
    status: Annotated[
        str | None,
        typer.Option("--status", help="candidate | active | rejected"),
    ] = None,
) -> None:
    """List known sources of hiring practice."""
    config = load_config()
    sources, _ = _recruiters_state(config)
    if status:
        wanted = status.strip().casefold()
        sources = [s for s in sources if s.status.value == wanted]
    if not sources:
        console.print("No sources yet. Try [bold]jobbot recruiters discover URL[/bold].")
        return
    table = Table(title="Hiring practice sources")
    table.add_column("Status")
    table.add_column("Practices", justify="right")
    table.add_column("Title")
    table.add_column("URL")
    for source in sources:
        table.add_row(
            source.status.value,
            str(len(source.practices)),
            (source.title or "—")[:38],
            source.url[:48],
        )
    console.print(table)


@recruiters_app.command("show")
def recruiters_show(
    url: Annotated[str, typer.Argument(help="Source URL")],
) -> None:
    """Show one source with every practice it taught."""
    from jobbot.companies.urls import canonical_key

    config = load_config()
    sources, _ = _recruiters_state(config)
    key = canonical_key(url)
    source = next((s for s in sources if canonical_key(s.url) == key), None)
    if source is None:
        err_console.print(f"[red]Unknown source: {url}[/red]")
        raise typer.Exit(VALIDATION_FAILURE)
    console.print(
        Panel(
            f"[bold]{source.title or source.url}[/bold]\n{source.url}\n"
            f"status: {source.status.value}   practices: {len(source.practices)}\n"
            f"{source.evidence}",
            title="Source",
        )
    )
    for practice in source.practices:
        console.print(f"  [{practice.kind.value}] {practice.text}")


@recruiters_app.command("promote")
def recruiters_promote(
    url: Annotated[str, typer.Argument(help="Source URL")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Promote without asking")] = False,
) -> None:
    """Let one source's practices reach `cv advise`."""
    from jobbot.recruiters.sources import promote_source, save_sources

    config = load_config()
    sources, path = _recruiters_state(config)
    if not yes and not typer.confirm(f"Let {url} advise your CV?", default=False):
        console.print("Aborted.")
        raise typer.Exit(SUCCESS)
    save_sources(promote_source(sources, url), path)
    console.print(f"[green]active[/green] {url} → its practices now reach cv advise")


@recruiters_app.command("reject")
def recruiters_reject(
    url: Annotated[str, typer.Argument(help="Source URL")],
) -> None:
    """Keep one source out for good."""
    from jobbot.recruiters.sources import reject_source, save_sources

    config = load_config()
    sources, path = _recruiters_state(config)
    save_sources(reject_source(sources, url), path)
    console.print(f"[yellow]rejected[/yellow] {url}")


@recruiters_app.command("export")
def recruiters_export(
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Destination file (default: output/recruiters/…)"),
    ] = None,
) -> None:
    """Write a shareable snapshot: active practices only, no people, no PII."""
    from jobbot.recruiters.sources import export_payload

    config = load_config()
    sources, _ = _recruiters_state(config)
    payload = export_payload(sources)
    target = (
        out.expanduser().resolve()
        if out
        else (config.output_dir / "recruiters" / "hiring_practice.yaml")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    listed = payload.get("sources")
    count = len(listed) if isinstance(listed, list) else 0
    console.print(f"[green]Exported[/green] {count} active source(s) → {target}")
