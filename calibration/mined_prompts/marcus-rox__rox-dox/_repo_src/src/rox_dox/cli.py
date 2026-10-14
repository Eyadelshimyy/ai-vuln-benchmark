from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from rox_dox.block_svg import MAX_LAYOUT_PROBLEMS, block_layout_problems
from rox_dox.features import (
    FeatureMap,
    is_excluded_path,
    is_feature_path,
)
from rox_dox.lint import (
    LintInputError,
    format_finding,
    lint_pages,
    load_component_ids,
    load_pages,
    summary_line,
)
from rox_dox.model import Page
from rox_dox.outline import (
    OutlineInputError,
    format_markdown,
    outline_sources,
    read_commit,
    read_worktree,
)
from rox_dox.plantuml import DiagramError
from rox_dox.relations import RelationCandidate, find_relation_candidates
from rox_dox.rebuild import (
    TOKEN_ENV,
    GitHubPublisher,
    run_rebuild,
)
from rox_dox.render import render_folder_page, render_page, render_uncovered_page
from rox_dox.repo_tree import list_entries, list_tree_paths
from rox_dox.schema import Table, extract_tables
from rox_dox.setup import run_setup
from rox_dox.sources import page_problems
from rox_dox.states import DEFAULT_PREFIXES, find_states
from rox_dox.tree import build_tree, tree_problems


def _parse_tables(value: str) -> tuple[str, ...]:
    tables = tuple(table.strip() for table in value.split(","))
    if not tables or any(not table for table in tables):
        raise argparse.ArgumentTypeError(
            "tables must be a comma-separated list of table names"
        )
    return tables


def _parse_page_globs(value: str) -> tuple[str, ...]:
    patterns = tuple(pattern.strip() for pattern in value.split(","))
    if not patterns or any(not pattern for pattern in patterns):
        raise argparse.ArgumentTypeError(
            "page globs must be a comma-separated list of non-empty patterns"
        )
    return patterns


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rox-dox")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("pages_dir", type=Path)
    check.add_argument("--repo", type=Path, required=True)

    build = commands.add_parser("build")
    build.add_argument("pages_dir", type=Path)
    build.add_argument("--repo", type=Path, required=True)
    build.add_argument("--repo-url", required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--features", type=Path, default=Path("features"))
    build.add_argument(
        "--plantuml-jar",
        type=Path,
        default=Path("tools/plantuml.jar"),
    )
    build.add_argument("--only", type=_parse_page_globs)
    build.add_argument("--no-folder-pages", action="store_true")

    relations = commands.add_parser("relations")
    relations.add_argument("--repo", type=Path, required=True)
    relations.add_argument("--commit", required=True)
    relations.add_argument("--tables", type=_parse_tables, required=True)

    lint_diagrams = commands.add_parser("lint-diagrams")
    lint_diagrams.add_argument("pages_dir", type=Path)
    lint_diagrams.add_argument(
        "--page",
        action="extend",
        nargs="+",
        dest="page_ids",
    )
    lint_diagrams.add_argument("--json", action="store_true")
    outline = commands.add_parser("outline")
    outline.add_argument("paths", nargs="+")
    outline.add_argument("--repo", type=Path)
    outline.add_argument("--commit")
    outline.add_argument("--json", action="store_true", dest="as_json")
    rebuild = commands.add_parser("rebuild")
    rebuild.add_argument("--repo", type=Path, required=True)
    rebuild.add_argument("--commit")
    rebuild.add_argument("--out", type=Path)
    rebuild.add_argument("--open-pr", action="store_true")
    rebuild.add_argument("--merge", action="store_true")
    states = commands.add_parser("states")
    states.add_argument("--repo", type=Path, required=True)
    states.add_argument("--commit", required=True)
    states.add_argument("--prefix", action="append", dest="prefixes")
    states.add_argument("--json", action="store_true")

    setup = commands.add_parser("setup")
    setup.add_argument("--workdir", type=Path, required=True)
    setup.add_argument("--rox-core", type=Path)
    setup.add_argument("--json", action="store_true")
    return parser


def _load_pages(pages_dir: Path) -> list[tuple[Path, Page | None, str | None]]:
    pages = []
    for page_file in sorted(pages_dir.rglob("*.json")):
        if page_file == pages_dir / "components.json":
            continue
        try:
            page_data = json.loads(page_file.read_text(encoding="utf-8"))
            page = Page.model_validate(page_data)
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            pages.append((page_file, None, str(error)))
        else:
            pages.append((page_file, page, None))
    return pages


def _check_pages(pages_dir: Path, repo: Path) -> int:
    pages = _load_pages(pages_dir)
    problems_found = False
    valid_pages = []
    table_cache: dict[tuple[Path, str], dict[str, Table]] = {}
    for page_file, page, load_error in pages:
        if load_error is not None:
            print(f"{page_file}: {load_error}")
            problems_found = True
            continue
        if page is None:
            continue
        valid_pages.append(page)

        tables: dict[str, Table] = {}
        if page.data.sql_tables or page.data.domains:
            cache_key = (repo, page.commit)
            if cache_key not in table_cache:
                try:
                    table_cache[cache_key] = extract_tables(repo, page.commit)
                except RuntimeError as error:
                    print(f"{page_file}: {error}")
                    problems_found = True
            tables = table_cache.get(cache_key, {})
            requested_tables = [
                *page.data.sql_tables,
                *(table for domain in page.data.domains for table in domain.tables),
            ]
            for table_name in dict.fromkeys(requested_tables):
                if table_name not in tables:
                    print(
                        f"{page_file}: SQL table '{table_name}' not found at "
                        f"commit {page.commit[:10]}"
                    )
                    problems_found = True

        for problem in page_problems(page, repo, tables=tables):
            print(f"{page_file}: {problem}")
            problems_found = True

    for problem in tree_problems(valid_pages):
        print(f"{pages_dir}: {problem}")
        problems_found = True

    if problems_found:
        return 1
    print(f"{len(pages)} pages OK")
    return 0


def _page_output_path(out_dir: Path, page: Page) -> Path | None:
    output_root = out_dir.resolve()
    output_path = (out_dir / f"{page.id}.html").resolve()
    if output_root not in output_path.parents:
        return None
    return output_path


def _load_feature_maps(features_dir: Path) -> tuple[list[FeatureMap], list[str]]:
    maps = []
    problems = []
    if not features_dir.is_dir():
        return maps, problems
    for map_file in sorted(features_dir.glob("*.json")):
        if map_file.stem == "names":
            continue
        try:
            feature_map = FeatureMap.model_validate(
                json.loads(map_file.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            problems.append(f"{map_file}: {error}")
        else:
            maps.append(feature_map)
    return maps, problems


def _folder_output_relative(folder: str) -> Path:
    return Path("folders/index.html" if folder == "." else f"folders/{folder}.html")


@dataclass
class _FolderFeatureStats:
    paths: set[str] = field(default_factory=set)
    primary_count: int = 0
    file_count: int = 0


def _folder_index(
    paths: Sequence[str],
) -> tuple[
    list[str],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, tuple[str, ...]],
]:
    directories = {"."}
    children: defaultdict[str, set[str]] = defaultdict(set)
    files_by_folder: defaultdict[str, list[str]] = defaultdict(list)
    ancestors_by_path = {}
    for path in paths:
        parts = PurePosixPath(path).parts
        ancestors = ["."]
        files_by_folder["."].append(path)
        for depth in range(1, len(parts)):
            directory = "/".join(parts[:depth])
            parent = "." if depth == 1 else "/".join(parts[: depth - 1])
            directories.add(directory)
            children[parent].add(directory)
            files_by_folder[directory].append(path)
            ancestors.append(directory)
        ancestors_by_path[path] = tuple(ancestors)
    return (
        sorted(directories),
        {
            folder: sorted(child_directories)
            for folder, child_directories in children.items()
        },
        dict(files_by_folder),
        ancestors_by_path,
    )


def _build_folder_pages(
    *,
    args: argparse.Namespace,
    tree,
    root_page: Page,
    entries: Mapping[str, Sequence],
    feature_maps: Sequence[FeatureMap],
    all_paths: Sequence[str],
) -> tuple[list[tuple[Path, str]], int] | list[str]:
    directories, child_directories, subtree_paths_by_folder, ancestors_by_path = (
        _folder_index(all_paths)
    )
    in_scope = {path for path in all_paths if is_feature_path(path)}
    placed = {
        feature_file.path
        for feature_map in feature_maps
        for feature in feature_map.features
        for feature_file in feature.files
    }
    globally_uncovered = sorted(in_scope - placed)
    uncovered_by_folder: defaultdict[str, int] = defaultdict(int)
    for path in globally_uncovered:
        for folder in ancestors_by_path[path]:
            uncovered_by_folder[folder] += 1

    per_map_reasons: dict[str, list[tuple[str, str]]] = {}
    map_reports = []
    for feature_map in sorted(feature_maps, key=lambda item: item.domain):
        current_report = sorted(
            (item.path, item.reason) for item in feature_map.uncovered
        )
        map_reports.append((feature_map.domain, current_report))
        for path, reason in current_report:
            per_map_reasons.setdefault(path, []).append((feature_map.domain, reason))

    feature_rows_by_folder: defaultdict[str, list[tuple]] = defaultdict(list)
    for feature_map in feature_maps:
        domain_id = f"domain-{feature_map.domain}"
        domain_page = tree.pages.get(domain_id)
        if domain_page is None:
            continue
        for feature in feature_map.features:
            feature_id = f"feature-{feature_map.domain}-{feature.id.replace('_', '-')}"
            feature_page = tree.pages.get(feature_id)
            if feature_page is None:
                continue

            folder_stats: defaultdict[str, _FolderFeatureStats] = defaultdict(
                _FolderFeatureStats
            )
            for item in feature.files:
                if item.path not in in_scope:
                    continue
                for folder in ancestors_by_path[item.path]:
                    stats = folder_stats[folder]
                    stats.paths.add(item.path)
                    stats.primary_count += item.primary
                    stats.file_count += 1
            for folder, stats in folder_stats.items():
                feature_rows_by_folder[folder].append(
                    (
                        feature_page.title,
                        feature_id,
                        domain_page.title,
                        domain_id,
                        len(stats.paths),
                        stats.primary_count,
                        stats.file_count - stats.primary_count,
                    )
                )

    output_root = args.out.resolve()
    rendered: list[tuple[Path, str]] = []
    for folder in directories:
        subtree_paths = subtree_paths_by_folder.get(folder, [])
        tests_only = bool(subtree_paths) and all(
            is_excluded_path(path) for path in subtree_paths
        )
        child_folders = [
            (str(PurePosixPath(child).name), child)
            for child in child_directories.get(folder, [])
        ]
        document = render_folder_page(
            folder_path=folder,
            tree=tree,
            entries=entries,
            repo_url=args.repo_url,
            feature_rows=sorted(feature_rows_by_folder.get(folder, [])),
            uncovered_count=uncovered_by_folder[folder],
            child_folders=child_folders,
            tests_only=tests_only,
        )
        output_path = (args.out / _folder_output_relative(folder)).resolve()
        if output_root not in output_path.parents:
            return [f"folder path '{folder}' would write outside output directory"]
        rendered.append((output_path, document))

    uncovered = {path: per_map_reasons.get(path, []) for path in globally_uncovered}
    uncovered_document = render_uncovered_page(
        tree=tree,
        entries=entries,
        repo_url=args.repo_url,
        uncovered=uncovered,
        map_reports=map_reports,
    )
    uncovered_path = (args.out / "uncovered.html").resolve()
    if output_root not in uncovered_path.parents:
        return ["uncovered inventory would write outside output directory"]
    rendered.append((uncovered_path, uncovered_document))
    return rendered, len(globally_uncovered)


def _build_pages(args: argparse.Namespace) -> int:
    problems: list[tuple[Path, str]] = []
    pages_to_render: list[tuple[Path, Page, dict[str, Table], Path]] = []
    output_paths: dict[Path, Path] = {}
    table_cache: dict[tuple[Path, str], dict[str, Table]] = {}
    table_errors: dict[tuple[Path, str], str] = {}
    valid_pages = []

    for page_file, page, load_error in _load_pages(args.pages_dir):
        if load_error is not None:
            problems.append((page_file, load_error))
            continue
        if page is None:
            continue
        valid_pages.append(page)

        tables: dict[str, Table] = {}
        cache_key = (args.repo, page.commit)
        if page.data.sql_tables or page.data.domains:
            if cache_key not in table_cache and cache_key not in table_errors:
                try:
                    table_cache[cache_key] = extract_tables(args.repo, page.commit)
                except RuntimeError as error:
                    table_errors[cache_key] = str(error)
            if cache_key in table_errors:
                problems.append((page_file, table_errors[cache_key]))
            else:
                tables = table_cache[cache_key]
                requested_tables = [
                    *page.data.sql_tables,
                    *(table for domain in page.data.domains for table in domain.tables),
                ]
                for table_name in dict.fromkeys(requested_tables):
                    if table_name not in tables:
                        table_kind = "SQL table"
                        if any(
                            table_name in domain.tables for domain in page.data.domains
                        ):
                            table_kind = "schema domain table"
                        problems.append(
                            (
                                page_file,
                                f"{table_kind} '{table_name}' not found at "
                                f"commit {page.commit[:10]}",
                            )
                        )
                    elif (
                        table_name in page.data.sql_tables
                        and tables[table_name].duplicate_paths
                    ):
                        duplicate_paths = ", ".join(tables[table_name].duplicate_paths)
                        print(
                            f"warning: {page_file}: SQL table '{table_name}' also "
                            f"declared in {duplicate_paths}; using "
                            f"{tables[table_name].source.path}",
                            file=sys.stderr,
                        )

        citation_problems = page_problems(page, args.repo, tables=tables)
        problems.extend((page_file, problem) for problem in citation_problems)
        output_path = _page_output_path(args.out, page)
        if output_path is None:
            problems.append(
                (
                    page_file,
                    f"page id '{page.id}' would write outside output directory",
                )
            )
        elif output_path in output_paths:
            problems.append(
                (
                    page_file,
                    f"output path '{output_path}' is already used by "
                    f"{output_paths[output_path]}",
                )
            )
        else:
            output_paths[output_path] = page_file
            if args.only is None or any(
                fnmatch.fnmatchcase(page.id, pattern) for pattern in args.only
            ):
                pages_to_render.append((page_file, page, tables, output_path))

    problems.extend((args.pages_dir, problem) for problem in tree_problems(valid_pages))

    if problems:
        for page_file, problem in problems:
            print(f"{page_file}: {problem}")
        return 1

    layout_limit_exceeded = False
    for page in valid_pages:
        diagrams = [
            ("overview", page.block),
            *((figure.id, figure.block) for figure in page.block_figures),
        ]
        for figure_id, diagram in diagrams:
            layout_problems = block_layout_problems(diagram)
            for problem in layout_problems:
                print(f"warning: {page.id} {figure_id}: {problem}")
            if len(layout_problems) > MAX_LAYOUT_PROBLEMS:
                print(
                    f"error: {page.id} {figure_id}: {len(layout_problems)} layout "
                    f"problems (max {MAX_LAYOUT_PROBLEMS})"
                )
                layout_limit_exceeded = True

    if layout_limit_exceeded:
        return 1

    tree = build_tree(valid_pages)
    root_page = tree.pages[tree.root]
    entries = {
        root_page.id: list_entries(args.repo, root_page.commit, "."),
    }
    extra_pages: list[tuple[Path, str]] = []
    uncovered_count = 0
    if not args.no_folder_pages:
        feature_maps, map_problems = _load_feature_maps(args.features)
        if map_problems:
            for problem in map_problems:
                print(problem)
            return 1
        feature_maps = [
            feature_map
            for feature_map in feature_maps
            if feature_map.commit == root_page.commit
        ]
        all_paths = list_tree_paths(args.repo, root_page.commit)
        folder_result = _build_folder_pages(
            args=args,
            tree=tree,
            root_page=root_page,
            entries=entries,
            feature_maps=feature_maps,
            all_paths=all_paths,
        )
        if isinstance(folder_result, list):
            for problem in folder_result:
                print(problem)
            return 1
        extra_pages, uncovered_count = folder_result
    rendered_pages: list[tuple[Path, str]] = []
    render_problems: list[tuple[Path, str]] = []
    for page_file, page, tables, output_path in pages_to_render:
        try:
            rendered = render_page(
                page,
                tree=tree,
                repo_url=args.repo_url,
                tables=tables,
                jar=args.plantuml_jar,
                entries=entries,
            )
        except DiagramError as error:
            render_problems.append((page_file, str(error)))
        else:
            rendered_pages.append((output_path, rendered))

    if render_problems:
        for page_file, problem in render_problems:
            print(f"{page_file}: {problem}")
        return 1

    all_rendered = [*rendered_pages, *extra_pages]
    seen_output_paths: set[Path] = set()
    for output_path, _document in all_rendered:
        if output_path in seen_output_paths:
            print(f"output path '{output_path}' is already used")
            return 1
        seen_output_paths.add(output_path)
    for output_path, document in all_rendered:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
    print(
        f"{len(all_rendered)} pages written to {args.out}; "
        f"{uncovered_count} in-scope files uncovered"
    )
    return 0


def _print_relation_candidates(
    repo: Path,
    commit: str,
    tables: Sequence[str],
) -> int:
    try:
        candidates: list[RelationCandidate] = find_relation_candidates(
            repo,
            commit,
            tables,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    for candidate in candidates:
        print(candidate.to_tsv())
    return 0


def _lint_diagrams(args: argparse.Namespace) -> int:
    pages_dir: Path = args.pages_dir
    if not pages_dir.is_dir():
        print(f"{pages_dir}: pages directory not found", file=sys.stderr)
        return 2
    try:
        component_ids = load_component_ids(pages_dir)
        pages = load_pages(pages_dir)
    except LintInputError as error:
        print(error, file=sys.stderr)
        return 2

    if args.page_ids:
        requested_ids = set(args.page_ids)
        known_ids = {page.id for page in pages}
        unknown_ids = sorted(requested_ids - known_ids)
        if unknown_ids:
            print(
                f"unknown page id(s): {', '.join(unknown_ids)}",
                file=sys.stderr,
            )
            return 2
        pages = [page for page in pages if page.id in requested_ids]

    findings = lint_pages(pages, component_ids)
    if args.json:
        print(
            json.dumps(
                {
                    "pages": len(pages),
                    "errors": sum(finding.severity == "error" for finding in findings),
                    "warnings": sum(
                        finding.severity == "warning" for finding in findings
                    ),
                    "findings": [finding.to_dict() for finding in findings],
                },
                ensure_ascii=False,
            )
        )
    else:
        for finding in findings:
            print(format_finding(finding))
        print(summary_line(findings, len(pages)))
    return int(any(finding.severity == "error" for finding in findings))


def _print_states(
    repo: Path, commit: str, prefixes: Sequence[str] | None, as_json: bool
) -> int:
    if not repo.is_dir():
        print(f"{repo}: repository not found", file=sys.stderr)
        return 2
    try:
        report = find_states(repo, commit, prefixes or DEFAULT_PREFIXES)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), indent=2) if as_json else report.to_text())
    return 1 if report.parse_errors else 0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _setup_input_problem(args: argparse.Namespace) -> str | None:
    if not os.environ.get(TOKEN_ENV):
        return f"{TOKEN_ENV} is required"
    if args.workdir.exists() and not args.workdir.is_dir():
        return f"--workdir {args.workdir} exists and is not a directory"
    if args.rox_core is not None and not args.rox_core.is_dir():
        return f"--rox-core {args.rox_core} is not a directory"
    return None


def _setup(args: argparse.Namespace) -> int:
    problem = _setup_input_problem(args)
    if problem is not None:
        print(problem, file=sys.stderr)
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": {"step": "input", "message": problem},
                    },
                    indent=2,
                )
            )
        return 2
    result = run_setup(
        args.workdir,
        rox_core=args.rox_core,
        token=os.environ[TOKEN_ENV],
        command_runner=subprocess.run,
    )
    if args.json:
        print(json.dumps(result.to_json(), indent=2))
    else:
        for step in result.steps:
            print(f"{step.step}: {step.action} — {step.detail}")
        if result.error is not None:
            print(f"error in {result.error['step']}: {result.error['message']}")
    return 0 if result.error is None else 1


def _rebuild(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[2]
    publisher = None
    if args.open_pr:
        publisher = GitHubPublisher(
            project_root,
            token=os.environ.get(TOKEN_ENV),
            unix_timestamp=int(time.time()),
            command_runner=subprocess.run,
        )
    return run_rebuild(
        repo=args.repo,
        requested_commit=args.commit,
        output_dir=args.out,
        open_pr=args.open_pr,
        merge=args.merge,
        project_root=project_root,
        command_runner=subprocess.run,
        clock=_utc_now,
        publisher=publisher,
    )


def _outline(args: argparse.Namespace) -> int:
    if (args.repo is None) != (args.commit is None):
        print("--repo and --commit must be given together", file=sys.stderr)
        return 2
    try:
        if args.repo is None:
            sources = read_worktree([Path(path) for path in args.paths])
        else:
            sources = read_commit(args.repo, args.commit, args.paths)
    except (OutlineInputError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 2
    outlines = outline_sources(sources)
    if args.as_json:
        print(
            json.dumps({"files": [outline.to_json() for outline in outlines]}, indent=2)
        )
    else:
        print(format_markdown(outlines), end="")
    if any(outline.error is not None for outline in outlines):
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "rebuild" and args.merge and not args.open_pr:
        parser.error("--merge requires --open-pr")
    if args.command == "relations":
        return _print_relation_candidates(args.repo, args.commit, args.tables)
    if args.command == "rebuild":
        return _rebuild(args)
    if args.command == "lint-diagrams":
        return _lint_diagrams(args)
    if args.command == "outline":
        return _outline(args)
    if args.command == "states":
        return _print_states(args.repo, args.commit, args.prefixes, args.json)
    if args.command == "setup":
        return _setup(args)
    pages_dir: Path = args.pages_dir
    if not pages_dir.is_dir():
        print(f"{pages_dir}: pages directory not found")
        return 1

    if args.command == "check":
        return _check_pages(pages_dir, args.repo)
    return _build_pages(args)


if __name__ == "__main__":
    raise SystemExit(main())
