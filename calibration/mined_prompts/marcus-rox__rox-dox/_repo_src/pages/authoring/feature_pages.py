import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rox_core_domains
from rox_dox.components import FileFacts, extract_file_facts
from rox_dox.feature_pages import feature_pages
from rox_dox.features import FeatureMap, _cached_parse_graph, _snapshot, parallel_map
from rox_dox.schema import Table, extract_tables

DEFAULT_REPO = Path("/home/ubuntu/repos/rox-core")


@dataclass(frozen=True)
class _PageInputs:
    feature_maps: Sequence[FeatureMap]
    root_id: str
    domain_titles: Mapping[str, str]
    names: Mapping[str, Mapping[str, str]]
    tables_by_commit: Mapping[str, Mapping[str, Table]]
    components_by_commit: Mapping[str, Mapping[str, FileFacts]]
    imports_by_commit: Mapping[str, Mapping[str, Mapping[str, int]]]
    table_accesses_by_commit: Mapping[
        str, Mapping[str, Mapping[str, tuple[int | None, int | None]]]
    ]
    primary_features: Mapping[str, tuple[str, str]]
    table_domains: Mapping[str, str]
    component_catalog: Sequence[Mapping[str, Any]]


def _domain_page_dumps(inputs: _PageInputs, map_index: int) -> list[dict]:
    feature_map = inputs.feature_maps[map_index]
    return [
        page.model_dump(exclude_none=True)
        for page in feature_pages(
            feature_map,
            root_id=inputs.root_id,
            domain_title=inputs.domain_titles[feature_map.domain],
            names=inputs.names.get(feature_map.domain, {}),
            tables=inputs.tables_by_commit[feature_map.commit],
            component_facts=inputs.components_by_commit[feature_map.commit],
            component_imports=inputs.imports_by_commit[feature_map.commit],
            table_accesses=inputs.table_accesses_by_commit[feature_map.commit],
            primary_features=inputs.primary_features,
            domain_titles=inputs.domain_titles,
            table_domains=inputs.table_domains,
            component_catalog=inputs.component_catalog,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cited domain and feature pages from feature maps."
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--domain", default="all")
    args = parser.parse_args()
    repo = args.repo.resolve()

    project_root = Path(__file__).resolve().parents[2]
    pages_dir = project_root / "pages"
    feature_maps = []
    for map_file in sorted((project_root / "features").glob("*.json")):
        if map_file.stem == "names":
            continue
        feature_maps.append(
            FeatureMap.model_validate(json.loads(map_file.read_text(encoding="utf-8")))
        )
    if args.domain != "all" and not any(
        feature_map.domain == args.domain for feature_map in feature_maps
    ):
        raise SystemExit(f"unknown domain: {args.domain}")
    selected_maps = [
        feature_map
        for feature_map in feature_maps
        if args.domain == "all" or feature_map.domain == args.domain
    ]

    root_page = json.loads((pages_dir / "rox-core.json").read_text(encoding="utf-8"))
    component_catalog = json.loads(
        (pages_dir / "components.json").read_text(encoding="utf-8")
    )
    names = json.loads(
        (project_root / "features" / "names.json").read_text(encoding="utf-8")
    )
    domain_titles = {
        domain_id: title for domain_id, title, *_ in rox_core_domains.DOMAINS
    }
    tables_by_commit = {}
    components_by_commit = {}
    imports_by_commit = {}
    table_accesses_by_commit = {}
    for feature_map in selected_maps:
        if feature_map.commit not in tables_by_commit:
            tables_by_commit[feature_map.commit] = extract_tables(
                repo, feature_map.commit
            )
            snapshot = _snapshot(repo, feature_map.commit)
            graph, _ = _cached_parse_graph(
                repo,
                feature_map.commit,
                snapshot,
                tables_by_commit[feature_map.commit],
                set(tables_by_commit[feature_map.commit]),
            )
            source_paths = sorted(graph)
            components_by_commit[feature_map.commit] = extract_file_facts(
                repo,
                feature_map.commit,
                source_paths,
            )
            imports_by_commit[feature_map.commit] = {
                path: {
                    imported: graph_file.import_lines[imported]
                    for imported in graph_file.imports
                }
                for path, graph_file in sorted(graph.items())
            }
            table_accesses_by_commit[feature_map.commit] = {
                path: graph_file.table_accesses
                for path, graph_file in sorted(graph.items())
            }

    primary_features = {}
    table_domains = {
        table: feature_map.domain
        for feature_map in feature_maps
        for table in feature_map.tables
    }
    for candidate in feature_maps:
        for feature in candidate.features:
            display_name = names.get(candidate.domain, {}).get(feature.id)
            if display_name is None:
                display_name = " ".join(
                    part.capitalize()
                    for part in feature.id.replace("-", "_").split("_")
                )
            page_id = f"feature-{candidate.domain}-{feature.id.replace('_', '-')}"
            for file in feature.files:
                if file.primary and (
                    file.path not in primary_features
                    or page_id < primary_features[file.path][1]
                ):
                    primary_features[file.path] = (display_name, page_id)

    inputs = _PageInputs(
        feature_maps=selected_maps,
        root_id=root_page["id"],
        domain_titles=domain_titles,
        names=names,
        tables_by_commit=tables_by_commit,
        components_by_commit=components_by_commit,
        imports_by_commit=imports_by_commit,
        table_accesses_by_commit=table_accesses_by_commit,
        primary_features=primary_features,
        table_domains=table_domains,
        component_catalog=component_catalog,
    )
    generated = [
        page
        for domain_pages in parallel_map(
            _domain_page_dumps,
            inputs,
            range(len(selected_maps)),
            desc="Generating domain pages",
            unit="domain",
        )
        for page in domain_pages
    ]

    output_dirs = {
        "domain": pages_dir / "domains",
        "feature": pages_dir / "features",
    }
    expected = {directory: set() for directory in output_dirs.values()}
    for page in generated:
        directory = output_dirs[page["kind"]]
        output_path = directory / (
            f"{page['id'].removeprefix('domain-')}.json"
            if page["kind"] == "domain"
            else f"{page['id']}.json"
        )
        expected[directory].add(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                page,
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    for directory, expected_paths in expected.items():
        for old_path in directory.glob("*.json"):
            belongs_to_selected_domain = (
                directory == output_dirs["domain"] and old_path.stem == args.domain
            ) or (
                directory == output_dirs["feature"]
                and old_path.stem.startswith(f"feature-{args.domain}-")
            )
            if old_path not in expected_paths and (
                args.domain == "all" or belongs_to_selected_domain
            ):
                old_path.unlink()


if __name__ == "__main__":
    main()
