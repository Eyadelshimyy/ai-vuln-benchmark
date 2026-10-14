from pathlib import Path, PurePosixPath

from rox_dox.features import FeatureMap


def test_R6_domain_files_are_placed_or_uncovered_without_excluded_paths(
    synthetic_feature_map: tuple[Path, str, FeatureMap],
) -> None:
    _, _, feature_map = synthetic_feature_map
    expected_domain_files = {
        "backend/src/pkg/sequence.py",
        "backend/src/pkg/sequence_service.py",
        "backend/src/pkg/isolated.py",
    }
    excluded_paths = {
        "backend/src/pkg/tests/ignored.py",
        "backend/src/pkg/test/ignored.py",
        "backend/src/pkg/__tests__/ignored.py",
        "backend/src/pkg/tests_unit/ignored.py",
        "backend/src/pkg/migrations/001_hidden.py",
        "backend/src/pkg/alembic/versions/001_hidden.py",
        "backend/src/pkg/conftest.py",
        "backend/src/pkg/test_hidden.py",
        "backend/src/pkg/hidden_test.py",
        "backend/src/pkg/thing.test.py",
        "backend/src/pkg/thing.spec.py",
    }
    placed_paths = {
        file.path for feature in feature_map.features for file in feature.files
    }
    uncovered_paths = {file.path for file in feature_map.uncovered}

    assert feature_map.counts.domain_files == len(expected_domain_files)
    assert placed_paths | uncovered_paths == expected_domain_files
    assert (placed_paths | uncovered_paths).isdisjoint(excluded_paths)

    output_paths = placed_paths | uncovered_paths
    for feature in feature_map.features:
        output_paths.update(
            evidence.path for file in feature.files for evidence in file.evidence
        )
        output_paths.update(link.path for link in feature.table_links)
    assert all(
        not (
            {"tests", "test", "__tests__", "migrations", "alembic"} & set(path.parts)
            or any(part.startswith("tests_") for part in path.parts)
            or path.name == "conftest.py"
            or path.name.startswith("test_")
            or path.name.endswith("_test.py")
            or ".test." in path.name
            or ".spec." in path.name
        )
        for path in map(PurePosixPath, output_paths)
    )
