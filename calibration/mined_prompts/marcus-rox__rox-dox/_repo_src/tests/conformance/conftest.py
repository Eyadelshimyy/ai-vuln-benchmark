from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rox_dox.features import FeatureMap, build_feature_map


@pytest.fixture
def synthetic_feature_map(tmp_path: Path) -> tuple[Path, str, FeatureMap]:
    sources = {
        "backend/src/pkg/sequence.py": (
            "from sqlalchemy import Column, Integer\n"
            "\n"
            "class Sequence:\n"
            '    __tablename__ = "sequence"\n'
            "    id = Column(Integer, primary_key=True)\n"
        ),
        "backend/src/pkg/sequence_service.py": (
            "from .sequence import Sequence\n"
            "\n"
            "def get_sequence() -> Sequence:\n"
            "    return Sequence()\n"
        ),
        "backend/src/pkg/isolated.py": (
            "from sqlalchemy import Column, Integer\n"
            "\n"
            "class Isolated:\n"
            '    __tablename__ = "isolated"\n'
            "    id = Column(Integer, primary_key=True)\n"
        ),
        **{
            path: "from .sequence import Sequence\n"
            for path in (
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
            )
        },
    }
    repo = tmp_path / "feature-source"
    repo.mkdir()
    for relative_path, source in sources.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "add", *sources],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add feature conformance fixture"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    feature_map = build_feature_map(
        repo,
        commit,
        {"seq": ["isolated", "sequence"]},
        "seq",
    )
    return repo, commit, feature_map
