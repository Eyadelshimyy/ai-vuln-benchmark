from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def plantuml_jar() -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    configured_path = os.environ.get("ROX_DOX_PLANTUML_JAR")
    jar = (
        Path(configured_path)
        if configured_path
        else repository_root / "tools/plantuml.jar"
    )
    if not jar.is_file():
        pytest.fail(f"PlantUML jar not found at {jar}; run scripts/fetch_plantuml.sh")
    return jar


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source-repo"
    repo.mkdir()
    (repo / "pkg").mkdir()
    (repo / "pkg" / "sub").mkdir()
    (repo / "README.md").write_text("Repository fixture\n", encoding="utf-8")
    other_path = repo / "other" / "thing"
    other_path.mkdir(parents=True)
    (other_path / "c.py").write_text("distant feature source\n", encoding="utf-8")
    source_file = repo / "pkg" / "a.py"
    source_file.write_text(
        "\n".join(f"line {line_number}" for line_number in range(1, 11)) + "\n",
        encoding="utf-8",
    )
    (repo / "pkg" / "sub" / "b.py").write_text("leaf source\n", encoding="utf-8")
    models_dir = repo / "models"
    models_dir.mkdir()
    (models_dir / "user.py").write_text(
        "from sqlalchemy import Column, ForeignKey, Integer, String\n"
        "from sqlalchemy.orm import Mapped, mapped_column\n"
        "\n"
        "class User:\n"
        '    __tablename__ = "users"\n'
        "    id = Column(Integer, primary_key=True)\n"
        "    email = Column(String)\n"
        "\n"
        "class Session:\n"
        '    __tablename__ = "sessions"\n'
        "    id = Column(Integer, primary_key=True)\n"
        '    user_id = Column("user_id", Integer, ForeignKey("users.id"))\n'
        "    token: Mapped[str] = mapped_column()\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "add",
            "pkg/a.py",
            "pkg/sub/b.py",
            "other/thing/c.py",
            "models/user.py",
            "README.md",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add source fixture"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit


@pytest.fixture
def page_data(git_repo: tuple[Path, str]) -> dict[str, object]:
    _, commit = git_repo
    code_source = {"path": "pkg/a.py", "lines": [1, 3]}
    notion_source = {"notion": "https://www.notion.so/rox/Foo-123"}
    return {
        "id": "rox-core",
        "title": "Rox Core",
        "kind": "root",
        "commit": commit,
        "parent": None,
        "paths": ["."],
        "tldr": {
            "summary": [{"text": "Handles requests.", "sources": [code_source]}],
            "key_points": [],
            "table": {
                "columns": ["Component"],
                "rows": [["API"]],
                "sources": [code_source],
            },
            "notes": [{"text": "Design notes.", "sources": [notion_source]}],
        },
        "block": {
            "nodes": [
                {"id": "api", "label": "API", "source": code_source},
                {"id": "store", "label": "Store", "source": code_source},
            ],
            "edges": [
                {
                    "src": "api",
                    "dst": "store",
                    "label": "reads",
                    "source": code_source,
                }
            ],
        },
        "data": {
            "sql_tables": ["users", "sessions"],
            "nosql": [
                {
                    "name": "cache",
                    "kind": "redis",
                    "fields": ["key", "value"],
                    "source": code_source,
                }
            ],
        },
        "sequences": [
            {
                "title": "send message",
                "participants": [
                    {"id": "api", "label": "API", "source": code_source},
                    {"id": "store", "label": "Store", "source": code_source},
                ],
                "steps": [
                    {
                        "src": "api",
                        "dst": "store",
                        "message": "write",
                        "source": code_source,
                    }
                ],
            }
        ],
        "states": [
            {
                "title": "message lifecycle",
                "entity": "Message",
                "states": [
                    {"id": "queued", "label": "Queued", "source": code_source},
                    {"id": "sent", "label": "Sent", "source": code_source},
                ],
                "transitions": [
                    {
                        "src": "queued",
                        "dst": "sent",
                        "event": "deliver",
                        "source": code_source,
                    }
                ],
            }
        ],
        "related": [],
    }


@pytest.fixture
def site_pages(page_data: dict[str, object]) -> list[dict[str, object]]:
    root = copy.deepcopy(page_data)
    root["title"] = "rox-core"

    child = copy.deepcopy(page_data)
    child["id"] = "rox-core/pkg"
    child["title"] = "pkg"
    child["kind"] = "domain"
    child["parent"] = "rox-core"
    child["paths"] = ["pkg"]
    child["block"]["nodes"][0]["link"] = "rox-core/pkg/sub"

    leaf = copy.deepcopy(page_data)
    leaf["id"] = "rox-core/pkg/sub"
    leaf["title"] = "sub"
    leaf["kind"] = "feature"
    leaf["parent"] = "rox-core/pkg"
    leaf["paths"] = ["other/thing"]

    return [root, child, leaf]
