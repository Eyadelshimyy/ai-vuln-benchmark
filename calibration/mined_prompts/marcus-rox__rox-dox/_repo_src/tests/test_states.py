from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import rox_dox.cli as cli
from rox_dox.states import find_states


def _write_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "source-repo"
    repo.mkdir()
    for path, source in files.items():
        source_file = repo / path
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "add", *files],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add fixture"],
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


def test_enum_detection_requires_state_suffix(tmp_path: Path) -> None:
    repo, commit = _write_repo(
        tmp_path,
        {
            "backend/src/pkg/enums.py": (
                "import enum\n"
                "from enum import Enum, IntEnum, StrEnum\n"
                "from pydantic import BaseModel\n"
                "\n"
                "class TaskStatus(str, Enum):\n"
                "    PENDING = 'pending'\n"
                "    DONE = 'done'\n"
                "\n"
                "class RunPhase(StrEnum):\n"
                "    QUEUED = 'queued'\n"
                "\n"
                "class JobStage(IntEnum):\n"
                "    FIRST = 1\n"
                "\n"
                "class OrderState(enum.Enum):\n"
                "    OPEN = 'open'\n"
                "\n"
                "class Color(Enum):\n"
                "    RED = 'red'\n"
                "\n"
                "class FooStatus(BaseModel):\n"
                "    name: str\n"
            )
        },
    )

    report = find_states(repo, commit)

    by_name = {state_enum.enum: state_enum for state_enum in report.enums}
    assert set(by_name) == {"TaskStatus", "RunPhase", "JobStage", "OrderState"}
    task_status = by_name["TaskStatus"]
    assert task_status.path == "backend/src/pkg/enums.py"
    assert task_status.line == 5
    assert [(member.name, member.line) for member in task_status.members] == [
        ("PENDING", 6),
        ("DONE", 7),
    ]
    assert [member.name for member in by_name["OrderState"].members] == ["OPEN"]
    assert report.parse_errors == []


def test_columns_in_all_three_styles_and_imported_enum(tmp_path: Path) -> None:
    repo, commit = _write_repo(
        tmp_path,
        {
            "backend/src/pkg/enums.py": (
                "from enum import Enum\n"
                "\n"
                "class TaskStatus(str, Enum):\n"
                "    PENDING = 'pending'\n"
                "\n"
                "class OrderState(Enum):\n"
                "    OPEN = 'open'\n"
                "\n"
                "class JobStage(Enum):\n"
                "    FIRST = 'first'\n"
            ),
            "backend/src/app/models.py": (
                "import sqlalchemy\n"
                "from sqlalchemy import Column, Enum, Integer\n"
                "from sqlalchemy.orm import Mapped, mapped_column\n"
                "from pkg.enums import JobStage, OrderState, TaskStatus\n"
                "\n"
                "class Task:\n"
                "    __tablename__ = 'tasks'\n"
                "    id = Column(Integer, primary_key=True)\n"
                "    status = Column(Enum(TaskStatus), nullable=False)\n"
                "    state = mapped_column(sqlalchemy.Enum(OrderState))\n"
                "    stage: Mapped[JobStage] = mapped_column()\n"
                "    kind: Mapped[TaskStatus] = mapped_column()\n"
            ),
        },
    )

    report = find_states(repo, commit)
    by_name = {state_enum.enum: state_enum for state_enum in report.enums}

    columns = [
        (column.table, column.column, column.path, column.line)
        for enum_name in ("TaskStatus", "OrderState", "JobStage")
        for column in by_name[enum_name].columns
    ]
    assert columns == [
        ("tasks", "status", "backend/src/app/models.py", 9),
        ("tasks", "state", "backend/src/app/models.py", 10),
        ("tasks", "stage", "backend/src/app/models.py", 11),
    ]


def test_set_detection(tmp_path: Path) -> None:
    repo, commit = _write_repo(
        tmp_path,
        {
            "backend/src/pkg/enums.py": (
                "from enum import Enum\n"
                "\n"
                "class TaskStatus(str, Enum):\n"
                "    PENDING = 'pending'\n"
                "    DONE = 'done'\n"
                "\n"
                "class OrderState(Enum):\n"
                "    OPEN = 'open'\n"
            ),
            "backend/src/app/jobs.py": (
                "from pkg.enums import OrderState, TaskStatus\n"
                "\n"
                "def close(task, worker):\n"
                "    task.status = TaskStatus.DONE\n"
                "    worker.state = OrderState.OPEN.value\n"
                "    status = TaskStatus.DONE\n"
            ),
        },
    )

    report = find_states(repo, commit)
    by_name = {state_enum.enum: state_enum for state_enum in report.enums}

    assert [
        (state_set.member, state_set.path, state_set.line)
        for state_set in by_name["TaskStatus"].sets
    ] == [("DONE", "backend/src/app/jobs.py", 4)]
    assert [
        (state_set.member, state_set.path, state_set.line)
        for state_set in by_name["OrderState"].sets
    ] == [("OPEN", "backend/src/app/jobs.py", 5)]


def test_transitions(tmp_path: Path) -> None:
    repo, commit = _write_repo(
        tmp_path,
        {
            "backend/src/pkg/enums.py": (
                "from enum import Enum\n"
                "\n"
                "class TaskStatus(str, Enum):\n"
                "    PENDING = 'pending'\n"
                "    RUNNING = 'running'\n"
                "    DONE = 'done'\n"
            ),
            "backend/src/app/jobs.py": (
                "from pkg.enums import TaskStatus\n"
                "\n"
                "def advance(task, other):\n"
                "    if task.status == TaskStatus.PENDING:\n"
                "        task.status = TaskStatus.RUNNING\n"
                "        other.status = TaskStatus.DONE\n"
                "    if task.status in (TaskStatus.RUNNING, TaskStatus.PENDING):\n"
                "        task.status = TaskStatus.DONE\n"
            ),
        },
    )

    report = find_states(repo, commit)
    (task_status,) = report.enums

    transitions = [
        (transition.from_member, transition.to_member, transition.line)
        for transition in task_status.transitions
    ]
    assert transitions == [
        ("PENDING", "RUNNING", 5),
        ("PENDING", "DONE", 8),
        ("RUNNING", "DONE", 8),
    ]


def test_unresolved_and_unknown_members_are_skipped(tmp_path: Path) -> None:
    repo, commit = _write_repo(
        tmp_path,
        {
            "backend/src/pkg/a.py": (
                "from enum import Enum\n\nclass CellStatus(str, Enum):\n    X = 'x'\n"
            ),
            "backend/src/pkg/b.py": (
                "from enum import Enum\n\nclass CellStatus(str, Enum):\n    X = 'x'\n"
            ),
            "backend/src/pkg/enums.py": (
                "from enum import Enum\n"
                "\n"
                "class TaskStatus(str, Enum):\n"
                "    DONE = 'done'\n"
            ),
            "backend/src/app/unresolved.py": (
                "from somewhere.unknown import *\n"
                "\n"
                "def go(job):\n"
                "    job.status = CellStatus.X\n"
            ),
            "backend/src/app/resolved.py": (
                "from pkg.a import CellStatus\n"
                "\n"
                "def go(job):\n"
                "    job.status = CellStatus.X\n"
            ),
            "backend/src/app/unknown_member.py": (
                "from pkg.enums import TaskStatus\n"
                "\n"
                "def go(task):\n"
                "    task.status = TaskStatus.NOT_A_MEMBER\n"
            ),
        },
    )

    report = find_states(repo, commit)
    by_path = {state_enum.path: state_enum for state_enum in report.enums}

    cell_a = by_path["backend/src/pkg/a.py"]
    cell_b = by_path["backend/src/pkg/b.py"]
    task_status = by_path["backend/src/pkg/enums.py"]
    assert [
        (state_set.member, state_set.path, state_set.line) for state_set in cell_a.sets
    ] == [("X", "backend/src/app/resolved.py", 4)]
    assert cell_b.sets == []
    assert task_status.sets == []


def test_cli_states_json_and_bad_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, commit = _write_repo(
        tmp_path,
        {
            "backend/src/pkg/enums.py": (
                "from enum import Enum\n"
                "\n"
                "class TaskStatus(str, Enum):\n"
                "    DONE = 'done'\n"
            )
        },
    )

    assert cli.main(["states", "--repo", str(repo), "--commit", commit, "--json"]) == 0
    output = capsys.readouterr().out
    document = json.loads(output)
    assert isinstance(document, dict)
    assert [item["enum"] for item in document["enums"]] == ["TaskStatus"]

    assert cli.main(["states", "--repo", str(repo), "--commit", "0" * 40]) == 2
