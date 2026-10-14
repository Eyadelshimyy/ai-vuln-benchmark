from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import rox_dox.cli as cli
from rox_dox.relations import find_relation_candidates


def _commit_relation_fixture(repo: Path) -> str:
    sources = {
        "backend/src/models.py": (
            "from sqlalchemy.orm import Mapped, mapped_column, relationship\n"
            "\n"
            "class TaskRun:\n"
            '    __tablename__ = "task_run"\n'
            "    run_id: Mapped[str] = mapped_column()\n"
            "\n"
            "class TaskRunLog:\n"
            '    __tablename__ = "task_run_log"\n'
            '    run_id: Mapped[str] = mapped_column()  # ForeignKey("task_run.run_id")\n'
            "    task_run = relationship(\n"
            '        "TaskRun",\n'
            '        primaryjoin="foreign(TaskRunLog.run_id) == remote(TaskRun.run_id)",\n'
            "    )\n"
            "\n"
            "class Outside:\n"
            '    __tablename__ = "outside"\n'
            "    run_id: Mapped[str] = mapped_column()\n"
        ),
        "backend/src/queries.py": (
            "first = TaskRunLog.run_id == TaskRun.run_id\n"
            "filtered = TaskRunLog.run_id == Outside.run_id\n"
        ),
        "backend/src/z_queries.py": (
            "duplicate = TaskRunLog.run_id == TaskRun.run_id\n"
        ),
    }
    for relative_path, source in sources.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    subprocess.run(
        ["git", "-C", str(repo), "add", *sources],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add relation fixture"],
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_relations_cli_reports_pinned_candidates_sorted_and_filtered(
    git_repo: tuple[Path, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, _ = git_repo
    commit = _commit_relation_fixture(repo)
    uncommitted_path = repo / "backend/src/worktree.py"
    uncommitted_path.write_text(
        "uncommitted = TaskRunLog.run_id == TaskRun.run_id\n",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "relations",
            "--repo",
            str(repo),
            "--commit",
            commit,
            "--tables",
            "task_run,task_run_log",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.splitlines() == [
        "task_run_log.run_id\ttask_run.run_id\tcommented_fk\tbackend/src/models.py:9",
        "task_run_log.run_id\ttask_run.run_id\tcomparison\tbackend/src/queries.py:1",
        "task_run_log.run_id\ttask_run.run_id\tprimaryjoin\tbackend/src/models.py:12",
    ]
    assert "outside.run_id" not in output.out
    assert "worktree.py" not in output.out


def test_relations_resolve_duplicate_class_names_from_imports_and_mark_ambiguity(
    git_repo: tuple[Path, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, _ = git_repo
    sources = {
        "backend/src/a/person.py": (
            'class Person:\n    __tablename__ = "person"\n    id = Column()\n'
        ),
        "backend/src/b/person.py": (
            'class Person:\n    __tablename__ = "entity_person"\n    id = Column()\n'
        ),
        "backend/src/imported_query.py": (
            "from a.person import Person\n"
            "from b.person import Person as EntityPerson\n"
            "\n"
            "resolved = Person.id == EntityPerson.id\n"
        ),
        "backend/src/ambiguous_query.py": ("ambiguous = Person.id == Person.id\n"),
    }
    for relative_path, source in sources.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", *sources],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add duplicate class fixture"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert (
        cli.main(
            [
                "relations",
                "--repo",
                str(repo),
                "--commit",
                commit,
                "--tables",
                "person,entity_person",
            ]
        )
        == 0
    )
    output = capsys.readouterr()

    assert output.err == ""
    assert "person.id\tentity_person.id\tcomparison\t" in output.out
    ambiguous_candidates = {
        line for line in output.out.splitlines() if "ambiguous_query.py" in line
    }
    assert ambiguous_candidates == {
        f"{src}.id\t{dst}.id\tcomparison:ambiguous\tbackend/src/ambiguous_query.py:1"
        for src in ("person", "entity_person")
        for dst in ("person", "entity_person")
    }


def test_id_column_candidates_use_longest_table_suffix_and_column_match(
    git_repo: tuple[Path, str],
) -> None:
    repo, _ = git_repo
    path = repo / "backend/src/id_models.py"
    source = (
        "from sqlalchemy import Column, String\n"
        "\n"
        "class User:\n"
        '    __tablename__ = "user"\n'
        "    rox_user_id = Column(String)\n"
        "    user_id = Column(String)\n"
        "    public_id = Column(String)\n"
        "    rox_org_id = Column(String)\n"
        "\n"
        "class Integration:\n"
        '    __tablename__ = "integration"\n'
        "    public_id = Column(String)\n"
        "\n"
        "class CalendarEvent:\n"
        '    __tablename__ = "calendar_event"\n'
        "    id = Column(String)\n"
        "\n"
        "class Event:\n"
        '    __tablename__ = "event"\n'
        "    id = Column(String)\n"
        "\n"
        "class Public:\n"
        '    __tablename__ = "public"\n'
        "    id = Column(String)\n"
        "\n"
        "class RoxOrg:\n"
        '    __tablename__ = "rox_org"\n'
        "    id = Column(String)\n"
        "\n"
        "class EmailMessageRecipient:\n"
        '    __tablename__ = "email_message_recipient"\n'
        "    recipient_rox_user_id = Column(String)\n"
        "    integration_public_id = Column(String)\n"
        "    calendar_event_id = Column(String)\n"
        "    missing_id = Column(String)\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "backend/src/id_models.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add ID relation fixture"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    table_names = [
        "user",
        "integration",
        "calendar_event",
        "event",
        "public",
        "rox_org",
        "email_message_recipient",
    ]
    candidates = find_relation_candidates(repo, commit, table_names)
    id_candidates = {
        (candidate.src, candidate.dst, candidate.line)
        for candidate in candidates
        if candidate.signal == "id_column"
    }
    source_lines = source.splitlines()
    line_numbers = {
        definition: source_lines.index(definition) + 1
        for definition in (
            "    recipient_rox_user_id = Column(String)",
            "    integration_public_id = Column(String)",
            "    calendar_event_id = Column(String)",
        )
    }

    assert id_candidates == {
        (
            "email_message_recipient.recipient_rox_user_id",
            "user.rox_user_id",
            line_numbers["    recipient_rox_user_id = Column(String)"],
        ),
        (
            "email_message_recipient.integration_public_id",
            "integration.public_id",
            line_numbers["    integration_public_id = Column(String)"],
        ),
        (
            "email_message_recipient.calendar_event_id",
            "calendar_event.id",
            line_numbers["    calendar_event_id = Column(String)"],
        ),
    }
