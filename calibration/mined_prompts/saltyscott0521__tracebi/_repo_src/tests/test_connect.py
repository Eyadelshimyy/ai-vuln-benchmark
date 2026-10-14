"""``tracebi connect``: secrets stay in ``.env``, a failed test writes nothing."""

import ast
from pathlib import Path

from tracebi.cli import main


def _run(monkeypatch, tmp_path, *argv):
    monkeypatch.chdir(tmp_path)
    return main(list(argv))


def _snowflake(*extra):
    return [
        "connect", "wh", "--kind", "snowflake", "--no-test",
        "--account", "acct", "--user", "usr", "--password", "s3cret-token",
        "--warehouse", "WH", "--database", "DB", "--schema", "PUBLIC",
        *extra,
    ]


class TestConnectFlags:
    def test_missing_flags_without_a_tty_exit_2_and_name_them(self, tmp_path, monkeypatch, capsys):
        code = _run(monkeypatch, tmp_path, "connect", "wh", "--kind", "snowflake")
        assert code == 2
        err = capsys.readouterr().err
        for flag in ("--account", "--user", "--password", "--warehouse",
                     "--database", "--schema"):
            assert flag in err
        assert not (tmp_path / ".env").exists()
        assert not (tmp_path / "models").exists()

    def test_missing_kind_names_the_flag(self, tmp_path, monkeypatch, capsys):
        code = _run(monkeypatch, tmp_path, "connect", "wh")
        assert code == 2
        assert "--kind" in capsys.readouterr().err

    def test_failing_test_writes_nothing(self, tmp_path, monkeypatch, capsys):
        missing = tmp_path / "missing.duckdb"
        code = _run(
            monkeypatch, tmp_path,
            "connect", "wh", "--kind", "duckdb", "--database", str(missing),
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "FileNotFoundError" in err or "does not exist" in err
        assert not (tmp_path / ".env").exists()
        assert not (tmp_path / "models").exists()
        assert not (tmp_path / ".gitignore").exists()

    def test_existing_env_key_is_not_overwritten_without_force(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "wh.duckdb"
        import duckdb
        duckdb.connect(str(db)).close()
        (tmp_path / ".env").write_text("OTHER=keep\nTRACEBI_WH_DATABASE=old-value\n")
        code = _run(
            monkeypatch, tmp_path,
            "connect", "--python", "wh", "--kind", "duckdb", "--database", str(db),
        )
        assert code == 1
        text = (tmp_path / ".env").read_text()
        assert "OTHER=keep" in text
        assert "old-value" in text
        assert str(db) not in text
        assert not (tmp_path / "models" / "_connections" / "wh.py").exists()
        captured = capsys.readouterr()
        assert "old-value" not in captured.out
        assert str(db) not in captured.out

    def test_force_replaces_the_key_and_keeps_other_lines(self, tmp_path, monkeypatch):
        db = tmp_path / "wh.duckdb"
        import duckdb
        duckdb.connect(str(db)).close()
        (tmp_path / ".env").write_text("# comment\nOTHER=keep\nTRACEBI_WH_DATABASE=old-value\n")
        code = _run(
            monkeypatch, tmp_path,
            "connect", "--python", "wh", "--kind", "duckdb", "--database", str(db), "--force",
        )
        assert code == 0
        lines = (tmp_path / ".env").read_text().splitlines()
        assert lines[0] == "# comment"
        assert "OTHER=keep" in lines
        assert any(line.startswith("TRACEBI_WH_DATABASE=") and "old-value" not in line for line in lines)
        assert (tmp_path / "models" / "_connections" / "wh.py").is_file()

    def test_secret_never_appears_in_stdout_or_the_module(self, tmp_path, monkeypatch, capsys):
        url = "postgresql://analyst:s3cret-token@localhost:5432/sales"
        code = _run(
            monkeypatch, tmp_path,
            "connect", "--python", "wh", "--kind", "postgres", "--url", url, "--no-test",
        )
        assert code == 0
        captured = capsys.readouterr()
        assert "s3cret-token" not in captured.out
        assert "s3cret-token" not in captured.err
        src = (tmp_path / "models" / "_connections" / "wh.py").read_text()
        assert "s3cret-token" not in src
        assert "os.environ" in src
        assert "load_dotenv" in src
        assert "s3cret-token" in (tmp_path / ".env").read_text()
        assert "TRACEBI_WH_URL" in (tmp_path / ".env").read_text()

    def test_duckdb_test_prints_the_count_not_the_path(self, tmp_path, monkeypatch, capsys):
        import duckdb
        db = tmp_path / "wh.duckdb"
        con = duckdb.connect(str(db))
        con.execute("create table t (id integer)")
        con.close()
        code = _run(
            monkeypatch, tmp_path,
            "connect", "--python", "wh", "--kind", "duckdb", "--database", str(db),
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "1 table" in out
        assert str(db) not in out
        assert "TRACEBI_WH_DATABASE" in (tmp_path / ".env").read_text()
        src = (tmp_path / "models" / "_connections" / "wh.py").read_text()
        ast.parse(src)
        assert str(db) not in src

    def test_gitignore_gains_dotenv_without_dropping_other_lines(self, tmp_path, monkeypatch):
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        code = _run(
            monkeypatch, tmp_path,
            "connect", "wh", "--kind", "postgres",
            "--url", "postgresql://u:s3cret-token@h/db", "--no-test",
        )
        assert code == 0
        text = (tmp_path / ".gitignore").read_text()
        assert text.startswith("*.pyc\n")
        assert ".env" in text.splitlines()


class TestGeneratedConnections:
    def _source(self, tmp_path, monkeypatch, argv) -> str:
        assert _run(monkeypatch, tmp_path, argv[0], "--python", *argv[1:]) == 0
        return (tmp_path / "models" / "_connections" / "wh.py").read_text()

    def test_each_kind_parses(self, tmp_path, monkeypatch):
        specs = [
            (["connect", "wh", "--kind", "postgres", "--no-test",
              "--url", "postgresql://u:s3cret-token@h/db"], "SQLConnector"),
            (_snowflake(), "SnowflakeConnector"),
            (["connect", "wh", "--kind", "bigquery", "--no-test",
              "--project", "proj", "--dataset", "analytics"], "BigQueryConnector"),
        ]
        for i, (argv, marker) in enumerate(specs):
            project = tmp_path / f"p{i}"
            project.mkdir()
            src = self._source(project, monkeypatch, argv)
            ast.parse(src)
            assert marker in src
            assert "os.environ" in src
            assert "s3cret-token" not in src
            if marker == "SnowflakeConnector":
                assert "TRACEBI_WH_ROLE" in src
                assert "TRACEBI_WH_PASSWORD" in src


def test_connect_module_does_not_import_dotenv():
    """The framework never loads ``.env``. The string lives only in the template."""
    import tracebi.connect as connect

    tree = ast.parse(Path(connect.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "dotenv"
        if isinstance(node, ast.Import):
            assert all(alias.name != "dotenv" for alias in node.names)


def test_new_model_from_without_tables_exits_2(tmp_path, monkeypatch, capsys):
    code = _run(monkeypatch, tmp_path, "new-model", "Sales", "--from", "wh")
    assert code == 2
    assert "--tables" in capsys.readouterr().err
    assert not (tmp_path / "models" / "sales.py").exists()


def test_missing_table_writes_no_model(tmp_path, monkeypatch, capsys):
    import duckdb
    db = tmp_path / "wh.duckdb"
    duckdb.connect(str(db)).close()
    assert _run(
        monkeypatch, tmp_path,
        "connect", "wh", "--kind", "duckdb", "--database", str(db), "--no-test",
    ) == 0
    capsys.readouterr()
    code = _run(
        monkeypatch, tmp_path,
        "new-model", "Sales", "--from", "wh", "--tables", "no_such",
    )
    assert code == 1
    assert not (tmp_path / "models" / "sales.py").exists()
    assert (tmp_path / "connections" / "wh.yaml").is_file()


def test_help_lists_connect(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "connect" in capsys.readouterr().out
