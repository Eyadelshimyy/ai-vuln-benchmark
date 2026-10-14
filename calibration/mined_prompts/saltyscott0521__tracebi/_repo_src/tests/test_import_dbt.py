"""``tracebi import dbt`` drafts a model from a manifest and does not run dbt."""

import ast
import json
import runpy
from pathlib import Path

from tracebi.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "dbt" / "manifest.json"


def _run(monkeypatch, tmp_path, *argv):
    monkeypatch.chdir(tmp_path)
    return main(list(argv))


def _connection(tmp_path, name="demo"):
    path = tmp_path / "models" / "_connections" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "from tracebi.connectors.memory_connector import MemoryConnector\n"
        f"connector = MemoryConnector({name!r}, tables={{}})\n",
        encoding="utf-8",
    )
    return path


def _calls(src: str, attr: str) -> list[ast.Call]:
    found = []
    for node in ast.walk(ast.parse(src)):
        func = getattr(node, "func", None)
        if isinstance(node, ast.Call) and isinstance(func, ast.Attribute) and func.attr == attr:
            found.append(node)
    return found


def test_import_dbt_drafts_tables_and_leaves_guesses_commented(tmp_path, monkeypatch):
    _connection(tmp_path)
    code = _run(
        monkeypatch, tmp_path,
        "import", "dbt", str(FIXTURE),
        "--connection", "demo", "--name", "dbt_marts",
    )
    assert code == 0
    path = tmp_path / "models" / "dbt_marts.py"
    src = path.read_text(encoding="utf-8")
    ast.parse(src)
    assert "load_dotenv" not in src
    assert "_connections/demo.py" in src
    assert "# DRAFT: review" in src
    assert "int_ephemeral" not in src
    assert "old_orders" not in src
    assert "raw_customers" not in src
    assert "stg_payments" in src
    assert not _calls(src, "add_relationship")
    assert not _calls(src, "add_measure")
    assert not _calls(src, "add_fact")
    assert not _calls(src, "add_dimension")
    assert any(
        line.lstrip().startswith("#") and "add_measure" in line and "amount" in line
        for line in src.splitlines()
    )
    assert not any("margin_pct" in line and "add_measure" in line for line in src.splitlines())
    assert any(line.lstrip().startswith("#") and "add_relationship" in line for line in src.splitlines())

    model = runpy.run_path(str(path))["model"]
    info = model.info()
    assert info["name"] == "dbt_marts"
    assert info["connectors"] == ["demo"]
    tables = {table["name"]: table["source"] for table in info["tables"]}
    assert tables["customers"] == "customers"
    assert tables["other_pkg__customers"] == "customers"
    assert tables["dim_customer"] == "dim_customer_v2"
    assert tables["orders"] == "orders"
    assert tables["stg_payments"] == "stg_payments"
    assert info["relationships"] == []
    assert info["measures"] == []


def test_schema_filter_limits_models(tmp_path, monkeypatch):
    _connection(tmp_path)
    code = _run(
        monkeypatch, tmp_path,
        "import", "dbt", str(FIXTURE),
        "--connection", "demo", "--name", "analytics_only", "--schema", "Analytics",
    )
    assert code == 0
    src = (tmp_path / "models" / "analytics_only.py").read_text(encoding="utf-8")
    assert "stg_payments" not in src
    assert "orders" in src


def test_project_root_uses_the_dbt_project_name(tmp_path, monkeypatch):
    _connection(tmp_path)
    project = tmp_path / "jaffle"
    target = project / "target"
    target.mkdir(parents=True)
    target.joinpath("manifest.json").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    code = _run(monkeypatch, tmp_path, "import", "dbt", str(project), "--connection", "demo")
    assert code == 0
    assert (tmp_path / "models" / "jaffle_shop.py").is_file()


def test_folder_name_when_the_manifest_has_no_project_name(tmp_path, monkeypatch):
    _connection(tmp_path)
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del payload["metadata"]["project_name"]
    project = tmp_path / "acme_marts" / "target"
    project.mkdir(parents=True)
    project.joinpath("manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    code = _run(
        monkeypatch, tmp_path,
        "import", "dbt", str(tmp_path / "acme_marts"), "--connection", "demo",
    )
    assert code == 0
    assert (tmp_path / "models" / "acme_marts.py").is_file()


def test_missing_manifest_writes_nothing(tmp_path, monkeypatch, capsys):
    code = _run(monkeypatch, tmp_path, "import", "dbt", str(tmp_path / "nope"))
    assert code == 1
    assert "manifest.json" in capsys.readouterr().err
    assert not (tmp_path / "models").exists()

    project = tmp_path / "empty_project"
    project.mkdir()
    code = _run(monkeypatch, tmp_path, "import", "dbt", str(project))
    assert code == 1
    assert "target/manifest.json" in capsys.readouterr().err
    assert list(project.iterdir()) == []

    bad = tmp_path / "manifest.json"
    bad.write_text("{not json", encoding="utf-8")
    code = _run(monkeypatch, tmp_path, "import", "dbt", str(bad))
    assert code == 1
    assert "unreadable" in capsys.readouterr().err
    assert not (tmp_path / "models" / "manifest.py").exists()


def test_existing_file_is_kept_without_force(tmp_path, monkeypatch, capsys):
    _connection(tmp_path)
    dest = tmp_path / "models" / "dbt_marts.py"
    dest.write_text("# keep\n", encoding="utf-8")
    code = _run(
        monkeypatch, tmp_path,
        "import", "dbt", str(FIXTURE),
        "--connection", "demo", "--name", "dbt_marts",
    )
    assert code == 1
    assert dest.read_text(encoding="utf-8") == "# keep\n"
    assert "--force" in capsys.readouterr().err

    code = _run(
        monkeypatch, tmp_path,
        "import", "dbt", str(FIXTURE),
        "--connection", "demo", "--name", "dbt_marts", "--force",
    )
    assert code == 0
    text = dest.read_text(encoding="utf-8")
    assert text != "# keep\n"
    assert "DataModel" in text


def test_missing_connection_writes_nothing(tmp_path, monkeypatch, capsys):
    code = _run(
        monkeypatch, tmp_path,
        "import", "dbt", str(FIXTURE),
        "--connection", "demo", "--name", "dbt_marts",
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "demo" in err
    assert "connections/" in err
    assert not (tmp_path / "models" / "dbt_marts.py").exists()


def test_without_connection_the_tables_stay_commented(tmp_path, monkeypatch):
    code = _run(
        monkeypatch, tmp_path,
        "import", "dbt", str(FIXTURE), "--name", "dbt_marts",
    )
    assert code == 0
    src = (tmp_path / "models" / "dbt_marts.py").read_text(encoding="utf-8")
    assert "load_dotenv" not in src
    assert not _calls(src, "add_table")
    assert "customers" in src
    model = runpy.run_path(str(tmp_path / "models" / "dbt_marts.py"))["model"]
    info = model.info()
    assert info["name"] == "dbt_marts"
    assert info["tables"] == []
    assert info["connectors"] == []


def test_import_dbt_module_does_not_load_dotenv_or_run_dbt():
    import tracebi.dbt_import as dbt_import

    tree = ast.parse(Path(dbt_import.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "dotenv"
            assert node.module != "subprocess"
        if isinstance(node, ast.Import):
            assert all(alias.name not in {"dotenv", "subprocess"} for alias in node.names)


def test_help_lists_import_dbt(capsys):
    try:
        main(["import", "dbt", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "manifest.json" in out
    assert "--connection" in out
    assert "--schema" in out
    assert "does not run dbt" in out
