"""Declarative model schema: the refusals a journey can't pin precisely."""

import json

import pytest

from tracebi.model.model_spec import load_model_spec, validate_model_file, validate_model_spec


def _doc(**over):
    doc = {
        "name": "m",
        "connectors": [{"name": "w", "type": "duckdb", "database": "data/w.duckdb"}],
        "tables": [{"name": "t", "connector": "w", "source": "t"}],
        "measures": [{"name": "n", "column": "x", "agg": "sum"}],
    }
    doc.update(over)
    return doc


def _write(tmp_path, doc, stem="m"):
    (tmp_path / "models").mkdir(exist_ok=True)
    path = tmp_path / "models" / f"{stem}.json"
    path.write_text(json.dumps(doc))
    return path


def test_a_valid_document_has_no_errors():
    assert validate_model_spec(_doc()) == []


def test_unknown_keys_are_errors_with_their_path():
    errors = validate_model_spec(_doc(measures=[{"name": "n", "column": "x", "agg": "sum", "colum": 1}]))
    assert len(errors) == 1 and errors[0].startswith("measures[0]: unknown key 'colum'")
    assert "Did you mean 'column'" in errors[0]
    assert validate_model_spec(_doc(extra=1))[0].startswith("model: unknown key 'extra'")


@pytest.mark.parametrize("key", ["url", "password", "token", "credentials"])
def test_a_connector_cannot_carry_credentials(key):
    doc = _doc(connectors=[{"name": "w", "type": "duckdb", "database": "d.duckdb", key: "s3cret"}])
    errors = validate_model_spec(doc)
    assert any(f"connectors[0]: unknown key '{key}'" in e for e in errors)


@pytest.mark.parametrize("database", ["/etc/x.duckdb", "../x.duckdb", "data/../../x.duckdb", "C:\\x.duckdb"])
def test_a_database_must_stay_inside_the_project(database):
    doc = _doc(connectors=[{"name": "w", "type": "duckdb", "database": database}])
    assert any(e.startswith("connectors[0].database") for e in validate_model_spec(doc))


def test_the_name_must_equal_the_file_stem(tmp_path):
    path = _write(tmp_path, _doc(name="other"))
    [err] = validate_model_file(path)
    assert "must equal the file name 'm'" in err


def test_a_python_model_wins_over_a_json_of_the_same_name(tmp_path):
    from tracebi.model_registry import ModelRegistry

    _write(tmp_path, _doc())
    (tmp_path / "models" / "m.py").write_text("from tracebi import DataModel\nmodel = DataModel('m')\n")
    reg = ModelRegistry()
    assert reg.auto_discover(str(tmp_path / "models")) == ["m"]
    assert reg.clashes() == {str(tmp_path / "models" / "m.json"):
                             "m.json refused: m.py exists; a model is one file, so delete one"}
    assert reg.get("m").measures() == {}          # the Python one


def test_a_good_file_compiles_without_reading_the_warehouse(tmp_path):
    model = load_model_spec(_write(tmp_path, _doc()))
    assert list(model.measures()) == ["n"]
    assert not (tmp_path / "data").exists()


def _yaml(tmp_path, text, stem="m"):
    (tmp_path / "models").mkdir(exist_ok=True)
    path = tmp_path / "models" / f"{stem}.yaml"
    path.write_text(text)
    return path


def test_a_repeated_yaml_key_is_an_error_not_last_wins(tmp_path):
    path = _yaml(tmp_path, "name: m\nmeasures:\n  - name: n\n    agg: sum\n    agg: mean\n")
    errors = validate_model_file(path)
    assert len(errors) == 1 and "duplicate key 'agg'" in errors[0]


def test_a_yaml_object_tag_is_refused_not_constructed(tmp_path):
    marker = tmp_path / "ran"
    path = _yaml(tmp_path, "name: !!python/object/apply:pathlib.Path.touch "
                           f"[!!python/object/apply:pathlib.Path [{marker}]]\n")
    errors = validate_model_file(path)
    assert errors and "not valid YAML" in errors[0]
    assert not marker.exists()


def test_one_model_one_file_across_the_declarative_forms(tmp_path):
    from tracebi.model_registry import ModelRegistry

    models = tmp_path / "models"
    _yaml(tmp_path, "name: both\n", stem="both")
    (models / "both.json").write_text('{"name": "both"}')
    _yaml(tmp_path, "name: solo\n", stem="solo")
    _yaml(tmp_path, "name: py\n", stem="py")
    (models / "py.py").write_text("from tracebi import DataModel\nmodel = DataModel('py')\n")
    reg = ModelRegistry()
    assert reg.auto_discover(str(models)) == ["py", "solo"]
    refused = {p.rsplit("/", 1)[1]: why for p, why in reg.clashes().items()}
    assert set(refused) == {"both.yaml", "both.json", "py.yaml"}
    assert "both.json" in refused["both.yaml"] and "both.yaml" in refused["both.json"]
    assert "py.py exists" in refused["py.yaml"]
    (models / "both.json").unlink()                 # delete one: the other is the model
    assert reg.auto_discover(str(models)) == ["both", "py", "solo"]
    assert set(reg.clashes()) == {str(models / "py.yaml")}
