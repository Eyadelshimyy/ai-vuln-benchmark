"""Public YAML parsing preserves safe construction and fresh object ownership."""

import json
import subprocess
import sys
from datetime import date

import pytest
import yaml

from ellements.yaml import load_yaml


@pytest.mark.parametrize("content", [
    "",
    "held: true\ncount: 012\nvalue: null\n",
    "date: 2026-09-12\n",
    "a: &base {name: owner}\nb: *base\n",
    "value: ! text\n",
    "text: |\n  indented\ttab\n",
    b"\xef\xbb\xbfvalue: true\n",
    "value: text\n".encode("utf-16"),
])
def test_safe_values_and_encoding(content):
    assert load_yaml(content) == yaml.safe_load(content)


@pytest.mark.parametrize("content", [
    "value: [unfinished\n",
    "value: \ttab\n",
    "value: |# comment\n",
    "!!python/object/apply:builtins.str [untrusted]",
    b"value: \xff\n",
    "value: text\n".encode("utf-32"),
])
def test_safe_rejection_and_portable_diagnostics(content):
    with pytest.raises(yaml.YAMLError) as expected:
        yaml.safe_load(content)
    with pytest.raises(type(expected.value)) as actual:
        load_yaml(content)
    assert str(actual.value) == str(expected.value)


def test_parsed_objects_are_not_cached():
    source = "a: &base {held: true}\nb: *base\n"
    first = load_yaml(source)
    first["a"]["held"] = False
    second = load_yaml(source)
    assert second["a"]["held"] is True
    assert second["a"] is second["b"]
    assert type(load_yaml("date: 2026-09-12\n")["date"]) is date


def test_cold_parser_does_not_load_provider_integrations():
    result = subprocess.run(
        [
            sys.executable, "-I", "-c",
            "import json,sys;from ellements.yaml import load_yaml;"
            "assert load_yaml('held: true') == {'held': True};"
            "print(json.dumps(sorted(set(sys.modules)&"
            "{'ellements.core','litellm','openai','anthropic','pandas','numpy'})))",
        ],
        capture_output=True, check=True, timeout=10,
    )
    assert json.loads(result.stdout) == []
