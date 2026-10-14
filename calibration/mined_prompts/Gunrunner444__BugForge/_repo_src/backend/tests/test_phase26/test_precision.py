"""Python keyword arguments, lexical framework binding, secrets, and crypto."""

from __future__ import annotations

from pathlib import Path

from app.domain.security import VulnerabilityClass
from app.security.engine import SecurityAnalysisEngine

SQL = VulnerabilityClass.SQL_INJECTION
CMD = VulnerabilityClass.COMMAND_INJECTION
CRYPTO = VulnerabilityClass.WEAK_CRYPTOGRAPHY
SECRET = VulnerabilityClass.HARDCODED_SECRET


def _scan(tmp_path: Path, name: str, source: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return SecurityAnalysisEngine().analyze_repository(tmp_path, [path])


def _has(result, kind: VulnerabilityClass) -> bool:
    return any(obs.vulnerability_class is kind for obs in result.observations)


def test_python_keyword_shell_does_not_hide_command_injection(tmp_path: Path) -> None:
    source = """import subprocess
def run(request):
    user = request.args.get('q')
    subprocess.run(['git', user])
    subprocess.run(['git', user], shell=True)
    subprocess.run([user])
    subprocess.run([user], shell=True)
    subprocess.run(['/bin/sh', '-c', user])
    subprocess.run(['/bin/bash', '-c', user])
"""
    result = _scan(tmp_path, "cmd.py", source)
    lines = {obs.line for obs in result.observations if obs.vulnerability_class is CMD}
    assert 4 not in lines
    assert lines == {5, 6, 7, 8, 9}


def test_fastapi_query_respects_lexical_scope(tmp_path: Path) -> None:
    recognized = _scan(
        tmp_path / "ok",
        "ok.py",
        "from fastapi import Query\n"
        "def f(user_value):\n"
        "    q = Query(user_value)\n"
        "    cursor.execute(q)\n",
    )
    assert _has(recognized, SQL)

    nested = _scan(
        tmp_path / "nested",
        "nested.py",
        "from fastapi import Query\n"
        "def f(user_value):\n"
        "    def Query(value):\n"
        "        return value\n"
        "    q = Query(user_value)\n"
        "    cursor.execute(q)\n"
        "def g(user_value):\n"
        "    q = Query(user_value)\n"
        "    cursor.execute(q)\n",
    )
    sql_lines = {obs.line for obs in nested.observations if obs.vulnerability_class is SQL}
    assert 6 not in sql_lines
    assert sql_lines == {9}

    rebound = _scan(
        tmp_path / "re",
        "re.py",
        "from fastapi import Query\n"
        "def f(user_value):\n"
        "    Query = user_value\n"
        "    cursor.execute(Query)\n",
    )
    assert not _has(rebound, SQL)

    alias = _scan(
        tmp_path / "alias",
        "alias.py",
        "import fastapi as fa\n"
        "def f(user_value):\n"
        "    q = fa.Query(user_value)\n"
        "    cursor.execute(q)\n",
    )
    assert _has(alias, SQL)

    module_rebind = _scan(
        tmp_path / "mod",
        "mod.py",
        "import fastapi as fa\n"
        "def f(user_value):\n"
        "    fa = user_value\n"
        "    q = fa.Query(user_value)\n"
        "    cursor.execute(q)\n",
    )
    assert not _has(module_rebind, SQL)

    similar = _scan(
        tmp_path / "sim",
        "sim.py",
        "from notfastapi import Query\n"
        "def f(user_value):\n"
        "    q = Query(user_value)\n"
        "    cursor.execute(q)\n",
    )
    assert not _has(similar, SQL)

    unknown = _scan(
        tmp_path / "unk",
        "unk.py",
        "from app.local import Query\n"
        "def f(user_value):\n"
        "    q = Query(user_value)\n"
        "    cursor.execute(q)\n",
    )
    assert not _has(unknown, SQL)


def test_same_line_secrets_are_redacted(tmp_path: Path) -> None:
    secret = "sk_live_same_line_secret"
    pem = "-----BEGIN PRIVATE KEY-----MII_SAME_LINE-----END PRIVATE KEY-----"
    source = (
        f'API_KEY = "{secret}"\n'
        f"password = '{secret}'\n"
        f'config = {{"password": "{secret}"}}\n'
        'DB = "postgres://app:s3cret-pass@db.internal/app"\n'
        'TOKEN = "Bearer supersecrettokenvalue"\n'
        'AWS = "AKIAIOSFODNN7EXAMPLE"\n'
        f'KEY = "{pem}"\n'
    )
    result = _scan(tmp_path, "secrets.py", source)
    assert _has(result, SECRET)
    blob = " ".join(obs.evidence_text for obs in result.observations)
    for hidden in (
        secret,
        "s3cret-pass",
        "supersecrettokenvalue",
        "AKIAIOSFODNN7EXAMPLE",
        "MII_SAME_LINE",
    ):
        assert hidden not in blob
    assert "***" in blob


def test_weak_crypto_matches_algorithm_values(tmp_path: Path) -> None:
    bad = _scan(
        tmp_path / "bad",
        "bad.py",
        "import hashlib\n"
        "def digest(data):\n"
        "    hashlib.md5(data)\n"
        "    hashlib.sha1(data)\n"
        "    hashlib.new('md5')\n"
        "    Cipher.getInstance('AES/ECB/PKCS5Padding')\n",
    )
    assert _has(bad, CRYPTO)
    clean = _scan(
        tmp_path / "clean",
        "clean.py",
        "import hashlib\n"
        "def digest(data, variable_name):\n"
        "    message = 'this text mentions md5'\n"
        "    hashlib.new('not-md5')\n"
        "    hashlib.new(variable_name)\n"
        "    return message\n",
    )
    assert not _has(clean, CRYPTO)
