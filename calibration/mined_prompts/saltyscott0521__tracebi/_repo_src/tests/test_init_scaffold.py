"""
The init scaffold is the product's first run — it must walk the three-phase
workflow end to end and finish with `tracebi verify` reading REPRODUCES.

These tests run the scaffolded project through subprocesses (the exact
commands the scaffolded README gives a new user), so they are hermetic: no
shared registry state with the rest of the suite.
"""

import compileall
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tracebi import cli

pytest.importorskip("duckdb")


def _run(args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "tracebi.cli", *args],
        capture_output=True, text=True, cwd=str(cwd),
    )


class TestInitScaffold:
    def test_init_wires_the_gateway(self, tmp_path):
        proj = tmp_path / "proj"
        assert cli.main(["init", str(proj)]) == 0
        expected = {
            "mcpServers": {
                "tracebi": {"command": "tracebi", "args": ["mcp"]},
            },
        }
        for rel in (".mcp.json", Path(".cursor") / "mcp.json"):
            assert json.loads((proj / rel).read_text()) == expected
        readme = (proj / "README.md").read_text()
        assert "tracebi mcp config --client" in readme
        (proj / ".mcp.json").write_text('{"edited": true}\n', encoding="utf-8")
        assert cli.main(["init", str(proj)]) == 1
        assert (proj / ".mcp.json").read_text() == '{"edited": true}\n'

    def test_mcp_config_prints_json_and_never_a_token(self, monkeypatch, capsys):
        monkeypatch.setenv("TRACEBI_MCP_TOKEN", "super-secret-token-xyz")
        assert cli.main(["mcp", "config", "--client", "cursor"]) == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["mcpServers"]["tracebi"] == {
            "command": "tracebi", "args": ["mcp"],
        }
        assert "super-secret-token-xyz" not in out

        assert cli.main([
            "mcp", "config", "--client", "claude-code",
            "--http", "http://127.0.0.1:8765/mcp",
        ]) == 0
        http_out = capsys.readouterr().out
        http = json.loads(http_out)
        server = http["mcpServers"]["tracebi"]
        assert server["type"] == "http"
        assert server["url"] == "http://127.0.0.1:8765/mcp"
        assert server["headers"]["Authorization"] == "Bearer ${TRACEBI_MCP_TOKEN}"
        assert "super-secret-token-xyz" not in http_out
        assert "command" not in server

        assert cli.main([
            "mcp", "config", "--client", "cursor",
            "--http", "http://127.0.0.1:8765/mcp",
        ]) == 0
        cursor_out = capsys.readouterr().out
        cursor = json.loads(cursor_out)
        assert cursor["mcpServers"]["tracebi"]["headers"]["Authorization"] == (
            "Bearer ${env:TRACEBI_MCP_TOKEN}"
        )
        assert "super-secret-token-xyz" not in cursor_out

        assert cli.main(["mcp", "config", "--client", "claude-desktop"]) == 0
        desktop = json.loads(capsys.readouterr().out)
        command = desktop["mcpServers"]["tracebi"]["command"]
        assert Path(command).is_absolute()

        assert cli.main([
            "mcp", "config", "--client", "claude-desktop",
            "--http", "http://127.0.0.1:8765/mcp",
        ]) == 1
        refused = capsys.readouterr()
        assert refused.out == ""
        assert "{" not in refused.err
        with pytest.raises(json.JSONDecodeError):
            json.loads(refused.err)
        assert refused.err.strip() == (
            "Add the URL as a custom connector in Claude Desktop "
            "(Settings → Connectors)."
        )
        assert "super-secret-token-xyz" not in refused.err

    def test_existing_scheduled_dir_still_starts(self, tmp_path):
        proj = tmp_path / "proj"
        (proj / "scheduled").mkdir(parents=True)
        (proj / "scheduled" / "old.py").write_text("VALUE = 1\n", encoding="utf-8")
        for name in ("reports", "models", "pipelines"):
            (proj / name).mkdir()
        code = (
            "import os\n"
            f"os.chdir({str(proj)!r})\n"
            "os.environ['TRACEBI_APP'] = ''\n"
            "os.environ['TRACEBI_SCHEDULED_DIR'] = 'scheduled'\n"
            "os.environ['TRACEBI_REPORTS_DIR'] = 'reports'\n"
            "os.environ['TRACEBI_MODELS_DIR'] = 'models'\n"
            "os.environ['TRACEBI_PIPELINES_DIR'] = 'pipelines'\n"
            "from tracebi.web.api.main import app\n"
            "print('STARTED', app.title)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=str(proj))
        assert out.returncode == 0, out.stderr
        assert "STARTED" in out.stdout
        assert "deprecated" in out.stderr

    def test_scaffolded_model_is_lazy(self, tmp_path):
        """Importing the scaffolded model must not touch the warehouse —
        it has to load before phase ① has ever run."""
        proj = tmp_path / "proj"
        assert cli.main(["init", str(proj)]) == 0
        # No warehouse exists yet; a connect-at-import would fail here.
        out = subprocess.run(
            [sys.executable, "-c",
             "from tracebi.model_registry import get_model; "
             "m = get_model('sample_model'); print('lazy-ok', m.name)"],
            capture_output=True, text=True, cwd=str(proj),
        )
        assert out.returncode == 0, out.stderr
        assert "lazy-ok" in out.stdout


class TestNewTransform:
    def test_scaffold_compiles(self, tmp_path):
        out = _run(["new-transform", "Orders Clean",
                    "--transforms-dir", str(tmp_path / "transforms")], tmp_path)
        assert out.returncode == 0, out.stderr
        f = tmp_path / "transforms" / "orders_clean.py"
        assert f.is_file()
        assert compileall.compile_file(str(f), quiet=2)
        text = f.read_text()
        assert "DuckDBConnector" in text
        # Notebook-shaped: percent cells, top-level execution — every
        # notebook editor opens it as a notebook; python runs it as a script.
        assert "# %%" in text and "# %% [markdown]" in text


class TestNewModelScaffold:
    def test_no_connect_at_import(self, tmp_path):
        """The new-model template must construct lazily — discovery imports
        every model file, and a connect at import opens a connection (or
        fails outright) on every scan."""
        out = _run(["--models-dir", str(tmp_path / "models"),
                    "new-model", "My Model", "--python"], tmp_path)
        assert out.returncode == 0, out.stderr
        text = (tmp_path / "models" / "my_model.py").read_text()
        assert "\nmodel.connect()" not in text


class TestNotebookShapedTransforms:
    """Transforms are notebook-shaped .py (percent cells) — every notebook
    editor opens them as notebooks while the file stays reviewable Python —
    and literal .ipynb runs top-to-bottom fresh via run-transform."""

    def test_run_transform_executes_py_and_ipynb(self, tmp_path, monkeypatch):
        import json as _json
        proj = tmp_path / "proj"
        assert cli.main(["init", str(proj)]) == 0
        monkeypatch.chdir(proj)
        # the scaffolded notebook-shaped .py runs top-to-bottom
        assert cli.main(["run-transform", "sample_transform"]) == 0
        assert (proj / "data" / "warehouse.duckdb").exists()
        # a literal .ipynb transform runs fresh, cells concatenated in order
        nb = {"cells": [
            {"cell_type": "markdown", "source": ["# methodology\n"]},
            {"cell_type": "code", "source": ["x = 2\n"]},
            {"cell_type": "code",
             "source": ["open('nb_ran.txt', 'w').write(str(x * 21))\n"]},
        ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        (proj / "transforms" / "probe.ipynb").write_text(_json.dumps(nb))
        assert cli.main(["run-transform", "probe"]) == 0
        assert (proj / "nb_ran.txt").read_text() == "42"

    def test_run_transform_missing_is_a_clean_error(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"
        assert cli.main(["init", str(proj)]) == 0
        monkeypatch.chdir(proj)
        assert cli.main(["run-transform", "nope"]) == 1


class TestReportSend:
    """`tracebi report send` — scheduled-delivery v1: distribution with the
    receipt attached, gated by verification.

    The honesty rule under test: distribution never outruns verification.
    A receipt that does not verify is refused; --force pastes the failing
    verdict INTO the body, so the red flag travels WITH the report — never
    silently. No test opens a socket: SMTP is faked at the smtplib seam or
    send_report is captured at the CLI seam.
    """

    _PASSING = {
        "verdict": "reproduces",
        "verdict_detail": "REPRODUCES — every checked section matches "
                          "the manifest",
        "exit_code": 0, "ok": True, "figures": [{"id": "kpi_total"}],
    }
    _FAILING = {
        "verdict": "not_reproduced",
        "verdict_detail": "NOT REPRODUCED — section(s) could not be shown "
                          "to reproduce; explain before anyone reads the "
                          "number",
        "exit_code": 1, "ok": False, "figures": [{"id": "kpi_total"}],
    }

    @pytest.fixture(scope="class")
    def proj(self, tmp_path_factory):
        """One scaffolded project with a sunk warehouse, shared by the
        class — each test rebuilds the report in-process (that is what
        `send` does) but the transform runs once."""
        proj = tmp_path_factory.mktemp("send") / "proj"
        assert cli.main(["init", str(proj)]) == 0
        out = subprocess.run(
            [sys.executable, "transforms/sample_transform.py"],
            capture_output=True, text=True, cwd=str(proj),
        )
        assert out.returncode == 0, out.stderr
        return proj

    def _env(self, monkeypatch, proj):
        monkeypatch.chdir(proj)
        monkeypatch.setenv("TRACEBI_SMTP_URL", "smtp://mail.example.com:2525")
        monkeypatch.setenv("TRACEBI_SMTP_FROM", "reports@example.com")
        monkeypatch.delenv("TRACEBI_SLACK_WEBHOOK", raising=False)

    def _fake_smtp(self, monkeypatch):
        """Capture the EmailMessage at the smtplib seam — no sockets."""
        import tracebi._delivery as delivery
        sent = {}

        class FakeSMTP:
            def __init__(self, host, port, timeout=None, context=None):
                sent["host"], sent["port"] = host, port
                sent["ssl_context"] = context   # smtps must pass a verified ctx

            def starttls(self, context=None):
                sent["starttls_context"] = context

            def login(self, user, password):
                sent["login"] = (user, password)

            def send_message(self, msg):
                sent["msg"] = msg

            def quit(self):
                pass

        monkeypatch.setattr(delivery.smtplib, "SMTP", FakeSMTP)
        monkeypatch.setattr(delivery.smtplib, "SMTP_SSL", FakeSMTP)
        return sent

    def test_send_refuses_when_verify_fails(self, proj, monkeypatch, capsys):
        import tracebi._delivery as delivery
        import tracebi.verify as verify_mod
        self._env(monkeypatch, proj)
        monkeypatch.setattr(verify_mod, "verify_manifest",
                            lambda *a, **k: dict(self._FAILING))
        attempted = []
        monkeypatch.setattr(delivery, "send_report",
                            lambda *a, **k: attempted.append(a))
        rc = cli.main(["report", "send", "sample_model/sample_dashboard",
                       "--to", "a@example.com"])
        assert rc == 1
        assert not attempted, "a failing receipt must never be sent"
        assert "NOT REPRODUCED" in capsys.readouterr().err

    def test_force_sends_with_the_verdict_in_the_body(self, proj, monkeypatch):
        import tracebi.verify as verify_mod
        self._env(monkeypatch, proj)
        monkeypatch.setattr(verify_mod, "verify_manifest",
                            lambda *a, **k: dict(self._FAILING))
        sent = self._fake_smtp(monkeypatch)
        rc = cli.main(["report", "send", "sample_model/sample_dashboard",
                       "--to", "a@example.com", "--force"])
        assert rc == 0
        body = sent["msg"].get_body(
            preferencelist=("plain",)).get_content()
        assert "DID NOT VERIFY" in body    # the red flag banner, up top
        assert "NOT REPRODUCED" in body    # the verdict itself

    def test_starttls_uses_a_verified_context(self, proj, monkeypatch):
        """STARTTLS must pass an SSL context so the server certificate is
        verified — an unverified upgrade lets a MITM capture credentials."""
        import ssl

        import tracebi.verify as verify_mod
        self._env(monkeypatch, proj)
        monkeypatch.setattr(verify_mod, "verify_manifest",
                            lambda *a, **k: dict(self._PASSING))
        sent = self._fake_smtp(monkeypatch)
        cli.main(["report", "send", "sample_model/sample_dashboard", "--to", "a@example.com"])
        assert isinstance(sent.get("starttls_context"), ssl.SSLContext)

    def test_refuses_cleartext_credentials_when_no_starttls(self, tmp_path,
                                                            monkeypatch):
        """Server offers no STARTTLS and credentials are set: refuse to send
        them over an unencrypted link rather than leaking them."""
        import smtplib

        import tracebi._delivery as delivery
        from tracebi._delivery import send_report

        class NoTLS:
            def __init__(self, host, port, timeout=None, context=None):
                pass

            def starttls(self, context=None):
                raise smtplib.SMTPNotSupportedError("no starttls")

            def login(self, *a):
                raise AssertionError("credentials must not be sent in cleartext")

            def send_message(self, m):
                raise AssertionError("must not send after refusing")

            def quit(self):
                pass

        monkeypatch.setattr(delivery.smtplib, "SMTP", NoTLS)
        monkeypatch.setenv("TRACEBI_SMTP_URL",
                           "smtp://user:pass@mail.example.com:2525")
        monkeypatch.setenv("TRACEBI_SMTP_FROM", "reports@example.com")
        html = tmp_path / "r.html"
        html.write_text("<p>x</p>")
        man = tmp_path / "r.html.manifest.json"
        man.write_text("{}")
        with pytest.raises(RuntimeError, match="STARTTLS"):
            send_report(html, man, ["a@example.com"])
