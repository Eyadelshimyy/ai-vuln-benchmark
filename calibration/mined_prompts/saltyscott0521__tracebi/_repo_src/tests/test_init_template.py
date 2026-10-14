"""
``tracebi init --template`` scaffolds a project that verifies.

Default ``tracebi init`` (no flag) stays the sample orders dashboard.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from tracebi import cli
from tracebi.capabilities import describe

pytest.importorskip("duckdb")

_REPO = Path(__file__).parent.parent


def _run(args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "tracebi.cli", *args],
        capture_output=True, text=True, cwd=str(cwd),
    )


class TestTemplateFlag:
    def test_unknown_template_refuses_with_the_known_list(self, tmp_path, capsys):
        proj = tmp_path / "nope"
        assert cli.main(["init", str(proj), "--template", "not-a-template"]) == 1
        err = capsys.readouterr().err
        assert "unknown template" in err
        assert "saas-metrics" in err
        assert "sales-pipeline" in err
        assert not proj.exists()

    def test_default_init_is_unchanged(self, tmp_path):
        proj = tmp_path / "plain"
        assert cli.main(["init", str(proj)]) == 0
        assert (proj / "models" / "sample_model.yaml").is_file()
        assert (proj / "reports" / "sample_model" / "sample_dashboard"
                / "template.html").is_file()
        assert (proj / "pipelines" / "sample_model.yaml").is_file()
        assert (proj / "inputs" / "orders.csv").is_file()
        assert not (proj / "models" / "saas_model.yaml").exists()
        assert not (proj / "reports" / "saas_model").exists()
        assert not (proj / "models" / "sales_pipeline_model.yaml").exists()
        assert not (proj / "reports" / "sales_pipeline_model").exists()
        readme = (proj / "README.md").read_text(encoding="utf-8")
        assert "sample_dashboard" in readme
        assert "saas-metrics" not in readme
        assert "sales-pipeline" not in readme

    def test_alias_and_force(self, tmp_path, capsys):
        proj = tmp_path / "saas"
        assert cli.main(
            ["init", str(proj), "--template", "saas_metrics"]) == 0
        assert (proj / "models" / "saas_model.yaml").is_file()
        assert (proj / "transforms" / "saas_transform.py").is_file()
        assert (proj / "reports" / "saas_model" / "mrr_dashboard"
                / "template.html").is_file()
        assert (proj / "reports" / "saas_model" / "cohort_brief"
                / "report.json").is_file()
        readme = (proj / "README.md").read_text(encoding="utf-8")
        assert "saas-metrics" in readme
        assert "tracebi connect" in readme
        model = proj / "models" / "saas_model.yaml"
        model.write_text("# edited\n", encoding="utf-8")
        assert cli.main(["init", str(proj), "--template", "saas-metrics"]) == 1
        assert model.read_text(encoding="utf-8") == "# edited\n"
        assert cli.main(
            ["init", str(proj), "--template", "saas-metrics", "--force"]) == 0
        assert "measures:" in model.read_text(encoding="utf-8")
        # capsys drains the init prints so they don't leak into later asserts
        capsys.readouterr()

    def test_context_and_both_guides_name_the_template(self):
        payload = describe(brief=True)["templates"]
        assert payload["names"] == ["saas-metrics", "sales-pipeline"]
        known = {item["name"]: item for item in payload["known"]}
        assert list(known) == ["saas-metrics", "sales-pipeline"]
        assert "saas_metrics" in known["saas-metrics"]["aliases"]
        assert "saas_model/mrr_dashboard" in known["saas-metrics"]["reports"]
        assert "saas_model/cohort_brief" in known["saas-metrics"]["reports"]
        assert "sales_pipeline" in known["sales-pipeline"]["aliases"]
        assert "sales_pipeline_model/pipeline_dashboard" in (
            known["sales-pipeline"]["reports"])
        assert "sales_pipeline_model/rep_scorecard" in (
            known["sales-pipeline"]["reports"])
        assert "sales-pipeline" in payload["aliases"]
        assert "sales_pipeline" in payload["aliases"]
        guides = {
            "repo AGENTS.md": (_REPO / "AGENTS.md").read_text(encoding="utf-8"),
            "scaffolded AGENTS.md": (
                _REPO / "tracebi" / "_scaffold" / "init_agents.md"
            ).read_text(encoding="utf-8"),
        }
        for name, text in guides.items():
            assert "saas-metrics" in text, name
            assert "saas_metrics" in text, name
            assert "sales-pipeline" in text, name
            assert "sales_pipeline" in text, name
            assert "tracebi connect" in text, name


class TestSaasTemplateVerifies:
    def test_transform_build_and_verify(self, tmp_path):
        proj = tmp_path / "saas"
        assert cli.main(
            ["init", str(proj), "--template", "saas-metrics"]) == 0
        sample = (proj / "inputs" / "subscriptions.csv").read_text(
            encoding="utf-8").splitlines()
        assert 1 < len(sample) <= 301  # header + a few hundred rows at most

        # The model must load before phase ① has run.
        lazy = subprocess.run(
            [sys.executable, "-c",
             "from tracebi.model_registry import get_model; "
             "m = get_model('saas_model'); print('lazy-ok', m.name)"],
            capture_output=True, text=True, cwd=str(proj),
        )
        assert lazy.returncode == 0, lazy.stderr
        assert "lazy-ok saas_model" in lazy.stdout

        for _ in range(2):  # idempotent: a second sink replaces the tables
            sunk = subprocess.run(
                [sys.executable, "transforms/saas_transform.py"],
                capture_output=True, text=True, cwd=str(proj),
            )
            assert sunk.returncode == 0, sunk.stderr
            assert "account-months:" in sunk.stdout

        # The printed step: the model's pipeline runs the transform, then
        # builds every report in reports/saas_model/.
        built = _run(["run-pipeline", "saas_model"], proj)
        assert built.returncode == 0, built.stdout + built.stderr
        reports = (
            "saas_model/mrr_dashboard",
            "saas_model/cohort_brief",
        )
        for name in reports:
            manifest = proj / "output" / f"{name}.html.manifest.json"
            assert manifest.is_file()
            verified = _run(
                ["verify", str(manifest), "--strict", "--contracts"], proj)
            assert verified.returncode == 0, verified.stdout + verified.stderr
            assert "REPRODUCES" in verified.stdout


class TestSalesTemplateVerifies:
    def test_alias_scaffolds_the_sales_files(self, tmp_path, capsys):
        proj = tmp_path / "sales"
        assert cli.main(
            ["init", str(proj), "--template", "sales_pipeline"]) == 0
        assert (proj / "models" / "sales_pipeline_model.yaml").is_file()
        assert (proj / "transforms" / "sales_pipeline_transform.py").is_file()
        assert (proj / "inputs" / "opportunities.csv").is_file()
        assert (proj / "reports" / "sales_pipeline_model" / "pipeline_dashboard"
                / "template.html").is_file()
        assert (proj / "reports" / "sales_pipeline_model" / "rep_scorecard"
                / "report.json").is_file()
        readme = (proj / "README.md").read_text(encoding="utf-8")
        assert "sales-pipeline" in readme
        assert "sales_pipeline_model" in readme
        assert "tracebi connect" in readme
        capsys.readouterr()

    def test_transform_build_and_verify(self, tmp_path):
        proj = tmp_path / "sales"
        assert cli.main(
            ["init", str(proj), "--template", "sales-pipeline"]) == 0
        sample = (proj / "inputs" / "opportunities.csv").read_text(
            encoding="utf-8").splitlines()
        assert 1 < len(sample) <= 301  # header + a few hundred rows at most

        # The model must load before phase ① has run.
        lazy = subprocess.run(
            [sys.executable, "-c",
             "from tracebi.model_registry import get_model; "
             "m = get_model('sales_pipeline_model'); print('lazy-ok', m.name)"],
            capture_output=True, text=True, cwd=str(proj),
        )
        assert lazy.returncode == 0, lazy.stderr
        assert "lazy-ok sales_pipeline_model" in lazy.stdout

        for _ in range(2):  # idempotent: a second sink replaces the tables
            sunk = subprocess.run(
                [sys.executable, "transforms/sales_pipeline_transform.py"],
                capture_output=True, text=True, cwd=str(proj),
            )
            assert sunk.returncode == 0, sunk.stderr
            assert "opportunities:" in sunk.stdout

        # Win rate is won/closed, not the unweighted mean of the rep rates.
        rated = subprocess.run(
            [sys.executable, "-c",
             "from tracebi.model_registry import get_model\n"
             "m = get_model('sales_pipeline_model')\n"
             "total = m.query(fact='fact_opportunity', "
             "measures=['pipeline_value', 'win_rate', 'won_deals', "
             "'closed_deals']).to_pandas()\n"
             "by_rep = m.query(fact='fact_opportunity', measures=['win_rate'], "
             "dimensions=['dim_rep.rep']).to_pandas()\n"
             "top = m.query(fact='fact_opportunity', "
             "measures=['pipeline_value'], dimensions=['dim_stage.stage'], "
             "having={'pipeline_value': {'gt': 0}}, "
             "order_by=['-pipeline_value'], limit=1).to_pandas()\n"
             "row = total.iloc[0]\n"
             "overall = float(row['win_rate'])\n"
             "mean_of_rates = float(by_rep['win_rate'].mean())\n"
             "assert abs(overall - (row['won_deals'] / row['closed_deals'])) "
             "< 1e-9\n"
             "assert abs(overall - mean_of_rates) > 1e-6, "
             "(overall, mean_of_rates)\n"
             "assert float(row['pipeline_value']) == 463000\n"
             "assert top.iloc[0]['dim_stage.stage'] == 'Negotiation'\n"
             "assert by_rep.sort_values('win_rate').iloc[-1]['dim_rep.rep'] "
             "== 'Cara Singh'\n"
             "print('win-rate-ok', overall)\n"],
            capture_output=True, text=True, cwd=str(proj),
        )
        assert rated.returncode == 0, rated.stdout + rated.stderr
        assert "win-rate-ok" in rated.stdout

        built = _run(["run-pipeline", "sales_pipeline_model"], proj)
        assert built.returncode == 0, built.stdout + built.stderr
        reports = (
            "sales_pipeline_model/pipeline_dashboard",
            "sales_pipeline_model/rep_scorecard",
        )
        for name in reports:
            manifest = proj / "output" / f"{name}.html.manifest.json"
            assert manifest.is_file()
            verified = _run(
                ["verify", str(manifest), "--strict", "--contracts"], proj)
            assert verified.returncode == 0, verified.stdout + verified.stderr
            assert "REPRODUCES" in verified.stdout
