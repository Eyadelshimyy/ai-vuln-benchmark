"""model_pipeline stamps the connected model for the shared UI scope."""

from tracebi.pipeline.model_pipeline import model_pipeline


def test_model_pipeline_stamps_the_connected_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    # No transform/reports needed — we only assert the stamp on the runner.
    runner = model_pipeline("portfolio_model", transform="holdings_transform",
                            db_url=f"sqlite:///{tmp_path / 'runs.db'}")
    assert runner.model == "portfolio_model"
