"""Smoke test configuration — isolated in-memory DB, server dir via pyproject.toml pythonpath."""
import pytest


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch, tmp_path):
    """Redirect DB + docs root to a temp directory for isolation."""
    db_file = tmp_path / "test.db"

    import tools.db as db_module
    monkeypatch.setattr(db_module, "get_db_path", lambda: db_file)
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path / "docs")

    from tools.db import init_db
    init_db()

    yield db_file
