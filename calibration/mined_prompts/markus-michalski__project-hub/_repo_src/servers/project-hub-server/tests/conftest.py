"""Test configuration — uses an in-memory SQLite database."""
import pytest


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch, tmp_path):
    """Redirect all DB operations to a temporary file-based SQLite database.

    Uses tmp_path (unique per test) so each test starts with a clean slate.
    """
    db_file = tmp_path / "test.db"

    import tools.db as db_module
    monkeypatch.setattr(db_module, "get_db_path", lambda: db_file)

    from tools.db import init_db
    init_db()

    yield db_file
