"""
Bypass auth for all integration tests that test endpoint business logic.
Auth-specific behaviour is tested in test_auth_endpoints.py instead.
"""
import pytest

from qsf.api.auth import get_current_user
from qsf.api.main import app

MOCK_USER = {"email": "test@example.com", "name": "Test User"}


@pytest.fixture(autouse=True)
def bypass_auth():
    app.dependency_overrides[get_current_user] = lambda: MOCK_USER
    yield
    app.dependency_overrides.clear()
