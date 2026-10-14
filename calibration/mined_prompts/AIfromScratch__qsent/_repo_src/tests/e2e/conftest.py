"""
Playwright E2E test configuration.

Requires the server to be running with TEST_MODE=true:
    TEST_MODE=true uvicorn qsf.api.main:app --reload

Run with:
    pytest tests/e2e/
"""
import pytest


@pytest.fixture(scope="session")
def base_url():
    return "http://localhost:8000"
