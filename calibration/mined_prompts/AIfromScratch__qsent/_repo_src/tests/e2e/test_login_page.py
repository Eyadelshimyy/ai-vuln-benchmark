"""
E2E tests for the login page UI.

Requires server running: TEST_MODE=true uvicorn qsf.api.main:app --reload
"""
import pytest
from playwright.sync_api import Page, expect


def test_login_page_shows_google_button(page: Page, base_url: str):
    page.goto(f"{base_url}/login.html")
    btn = page.get_by_text("Sign in with Google")
    expect(btn).to_be_visible()


def test_unauthenticated_visit_redirects_to_login(page: Page, base_url: str):
    # Clear all cookies to ensure no session
    page.context.clear_cookies()
    page.goto(base_url)
    # index.html calls /auth/me on load and redirects if 401
    page.wait_for_url(f"{base_url}/login.html", timeout=5000)
    expect(page).to_have_url(f"{base_url}/login.html")


def test_google_button_points_to_auth_login(page: Page, base_url: str):
    page.goto(f"{base_url}/login.html")
    btn = page.get_by_text("Sign in with Google")
    href = btn.get_attribute("href")
    assert href == "/auth/login"
