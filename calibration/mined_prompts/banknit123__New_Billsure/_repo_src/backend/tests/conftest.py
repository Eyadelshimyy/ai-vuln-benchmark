"""Shared test configuration — loads credentials from environment."""
import os

API_URL = os.environ.get("TEST_API_URL", "http://localhost:8001/api")

TEST_USER_EMAIL = os.environ.get("TEST_USER_EMAIL", "")
TEST_USER_PASSWORD = os.environ.get("TEST_USER_PASSWORD", "")

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

BASIC_USER_EMAIL = os.environ.get("BASIC_USER_EMAIL", "")
BASIC_USER_PASSWORD = os.environ.get("BASIC_USER_PASSWORD", "")

def get_customer_creds():
    return {"email": TEST_USER_EMAIL, "pw": TEST_USER_PASSWORD}

def get_admin_creds():
    return {"email": ADMIN_EMAIL, "pw": ADMIN_PASSWORD}
