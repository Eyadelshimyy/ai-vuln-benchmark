"""Indeed selector hints (prefer role/label over obfuscated CSS)."""

INDEED_HOME = "https://www.indeed.com/"
INDEED_PROFILE = "https://profile.indeed.com/"
INDEED_RESUME = "https://profile.indeed.com/resume"
INDEED_LOGIN = "https://secure.indeed.com/auth"

_COUNTRY_HOSTS = {
    "cl": "https://cl.indeed.com",
    "us": "https://www.indeed.com",
    "mx": "https://mx.indeed.com",
    "ar": "https://ar.indeed.com",
    "es": "https://es.indeed.com",
}


def indeed_jobs_base(country: str = "cl") -> str:
    return _COUNTRY_HOSTS.get(country.lower(), f"https://{country.lower()}.indeed.com")
