"""Compare the public PyPI release with the admitted distribution bytes."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def admitted_files(path: Path, version: str) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None:
            raise ValueError("malformed admission manifest")
        digest, name = match.groups()
        if (name in expected or (not name.startswith(f"fast_mlsirm-{version}-")
                                 and name != f"fast_mlsirm-{version}.tar.gz")):
            raise ValueError("duplicate or foreign admitted distribution")
        expected[name] = digest
    if (len(expected) != 13 or sum(name.endswith(".whl") for name in expected) != 12
            or f"fast_mlsirm-{version}.tar.gz" not in expected):
        raise ValueError("admission manifest lacks twelve wheels and one sdist")
    return expected


def verify_published(release: object, expected: dict[str, str], version: str) -> None:
    if (not isinstance(release, dict) or not isinstance(release.get("info"), dict)
            or release["info"].get("version") != version):
        raise ValueError("PyPI returned a different release")
    files = release.get("urls")
    if not isinstance(files, list) or len(files) != 13:
        raise ValueError("PyPI does not list thirteen distributions")
    found: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("digests"), dict):
            raise ValueError("malformed PyPI distribution")
        name, digest = item.get("filename"), item["digests"].get("sha256")
        if (not isinstance(name, str) or name in found or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or item.get("yanked") is not False):
            raise ValueError("duplicate, malformed, or yanked PyPI distribution")
        found[name] = digest
    if found != expected:
        raise ValueError("published filenames or SHA256 digests differ from admission")


def main() -> None:
    if len(sys.argv) != 3 or not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", sys.argv[2]):
        raise SystemExit("usage: verify_published_pypi.py MANIFEST vMAJOR.MINOR.PATCH")
    version = sys.argv[2][1:]
    expected = admitted_files(Path(sys.argv[1]), version)
    url = f"https://pypi.org/pypi/fast-mlsirm/{version}/json"
    last_error: Exception | None = None
    for attempt in range(24):
        try:
            # Fixed HTTPS origin; the CLI version is restricted to numeric vX.Y.Z.
            with urlopen(url, timeout=10) as response:  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
                verify_published(json.load(response), expected, version)
            print(f"PyPI {version}: all twelve wheels and the sdist match admitted SHA256 bytes")
            return
        except (HTTPError, URLError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if isinstance(error, HTTPError) and error.code < 500 and error.code != 404:
                break
            if attempt < 23:
                time.sleep(5)
    raise SystemExit(f"PyPI published-matrix readback failed: {last_error}")


if __name__ == "__main__":
    main()
