#!/usr/bin/env python3
"""Compare a hosted dashboard with the locally rendered file; never renders or invokes owners."""

import argparse
import hashlib
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def digest(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dashboard", type=Path, help="locally rendered dashboard HTML")
    parser.add_argument("url", help="URL the audience opens")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    local = digest(args.dashboard.read_bytes())
    try:
        with urlopen(Request(args.url, headers={"Cache-Control": "no-cache"}), timeout=args.timeout) as response:
            served = digest(response.read())
    except (URLError, OSError) as error:
        parser.exit(2, f"LIMITED: could not fetch {args.url} ({error}); record the limitation instead of a screenshot check\n")
    if served != local:
        parser.exit(1, f"FAIL: served {served[:12]} differs from rendered {local[:12]}; update the destination\n")
    print(f"PASS: served page matches rendered {local[:12]}")


if __name__ == "__main__":
    main()
