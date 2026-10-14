#!/usr/bin/env python3
"""
Job board crawler. Fetches all job postings from configured career pages
and saves each as a JSON file. Git history tracks when jobs were active.

Supported platforms: greenhouse, lever, ashby, workable, recruitee, smartrecruiters, rippling, gem
"""

import csv
import os
import sys

from src.crawl import PLATFORMS, crawl


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "companies.csv"

    with open(config_path, newline="") as f:
        reader = csv.DictReader(f)
        companies = [(row["slug"], row["ats"]) for row in reader if row["slug"] and row["ats"]]

    total = 0
    failures = []
    for name, platform in companies:
        if platform not in PLATFORMS:
            print(f"Skipping unsupported platform: {platform}")
            continue

        print(f"Crawling {name} ({platform})")
        output_dir = os.path.join("data", name)
        try:
            total += crawl(platform, name, output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            failures.append((name, platform, e))

    print(f"\nDone. Total jobs processed: {total}")

    if failures:
        print(f"\nFailed ({len(failures)}):")
        for name, platform, e in failures:
            print(f"  {name} ({platform}): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
