"""
Test for biller_allowlist.py — checks internal consistency of the seed
data (categories match pilot_config's approved categories, no duplicate
names, every entry has a BPAY code recorded).

Run: python3 test_biller_allowlist.py
"""
import sys

import biller_allowlist as ba

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


APPROVED_CATEGORIES = {"electricity", "gas", "water", "telecommunications"}


def main():
    check("allowlist is non-empty", len(ba.PILOT_BILLER_ALLOWLIST) > 0)

    names = [b.name for b in ba.PILOT_BILLER_ALLOWLIST]
    check("no duplicate biller names in the seed list", len(names) == len(set(names)))

    bad_categories = [b for b in ba.PILOT_BILLER_ALLOWLIST if b.category not in APPROVED_CATEGORIES]
    check("every biller's category is one of the pilot's approved categories", bad_categories == [])

    missing_codes = [b for b in ba.PILOT_BILLER_ALLOWLIST if not b.bpay_biller_code]
    check("every biller has a recorded BPAY biller code", missing_codes == [])

    missing_states = [b for b in ba.PILOT_BILLER_ALLOWLIST if not b.state_coverage]
    check("every biller has at least one state of coverage recorded", missing_states == [])

    check("all four approved categories have at least one biller", all(
        len(ba.allowlist_names_for_category(cat)) > 0 for cat in APPROVED_CATEGORIES
    ))

    check("VIC is covered by at least one biller in every category (pilot's initial geographic area)", all(
        any("VIC" in b.state_coverage for b in ba.PILOT_BILLER_ALLOWLIST if b.category == cat)
        for cat in APPROVED_CATEGORIES
    ))

    check("allowlist_names() returns the same count as the raw tuple (no accidental collapsing)",
          len(ba.allowlist_names()) == len(ba.PILOT_BILLER_ALLOWLIST))

    entry = ba.get_entry("AusNet Electricity")
    check("get_entry() finds a known biller", entry is not None and entry.category == "electricity")

    check("get_entry() returns None for an unknown biller", ba.get_entry("Definitely Not A Real Biller Pty Ltd") is None)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
