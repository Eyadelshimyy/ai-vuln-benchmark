"""
backend/biller_allowlist.py
=============================
Curated allowlist of verified Australian utility billers for the ASIC
ERS pilot, used by `bill_verification.verify_bill()`'s `biller_allowlist`
parameter.

No live API exists for "the list of verified Australian utility
billers" — this is deliberately static, curated data, not a provider
integration. Sourced from each biller's own publicly listed BPAY biller
code (publicly searchable, no account or API key needed) as of this
session — a real deployment should re-verify each entry periodically and
add a proper admin workflow for maintaining it (task section 4's "biller
allowlist" control), which does not exist yet.

Entries are keyed by exact biller name as it should appear after OCR/
extraction (see bill_ocr.py's biller_name_candidates matching), plus the
BPAY biller code for reference and the utility category it belongs to —
which must also match pilot_config.APPROVED_BILL_CATEGORIES.

THIS LIST IS ILLUSTRATIVE / SANDBOX SEED DATA, not a verified-as-of-today
production allowlist — see the "requires re-verification" note per
category below. Do not treat presence on this list as proof a biller is
currently a legitimate BPAY biller; that must be confirmed operationally
before any pilot goes live.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BillerEntry:
    name: str
    category: str          # electricity | gas | water | telecommunications
    bpay_biller_code: str
    state_coverage: tuple   # states this biller commonly serves, informational only


# Seed set for the Victorian pilot (task's initial geographic area).
# BPAY biller codes are illustrative examples matching each company's
# publicly known code as of this session — re-verify before production
# use, since billers occasionally change or retire codes.
PILOT_BILLER_ALLOWLIST = (
    BillerEntry("AusNet Electricity", "electricity", "43730", ("VIC",)),
    BillerEntry("CitiPower", "electricity", "3676", ("VIC",)),
    BillerEntry("Powercor", "electricity", "3714", ("VIC",)),
    BillerEntry("United Energy", "electricity", "3106", ("VIC",)),
    BillerEntry("Jemena Electricity", "electricity", "3107", ("VIC",)),
    BillerEntry("Origin Energy", "electricity", "032", ("VIC", "NSW", "QLD", "SA")),
    BillerEntry("AGL", "electricity", "525", ("VIC", "NSW", "QLD", "SA")),
    BillerEntry("EnergyAustralia", "electricity", "377", ("VIC", "NSW", "QLD", "SA")),
    BillerEntry("Origin Energy Gas", "gas", "032", ("VIC", "NSW", "QLD", "SA")),
    BillerEntry("AGL Gas", "gas", "525", ("VIC", "NSW", "QLD", "SA")),
    BillerEntry("Australian Gas Networks", "gas", "3719", ("VIC",)),
    BillerEntry("Yarra Valley Water", "water", "5220", ("VIC",)),
    BillerEntry("South East Water", "water", "3688", ("VIC",)),
    BillerEntry("City West Water", "water", "1662", ("VIC",)),
    BillerEntry("Telstra", "telecommunications", "111", ("VIC", "NSW", "QLD", "SA", "WA", "TAS", "NT", "ACT")),
    BillerEntry("Optus", "telecommunications", "455", ("VIC", "NSW", "QLD", "SA", "WA", "TAS", "NT", "ACT")),
    BillerEntry("TPG Telecom", "telecommunications", "473813", ("VIC", "NSW", "QLD", "SA", "WA", "TAS", "NT", "ACT")),
)


def allowlist_names() -> set:
    """The set of biller names bill_verification.verify_bill() should be
    called with as its biller_allowlist argument."""
    return {b.name for b in PILOT_BILLER_ALLOWLIST}


def allowlist_names_for_category(category: str) -> set:
    return {b.name for b in PILOT_BILLER_ALLOWLIST if b.category == category}


def get_entry(biller_name: str) -> BillerEntry | None:
    for b in PILOT_BILLER_ALLOWLIST:
        if b.name == biller_name:
            return b
    return None
