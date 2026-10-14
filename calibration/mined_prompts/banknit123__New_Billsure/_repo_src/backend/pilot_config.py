"""
backend/pilot_config.py
========================
Centrally managed, versioned configuration for the ASIC Enhanced Regulatory
Sandbox (ERS) pilot: a non-cash payment facility + a continuing
consumer-credit facility used exclusively to pay verified household utility
bills.

STATUS: this credit/product structure is a PILOT DESIGN ONLY. It is
subject to final Australian legal confirmation and is NOT represented
anywhere in this module, its callers, or generated documentation as
legally approved, ASIC-approved, or ready for real customers. See
docs/asic-ers-readiness/regulatory-assumptions.md.

Design principles:

- Pilot parameters are never hard-coded at call sites. Every limit check
  in onboarding, credit assessment, bill payment, and exposure monitoring
  must read from `get_active_config()` (or the equivalent DB row), not a
  literal number, so a single reviewed change here or in the DB updates
  every enforcement point consistently.
- Configuration is versioned and immutable once created — a "change" is a
  new version row, never an UPDATE of an existing one. `active_version_id`
  moves; history never disappears. This mirrors ledger.py's "never mutate,
  only append" discipline for the same audit reasons.
- `validate_config_change()` is a hard ceiling, not a guideline. It runs
  against every proposed new version — including the very first one — and
  rejects anything that would breach a pilot-wide cap, enable a prohibited
  capability, or raise a limit without a recorded approver distinct from
  the proposer. This function must be called before a new version is
  persisted; nothing else in the system is trusted to do that check.
- Activating real-money functionality is NOT controlled here — this module
  only defines and validates the numeric/product envelope. Whether the
  pilot is allowed to move real money at all is decided exclusively by
  launch_gates.is_production_authorized(), which fails closed. A pilot
  config with real_money_enabled=True but launch gates unsatisfied must
  still result in production being blocked; enforce that at every call
  site, don't rely on this module alone.
"""

import logging
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional

import supabase_db as sdb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard pilot-wide ceilings. These are NOT the active config — they are the
# absolute maximum the ERS notification described, and no config version,
# however approved, may exceed them. Changing these constants is a code
# change requiring its own review; it is deliberately not something a
# runtime "config change" can touch.
# ---------------------------------------------------------------------------
HARD_MAX_PILOT_CUSTOMERS = 25
HARD_MAX_CONTRACTUAL_CREDIT_LIMIT = Decimal("2500.00")
HARD_MAX_INITIAL_AVAILABLE_CREDIT = Decimal("500.00")
HARD_MIN_INITIAL_AVAILABLE_CREDIT = Decimal("300.00")
HARD_MAX_SINGLE_BILL_PAYMENT = Decimal("1500.00")
HARD_MAX_OUTSTANDING_BALANCE = Decimal("2500.00")
HARD_MAX_AGGREGATE_EXPOSURE = Decimal("62500.00")
HARD_CONTRACT_TERM_MONTHS = 12
HARD_MAX_PILOT_DURATION_MONTHS = 6

APPROVED_BILL_CATEGORIES = frozenset({"electricity", "gas", "water", "telecommunications"})
APPROVED_GEOGRAPHIC_AREAS = frozenset({"VIC"})

WARNING_THRESHOLDS = (Decimal("0.70"), Decimal("0.80"), Decimal("0.90"))


@dataclass
class PilotConfig:
    """One versioned snapshot of the pilot product configuration.

    `id` and `created_at` are assigned on persistence; construct without
    them when proposing a new version.
    """
    max_pilot_customers: int = HARD_MAX_PILOT_CUSTOMERS
    contractual_credit_limit: Decimal = HARD_MAX_CONTRACTUAL_CREDIT_LIMIT
    initial_available_credit_min: Decimal = HARD_MIN_INITIAL_AVAILABLE_CREDIT
    initial_available_credit_max: Decimal = HARD_MAX_INITIAL_AVAILABLE_CREDIT
    max_single_bill_payment: Decimal = HARD_MAX_SINGLE_BILL_PAYMENT
    max_outstanding_balance: Decimal = HARD_MAX_OUTSTANDING_BALANCE
    aggregate_contractual_exposure_cap: Decimal = HARD_MAX_AGGREGATE_EXPOSURE
    contract_term_months: int = HARD_CONTRACT_TERM_MONTHS
    interest_rate_percent: Decimal = Decimal("0.00")
    late_fee_amount: Decimal = Decimal("0.00")
    early_repayment_fee_amount: Decimal = Decimal("0.00")
    cash_withdrawals_enabled: bool = False
    customer_transfers_enabled: bool = False
    approved_bill_categories: frozenset = field(default_factory=lambda: frozenset(APPROVED_BILL_CATEGORIES))
    geographic_areas: frozenset = field(default_factory=lambda: frozenset(APPROVED_GEOGRAPHIC_AREAS))
    pilot_duration_months: int = HARD_MAX_PILOT_DURATION_MONTHS
    real_money_enabled: bool = False
    label: str = "subject to final Australian legal confirmation"

    id: Optional[str] = None
    version: Optional[int] = None
    created_at: Optional[str] = None
    proposed_by: Optional[str] = None
    approved_by: Optional[str] = None
    is_active: bool = False


class ConfigValidationError(Exception):
    """Raised when a proposed pilot config change would breach a pilot cap
    or capability restriction. Always fail closed: catching this and
    proceeding anyway defeats its entire purpose."""


def validate_config_change(
    new_config: PilotConfig,
    previous_config: Optional[PilotConfig],
    proposed_by: str,
    approved_by: Optional[str],
) -> None:
    """Raises ConfigValidationError if the proposed config is invalid.
    Call this before persisting ANY new config version, including the
    first one ever created (previous_config=None in that case).
    """
    errors = []

    # --- absolute hard ceilings, independent of any prior version ---
    if new_config.max_pilot_customers > HARD_MAX_PILOT_CUSTOMERS:
        errors.append(f"max_pilot_customers {new_config.max_pilot_customers} exceeds hard cap {HARD_MAX_PILOT_CUSTOMERS}")
    if new_config.contractual_credit_limit > HARD_MAX_CONTRACTUAL_CREDIT_LIMIT:
        errors.append("contractual_credit_limit exceeds hard cap AUD 2,500")
    if not (HARD_MIN_INITIAL_AVAILABLE_CREDIT <= new_config.initial_available_credit_max <= HARD_MAX_INITIAL_AVAILABLE_CREDIT):
        errors.append("initial_available_credit_max must be between AUD 300 and AUD 500")
    if new_config.initial_available_credit_min > new_config.initial_available_credit_max:
        errors.append("initial_available_credit_min cannot exceed initial_available_credit_max")
    if new_config.max_single_bill_payment > HARD_MAX_SINGLE_BILL_PAYMENT:
        errors.append("max_single_bill_payment exceeds hard cap AUD 1,500")
    if new_config.max_outstanding_balance > HARD_MAX_OUTSTANDING_BALANCE:
        errors.append("max_outstanding_balance exceeds hard cap AUD 2,500")
    if new_config.aggregate_contractual_exposure_cap > HARD_MAX_AGGREGATE_EXPOSURE:
        errors.append("aggregate_contractual_exposure_cap exceeds hard cap AUD 62,500")
    implied_aggregate = Decimal(new_config.max_pilot_customers) * new_config.contractual_credit_limit
    if implied_aggregate > HARD_MAX_AGGREGATE_EXPOSURE:
        errors.append(
            f"max_pilot_customers x contractual_credit_limit = {implied_aggregate} exceeds "
            f"aggregate hard cap {HARD_MAX_AGGREGATE_EXPOSURE}"
        )
    if new_config.contract_term_months != HARD_CONTRACT_TERM_MONTHS:
        errors.append("contract_term_months must equal 12 for this pilot")
    if new_config.pilot_duration_months > HARD_MAX_PILOT_DURATION_MONTHS:
        errors.append("pilot_duration_months exceeds the 6-month ERS pilot window")

    # --- prohibited capabilities: never enableable via config, period ---
    if new_config.cash_withdrawals_enabled:
        errors.append("cash_withdrawals_enabled cannot be set true — cash withdrawals are prohibited for this pilot")
    if new_config.customer_transfers_enabled:
        errors.append("customer_transfers_enabled cannot be set true — customer-to-customer transfers are prohibited")
    if new_config.interest_rate_percent != Decimal("0.00"):
        errors.append("interest_rate_percent must be 0 for this pilot")
    if new_config.late_fee_amount != Decimal("0.00"):
        errors.append("late_fee_amount must be 0 for this pilot")
    if new_config.early_repayment_fee_amount != Decimal("0.00"):
        errors.append("early_repayment_fee_amount must be 0 for this pilot")

    disallowed_categories = set(new_config.approved_bill_categories) - APPROVED_BILL_CATEGORIES
    if disallowed_categories:
        errors.append(f"approved_bill_categories includes disallowed categories: {sorted(disallowed_categories)}")
    disallowed_areas = set(new_config.geographic_areas) - APPROVED_GEOGRAPHIC_AREAS
    if disallowed_areas:
        errors.append(f"geographic_areas includes areas outside the approved pilot area: {sorted(disallowed_areas)}")

    # --- maker-checker on the change itself ---
    if not proposed_by:
        errors.append("proposed_by is required")
    is_increase = False
    if previous_config is not None:
        is_increase = (
            new_config.max_pilot_customers > previous_config.max_pilot_customers
            or new_config.contractual_credit_limit > previous_config.contractual_credit_limit
            or new_config.max_single_bill_payment > previous_config.max_single_bill_payment
            or new_config.max_outstanding_balance > previous_config.max_outstanding_balance
            or new_config.aggregate_contractual_exposure_cap > previous_config.aggregate_contractual_exposure_cap
            or (new_config.real_money_enabled and not previous_config.real_money_enabled)
        )
    else:
        # First-ever config: treat as an "increase" from a null baseline so
        # it always requires a distinct, documented approver too.
        is_increase = True

    if is_increase:
        if not approved_by:
            errors.append("this change increases a limit or enables real-money functionality and requires an approved_by")
        elif approved_by == proposed_by:
            errors.append("approved_by must be a different person from proposed_by for any limit increase (maker-checker)")

    # --- real-money activation cannot be decided here at all ---
    # This module never checks launch gates directly (avoid a circular
    # dependency with launch_gates.py) — callers MUST additionally check
    # launch_gates.is_production_authorized() before treating
    # real_money_enabled=True as meaning anything. Documented, not enforced
    # here, deliberately: see module docstring.

    if errors:
        raise ConfigValidationError("; ".join(errors))


def warning_level(current: Decimal, limit: Decimal) -> Optional[str]:
    """Returns '70pct' / '80pct' / '90pct' / 'breach' / None for a
    current-vs-limit ratio, for exposure/limit monitoring dashboards."""
    if limit <= 0:
        return None
    ratio = current / limit
    if ratio >= Decimal("1.0"):
        return "breach"
    if ratio >= WARNING_THRESHOLDS[2]:
        return "90pct"
    if ratio >= WARNING_THRESHOLDS[1]:
        return "80pct"
    if ratio >= WARNING_THRESHOLDS[0]:
        return "70pct"
    return None


def _row_to_config(row: dict) -> PilotConfig:
    return PilotConfig(
        max_pilot_customers=row["max_pilot_customers"],
        contractual_credit_limit=Decimal(str(row["contractual_credit_limit"])),
        initial_available_credit_min=Decimal(str(row["initial_available_credit_min"])),
        initial_available_credit_max=Decimal(str(row["initial_available_credit_max"])),
        max_single_bill_payment=Decimal(str(row["max_single_bill_payment"])),
        max_outstanding_balance=Decimal(str(row["max_outstanding_balance"])),
        aggregate_contractual_exposure_cap=Decimal(str(row["aggregate_contractual_exposure_cap"])),
        contract_term_months=row["contract_term_months"],
        interest_rate_percent=Decimal(str(row["interest_rate_percent"])),
        late_fee_amount=Decimal(str(row["late_fee_amount"])),
        early_repayment_fee_amount=Decimal(str(row["early_repayment_fee_amount"])),
        cash_withdrawals_enabled=row["cash_withdrawals_enabled"],
        customer_transfers_enabled=row["customer_transfers_enabled"],
        approved_bill_categories=frozenset(row["approved_bill_categories"]),
        geographic_areas=frozenset(row["geographic_areas"]),
        pilot_duration_months=row["pilot_duration_months"],
        real_money_enabled=row["real_money_enabled"],
        label=row.get("label", "subject to final Australian legal confirmation"),
        id=row.get("id"),
        version=row.get("version"),
        created_at=row.get("created_at"),
        proposed_by=row.get("proposed_by"),
        approved_by=row.get("approved_by"),
        is_active=row.get("is_active", False),
    )


async def get_active_config() -> Optional[PilotConfig]:
    row = await sdb.find_one("pilot_config_versions", {"is_active": True})
    return _row_to_config(row) if row else None


async def propose_config_version(
    new_config: PilotConfig,
    proposed_by: str,
    approved_by: Optional[str] = None,
    activate: bool = False,
) -> PilotConfig:
    """Validates and persists a new, immutable config version. Does NOT
    activate it (does not touch is_active) unless activate=True AND
    approved_by is set and distinct from proposed_by for any increase —
    validate_config_change() already enforces the maker-checker rule,
    this just refuses to silently promote an unapproved draft."""
    previous = await get_active_config()
    validate_config_change(new_config, previous, proposed_by, approved_by)

    prev_version = previous.version if previous else 0
    row = {
        "version": (prev_version or 0) + 1,
        "max_pilot_customers": new_config.max_pilot_customers,
        "contractual_credit_limit": str(new_config.contractual_credit_limit),
        "initial_available_credit_min": str(new_config.initial_available_credit_min),
        "initial_available_credit_max": str(new_config.initial_available_credit_max),
        "max_single_bill_payment": str(new_config.max_single_bill_payment),
        "max_outstanding_balance": str(new_config.max_outstanding_balance),
        "aggregate_contractual_exposure_cap": str(new_config.aggregate_contractual_exposure_cap),
        "contract_term_months": new_config.contract_term_months,
        "interest_rate_percent": str(new_config.interest_rate_percent),
        "late_fee_amount": str(new_config.late_fee_amount),
        "early_repayment_fee_amount": str(new_config.early_repayment_fee_amount),
        "cash_withdrawals_enabled": new_config.cash_withdrawals_enabled,
        "customer_transfers_enabled": new_config.customer_transfers_enabled,
        "approved_bill_categories": sorted(new_config.approved_bill_categories),
        "geographic_areas": sorted(new_config.geographic_areas),
        "pilot_duration_months": new_config.pilot_duration_months,
        "real_money_enabled": new_config.real_money_enabled,
        "label": "subject to final Australian legal confirmation",
        "proposed_by": proposed_by,
        "approved_by": approved_by,
        "is_active": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    created = await sdb.insert_one("pilot_config_versions", row)

    if activate:
        if not approved_by or approved_by == proposed_by:
            raise ConfigValidationError("cannot activate a config version without a distinct approved_by")
        await sdb.update_many("pilot_config_versions", {"is_active": True}, {"is_active": False})
        await sdb.update_one("pilot_config_versions", {"id": created["id"]}, {"is_active": True})
        created["is_active"] = True

    return _row_to_config(created)


async def check_customer_cap(active_pilot_customer_count: int) -> None:
    """Raises ConfigValidationError (treat as a hard block, not a warning)
    if activating one more customer would exceed the pilot cap. Callers
    must call this immediately before activating a 26th (or Nth+1)
    customer's credit facility, inside the same transaction/lock that
    counts active customers, to avoid a race between the count and the
    activation."""
    cfg = await get_active_config()
    limit = cfg.max_pilot_customers if cfg else HARD_MAX_PILOT_CUSTOMERS
    if active_pilot_customer_count >= limit:
        raise ConfigValidationError(
            f"pilot customer cap reached ({active_pilot_customer_count}/{limit}); cannot activate another customer"
        )


async def check_aggregate_exposure(current_aggregate_contractual_exposure: Decimal, additional: Decimal) -> None:
    cfg = await get_active_config()
    cap = cfg.aggregate_contractual_exposure_cap if cfg else HARD_MAX_AGGREGATE_EXPOSURE
    projected = current_aggregate_contractual_exposure + additional
    if projected > cap:
        raise ConfigValidationError(
            f"aggregate contractual exposure would reach {projected}, exceeding cap {cap}"
        )
