"""
BillSure v2 — Billing Smoothing Engine
==========================================
Core engine for annual bill prediction, monthly equalised payments,
excess/deficit balancing, and subscription tier management.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import math

# Australian seasonal weighting factors by category
# Based on typical Australian utility usage patterns
SEASONAL_WEIGHTS = {
    "Electricity": [1.15, 1.10, 0.95, 0.85, 0.80, 0.85, 0.90, 0.95, 0.95, 0.90, 1.00, 1.10],  # High summer/winter
    "Gas": [0.60, 0.60, 0.70, 0.90, 1.20, 1.40, 1.50, 1.40, 1.20, 0.90, 0.70, 0.60],  # High winter
    "Water": [1.30, 1.25, 1.10, 0.90, 0.80, 0.75, 0.75, 0.80, 0.90, 1.00, 1.10, 1.25],  # High summer
    "Internet": [1.0] * 12,  # Flat
    "Mobile": [1.0] * 12,  # Flat
    "Council": [1.0] * 12,  # Flat (quarterly but even)
    "Insurance": [1.0] * 12,  # Flat
    "Other": [1.0] * 12,
}

# Subscription tiers
SUBSCRIPTION_TIERS = {
    "basic": {
        "id": "basic",
        "name": "Basic",
        "monthly_fee": 0,
        "description": "Free plan — manage up to 5 bills",
        "max_bills": 5,
        "features": [
            "Up to 5 bills",
            "Basic payment plan",
            "Manual bill upload",
            "Email reminders",
        ],
        "has_ai_insights": False,
        "has_forecasting": False,
        "has_smoothing": True,
        "buffer_pct": 8,
    },
    "standard": {
        "id": "standard",
        "name": "Standard",
        "monthly_fee": 9.90,
        "description": "For households — unlimited bills with AI insights",
        "max_bills": 50,
        "features": [
            "Unlimited bills",
            "Smart smoothing engine",
            "AI bill intelligence",
            "12-month forecasting",
            "Provider comparison",
            "Priority email support",
        ],
        "has_ai_insights": True,
        "has_forecasting": True,
        "has_smoothing": True,
        "buffer_pct": 8,
    },
    "premium": {
        "id": "premium",
        "name": "Premium",
        "monthly_fee": 19.90,
        "description": "For families & landlords — advanced analytics & multi-property",
        "max_bills": 200,
        "features": [
            "Everything in Standard",
            "Multi-property support",
            "Advanced financial analytics",
            "Seasonal prediction engine",
            "Auto-negotiation alerts",
            "Dedicated account manager",
            "Reduced safety buffer (5%)",
        ],
        "has_ai_insights": True,
        "has_forecasting": True,
        "has_smoothing": True,
        "buffer_pct": 5,
    },
}


def predict_annual_bills(bills: List[dict]) -> dict:
    """
    Predict annual bills with seasonal weighting.
    Returns month-by-month forecast for the next 12 months.
    """
    now = datetime.now(timezone.utc)
    current_month = now.month  # 1-12

    # Aggregate by category
    category_annual = {}
    for b in bills:
        cat = b.get("category", "Other")
        amt = b.get("amount", 0) or 0
        freq = b.get("frequency", "monthly")

        annual = _annualize(amt, freq)
        if cat not in category_annual:
            category_annual[cat] = {"annual": 0, "bills": [], "avg_bill": 0, "count": 0}
        category_annual[cat]["annual"] += annual
        category_annual[cat]["bills"].append(b)
        category_annual[cat]["count"] += 1

    for cat in category_annual:
        d = category_annual[cat]
        d["avg_bill"] = round(d["annual"] / max(d["count"], 1) / 12, 2)

    # Build 12-month forecast
    monthly_forecast = []
    total_predicted = 0

    for offset in range(12):
        month_idx = (current_month - 1 + offset) % 12  # 0-indexed
        month_num = month_idx + 1
        month_date = _offset_month(now, offset)
        month_label = month_date.strftime("%b %Y")

        month_total = 0
        month_categories = {}

        for cat, data in category_annual.items():
            weights = SEASONAL_WEIGHTS.get(cat, SEASONAL_WEIGHTS["Other"])
            base_monthly = data["annual"] / 12
            weighted = round(base_monthly * weights[month_idx], 2)
            month_total += weighted
            month_categories[cat] = weighted

        month_total = round(month_total, 2)
        total_predicted += month_total
        monthly_forecast.append({
            "month": month_label,
            "month_num": month_num,
            "offset": offset,
            "total": month_total,
            "by_category": month_categories,
        })

    return {
        "total_predicted_annual": round(total_predicted, 2),
        "monthly_forecast": monthly_forecast,
        "category_breakdown": {
            cat: {
                "annual": round(d["annual"], 2),
                "monthly_avg": round(d["annual"] / 12, 2),
                "bill_count": d["count"],
            }
            for cat, d in category_annual.items()
        },
    }


def calculate_smoothed_payment(bills: List[dict], frequency: str = "monthly", buffer_pct: float = 8.0) -> dict:
    """
    Calculate true smoothed payment — equalised across all months,
    accounting for seasonal variations so users pay the SAME amount every period.
    """
    prediction = predict_annual_bills(bills)
    annual_total = prediction["total_predicted_annual"]
    buffer_multiplier = 1 + (buffer_pct / 100)
    buffered_annual = round(annual_total * buffer_multiplier, 2)

    divisors = {"weekly": 52, "fortnightly": 26, "monthly": 12}
    periods = divisors.get(frequency, 12)
    smoothed_amount = round(buffered_annual / periods, 2)

    # Calculate peak and trough months
    forecasts = prediction["monthly_forecast"]
    peak_month = max(forecasts, key=lambda m: m["total"])
    trough_month = min(forecasts, key=lambda m: m["total"])
    variance = round(peak_month["total"] - trough_month["total"], 2)

    # Without smoothing, max monthly bill would be peak_month
    # With smoothing, it's always smoothed_amount — that's the saving
    monthly_smoothed = round(buffered_annual / 12, 2)

    return {
        "smoothed_amount": smoothed_amount,
        "frequency": frequency,
        "periods_per_year": periods,
        "annual_predicted": annual_total,
        "buffered_annual": buffered_annual,
        "buffer_pct": buffer_pct,
        "monthly_equivalent": monthly_smoothed,
        "peak_month": {"month": peak_month["month"], "amount": peak_month["total"]},
        "trough_month": {"month": trough_month["month"], "amount": trough_month["total"]},
        "seasonal_variance": variance,
        "monthly_forecast": forecasts,
    }


def compute_plan_health(
    plan: dict,
    bills: List[dict],
    wallet_balance: float,
    transactions: List[dict],
) -> dict:
    """
    Compute excess/deficit balancing — are we on track?
    Compares what's been collected vs what's been paid out,
    and projects whether the wallet will cover upcoming bills.
    """
    now = datetime.now(timezone.utc)

    total_collected = plan.get("total_collected", 0)
    total_paid_out = plan.get("total_paid_out", 0)
    deduction_amount = plan.get("deduction_amount", 0)
    frequency = plan.get("frequency", "monthly")

    # Current surplus/deficit
    surplus = round(total_collected - total_paid_out, 2)

    # Upcoming bills in next 90 days
    upcoming_total = 0
    upcoming_bills = []
    for b in bills:
        if b.get("status") != "pending":
            continue
        try:
            due = datetime.fromisoformat(b["due_date"].replace("Z", "+00:00"))
            if due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            days_until = (due - now).days
            if 0 <= days_until <= 90:
                upcoming_total += b.get("amount", 0)
                upcoming_bills.append({
                    "provider": b.get("provider"),
                    "amount": b.get("amount", 0),
                    "due_date": b.get("due_date"),
                    "days_until": days_until,
                })
        except Exception:
            pass

    # Project collections in next 90 days
    days_map = {"weekly": 7, "fortnightly": 14, "monthly": 30}
    period_days = days_map.get(frequency, 30)
    collections_in_90d = math.floor(90 / period_days) * deduction_amount

    # Projected balance in 90 days
    projected_balance = round(wallet_balance + collections_in_90d - upcoming_total, 2)

    # Health status
    if projected_balance >= upcoming_total * 0.2:
        status = "healthy"
        message = "Your plan is on track. Wallet balance will comfortably cover upcoming bills."
    elif projected_balance >= 0:
        status = "tight"
        message = "Your plan is tight. Consider topping up your wallet for extra buffer."
    else:
        shortfall = abs(projected_balance)
        status = "deficit"
        message = f"Projected shortfall of ${shortfall:.2f} in the next 90 days. Top up recommended."

    return {
        "status": status,
        "message": message,
        "wallet_balance": round(wallet_balance, 2),
        "total_collected": round(total_collected, 2),
        "total_paid_out": round(total_paid_out, 2),
        "surplus": surplus,
        "upcoming_bills_90d": round(upcoming_total, 2),
        "upcoming_bill_count": len(upcoming_bills),
        "upcoming_bills": sorted(upcoming_bills, key=lambda x: x["days_until"]),
        "projected_collections_90d": round(collections_in_90d, 2),
        "projected_balance_90d": projected_balance,
    }


def calculate_savings_comparison(bills: List[dict], buffer_pct: float = 8.0) -> dict:
    """
    Compare smoothed billing vs traditional billing.
    Shows how much stress/variance the user avoids.
    """
    prediction = predict_annual_bills(bills)
    forecasts = prediction["monthly_forecast"]

    if not forecasts:
        return {"has_data": False}

    monthly_amounts = [m["total"] for m in forecasts]
    annual_total = prediction["total_predicted_annual"]
    smoothed_monthly = round(annual_total * (1 + buffer_pct / 100) / 12, 2)

    peak = max(monthly_amounts)
    trough = min(monthly_amounts)
    avg = round(annual_total / 12, 2)

    # Calculate "bill shock" avoidance — the max single-month bill that smoothing prevents
    bill_shock_avoided = round(peak - smoothed_monthly, 2)

    # Predictability score (0-100) — lower variance = higher score
    if avg > 0:
        cv = (sum((m - avg) ** 2 for m in monthly_amounts) / len(monthly_amounts)) ** 0.5 / avg
        predictability_traditional = max(0, round(100 * (1 - cv), 0))
    else:
        predictability_traditional = 100

    return {
        "has_data": True,
        "traditional": {
            "monthly_amounts": [round(m, 2) for m in monthly_amounts],
            "peak_month": round(peak, 2),
            "trough_month": round(trough, 2),
            "average": avg,
            "variance": round(peak - trough, 2),
            "predictability_score": predictability_traditional,
        },
        "smoothed": {
            "fixed_monthly": smoothed_monthly,
            "annual_total": round(smoothed_monthly * 12, 2),
            "buffer_included": round(smoothed_monthly * 12 - annual_total, 2),
            "predictability_score": 100,
        },
        "savings_analysis": {
            "bill_shock_avoided": max(bill_shock_avoided, 0),
            "predictability_gain": round(100 - predictability_traditional, 0),
            "max_monthly_saving": round(peak - smoothed_monthly, 2) if peak > smoothed_monthly else 0,
            "buffer_cost_monthly": round((annual_total * buffer_pct / 100) / 12, 2),
        },
        "monthly_comparison": [
            {
                "month": forecasts[i]["month"],
                "traditional": round(monthly_amounts[i], 2),
                "smoothed": smoothed_monthly,
                "difference": round(smoothed_monthly - monthly_amounts[i], 2),
            }
            for i in range(len(forecasts))
        ],
    }


# ---- Helpers ----

def _annualize(amount: float, frequency: str) -> float:
    multipliers = {
        "weekly": 52,
        "fortnightly": 26,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1,
    }
    return amount * multipliers.get(frequency, 12)


def _offset_month(dt: datetime, months: int) -> datetime:
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, 28)
    return dt.replace(year=year, month=month, day=day)
