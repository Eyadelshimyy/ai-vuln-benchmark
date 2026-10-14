"""
The audit that caught a $1 error in a beautiful page.

``artifacts/book-review-editorial.html`` is a free-form HTML report an agent
wrote by hand from three stamped gateway queries — assurance level L1: the
*data* is governed, the presentation is not. This script is the audit for
that level. It re-runs the recorded queries through the gateway and checks
two things:

1. **Fingerprints** — the hashes cited on the page still match the live
   result (data hasn't drifted since authorship);
2. **Figures** — every number displayed on the page equals the number the
   query actually returns (the agent transcribed faithfully).

The first run of this audit failed one check: the page's total unrealized
gain read +$523,045, the sum of the *rounded* per-row gains. The true total
is $523,044.32 — round-then-sum vs sum-then-round, one dollar apart,
invisible to any human reviewer, caught immediately here. The page was
corrected; this file keeps both the method and the story.

Run from the project root (needs `pip install 'tracebi[mcp]'`):

    python examples/agent_gateway/verify_report.py
"""
import asyncio
import json
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS = StdioServerParameters(command="tracebi", args=["mcp"], cwd=str(PROJECT_ROOT))

#: The three queries recorded in the page's Working Papers section, verbatim.
QUERIES = {
    "asset_class": {
        "model": "wealth_model", "fact": "fact_holdings",
        "measures": ["market_value", "cost_basis", "unrealized_gain", "gain_pct"],
        "dimensions": ["dim_product.asset_class"],
    },
    "advisor": {
        "model": "wealth_model", "fact": "fact_holdings",
        "measures": {"market_value": "sum"},
        "dimensions": ["dim_client.advisor"],
    },
    "segment": {
        "model": "wealth_model", "fact": "fact_holdings",
        "measures": {"market_value": "sum", "units": "count"},
        "dimensions": ["dim_client.segment", "dim_client.risk_profile"],
    },
}

#: Fingerprint prefixes cited on the page (git-style truncation).
CITED = {
    "asset_class": "b3d614c1d264",
    "advisor": "7d5f1ad1e088",
    "segment": "3b6a4e8879c2",
}


def checks(r):
    """Every figure asserted on the page, as (claim, passed) pairs."""
    ac = {x["dim_product.asset_class"]: x for x in r["asset_class"]["rows"]}
    adv = {x["dim_client.advisor"]: x["market_value"] for x in r["advisor"]["rows"]}
    seg = {(x["dim_client.segment"], x["dim_client.risk_profile"]): x
           for x in r["segment"]["rows"]}
    tot_mv = sum(x["market_value"] for x in ac.values())
    tot_cb = sum(x["cost_basis"] for x in ac.values())
    tot_g = sum(x["unrealized_gain"] for x in ac.values())

    yield "AUM total $7,521,674", round(tot_mv) == 7_521_674
    yield "cost basis $6,998,630", round(tot_cb) == 6_998_630
    # Round the true total, never sum the rounded rows — the distinction
    # this audit exists to enforce.
    yield "unrealized gain +$523,044", round(tot_g) == 523_044
    yield "return on basis +7.5%", abs(tot_g / tot_cb - 0.075) < 0.0005
    yield "120 positions", sum(x["units"] for x in seg.values()) == 120

    for name, mv, cb, g, pct in [
        ("fixed income", 2_726_364, 2_554_683, 171_681, 0.067),
        ("equity", 2_215_130, 2_071_033, 144_097, 0.070),
        ("money market", 1_688_904, 1_539_119, 149_785, 0.097),
        ("alternatives", 891_276, 833_795, 57_482, 0.069),
    ]:
        row = ac[name]
        yield f"{name} row", (
            round(row["market_value"]) == mv
            and round(row["cost_basis"]) == cb
            and round(row["unrealized_gain"]) == g
            and abs(row["gain_pct"] - pct) < 0.0005
        )

    for who, mv in [
        ("Lisa Chen", 2_400_240), ("Mike Torres", 1_919_901),
        ("Anna Park", 1_498_415), ("Sara Green", 1_179_197),
        ("Tom Hall", 523_921),
    ]:
        yield f"{who}", round(adv[who]) == mv

    for cell, mv, n in [
        (("affluent", "balanced"), 1_637_035, 20),
        (("retail", "balanced"), 1_404_909, 20),
        (("private", "conservative"), 956_710, 16),
        (("retail", "aggressive"), 205_094, 4),
    ]:
        yield f"{cell[0]}/{cell[1]} cell", (
            round(seg[cell]["market_value"]) == mv and seg[cell]["units"] == n
        )


async def main() -> int:
    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            results, prints = {}, {}
            for key, q in QUERIES.items():
                out = json.loads(
                    (await s.call_tool("query_model", dict(q, limit=100))).content[0].text
                )
                results[key], prints[key] = out, out["fingerprint"]

    print("— fingerprints —")
    drift = False
    for key, fp in prints.items():
        ok = fp.startswith(CITED[key])
        drift |= not ok
        print(f"  {key:<12} cited {CITED[key]}  live {fp[:12]}  "
              f"{'MATCH' if ok else 'DRIFT'}")

    print("— figures —")
    failed = 0
    for claim, ok in checks(results):
        if not ok:
            failed += 1
            print(f"  FAIL  {claim}")
    total = sum(1 for _ in checks(results))
    print(f"  {total - failed}/{total} verified")
    return 1 if (failed or drift) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
