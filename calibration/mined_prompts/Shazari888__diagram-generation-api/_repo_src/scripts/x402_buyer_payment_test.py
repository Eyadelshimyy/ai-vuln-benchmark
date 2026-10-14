"""
Full live payment matrix test for the diagram generation API.
Tests all diagram types (mermaid, d2, plantuml, graphviz) × formats (svg, png).
Each call makes a real x402 on-chain payment on Base mainnet.

Usage:
    python scripts/x402_buyer_payment_test.py

Required .env vars:
    EVM_PRIVATE_KEY   — buyer wallet private key (0x...)
    BASE_URL          — production base URL (defaults to Railway)
"""
import asyncio
import json
import os
import time

from dotenv import load_dotenv
from eth_account import Account
from x402 import x402Client
from x402.http import x402HTTPClient
from x402.http.clients import x402HttpxClient
from x402.mechanisms.evm import EthAccountSigner
from x402.mechanisms.evm.exact.register import register_exact_evm_client

BASE_URL = os.getenv(
    "BASE_URL",
    "https://diagram-generation-api-production.up.railway.app",
).rstrip("/")

# Prompt per diagram type (LLM generates the source from this)
DIAGRAM_PROMPTS = {
    "mermaid":  "Create a simple login flow with success and failure paths.",
    "d2":       "Create a simple service dependency map with API, database, and cache.",
    "plantuml": "Create a simple sequence diagram for user authentication.",
    "graphviz": "Create a simple state machine for an order processing system.",
}

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


async def test_one(
    http,
    http_helper,
    diagram_type: str,
    output_format: str,
) -> dict:
    endpoint = f"{BASE_URL}/paid/diagrams/generate/{output_format}"
    body = {
        "diagram_type": diagram_type,
        "prompt": DIAGRAM_PROMPTS[diagram_type],
        "format": output_format,
    }

    start = time.monotonic()
    try:
        response = await http.post(endpoint, json=body)
        await response.aread()
        elapsed = round(time.monotonic() - start, 2)

        if response.is_success:
            data = response.json()
            settle = http_helper.get_payment_settle_response(
                lambda name: response.headers.get(name)
            )
            tx = settle.get("transaction", {}) if isinstance(settle, dict) else {}
            return {
                "ok": True,
                "status": response.status_code,
                "elapsed": elapsed,
                "has_url": bool(data.get("url") or data.get("image_url") or data.get("svg") or data.get("rendered")),
                "tx": tx.get("hash", "")[:16] + "..." if tx.get("hash") else "settled",
            }
        else:
            return {
                "ok": False,
                "status": response.status_code,
                "elapsed": elapsed,
                "error": response.text[:400],
            }
    except Exception as exc:
        return {
            "ok": False,
            "status": 0,
            "elapsed": round(time.monotonic() - start, 2),
            "error": str(exc)[:200],
        }


async def main() -> None:
    load_dotenv()
    private_key = _required_env("EVM_PRIVATE_KEY")

    account = Account.from_key(private_key)
    print(f"\n{BOLD}{CYAN}=== LIVE PAYMENT MATRIX TEST ==={RESET}")
    print(f"  Buyer wallet : {account.address}")
    print(f"  Target API   : {BASE_URL}")
    print(f"  Tests        : 4 types × 2 formats = 8 paid requests\n")

    client = x402Client()
    register_exact_evm_client(client, EthAccountSigner(account))
    http_helper = x402HTTPClient(client)

    results = []
    passed = 0
    failed = 0

    async with x402HttpxClient(client, timeout=120.0) as http:
        for diagram_type in DIAGRAM_PROMPTS:
            for fmt in ("svg", "png"):
                label = f"{diagram_type}/{fmt}"
                print(f"  Testing {label:<20} ... ", end="", flush=True)
                result = await test_one(http, http_helper, diagram_type, fmt)
                result["label"] = label

                if result["ok"]:
                    passed += 1
                    print(f"{GREEN}PASS{RESET}  {result['elapsed']}s  tx={result['tx']}")
                else:
                    failed += 1
                    err_preview = result.get('error', '')[:120]
                    print(f"{RED}FAIL{RESET}  status={result['status']}  {err_preview}")

                results.append(result)

                # Brief pause between calls to avoid nonce collisions
                await asyncio.sleep(2)

    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"  {GREEN}PASSED: {passed}/8{RESET}  |  {RED}FAILED: {failed}/8{RESET}")

    if failed == 0:
        print(f"\n  {GREEN}{BOLD}ALL TESTS PASSED — READY TO SHIP ✓{RESET}\n")
    else:
        print(f"\n  {YELLOW}Failed tests:{RESET}")
        for r in results:
            if not r["ok"]:
                print(f"    {r['label']}: status={r['status']} — {r.get('error','')[:100]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
