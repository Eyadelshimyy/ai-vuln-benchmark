#!/usr/bin/env python
"""
One-time setup: Approve Permit2 contract to spend USDC from buyer wallet.
This is required before any Permit2-based x402 payment can succeed.
"""
import asyncio
import os
from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

load_dotenv()

# Config
PRIVATE_KEY = os.getenv("EVM_PRIVATE_KEY")
RPC_URL = os.getenv("BASE_RPC_URL", "https://mainnet.base.org")

USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
PERMIT2_ADDRESS = "0x000000000022D473030F116dDEE9F6B43aC78BA3"
MAX_UINT256 = 2**256 - 1  # max approval

USDC_ABI = [
    {
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    }
]

async def main():
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print(f"ERROR: Cannot connect to {RPC_URL}")
        return

    account = Account.from_key(PRIVATE_KEY)
    print(f"Wallet:   {account.address}")
    print(f"Chain:    Base mainnet (chainId={w3.eth.chain_id})")
    print(f"USDC:     {USDC_ADDRESS}")
    print(f"Permit2:  {PERMIT2_ADDRESS}")

    usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC_ADDRESS), abi=USDC_ABI)

    # Check current allowance
    current = usdc.functions.allowance(
        account.address,
        Web3.to_checksum_address(PERMIT2_ADDRESS)
    ).call()
    print(f"\nCurrent Permit2 allowance: {current}")

    if current > 10**10:  # already approved
        print("Already approved! No action needed.")
        return

    # Build approve tx
    print("\nBuilding approve transaction...")
    nonce = w3.eth.get_transaction_count(account.address)
    gas_price = w3.eth.gas_price

    tx = usdc.functions.approve(
        Web3.to_checksum_address(PERMIT2_ADDRESS),
        MAX_UINT256
    ).build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gasPrice": gas_price,
        "chainId": 8453,
    })

    # Estimate gas
    tx["gas"] = w3.eth.estimate_gas(tx)
    cost_eth = tx["gas"] * gas_price / 1e18
    print(f"Gas estimate: {tx['gas']} units (~{cost_eth:.6f} ETH)")

    # Sign and send
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"\nTransaction sent: 0x{tx_hash.hex()}")
    print("Waiting for confirmation...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status == 1:
        print(f"SUCCESS! Permit2 approved.")
        print(f"Tx: https://basescan.org/tx/0x{tx_hash.hex()}")
    else:
        print("FAILED! Transaction reverted.")

if __name__ == "__main__":
    asyncio.run(main())
