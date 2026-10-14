#!/usr/bin/env python
"""
Approve Dexter's contract via Permit2 AllowanceTransfer.
This sets a standing allowance so Dexter can pull USDC via Permit2.
"""
import os
import time
from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_typed_data
from web3 import Web3

load_dotenv()

PRIVATE_KEY = os.getenv("EVM_PRIVATE_KEY")
RPC_URL = os.getenv("BASE_RPC_URL", "https://mainnet.base.org")

USDC = Web3.to_checksum_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
PERMIT2 = Web3.to_checksum_address("0x000000000022D473030F116dDEE9F6B43aC78BA3")
DEXTER = Web3.to_checksum_address("0x402085c248EeA27D92E8b30b2C58ed07f9E20001")

MAX_ALLOWANCE = 2**160 - 1
EXPIRATION = int(time.time()) + 365 * 24 * 3600  # 1 year

w3 = Web3(Web3.HTTPProvider(RPC_URL))
account = Account.from_key(PRIVATE_KEY)

print(f"Wallet:   {account.address}")
print(f"Chain:    Base mainnet (chainId={w3.eth.chain_id})")

# Permit2 approve(token, spender, amount, expiration) via on-chain call
PERMIT2_ABI = [
    {
        "inputs": [
            {"name": "token", "type": "address"},
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint160"},
            {"name": "expiration", "type": "uint48"}
        ],
        "name": "approve",
        "outputs": [],
        "type": "function"
    },
    {
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "token", "type": "address"},
            {"name": "spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [
            {"name": "amount", "type": "uint160"},
            {"name": "expiration", "type": "uint48"},
            {"name": "nonce", "type": "uint48"}
        ],
        "type": "function"
    }
]

permit2 = w3.eth.contract(address=PERMIT2, abi=PERMIT2_ABI)

# Check existing allowance
existing = permit2.functions.allowance(account.address, USDC, DEXTER).call()
print(f"\nCurrent Permit2 allowance to Dexter: amount={existing[0]}, expiration={existing[1]}")

if existing[0] > 0 and existing[1] > int(time.time()):
    print("Already approved with sufficient allowance!")
else:
    print(f"\nSetting Permit2 allowance: Dexter can spend {MAX_ALLOWANCE} USDC for 1 year...")
    nonce = w3.eth.get_transaction_count(account.address)
    
    tx = permit2.functions.approve(
        USDC, DEXTER, MAX_ALLOWANCE, EXPIRATION
    ).build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gasPrice": w3.eth.gas_price,
        "chainId": 8453,
    })
    tx["gas"] = w3.eth.estimate_gas(tx)
    
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"Transaction sent: 0x{tx_hash.hex()}")
    
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status == 1:
        print(f"SUCCESS! Permit2->Dexter allowance set.")
        print(f"Tx: https://basescan.org/tx/0x{tx_hash.hex()}")
    else:
        print("FAILED! Transaction reverted.")
