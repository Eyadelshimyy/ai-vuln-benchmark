"""Keccak-256 for deterministic Solidity selectors.

This is Keccak, not NIST SHA3. Selectors are the first four bytes of
keccak256(signature).
"""

from __future__ import annotations

_ROUNDS = 24
_ROTATION_OFFSETS = (
    1,
    3,
    6,
    10,
    15,
    21,
    28,
    36,
    45,
    55,
    2,
    14,
    27,
    41,
    56,
    8,
    25,
    43,
    62,
    18,
    39,
    61,
    20,
    44,
)
_PI_LANE = (10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1)
_ROUND_CONSTANTS = (
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
)

_MASK = (1 << 64) - 1


def _rotl(value: int, shift: int) -> int:
    shift %= 64
    return ((value << shift) | (value >> (64 - shift))) & _MASK


def _keccak_f(state: list[int]) -> None:
    for round_index in range(_ROUNDS):
        columns = [
            state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20]
            for i in range(5)
        ]
        for i in range(5):
            mixed = columns[(i + 4) % 5] ^ _rotl(columns[(i + 1) % 5], 1)
            for row in range(0, 25, 5):
                state[row + i] ^= mixed
        lane = state[1]
        for index, rotation in enumerate(_ROTATION_OFFSETS):
            dest = _PI_LANE[index]
            saved = state[dest]
            state[dest] = _rotl(lane, rotation)
            lane = saved
        for row in range(0, 25, 5):
            current = state[row : row + 5]
            for i in range(5):
                state[row + i] ^= (~current[(i + 1) % 5]) & current[(i + 2) % 5]
                state[row + i] &= _MASK
        state[0] ^= _ROUND_CONSTANTS[round_index]


def keccak256(data: bytes) -> bytes:
    """Return 32-byte Keccak-256 digest."""
    rate = 136
    state = [0] * 25
    offset = 0
    while offset + rate <= len(data):
        for i in range(rate // 8):
            lane = int.from_bytes(data[offset + i * 8 : offset + (i + 1) * 8], "little")
            state[i] ^= lane
        _keccak_f(state)
        offset += rate
    block = bytearray(data[offset:])
    block.append(0x01)
    block.extend(b"\x00" * (rate - len(block)))
    block[-1] ^= 0x80
    for i in range(rate // 8):
        lane = int.from_bytes(block[i * 8 : (i + 1) * 8], "little")
        state[i] ^= lane
    _keccak_f(state)
    out = bytearray()
    for lane in state:
        out.extend(lane.to_bytes(8, "little"))
        if len(out) >= 32:
            break
    return bytes(out[:32])


def function_selector(signature: str) -> str:
    """4-byte selector, including the 0x prefix."""
    return "0x" + keccak256(signature.encode("utf-8")).hex()[:8]
