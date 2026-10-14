"""One module per sensor role, and the one bridge to the simulator.

An adapter is the only code in CLAVE that knows a camera exists. Everything
above it reads a fused record, which is what lets a line gain a sensor without
editing a consumer.

`render.py` is the only module in `clave.tracker` that imports MuJoCo, and it
does so inside its functions. Every other adapter takes data a test can write as
a literal, so the conversions are provable without a simulator.
"""

from __future__ import annotations
