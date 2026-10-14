"""The perception record, and the tracker that fills it.

`docs/perception-contract.md` specifies how CLAVE describes one piece of waste
without naming the sensor that saw it. This package builds that description.

What is here is data, geometry and arithmetic. The rule deciding which
observation belongs to which object is a protocol with one implementation that
reads the simulator, and `learned-tracker` replaces it with a trained model
without changing anything above the seam.

No module in this package imports MuJoCo, OpenCV or torch at module scope, and
none reads a clock. Every instant is an argument.
"""

from __future__ import annotations
