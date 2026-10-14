"""Public distribution contract for ``ellements.fslm``."""

from __future__ import annotations

import ellements.fslm as fslm


def test_weavemark_runtime_contract_is_public() -> None:
    """Export every stable primitive consumed by WeaveMark."""
    required = {
        "ActionResult",
        "DecisionResult",
        "FSLMContext",
        "FSLMEvent",
        "FSLMKernel",
        "MachineDefinition",
        "MachineSpec",
        "OutputRecord",
        "RuntimeBindings",
        "load_machine_definition",
    }

    assert required <= set(fslm.__all__)
    for name in required:
        assert getattr(fslm, name) is not None
