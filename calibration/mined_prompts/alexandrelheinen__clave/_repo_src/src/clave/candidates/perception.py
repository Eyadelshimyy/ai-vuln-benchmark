"""Adapters for the shortlisted perception architectures.

Each loads from its upstream library rather than from a reimplementation, so
what is measured is the software the review screened. Imports are inside the
builders: importing this module must not require torch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from clave.candidates.base import MATERIAL_CLASS_COUNT, Candidate, CandidateSpec, Stage
from clave.candidates.fixture import frame

FASTER_RCNN = CandidateSpec(
    name="faster-rcnn-mobilenetv3",
    stage=Stage.PERCEPTION,
    license="BSD-3-Clause",
    source="torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn",
    requires="torchvision",
)

RESNET50 = CandidateSpec(
    name="resnet50-baseline",
    stage=Stage.PERCEPTION,
    license="BSD-3-Clause",
    source="torchvision.models.resnet50",
    requires="torchvision",
)

SAM2 = CandidateSpec(
    name="sam2",
    stage=Stage.PERCEPTION,
    license="Apache-2.0",
    source="facebookresearch/sam2",
    requires="sam2",
)


def _build_faster_rcnn() -> Any:
    """Build a detector with its head resized to the taxonomy.

    The head carries one class per material plus a background class, which is
    torchvision's convention for detection.
    """
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

    return fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, num_classes=MATERIAL_CLASS_COUNT + 1
    ).eval()


def _build_resnet50() -> Any:
    """Build the classifier baseline with its output sized to the taxonomy."""
    from torchvision.models import resnet50

    return resnet50(weights=None, num_classes=MATERIAL_CLASS_COUNT).eval()


def _build_sam2() -> Any:
    """Build SAM 2 from its smallest published configuration.

    No checkpoint is loaded. SAM 2 advanced because it is used zero-shot, and
    what this step measures is architecture cost, not segmentation quality.
    """
    from sam2.build_sam import build_sam2

    return build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml", ckpt_path=None, device="cpu"
    ).eval()


def forward_detector(model: Any) -> Callable[[], Any]:
    """Return a callable running one detection forward pass."""
    import torch

    image = torch.from_numpy(frame())

    def run() -> Any:
        with torch.no_grad():
            return model([image])

    return run


def forward_classifier(model: Any) -> Callable[[], Any]:
    """Return a callable running one classification forward pass."""
    import torch

    batch = torch.from_numpy(frame()).unsqueeze(0)

    def run() -> Any:
        with torch.no_grad():
            return model(batch)

    return run


def forward_sam2(model: Any) -> Callable[[], Any]:
    """Return a callable running SAM 2's image encoder on one frame.

    Only the encoder is timed. It dominates cost and runs once per frame, while
    the mask decoder runs once per prompt, so encoder latency is what a conveyor
    budget has to absorb.
    """
    import torch
    import torch.nn.functional as functional

    image = torch.from_numpy(frame()).unsqueeze(0)
    resized = functional.interpolate(image, size=(1024, 1024), mode="bilinear")

    def run() -> Any:
        with torch.no_grad():
            return model.image_encoder(resized)

    return run


PERCEPTION_CANDIDATES: tuple[
    tuple[Candidate, Callable[[Any], Callable[[], Any]]], ...
] = (
    (Candidate(FASTER_RCNN, _build_faster_rcnn), forward_detector),
    (Candidate(RESNET50, _build_resnet50), forward_classifier),
    (Candidate(SAM2, _build_sam2), forward_sam2),
)
