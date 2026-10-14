"""Turning a decoded symbol into evidence.

The adapter is thin on purpose. Reading the symbol is the decoder's job and
resolving it is the catalog's, so what is left here is the envelope, which is
the part that has to know a camera exists.

A frame that decodes nothing produces no evidence. That is the ordinary case
rather than an error: the measured yield over five seeds is about one gate
crossing in twenty, because a barcode wrapped around a can foreshortens
non-linearly under a nadir view and most packages present no readable symbol
from directly above at all.
"""

from __future__ import annotations

from typing import Any

from clave.tracker.codes import Decoder
from clave.tracker.evidence import Evidence, Role


def codes_from_frame(
    frame: Any,
    source_id: str,
    observed_at_nanos: int,
    decoder: Decoder,
    confidence: float = 0.9,
) -> tuple[Evidence, ...]:
    """Turn whatever decodes in one frame into readings.

    Args:
        frame: The frame, height by width by three.
        source_id: Which sensor produced it.
        observed_at_nanos: When it was taken.
        decoder: What reads the symbol.
        confidence: How sure the adapter is. A symbol that passed its check
            digit is very likely right, and the residual doubt is that it was
            read off the neighbouring object rather than this one, which is
            what the quad would settle if the decoder reported one.

    Returns:
        One reading per decoded symbol, which is none on most frames.
    """
    read = decoder.decode(frame)
    if read is None:
        return ()
    return (
        Evidence(
            source_id=source_id,
            role=Role.CODE,
            observed_at_nanos=observed_at_nanos,
            confidence=confidence,
            payload=read,
        ),
    )
