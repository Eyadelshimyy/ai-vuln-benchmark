"""Reading a symbol off a frame, and resolving it to a bill of materials.

A GTIN does not name a material. A product resolves to a packaging bill of
materials, and a steel can with a paper label returns steel and paper, which
tells a tracker that one of them carries the symbol rather than which. So this
module resolves to components and never to a class, and the fusion rule that
consumes it spreads its weight over them.

The check digit is computed here rather than trusted from whatever decoded the
symbol, because a misread that survived would resolve to some other product's
bill of materials. That arithmetic lives on the `Code` payload, so a `Code` that
exists is one that checked out.

The gate proves the decoder works, so a zero rendered yield reads as a
property of the optics rather than as breakage. The fixture is a
symbol on a pinned submodule texture rather than a committed image, because
`docs/guidelines.md` says no binary lands in git and a submodule pin is a
reference rather than a binary.

Generating the fixture instead was tried and abandoned. OpenCV detects before it
decodes, and it will not detect a synthetically rendered symbol at any module
width, quiet zone or canvas padding, so a generated fixture would have proved
only that the generator and the decoder disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from clave.errors import ClaveError
from clave.taxonomy import BY_ID
from clave.tracker.evidence import Code, check_digit_holds
from clave.world.config import require


class CatalogError(ClaveError):
    """The packaging catalog describes something that cannot be routed."""


@dataclass(frozen=True)
class Packaging:
    """What one catalogued symbol says its package is made of.

    Attributes:
        components: Taxonomy identifiers the packaging is made of, in the order
            the catalog lists them.
        object_name: Which of this repository's objects the symbol was decoded
            off, so an entry nobody can trace back to the object set is visible
            as one somebody looked up.
        decoded_from: How it was read, being a texture or a rendered frame.
    """

    components: tuple[str, ...]
    object_name: str
    decoded_from: str


class Decoder(Protocol):
    """Anything that reads a symbol off one frame."""

    def decode(self, frame: Any) -> Code | None:
        """Read the first symbol in a frame.

        Args:
            frame: The frame, height by width by three.

        Returns:
            The symbol, or None when nothing decoded. Declining is the ordinary
            outcome: the measured yield says most gate crossings read nothing.
        """


class OpenCvDecoder:
    """Reads linear symbols with OpenCV's detector.

    OpenCV rather than a decoder written here, because a barcode decoder is a
    solved problem and the interesting part of this project is elsewhere. It
    lives behind `Decoder` so that a line-scan reader, or a purpose-built
    decoder that rectifies the quad first, replaces it without touching fusion.
    """

    def decode(self, frame: Any) -> Code | None:
        """Read the first symbol in a frame, refusing a failed check digit.

        The frame is tried as given and then at half scale. That is not a
        flourish: the sugar box texture in the pinned object set decodes at half
        resolution and not at full, so a single-scale reader would report the
        line carries four readable symbols where it carries five. Detection is
        the step that fails, and downsampling suppresses the print texture the
        detector mistakes for bars.
        """
        import cv2

        detector = cv2.barcode.BarcodeDetector()
        for scale in (1.0, 0.5):
            image = (
                frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale)
            )
            found, digits, symbology, _ = detector.detectAndDecodeWithType(image)
            if not found or not digits or not digits[0]:
                continue
            read = str(digits[0])
            if not read.isdigit() or not check_digit_holds(read):
                continue
            return Code(symbology=str(symbology[0]) if symbology else "", digits=read)
        return None


def load_catalog(path: Path) -> dict[str, Packaging]:
    """Read the GTIN to bill-of-materials table.

    Args:
        path: The catalog file.

    Returns:
        Packaging by decoded digits.

    Raises:
        CatalogError: If an entry names a class the taxonomy does not define,
            lists no components, or carries digits that fail their own check
            digit. A catalogued typo would resolve a real misread to a
            plausible bill of materials, which is worse than resolving nothing.
    """
    raw = yaml.safe_load(path.read_text())
    catalog: dict[str, Packaging] = {}
    for digits, entry in require(raw, "codes").items():
        read = str(digits)
        if not check_digit_holds(read):
            raise CatalogError(
                f"{read} fails its own check digit, so no reader could have produced it"
            )
        components = tuple(str(value) for value in require(entry, "components", read))
        if not components:
            raise CatalogError(f"{read} lists no components, so it resolves nothing")
        for component in components:
            if component not in BY_ID:
                raise CatalogError(
                    f"{read} names component {component!r}, which the taxonomy "
                    f"does not define and nothing could route"
                )
        catalog[read] = Packaging(
            components=components,
            object_name=str(require(entry, "object", read)),
            decoded_from=str(require(entry, "decoded_from", read)),
        )
    return catalog


def components_of(catalog: dict[str, Packaging], digits: str) -> tuple[str, ...]:
    """Return what a symbol's package is made of, or nothing.

    Args:
        catalog: The loaded catalog.
        digits: The decoded symbol.

    Returns:
        The components, or an empty tuple when the catalog does not carry the
        code. An unknown GTIN resolves to nothing rather than to a guess,
        because a real line reads symbols off products nobody catalogued every
        day.
    """
    entry = catalog.get(digits)
    return entry.components if entry is not None else ()


def resolver_for(catalog: dict[str, Packaging]) -> Any:
    """Return the resolver the fusion rule takes.

    Args:
        catalog: The loaded catalog.

    Returns:
        A callable from decoded digits to components, which is the shape
        `clave.tracker.fusion.fold` accepts so that fusion needs no catalog and
        the catalog needs no fusion.
    """

    def resolve(digits: str) -> tuple[str, ...]:
        return components_of(catalog, digits)

    return resolve
