"""Counting decodes, and naming every denominator they could be counted against.



The arithmetic is tested against a fake frame source, so the counting is
provable without a render. What a render is needed for is the number itself,
which `docs/measurements.md` carries.
"""

from __future__ import annotations

import pytest

from clave.tracker.evidence import Code
from clave.tracker.measure import Crossing, YieldReport, measure_decode_yield


class FakeDecoder:
    """Reads whatever the crossing was labelled with."""

    def decode(self, frame: object) -> Code | None:
        """Return the symbol the fake frame stands for."""
        return Code(symbology="UPC_A", digits=str(frame)) if frame else None


def crossing(
    digits: str | None, material_class: str = "M-06", readable: bool = True
) -> Crossing:
    """One object passing the gate."""
    return Crossing(
        frame=digits,
        source_id="gate_code_center" if readable else "",
        observed_at_nanos=0,
        material_class=material_class,
        inside_a_code_band=readable,
    )


def test_a_report_names_both_denominators() -> None:
    """A report names both denominators.

    The two differ by more than half depending on which is used, so a single
    figure misleads in one direction or the other.
    """
    report = measure_decode_yield(
        [
            crossing("037600138727"),
            crossing(None),
            crossing(None, readable=False),
            crossing(None, readable=False),
        ],
        FakeDecoder(),
    )
    assert report.crossings == 4
    assert report.readable == 2
    assert report.decoded == 1
    assert report.per_readable == pytest.approx(0.5)
    assert report.per_crossing == pytest.approx(0.25)


def test_an_object_outside_every_code_band_is_not_counted_as_a_failure() -> None:
    """An object outside every code band is not counted as a failure.

    An object no camera could see did not fail to decode; it was never
    presented. Counting it against the decoder would blame optics on software.
    """
    report = measure_decode_yield(
        [crossing(None, readable=False), crossing(None, readable=False)], FakeDecoder()
    )
    assert report.readable == 0
    assert report.per_readable is None
    assert report.per_crossing == pytest.approx(0.0)


def test_the_report_breaks_the_yield_down_by_material_class() -> None:
    """The report breaks the yield down by material class.

    The measured yield is carried entirely by steel cans, and a headline figure
    that hid that would suggest the line reads barcodes on everything.
    """
    report = measure_decode_yield(
        [
            crossing("037600138727", "M-06"),
            crossing(None, "M-06"),
            crossing(None, "M-02"),
        ],
        FakeDecoder(),
    )
    assert report.per_class["M-06"] == (2, 1)
    assert report.per_class["M-02"] == (1, 0)


def test_a_report_over_nothing_states_no_yield_rather_than_zero() -> None:
    """A report over nothing states no yield rather than zero.

    Zero decodes out of zero crossings is not a zero percent yield, and
    reporting it as one would put a number where there is no measurement.
    """
    report = measure_decode_yield([], FakeDecoder())
    assert report.crossings == 0
    assert report.per_crossing is None
    assert report.per_readable is None


def test_a_symbol_that_fails_its_check_digit_never_reaches_the_count() -> None:
    """A symbol that fails its check digit never reaches the count.

    A misread counted as a decode would inflate the published yield with
    readings that resolve to the wrong product.
    """

    class MisreadingDecoder:
        def decode(self, frame: object) -> Code | None:
            del frame
            return None

    report = measure_decode_yield([crossing("037600138727")], MisreadingDecoder())
    assert report.decoded == 0


def test_the_report_renders_as_plain_data() -> None:
    """The figure lands in a document, so it has to serialize."""
    report = measure_decode_yield([crossing("037600138727")], FakeDecoder())
    rendered = report.as_dict()
    assert rendered["crossings"] == 1
    assert rendered["decoded"] == 1
    assert isinstance(report, YieldReport)
