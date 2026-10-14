"""Tests for CER and WER metric utilities."""

from __future__ import annotations

import pytest

from womblex.utils.metrics import (
    cer,
    cer_spatial,
    levenshtein,
    reading_order_accuracy,
    spatial_sort_text,
    wer,
)


class TestLevenshtein:
    def test_identical_strings_zero(self) -> None:
        assert levenshtein("hello", "hello") == 0

    def test_one_substitution(self) -> None:
        assert levenshtein("hello", "hallo") == 1

    def test_insertion_and_deletion(self) -> None:
        assert levenshtein("cat", "cats") == 1
        assert levenshtein("cats", "cat") == 1

    def test_no_normalisation(self) -> None:
        # Unlike cer/wer, the raw distance is over the exact bytes: case differs.
        assert levenshtein("Hello", "hello") == 1

    def test_empty(self) -> None:
        assert levenshtein("", "") == 0
        assert levenshtein("abc", "") == 3

    def test_token_sequences(self) -> None:
        assert levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1


class TestCER:
    def test_identical_strings(self) -> None:
        assert cer("hello world", "hello world") == 0.0

    def test_complete_mismatch(self) -> None:
        # hypothesis is entirely wrong — distance equals len(ref)
        result = cer("abc", "xyz")
        assert result == pytest.approx(1.0)

    def test_one_substitution(self) -> None:
        # "hxllo" vs "hello" — 1 substitution in 5 chars
        assert cer("hello", "hxllo") == pytest.approx(1 / 5)

    def test_insertion(self) -> None:
        # hypothesis has an extra char
        assert cer("cat", "cats") == pytest.approx(1 / 3)

    def test_deletion(self) -> None:
        assert cer("cats", "cat") == pytest.approx(1 / 4)

    def test_empty_reference_empty_hypothesis(self) -> None:
        assert cer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self) -> None:
        assert cer("", "something") == 1.0

    def test_normalise_case(self) -> None:
        assert cer("Hello", "hello") == 0.0

    def test_normalise_whitespace(self) -> None:
        assert cer("foo  bar", "foo bar") == 0.0

    def test_normalise_disabled(self) -> None:
        # With normalise=False, case difference counts as edits
        assert cer("Hello", "hello", normalise=False) > 0.0

    def test_unicode_normalisation(self) -> None:
        # NFC-normalised equivalents should match
        nfc = "\u00e9"   # é as single codepoint
        nfd = "e\u0301"  # é as e + combining accent
        assert cer(nfc, nfd) == 0.0


class TestWER:
    def test_identical(self) -> None:
        assert wer("the cat sat", "the cat sat") == 0.0

    def test_one_substitution(self) -> None:
        # 1 word wrong out of 3
        assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)

    def test_one_deletion(self) -> None:
        # hypothesis missing one word
        assert wer("the cat sat", "the sat") == pytest.approx(1 / 3)

    def test_one_insertion(self) -> None:
        assert wer("the cat sat", "the big cat sat") == pytest.approx(1 / 3)

    def test_empty_reference_empty_hypothesis(self) -> None:
        assert wer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self) -> None:
        assert wer("", "extra words") == 1.0

    def test_normalise_case(self) -> None:
        assert wer("The Cat Sat", "the cat sat") == 0.0

    def test_normalise_whitespace(self) -> None:
        assert wer("the  cat\tsat", "the cat sat") == 0.0

    def test_perfect_ocr_on_iam_style_sentence(self) -> None:
        ref = "the quick brown fox jumps over the lazy dog"
        assert wer(ref, ref) == 0.0

    def test_cer_wer_both_detect_error(self) -> None:
        # Both metrics should be non-zero when there is an error
        ref = "a b c"
        hyp = "a x c"
        assert cer(ref, hyp) > 0.0
        assert wer(ref, hyp) > 0.0
        # CER and WER use different denominators so their values differ
        assert cer(ref, hyp) != pytest.approx(wer(ref, hyp))


class TestSpatialSortText:
    def test_empty_list(self) -> None:
        assert spatial_sort_text([]) == ""

    def test_single_word(self) -> None:
        assert spatial_sort_text([("hello", (0, 0, 50, 20))]) == "hello"

    def test_left_to_right_ordering(self) -> None:
        words = [
            ("world", (100, 0, 150, 20)),
            ("hello", (0, 0, 50, 20)),
        ]
        assert spatial_sort_text(words) == "hello world"

    def test_top_to_bottom_ordering(self) -> None:
        words = [
            ("line2", (0, 50, 50, 70)),
            ("line1", (0, 0, 50, 20)),
        ]
        assert spatial_sort_text(words) == "line1 line2"

    def test_multiline_document(self) -> None:
        words = [
            ("c", (200, 0, 250, 20)),
            ("a", (0, 0, 50, 20)),
            ("b", (100, 0, 150, 20)),
            ("d", (0, 50, 50, 70)),
            ("e", (100, 50, 150, 70)),
        ]
        assert spatial_sort_text(words) == "a b c d e"

    def test_same_line_tolerance(self) -> None:
        # Words with slight vertical offset should be on the same line.
        words = [
            ("b", (100, 2, 150, 22)),
            ("a", (0, 0, 50, 20)),
        ]
        assert spatial_sort_text(words) == "a b"


class TestCERSpatial:
    def test_identical_layout(self) -> None:
        words = [("hello", (0, 0, 50, 20)), ("world", (60, 0, 110, 20))]
        assert cer_spatial(words, words) == 0.0

    def test_reordered_words_same_text(self) -> None:
        # Same words, different reading order — CER-s should be 0.
        ref = [("hello", (0, 0, 50, 20)), ("world", (60, 0, 110, 20))]
        hyp = [("world", (60, 0, 110, 20)), ("hello", (0, 0, 50, 20))]
        assert cer_spatial(ref, hyp) == 0.0

    def test_recognition_error_detected(self) -> None:
        ref = [("hello", (0, 0, 50, 20))]
        hyp = [("hxllo", (0, 0, 50, 20))]
        assert cer_spatial(ref, hyp) > 0.0

    def test_empty_inputs(self) -> None:
        assert cer_spatial([], []) == 0.0


class TestReadingOrderAccuracy:
    def test_perfect_order(self) -> None:
        # Same words, same boxes, same order.
        words = [
            ("hello", (0, 0, 50, 20)),
            ("world", (60, 0, 110, 20)),
            ("foo", (0, 30, 50, 50)),
        ]
        assert reading_order_accuracy(words, words) == 1.0

    def test_fully_reversed(self) -> None:
        ref = [
            ("a", (0, 0, 50, 20)),
            ("b", (60, 0, 110, 20)),
            ("c", (0, 30, 50, 50)),
        ]
        hyp = list(reversed(ref))
        assert reading_order_accuracy(ref, hyp) == 0.0

    def test_one_swap(self) -> None:
        ref = [
            ("a", (0, 0, 50, 20)),
            ("b", (60, 0, 110, 20)),
            ("c", (0, 30, 50, 50)),
        ]
        # Swap first two — 1 discordant pair out of 3.
        hyp = [ref[1], ref[0], ref[2]]
        assert reading_order_accuracy(ref, hyp) == pytest.approx(2 / 3)

    def test_no_spatial_match(self) -> None:
        ref = [("a", (0, 0, 10, 10)), ("b", (20, 0, 30, 10))]
        hyp = [("x", (500, 500, 510, 510)), ("y", (600, 600, 610, 610))]
        # No IoU overlap → fewer than 2 matches → returns 1.0.
        assert reading_order_accuracy(ref, hyp) == 1.0

    def test_empty_inputs(self) -> None:
        assert reading_order_accuracy([], []) == 1.0

    def test_single_word(self) -> None:
        w = [("a", (0, 0, 50, 20))]
        assert reading_order_accuracy(w, w) == 1.0

    def test_partial_match(self) -> None:
        # 3 GT words, only 2 match in hypothesis. Those 2 are in order.
        ref = [
            ("a", (0, 0, 50, 20)),
            ("b", (60, 0, 110, 20)),
            ("c", (0, 30, 50, 50)),
        ]
        hyp = [
            ("a", (0, 0, 50, 20)),
            ("c", (0, 30, 50, 50)),
            # "b" is missing / in a non-overlapping position
        ]
        # 2 matched, in correct order → 1.0
        assert reading_order_accuracy(ref, hyp) == 1.0
