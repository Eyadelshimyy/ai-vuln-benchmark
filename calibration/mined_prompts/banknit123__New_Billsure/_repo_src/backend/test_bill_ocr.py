"""
Test for bill_ocr.py using REAL synthetic bill files (a generated PDF
with a text layer, and a generated PNG image run through actual
Tesseract OCR) — unlike the other test_*.py files in this repo, this one
does NOT mock its dependency, because the entire point is proving the
free, self-hosted extraction genuinely works end to end with zero API
keys and zero network calls.

Run: python3 test_bill_ocr.py
(requires: pip install pdfplumber pytesseract pillow reportlab
           apt-get install tesseract-ocr)
"""
import io
import sys
from decimal import Decimal

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image, ImageDraw, ImageFont

import bill_ocr as ocr

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


def make_synthetic_pdf_bill() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "AusNet Electricity")
    c.setFont("Helvetica", 12)
    c.drawString(50, 760, "Tax Invoice / Bill")
    c.drawString(50, 730, "Account Name: Jane Citizen")
    c.drawString(50, 700, "Reference Number: REF-99123")
    c.drawString(50, 670, "Amount Due: $245.67")
    c.drawString(50, 640, "Due Date: 15/09/2026")
    c.drawString(50, 610, "Thank you for being a customer.")
    c.save()
    return buf.getvalue()


def make_synthetic_pdf_bill_with_decoys() -> bytes:
    """Reconstructs the exact structure of a REAL bill (an EnergyAustralia
    gas bill) that was found live to trip up the original amount/date
    extraction: a genuine 'Amount due $303.38' and 'Bill due date
    25 Feb 2025', PLUS an unrelated '$10,000' card-payment-limit
    disclaimer (larger than the real amount) and an unrelated numeric
    '27/11/2024' billing-period date (appearing before the real due date
    in the document, and in a format the old date regex would match but
    the real due date's month-name format would not)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "EnergyAustralia")
    c.setFont("Helvetica", 12)
    c.drawString(50, 760, "Your gas account")
    c.drawString(50, 730, "Account number: 2773934108")
    c.drawString(50, 700, "Your bill")
    c.drawString(50, 670, "Amount due")
    c.drawString(50, 650, "$303.38")
    c.drawString(50, 620, "Bill due date")
    c.drawString(50, 600, "25 Feb 2025")
    c.drawString(50, 560, "Gas payment options")
    c.drawString(50, 540, "Call 1300 559 873 to pay by MasterCard, Visa")
    c.drawString(50, 520, "or American Express for payment amounts up to $10,000.")
    c.drawString(50, 480, "Energy charges")
    c.drawString(50, 460, "Balance Plan (Home)")
    c.drawString(50, 440, "27/11/2024 - 05/02/2025 - 71 Days")
    c.drawString(50, 420, "Total current charges (incl. GST of $27.58) $303.38")
    c.save()
    return buf.getvalue()


def make_blank_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.save()  # no text drawn at all
    return buf.getvalue()


def make_synthetic_bill_image() -> bytes:
    img = Image.new("RGB", (900, 500), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = font_big
    draw.text((30, 30), "Origin Energy", fill="black", font=font_big)
    draw.text((30, 100), "Account Name: Jane Citizen", fill="black", font=font_small)
    draw.text((30, 150), "Reference Number: REF-55667", fill="black", font=font_small)
    draw.text((30, 200), "Amount Due: $189.40", fill="black", font=font_small)
    draw.text((30, 250), "Due Date: 20/09/2026", fill="black", font=font_small)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    known_billers = {"AusNet Electricity", "Origin Energy", "Yarra Valley Water"}

    # ---------------------------------------------------------------
    # Real PDF text-layer extraction (pdfplumber), genuinely parsed
    # ---------------------------------------------------------------
    pdf_bytes = make_synthetic_pdf_bill()
    result = ocr.extract_bill_data(pdf_bytes, known_billers=known_billers, is_pdf=True)
    check("PDF text-layer extraction used the free pdf_text_layer method, not OCR", result.extraction_method == "pdf_text_layer")
    check("PDF extraction found a high confidence score (real text layer, most fields found)", result.extraction_confidence >= 0.85)
    check("PDF extraction correctly parsed the amount ($245.67)", result.guessed_amount == Decimal("245.67"))
    check("PDF extraction correctly parsed the due date", result.guessed_due_date == "15/09/2026")
    check("PDF extraction correctly parsed the reference number", result.guessed_biller_reference == "REF-99123")
    check("PDF extraction correctly identified the known biller by name match", "AusNet Electricity" in result.biller_name_candidates)

    # ---------------------------------------------------------------
    # Real bug, found live: a real bill's "$10,000" card-payment-limit
    # disclaimer and an unrelated "27/11/2024" billing-period date were
    # picked up instead of the real amount ($303.38) and real due date
    # (25 Feb 2025, month-name format). This reconstructs that exact
    # document structure -- a genuine regression test for a genuine
    # production bug, not a hypothetical.
    # ---------------------------------------------------------------
    decoy_pdf_bytes = make_synthetic_pdf_bill_with_decoys()
    decoy_result = ocr.extract_bill_data(decoy_pdf_bytes, known_billers={"EnergyAustralia"}, is_pdf=True)
    check("real-bug regression: amount is the genuine $303.38, NOT the $10,000 payment-limit decoy",
          decoy_result.guessed_amount == Decimal("303.38"))
    check("real-bug regression: due date is the genuine '25 Feb 2025' (month-name format), NOT the '27/11/2024' billing-period decoy",
          decoy_result.guessed_due_date == "25 Feb 2025")

    # ---------------------------------------------------------------
    # A PDF with no text layer at all -> confidence 0, method 'none',
    # never a crash, never a false-positive high confidence
    # ---------------------------------------------------------------
    blank_pdf = make_blank_pdf()
    blank_result = ocr.extract_from_pdf_text_layer(blank_pdf, known_billers)
    check("a blank PDF (no text layer) reports zero confidence, not a crash or a guess", blank_result.extraction_confidence == 0.0)
    check("a blank PDF's method is reported as 'none'", blank_result.extraction_method == "none")

    # ---------------------------------------------------------------
    # Real image OCR via actual Tesseract — no mocking
    # ---------------------------------------------------------------
    image_bytes = make_synthetic_bill_image()
    image_result = ocr.extract_from_image_ocr(image_bytes, known_billers=known_billers)
    check("image OCR used the tesseract_ocr method", image_result.extraction_method == "tesseract_ocr")
    check("image OCR produced some non-empty extracted text from a clean synthetic image", len(image_result.raw_text.strip()) > 0)
    check("image OCR reports a real confidence score between 0 and 1 (not a placeholder)", 0.0 <= image_result.extraction_confidence <= 1.0)
    # Tesseract on a clean, high-contrast synthetic image should do
    # reasonably well, though OCR is never perfect -- assert it's in a
    # sane range rather than expecting pixel-perfect extraction.
    check(f"image OCR confidence ({image_result.extraction_confidence:.2f}) is at least moderate on a clean synthetic image",
          image_result.extraction_confidence >= 0.3)

    # ---------------------------------------------------------------
    # extract_bill_data(): top-level entry point falls back correctly
    # ---------------------------------------------------------------
    top_level_pdf_result = ocr.extract_bill_data(pdf_bytes, known_billers, is_pdf=True)
    check("top-level extract_bill_data() uses the PDF text layer when available (doesn't waste time on OCR unnecessarily)",
          top_level_pdf_result.extraction_method == "pdf_text_layer")

    # An image passed with is_pdf=False should go straight to OCR.
    top_level_image_result = ocr.extract_bill_data(image_bytes, known_billers, is_pdf=False)
    check("top-level extract_bill_data() uses OCR for a non-PDF input", top_level_image_result.extraction_method == "tesseract_ocr")

    # ---------------------------------------------------------------
    # Corrupt/garbage input never crashes the caller
    # ---------------------------------------------------------------
    garbage = b"this is not a valid pdf or image file at all, just random bytes 12345"
    garbage_result = ocr.extract_bill_data(garbage, known_billers, is_pdf=True)
    check("garbage input returns a low-confidence result instead of raising", garbage_result.extraction_confidence == 0.0 and garbage_result.extraction_method == "none")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
