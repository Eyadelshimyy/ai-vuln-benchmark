"""Generate the synthetic test fixtures: Australian-animal government documents.

Run from the repository root:

    uv run python fixtures/synthetic/generate.py

Every file is written deterministically (seeded content, reportlab's invariant
mode, fixed zip timestamps), so regenerating reproduces the committed bytes and
the source hashes and digests the tests pin. The documents imitate the shapes
real government releases take (a portfolio statement, a redacted notice, an
audit report and its transcript, a register export, a statistics workbook);
their content is invented and refers to no real person or organisation.
"""

from __future__ import annotations

import csv
import io
import random
import re
import textwrap
import zipfile
from pathlib import Path

import numpy as np
from docx import Document
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parent
DOCUMENTS = ROOT / "documents"
SPREADSHEETS = ROOT / "spreadsheets"
SCANS = ROOT / "scans"
_ZIP_EPOCH = (2025, 7, 1, 0, 0, 0)

ANIMALS = [
    ("wombat", "Vombatus ursinus", "burrows", "temperate forest and heathland"),
    ("koala", "Phascolarctos cinereus", "eucalypt canopy", "coastal and inland woodland"),
    ("quokka", "Setonix brachyurus", "dense scrub", "Rottnest Island and the south-west"),
    ("platypus", "Ornithorhynchus anatinus", "creek banks", "freshwater streams of the east"),
    ("echidna", "Tachyglossus aculeatus", "log hollows", "almost every Australian habitat"),
    ("bilby", "Macrotis lagotis", "spiral burrows", "arid and semi-arid grassland"),
    ("numbat", "Myrmecobius fasciatus", "fallen logs", "wandoo woodland"),
    ("cassowary", "Casuarius casuarius", "rainforest floor", "the Wet Tropics"),
]
HEADS = ["Program", "2024-25 $'000", "2025-26 $'000", "Change %"]
PROGRAMS = ["monitoring", "habitat", "research", "outreach", "rescue"]
STATES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]
WATERWAYS = ["Molonglo River", "Murrumbidgee River", "Tidbinbilla Creek", "Cotter River",
             "Queanbeyan River", "Naas River", "Paddys River", "Gudgenby River"]


def _sentences(rng: random.Random, animal: tuple[str, str, str, str], n: int) -> str:
    name, latin, shelter, habitat = animal
    pool = [
        f"The {name} ({latin}) shelters in {shelter} across {habitat}.",
        f"Field officers recorded {rng.randint(12, 480)} {name} observations during the reporting period.",
        f"Funding of ${rng.randint(1, 90)}.{rng.randint(0, 9)} million supports {name} habitat restoration.",
        f"Community groups reported that {name} numbers were stable in {rng.choice(STATES)}.",
        f"The department will review {name} monitoring protocols by 30 June {rng.randint(2026, 2028)}.",
        f"Fencing, predator control and revegetation remain the main {name} recovery actions.",
        f"Survey effort for the {name} increased by {rng.randint(3, 40)} per cent on the previous year.",
    ]
    return " ".join(rng.choice(pool) for _ in range(n))


def _fixed_zip(path: Path) -> None:
    """Rewrite a zip-based office file with fixed member timestamps and core dates."""
    with zipfile.ZipFile(path) as src:
        members = [(info.filename, src.read(info.filename)) for info in src.infolist()]
    stamp = b"2025-07-01T00:00:00Z"
    members = [
        (name, re.sub(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*", rb"\g<1>" + stamp, data)
         if name == "docProps/core.xml" else data)
        for name, data in members
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, data in members:
            dst.writestr(zipfile.ZipInfo(name, _ZIP_EPOCH), data, zipfile.ZIP_DEFLATED)
    path.write_bytes(buf.getvalue())


def budget_docx(path: Path) -> None:
    """A portfolio budget statement: headings, prose and tables, one header-only."""
    rng = random.Random(1)
    doc = Document()
    doc.core_properties.author = "Department of Native Wildlife"
    doc.core_properties.title = "Portfolio Budget Statements 2025-26"
    doc.add_heading("Portfolio Budget Statements 2025-26", 0)
    doc.add_heading("Budget Related Paper No. 1.9", 1)
    doc.add_heading("NATIVE WILDLIFE Portfolio", 1)
    doc.add_paragraph("© Commonwealth of Wombatia 2025")
    doc.add_paragraph("This publication is available for your use under a Creative Commons licence.")
    ministers = doc.add_table(rows=1, cols=3)
    for cell, name in zip(ministers.rows[0].cells, ["WINNIE WOMBAT", "KEVIN KOALA", "PENNY PLATYPUS"], strict=True):
        cell.text = name
    for section, animal in enumerate(ANIMALS, start=1):
        doc.add_heading(f"Section {section}: {animal[0].title()} programs", 1)
        doc.add_heading(f"{section}.1 Strategic direction", 2)
        for _ in range(3):
            doc.add_paragraph(_sentences(rng, animal, 4))
        doc.add_heading(f"{section}.2 Budgeted expenses", 2)
        doc.add_paragraph(_sentences(rng, animal, 2))
        table = doc.add_table(rows=1, cols=4)
        for cell, head in zip(table.rows[0].cells, HEADS, strict=True):
            cell.text = head
        for k in range(1, 4 + section % 3):
            before, after = rng.randint(800, 9000), rng.randint(800, 9000)
            row = table.add_row().cells
            row[0].text = f"{section}.{k} {animal[0].title()} {PROGRAMS[k - 1]}"
            row[1].text, row[2].text = f"{before:,}", f"{after:,}"
            row[3].text = f"{(after - before) / before * 100:.1f}"
        for _ in range(2):
            doc.add_paragraph(_sentences(rng, animal, 3))
    doc.save(path)
    _fixed_zip(path)


def _wrap(canvas: Canvas, x: float, y: float, text: str, *, size: float = 11,
          width: int = 92, leading: float = 14) -> float:
    canvas.setFont("Helvetica", size)
    for line in textwrap.wrap(text, width):
        canvas.drawString(x, y, line)
        y -= leading
    return y - 6


def _logo() -> Image.Image:
    img = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((10, 10, 110, 110), fill=(70, 110, 60))
    draw.ellipse((40, 30, 80, 70), fill=(230, 210, 150))
    return img


def redacted_notice_pdf(path: Path) -> list[str]:
    """A three-page native decision notice with vector redactions on page one."""
    rng = random.Random(2)
    quokka = ANIMALS[2]
    c = Canvas(str(path), invariant=1)
    pages: list[str] = []
    c.drawImage(ImageReader(_logo()), 470, 742, 70, 70)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(56, 780, "Notice of Administrative Decision")
    lines = [
        "Quokka Cove Out of School Hours Care",
        "Service approval number: SE-40099001   Provider approval number: PR-00099017",
        "Approved provider: Rottnest Community Services Incorporated",
        "Address: 1 Thomson Bay Road, ROTTNEST, WA 6161",
    ]
    y = 750
    for line in lines:
        y = _wrap(c, 56, y, line, leading=13) + 4
    body = [
        "The Regulatory Authority has decided to issue a compliance direction to the approved "
        "provider under the Wildlife Care Services National Law.",
        _sentences(rng, quokka, 3),
        "The nominated supervisor, [name withheld], was informed of the decision on 25 August 2025.",
        _sentences(rng, quokka, 3),
    ]
    for para in body:
        y = _wrap(c, 56, y, para)
    for rect in [(150, 655, 90, 12), (300, 617, 120, 12), (56, 560, 160, 12),
                 (240, 520, 70, 12), (330, 480, 110, 12), (90, 440, 140, 12)]:
        c.setFillColorRGB(0, 0, 0)
        c.rect(*rect, stroke=0, fill=1)
    pages.append("\n".join(lines + body))
    for page_no in (2, 3):
        c.showPage()
        c.setFont("Helvetica-Bold", 13)
        heading = "Reasons for the decision" if page_no == 2 else "Review rights"
        c.drawString(56, 790, heading)
        y, text = 765, []
        for _ in range(5):
            para = _sentences(rng, quokka, 4)
            text.append(para)
            y = _wrap(c, 56, y, para)
        pages.append("\n".join([heading, *text]))
    c.save()
    return pages


def audit_pdf(path: Path, transcript: Path) -> None:
    """A native audit report with a ruled table, and its plain-text transcript."""
    rng = random.Random(3)
    c = Canvas(str(path), invariant=1)
    out: list[str] = []
    for page_no, animal in enumerate(ANIMALS[:6], start=1):
        heading = f"Chapter {page_no}. Audit of {animal[0]} habitat management"
        c.setFont("Helvetica-Bold", 14)
        c.drawString(56, 790, heading)
        out.append(heading)
        y = 765
        for _ in range(3):
            para = _sentences(rng, animal, 4)
            out.append(para)
            y = _wrap(c, 56, y, para)
        if page_no == 3:
            rows = [["Region", "Sites", "Koalas counted", "Trend"]] + [
                [s, str(rng.randint(4, 40)), str(rng.randint(20, 900)), rng.choice(["stable", "rising", "falling"])]
                for s in STATES[:6]
            ]
            top, row_h, cols = y - 10, 18, [56, 186, 286, 406, 530]
            for i, row in enumerate(rows):
                c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 10)
                for j, cell in enumerate(row):
                    c.drawString(cols[j] + 4, top - row_h * i - 13, cell)
                out.append("  ".join(row))
            c.setStrokeColorRGB(0.7, 0.7, 0.7)
            for i in range(len(rows) + 1):
                c.line(cols[0], top - row_h * i, cols[-1], top - row_h * i)
            for x in cols:
                c.line(x, top, x, top - row_h * len(rows))
            c.setStrokeColorRGB(0, 0, 0)
        c.setFont("Helvetica", 9)
        c.drawString(290, 30, f"Page {page_no}")
        c.showPage()
    c.save()
    transcript.write_text("\n\n".join(out) + "\n", encoding="utf-8")


def spacing_pdf(path: Path) -> None:
    """A native page per leading and paragraph spacing, with size and weight changes.

    Every other native PDF here is set at one leading; block grouping needs
    pages that vary it. Each page is a bold heading over paragraphs of four
    lines, then a run of lines whose pitch grows step by step.
    """
    rng = random.Random(5)
    c = Canvas(str(path), invariant=1)
    cases = [(11, 12.5, 0), (11, 14, 6), (11, 14, 10), (10, 12, 4), (12, 18, 12), (11, 16, 0)]
    for page_no, (size, leading, gap) in enumerate(cases, start=1):
        animal = ANIMALS[page_no % len(ANIMALS)]
        c.setFont("Helvetica-Bold", size + 3)
        c.drawString(56, 790, f"Section {page_no}. Notes on {animal[0]} spacing")
        y = 790 - (size + 3) - gap - 4
        for _ in range(3):
            c.setFont("Helvetica", size)
            for line in textwrap.wrap(_sentences(rng, animal, 3), 90):
                c.drawString(56, y, line)
                y -= leading
            y -= gap
        c.setFont("Helvetica-Bold", size)
        c.drawString(56, y, "Run-in label in bold")
        c.setFont("Helvetica", size)
        c.drawString(56, y - leading, "Body text of the same size directly beneath it")
        c.setFont("Helvetica", size + 4)
        c.drawString(56, y - 2 * leading - 6, "A larger line closing the page")
        c.showPage()
    c.save()


INDEX_COLUMNS = [  # (header lines, x in landscape points, value maker)
    (["Unique ID"], 30, lambda r, i: f"{i:05d}"),
    (["Directorate"], 78, lambda r, i: r.choice(["NWS", "PKS", "ENV"])),
    (["Service Name"], 128, lambda r, i: f"{r.choice(['Bilby', 'Numbat', 'Quoll'])} Care {r.randint(1, 9)}"),
    (["File Name"], 220, lambda r, i: f"doc-{i:05d}.pdf"),
    (["Document Type"], 300, lambda r, i: r.choice(["Email", "Letter", "Report", "Notice"])),
    (["Case Number"], 372, lambda r, i: f"CAS-{r.randint(100000, 999999)}"),
    (["Subsection", "code"], 452, lambda r, i: r.choice(["3bi", "3bii"])),
    (["Issue Date"], 512, lambda r, i: f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/2024"),
    (["Author"], 582, lambda r, i: r.choice(["NWSA", "Ranger", "Vet"])),
    (["Privilege"], 642, lambda r, i: r.choice(["No", "Yes"])),
    (["Out of Scope", "Exemption"], 702, lambda r, i: r.choice(["No", "Partial"])),
]


def _index_rows(rng: random.Random, n: int) -> list[list[str]]:
    return [[make(rng, i) for _, _, make in INDEX_COLUMNS] for i in range(1, n + 1)]


def foi_index_pdf(path: Path, *, pages: int = 4, per_page: int = 36) -> list[list[str]]:
    """A spreadsheet printed to PDF: landscape content on rotated portrait pages."""
    rng = random.Random(6)
    rows = _index_rows(rng, pages * per_page)
    c = Canvas(str(path), pagesize=(842, 595), invariant=1)
    for p in range(pages):
        c.setPageRotation(90)
        c.saveState()
        c.translate(595, 0)
        c.rotate(90)  # draw in a 842 x 595 landscape frame
        y = 560
        c.setFont("Helvetica-Bold", 9)
        for label, value in [("FOI reference", "FOI-2025-042"), ("Element #", "3(b)(i) - 3(b)(ii)")]:
            if p == 0:  # later pages leave the space, so the frozen header keeps its height
                c.drawString(30, y, label)
                c.drawString(128, y, value)
            y -= 14
        y -= 10
        c.setFont("Helvetica-Bold", 7)
        for lines, x, _ in INDEX_COLUMNS:
            for k, line in enumerate(lines):
                c.drawString(x, y - 8 * k, line)
        y -= 22
        c.setFont("Helvetica", 7)
        for row in rows[p * per_page:(p + 1) * per_page]:
            for (_, x, _), cell in zip(INDEX_COLUMNS, row, strict=True):
                c.drawString(x, y, cell)
            y -= 13
        c.restoreState()
        c.showPage()
    c.save()
    return rows


def schedule_pdf(path: Path, *, n: int = 44) -> None:
    """A single-page schedule of documents, printed from a spreadsheet, unrotated."""
    rng = random.Random(7)
    cols = [("No.", 40), ("Date", 70), ("Type", 130), ("Author", 185), ("Pages", 245),
            ("Decision", 285), ("Exemption", 350), ("Folio", 410), ("Reference", 460)]
    c = Canvas(str(path), invariant=1)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, 800, "FOI reference: FOI-2025-042")
    c.drawString(40, 786, "Schedule: Part 3(b)")
    c.setFont("Helvetica-Bold", 8)
    for head, x in cols:
        c.drawString(x, 760, head)
    c.setFont("Helvetica", 8)
    y = 744
    for i in range(1, n + 1):
        cells = [str(i), f"{rng.randint(1, 28):02d}/0{rng.randint(1, 9)}/2024",
                 rng.choice(["Email", "Letter", "Brief"]), rng.choice(["NWSA", "Ranger"]),
                 str(rng.randint(1, 12)), rng.choice(["Release", "Partial", "Exempt"]),
                 rng.choice(["s47F", "s47E", "nil"]), f"F{i * 3:03d}", f"FOI-042-{i:03d}"]
        for (_, x), cell in zip(cols, cells, strict=True):
            c.drawString(x, y, cell)
        y -= 15.5
    c.save()


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _scanned(img: Image.Image, seed: int) -> Image.Image:
    """Grey paper, speckle noise and a slight skew, so the image reads as a scan."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("L"), dtype=np.int16) - 20
    speckle = rng.random(arr.shape) < 0.004
    arr[speckle] = rng.integers(60, 160, int(speckle.sum()))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)).rotate(0.4, fillcolor=235)


def form_image(path: Path, title: str, fields: list[tuple[str, str]], seed: int) -> None:
    """A filled-in paper form: a title, then labelled boxes holding typed values."""
    img = Image.new("L", (1000, 1300), 255)
    draw = ImageDraw.Draw(img)
    draw.text((60, 50), title, font=_font(38), fill=0)
    lines, y = [title], 140
    for label, value in fields:
        draw.text((60, y + 12), label, font=_font(24), fill=0)
        draw.rectangle((380, y, 940, y + 52), outline=0, width=2)
        draw.text((394, y + 12), value, font=_font(26), fill=0)
        lines.append(f"{label} {value}")
        y += 82
    draw.rectangle((60, y + 20, 84, y + 44), outline=0, width=2)
    draw.text((100, y + 18), "I declare the information above is correct.", font=_font(22), fill=0)
    lines.append("I declare the information above is correct.")
    _scanned(img, seed).save(path)
    path.with_suffix(".gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def line_image(path: Path, text: str, seed: int) -> None:
    """One line of field-note text, the shape of a single handwriting line."""
    width = 40 + int(_font(40).getlength(text))
    img = Image.new("L", (width, 110), 255)
    ImageDraw.Draw(img).text((20, 30), text, font=_font(40), fill=20)
    _scanned(img, seed).save(path)
    path.with_suffix(".gt.txt").write_text(text + "\n", encoding="utf-8")


def page_image(path: Path, kind: str, seed: int) -> None:
    """A scanned report page: a heading, then prose, a table, or very little."""
    rng = random.Random(seed)
    img = Image.new("L", (1025, 1325), 255)
    draw = ImageDraw.Draw(img)
    animal = ANIMALS[seed % len(ANIMALS)]
    heading = f"{animal[0].title()} survey results"
    draw.text((70, 60), heading, font=_font(40), fill=0)
    words, y = heading.split(), 140
    paras = {"dense": 6, "table": 1, "sparse": 0}[kind]
    for _ in range(paras):
        for line in textwrap.wrap(_sentences(rng, animal, 3), 70):
            draw.text((70, y), line, font=_font(22), fill=0)
            words += line.split()
            y += 32
        y += 18
    if kind == "sparse":
        line = f"Prepared by the {animal[0]} recovery team."
        draw.text((70, y), line, font=_font(22), fill=0)
        words += line.split()
    if kind in ("dense", "table"):
        rows = [["Site", "Visits", "Animals", "Trend"]] + [
            [f"Site {k}", str(rng.randint(2, 30)), str(rng.randint(5, 400)), rng.choice(["stable", "rising"])]
            for k in range(1, 9 if kind == "table" else 6)
        ]
        cols, top = [70, 330, 560, 760, 955], y + 10
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                draw.text((cols[j] + 10, top + 44 * i + 10), cell, font=_font(22), fill=0)
                words.append(cell)
        for i in range(len(rows) + 1):
            draw.line((cols[0], top + 44 * i, cols[-1], top + 44 * i), fill=0, width=2)
        for x in cols:
            draw.line((x, top, x, top + 44 * len(rows)), fill=0, width=2)
    _scanned(img, seed).save(path)
    path.with_suffix(".gt.txt").write_text(" ".join(words) + "\n", encoding="utf-8")


def scanned_letter_pdf(path: Path, page: Path) -> None:
    """An image-only PDF of one scanned page: no text layer, so it is OCR'd."""
    img = Image.open(page)
    size = (595, 595 * img.height / img.width)
    c = Canvas(str(path), pagesize=size, invariant=1)
    c.drawImage(ImageReader(img), 0, 0, *size)
    c.save()


def register_csv(path: Path) -> None:
    """A register export: one row per platypus sighting."""
    rng = random.Random(4)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["SightingID", "Date", "Waterway", "Suburb", "State", "Postcode",
                    "Observer Organisation", "Count", "Notes"])
        for i in range(1, 301):
            w.writerow([
                f"PL-{i:05d}", f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                rng.choice(WATERWAYS), rng.choice(["THARWA", "UPPER COTTER", "PADDYS RIVER", "OAKS ESTATE"]),
                "ACT", rng.choice(["2620", "2611", "2620", "2911"]),
                rng.choice(["Platypus Watch Inc", "Waterwatch Volunteers", "Ranger Service"]),
                rng.randint(1, 4), rng.choice(["", "juvenile present", "dusk survey", "burrow entrance seen"]),
            ])


def statistics_xlsx(path: Path) -> None:
    """A statistics workbook: metadata rows above each table, three sheets."""
    rng = random.Random(5)
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    summary.append(["Echidna population statistics, September quarter 2025"])
    summary.append(["Source: Native Wildlife Survey Program"])
    summary.append([])
    summary.append(["Measure", "June qtr 2025", "Sept qtr 2025", "Change"])
    for measure in ["Sites surveyed", "Echidnas recorded", "Juveniles recorded", "Road strikes"]:
        a, b = rng.randint(50, 900), rng.randint(50, 900)
        summary.append([measure, a, b, b - a])
    by_state = wb.create_sheet("By State")
    by_state.append(["Table 2. Echidnas recorded by state"])
    by_state.append([])
    by_state.append(["State", "Sites", "Adults", "Juveniles", "Total"])
    for state in STATES:
        adults, young = rng.randint(10, 400), rng.randint(0, 60)
        by_state.append([state, rng.randint(2, 30), adults, young, adults + young])
    notes = wb.create_sheet("Notes")
    for line in ["Explanatory notes", "Counts are provisional and may be revised.",
                 "Road strikes are reported by state transport agencies."]:
        notes.append([line])
    wb.properties.creator = "Native Wildlife Survey Program"
    wb.save(path)
    _fixed_zip(path)


def main() -> None:
    DOCUMENTS.mkdir(parents=True, exist_ok=True)
    SPREADSHEETS.mkdir(parents=True, exist_ok=True)
    budget_docx(DOCUMENTS / "wombat-portfolio-budget-statements.docx")
    redacted_notice_pdf(DOCUMENTS / "quokka-care-decision-notice_redacted.pdf")
    audit_pdf(DOCUMENTS / "koala-habitat-audit.pdf", DOCUMENTS / "koala-habitat-audit_transcript.txt")
    spacing_pdf(DOCUMENTS / "emu-spacing-variants.pdf")
    foi_index_pdf(DOCUMENTS / "bilby-foi-documents-index.pdf")
    schedule_pdf(DOCUMENTS / "bilby-schedule-of-documents.pdf")
    SCANS.mkdir(parents=True, exist_ok=True)
    form_image(SCANS / "form-bilby-sighting-report.png", "Bilby Sighting Report",
               [("Observer name", "Jo Wattle"), ("Date of sighting", "14/03/2025"),
                ("Location", "Currawinya National Park"), ("Number seen", "3")], seed=11)
    form_image(SCANS / "form-wombat-burrow-permit.png", "Wombat Burrow Works Permit",
               [("Applicant", "Bushland Fencing Pty Ltd"), ("Property", "Lot 12 Gundaroo Road"),
                ("Burrows affected", "2"), ("Works start", "02/06/2025"), ("Works end", "20/06/2025"),
                ("Ranger approval", "Pending")], seed=12)
    form_image(SCANS / "form-koala-rescue-intake.png", "Koala Rescue Intake",
               [("Rescuer", "Sam Gumtree"), ("Found at", "Port Macquarie"), ("Condition", "Dehydrated")],
               seed=13)
    for k, text in enumerate([
        "Two echidnas crossed the track near the dam.",
        "Quokka seen at dusk by the jetty.",
        "Platypus burrow found on the eastern bank of the creek after the flood waters receded.",
    ]):
        line_image(SCANS / f"line-field-note-{k + 1}.png", text, seed=21 + k)
    for kind, seed in [("dense", 31), ("table", 32), ("sparse", 33)]:
        page_image(SCANS / f"page-{kind}.png", kind, seed=seed)
    scanned_letter_pdf(DOCUMENTS / "numbat-scanned-survey-page.pdf", SCANS / "page-dense.png")
    register_csv(SPREADSHEETS / "platypus-sightings-register.csv")
    statistics_xlsx(SPREADSHEETS / "echidna-population-statistics.xlsx")


if __name__ == "__main__":
    main()
