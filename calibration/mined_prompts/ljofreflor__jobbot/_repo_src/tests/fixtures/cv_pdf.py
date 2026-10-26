"""Build the PDF CVs the importer tests read.

The PDFs are written at test time instead of committed: the PII guard blocks every
.pdf outside latex/, and a tracked binary is a fixture nobody can review in a diff.
The data is invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PAGE1 = """\
 
Marta Soto Vera 
GEOFÍSICO | SCRUM MASTER Jr. 
Móvil +56 9 00000000    marta.soto@example.com 
https://www.linkedin.com/in/marta-soto-vera/ 
Santiago, Región Metropolitana 
 
Ingeniera de proyectos con más de 9 años de experiencia en el sector minero, especializada en
supervisión, planificación y gestión de actividades geofísicas.
 
C O M P E T E N C I A S    P R O F E S I O N A L E S 
Metodologías Agiles | Gestión de Proyectos | Liderazgo | Scrum
 
E X P E R I E N C I A   P R O F E S I O N A L  
 
NORTE CONSULTORES LTDA - SANTIAGO DE CHILE 
10/2024 –ACTUALIDAD Coordinadora Administrativa 
Responsabilidades: Supervisar las actividades diarias del centro de distribución y coordinar la
recepción de materiales.
Logros 
• Reportes: informe mensual de ubicaciones disponibles.
• Gestión de personal: control de asistencia y turnos.
 
GEODETEC INGENIERIA SPA - SANTIAGO DE CHILE 
07/2022 –11/2023 Coordinadora/Ingeniera de Proyectos 
Responsabilidades: Planificación y ejecución de proyectos multidisciplinarios mineros.
Logros 
• Implementación de metodologías ágiles (Scrum) en equipos de siete personas.
• Reducción de plazos de entrega y mejora de la calidad de los entregables.
"""

PAGE2 = """\
 
PETRÓLEOS DEL SUR S.A - VENEZUELA 
09/2010-06/2016  Especialista / Ingeniera de Proyectos 
Responsabilidades: Control de calidad de datos sísmicos y elaboración de mapas y modelos.
Logros 
• Planificación detallada de proyectos exploratorios.
 
09/2008-06/201  Ingeniera de Terreno 
Responsabilidades: Supervisión de adquisición sísmica 3D.
Logros 
• Reportes de avance e informes técnicos.
 
F O R M A C I Ó N    A C A D É M I C A  
Universidad del Litoral (UDL) Caracas, Venezuela, 09/2000 – 10/2007 Ingeniera Geofísica
Instituto Playa Norte, Santiago de Chile,(2023-2024) Diplomado Diseño y Gestión Ágil de Proyectos
 
C O N O C I M I E N T O S     E N     S I S T E M A S 
Microsoft 365 | AutoCAD | Oasis Montaj | Python
"""


def escape(line: str) -> str:
    return line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def content_stream(text: str) -> bytes:
    parts = ["BT", "/F1 9 Tf", "40 780 Td", "11 TL"]
    for line in text.splitlines():
        parts.append(f"({escape(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("cp1252", errors="replace")


def write_pdf(objects: list[bytes], path: Path) -> None:
    """Serialise object bodies (1-based numbering) with an xref table and trailer."""
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode("latin-1")
    path.write_bytes(bytes(out))


def stream_object(stream: bytes) -> bytes:
    return (
        b"<< /Length "
        + str(len(stream)).encode("latin-1")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream"
    )


def build_pdf(pages: list[bytes], path: Path) -> None:
    objects: list[bytes] = []
    page_count = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(page_count))

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("latin-1")
    )
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    for i, stream in enumerate(pages):
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
        ).encode("latin-1")
        objects.append(page_obj)
        objects.append(stream_object(stream))

    write_pdf(objects, path)


@dataclass
class SubsetFont:
    """A Word-style subset font: we choose the widths and which code stays unmapped.

    `to_unicode` is the whole /ToUnicode map, so a code left out of it is the bug the
    importer has to survive: pypdf falls back to the raw byte for that glyph.
    """

    name: str
    base_font: str
    widths: dict[int, int]
    to_unicode: dict[int, str]


def cmap_stream(mapping: dict[int, str]) -> bytes:
    entries = "\n".join(
        f"<{code:02x}> <{text.encode('utf-16-be').hex()}>"
        for code, text in sorted(mapping.items())
    )
    return (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CMapName /Fixture def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<00><ff>\nendcodespacerange\n"
        f"{len(mapping)} beginbfchar\n{entries}\nendbfchar\nendcmap\nend\nend\n"
    ).encode("latin-1")


def font_object(font: SubsetFont, to_unicode_ref: int) -> bytes:
    first, last = min(font.widths), max(font.widths)
    widths = " ".join(str(font.widths.get(code, 0)) for code in range(first, last + 1))
    return (
        f"<< /Type /Font /Subtype /TrueType /BaseFont /{font.base_font} "
        f"/FirstChar {first} /LastChar {last} /Widths [{widths}] "
        f"/ToUnicode {to_unicode_ref} 0 R >>"
    ).encode("latin-1")


def write_subset_font_cv(path: Path, fonts: list[SubsetFont], lines: list[tuple[str, str]]) -> Path:
    """One page whose text is drawn with the given subset fonts, byte by byte."""
    objects: list[bytes] = [b"", b""]
    resources: list[str] = []
    for index, font in enumerate(fonts):
        font_ref = 3 + 2 * index
        objects.append(font_object(font, font_ref + 1))
        objects.append(stream_object(cmap_stream(font.to_unicode)))
        resources.append(f"/{font.name} {font_ref} 0 R")

    page_ref = 3 + 2 * len(fonts)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{page_ref} 0 R] /Count 1 >>".encode("latin-1")

    parts = ["BT", "40 780 Td", "11 TL"]
    for font_name, text in lines:
        parts.append(f"/{font_name} 9 Tf")
        parts.append(f"({escape(text)}) Tj")
        parts.append("T*")
    parts.append("ET")

    objects.append(
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << {' '.join(resources)} >> >> "
            f"/Contents {page_ref + 1} 0 R >>"
        ).encode("latin-1")
    )
    objects.append(stream_object("\n".join(parts).encode("latin-1")))

    write_pdf(objects, path)
    return path


def build_blank_pdf(path: Path) -> None:
    stream = b"0 0 0 rg\n100 100 200 300 re f"
    build_pdf([stream], path)


def write_sample_cv(path: Path) -> Path:
    """A two-page Word-style CV with a text layer."""
    build_pdf([content_stream(PAGE1), content_stream(PAGE2)], path)
    return path


def write_scanned_cv(path: Path) -> Path:
    """A page with graphics only: what a scanned CV looks like to pypdf."""
    build_blank_pdf(path)
    return path
