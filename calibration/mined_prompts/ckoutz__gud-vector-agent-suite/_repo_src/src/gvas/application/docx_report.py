"""Generic Word rendering of a completed field-notes report.

The document is a plain OOXML package written with the standard library: one
title, a version line, a heading per section and a paragraph per block. No
letterhead or owner template is applied; the owner-supplied template path in
``docs/templates_and_site_plans.md`` is the follow-up. Output is deterministic
for a given report version so retries reproduce the identical file.
"""

import re
from io import BytesIO
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from gvas.domain.reporting import (
    DOCX_MEDIA_TYPE,
    FieldNotesReportVersion,
    RenderedReportArtifact,
)

_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
    'relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
    'officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-'
    'officedocument.wordprocessingml.styles+xml"/>'
    "</Types>"
)

_ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)

_DOCUMENT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/styles" Target="styles.xml"/>'
    "</Relationships>"
)

_STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<w:styles xmlns:w="{_W}">'
    '<w:docDefaults><w:rPrDefault><w:rPr><w:sz w:val="22"/></w:rPr></w:rPrDefault>'
    '<w:pPrDefault><w:pPr><w:spacing w:after="120"/></w:pPr></w:pPrDefault></w:docDefaults>'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
    '<w:name w:val="Normal"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
    '<w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="40"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
    '<w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="240"/><w:outlineLvl w:val="0"/></w:pPr>'
    '<w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>'
    "</w:styles>"
)


def _paragraph(text: str, style: str | None = None) -> str:
    properties = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{properties}<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def render_report_docx_body(version: FieldNotesReportVersion) -> str:
    """The ``word/document.xml`` part for a report version."""

    paragraphs = [
        _paragraph(version.document.title, "Title"),
        _paragraph(f"Report version {version.version}"),
    ]
    for section in version.document.sections:
        paragraphs.append(_paragraph(section.heading, "Heading1"))
        paragraphs.extend(_paragraph(block.text) for block in section.blocks)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W}"><w:body>{"".join(paragraphs)}</w:body></w:document>'
    )


def render_report_docx(version: FieldNotesReportVersion) -> bytes:
    """Byte-for-byte reproducible DOCX for a report version."""

    parts = (
        ("[Content_Types].xml", _CONTENT_TYPES),
        ("_rels/.rels", _ROOT_RELS),
        ("word/_rels/document.xml.rels", _DOCUMENT_RELS),
        ("word/styles.xml", _STYLES),
        ("word/document.xml", render_report_docx_body(version)),
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, content in parts:
            info = ZipInfo(name, date_time=_FIXED_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, content.encode("utf-8"))
    return buffer.getvalue()


def report_docx_filename(version: FieldNotesReportVersion) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", version.document.title.lower()).strip("-") or "report"
    return f"{slug}-v{version.version}.docx"


class DocxReportRenderer:
    """``ReportArtifactRendererPort`` producing the generic Word projection."""

    def render(self, version: FieldNotesReportVersion) -> RenderedReportArtifact:
        return RenderedReportArtifact(
            content=render_report_docx(version),
            media_type=DOCX_MEDIA_TYPE,
            filename=report_docx_filename(version),
        )
