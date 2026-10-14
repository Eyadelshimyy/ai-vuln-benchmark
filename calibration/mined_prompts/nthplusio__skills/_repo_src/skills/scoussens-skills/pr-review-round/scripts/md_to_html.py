#!/usr/bin/env python3
"""md_to_html.py — render a review body's Markdown to HTML for the artifact.

Why this exists rather than a JS parser in the page: the Artifact CSP blocks
external scripts, so a CDN markdown library never loads and the page silently
falls back to showing raw `**bold**` and un-rendered ``` fences. Rendering at
build time has no runtime dependency and cannot fail in the viewer.

Scope is deliberately the subset a PR review body actually uses — headings,
bold/italic, inline code, fenced code, lists, tables, rules, blockquotes,
links. It is not a CommonMark implementation and does not try to be.

Safety: the input is escaped *before* any markup is inserted, so a review that
quotes `<img src="0">` renders as visible text rather than becoming an element.
Inline code is extracted before emphasis so `` `**not bold**` `` stays literal.

Usage:
    from md_to_html import render
    html = render(body)
"""
import html
import re

__all__ = ["render"]


def _esc(s):
    return html.escape(s, quote=False)


def _inline(text):
    """Inline spans. Escapes first, then substitutes markup, protecting code."""
    text = _esc(text)

    # Pull inline code out before emphasis so backticked markdown stays literal.
    codes = []

    def _stash_code(m):
        codes.append(m.group(1))
        return f"\x00CODE{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _stash_code, text)

    # Links: [text](url). Only http(s)/# targets, so a body can't inject a
    # javascript: href.
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+|#[^\s)]*)\)",
        r'<a href="\2">\1</a>',
        text,
    )
    # Bare autolinks that markdown would leave alone but readers expect to click.
    text = re.sub(
        r"(?<![\"\'>=])\bhttps?://[^\s<>\"')\]]+",
        lambda m: f'<a href="{m.group(0)}">{m.group(0)}</a>',
        text,
    )

    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Single-asterisk emphasis, but not a lone `*` used as a bullet or glob.
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w`])_(?!\s)([^_]+?)(?<!\s)_(?![\w`])", r"<em>\1</em>", text)
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text)

    for i, c in enumerate(codes):
        text = text.replace(f"\x00CODE{i}\x00", f"<code>{c}</code>")
    return text


_FENCE = re.compile(r"^```+\s*([A-Za-z0-9_+-]*)\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:-]*-[\s:|-]*\|?\s*$")


def _split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def render(md):
    """Markdown -> HTML string. Block-level pass, with inline applied inside."""
    if not md:
        return ""
    lines = md.replace("\r\n", "\n").split("\n")
    out = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Fenced code — emitted verbatim, no inline processing at all.
        m = _FENCE.match(line.strip())
        if m:
            lang = m.group(1)
            i += 1
            buf = []
            while i < n and not _FENCE.match(lines[i].strip()):
                buf.append(lines[i])
                i += 1
            i += 1  # closing fence
            cls = f' class="lang-{lang}"' if lang else ""
            out.append(f"<pre><code{cls}>{_esc(chr(10).join(buf))}</code></pre>")
            continue

        if not line.strip():
            i += 1
            continue

        # Horizontal rule — must be checked before the table separator, which
        # looks similar, and before emphasis eats the dashes.
        if re.fullmatch(r"\s*([-*_])\s*(\1\s*){2,}", line):
            out.append("<hr>")
            i += 1
            continue

        # ATX heading. Reviews start at ## so h2 maps to h4 here — the artifact
        # already owns h1/h2/h3, and a review body must not outrank the page.
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = min(len(m.group(1)) + 2, 6)
            out.append(f"<h{lvl}>{_inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue

        # Table: a header row followed by a |---|---| separator.
        if "|" in line and i + 1 < n and _TABLE_SEP.match(lines[i + 1]) and "|" in lines[i + 1]:
            head = _split_row(line)
            i += 2
            body = []
            while i < n and "|" in lines[i] and lines[i].strip():
                body.append(_split_row(lines[i]))
                i += 1
            th = "".join(f"<th>{_inline(c)}</th>" for c in head)
            trs = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                for r in body
            )
            out.append(
                '<div class="md-table"><table><thead><tr>'
                f"{th}</tr></thead><tbody>{trs}</tbody></table></div>"
            )
            continue

        # Blockquote — consume the contiguous run, then render it recursively so
        # a quote containing a list or code still works.
        if re.match(r"^\s*>", line):
            buf = []
            while i < n and re.match(r"^\s*>", lines[i]):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append(f"<blockquote>{render(chr(10).join(buf))}</blockquote>")
            continue

        # Lists. A continuation line is any non-blank line that isn't itself a
        # new marker — review bullets routinely wrap over several lines.
        m = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", line)
        if m:
            ordered = not m.group(2)[0] in "-*+"
            items = []
            while i < n:
                mm = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", lines[i]) if i < n else None
                if not mm:
                    break
                cur = [mm.group(3)]
                i += 1
                while i < n and lines[i].strip() and not re.match(
                    r"^(\s*)([-*+]|\d+[.)])\s+", lines[i]
                ) and not _FENCE.match(lines[i].strip()) and not re.match(
                    r"^#{1,6}\s", lines[i]
                ):
                    cur.append(lines[i].strip())
                    i += 1
                items.append(" ".join(cur))
            tag = "ol" if ordered else "ul"
            lis = "".join(f"<li>{_inline(t)}</li>" for t in items)
            out.append(f"<{tag}>{lis}</{tag}>")
            continue

        # Paragraph: consume to the next blank line or block-level marker.
        buf = [line.strip()]
        i += 1
        while i < n and lines[i].strip():
            nxt = lines[i]
            if (_FENCE.match(nxt.strip())
                    or re.match(r"^#{1,6}\s", nxt)
                    or re.match(r"^\s*>", nxt)
                    or re.match(r"^(\s*)([-*+]|\d+[.)])\s+", nxt)
                    or re.fullmatch(r"\s*([-*_])\s*(\1\s*){2,}", nxt)
                    or ("|" in nxt and i + 1 < n and _TABLE_SEP.match(lines[i + 1]))):
                break
            buf.append(nxt.strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(buf))}</p>")

    return "".join(out)


if __name__ == "__main__":
    import sys
    sys.stdout.write(render(sys.stdin.read()))
