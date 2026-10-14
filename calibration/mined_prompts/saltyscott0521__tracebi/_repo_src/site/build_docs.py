#!/usr/bin/env python3
"""
Render the ``docs/`` vault into a static docs site under ``site/docs/``.

    python site/build_docs.py

The markdown in ``docs/`` is the single source. This script renders it; it
never edits it. Three consumers read the same files — Obsidian, the app's
Docs page (``/handbook``, which fetches raw markdown and renders it in the
browser), and this static site — so a docs change lands everywhere at once.

**The output is committed.** ``site/`` deploys to any static host with no
build step, which is the property that makes the marketing site trivial to
host; generating at deploy time would give that up. The cost is that the
generated HTML can drift from the markdown, so ``tests/test_docs_site.py``
re-runs this and fails if the tree it produces differs from the tree in git.
Run this script whenever you edit ``docs/``.

**The palette is extracted from ``site/index.html``**, not copied. A docs site
in a different blue than the landing page reads as a different product, and a
second hand-maintained copy of the tokens is exactly how that happens.
"""

from __future__ import annotations

import html
import re
import shutil
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
REPO = SITE.parent
DOCS = REPO / "docs"
OUT = SITE / "docs"

#: Internal working documents. The positioning doc is gitignored outright and
#: the roadmap is a control document — neither is product documentation.
HIDDEN = {"north-star", "ROADMAP"}

#: The vault's sections, in reading order: ideas before look-up tables.
SECTIONS = [
    ("concepts", "Concepts", "The ideas. Read these once."),
    ("guides", "Guides", "Task-shaped walkthroughs."),
    ("reference", "Reference", "Look things up."),
    ("architecture", "Architecture", "For changing the framework."),
    ("agents", "Agent SOPs", "Standard operating procedures."),
]

#: Within a section, the page that orients you comes first.
FIRST = {"concepts": "the-three-phase-workflow", "guides": "plain-english"}


def _require_markdown():
    try:
        import markdown  # noqa: F401
    except ImportError:
        sys.exit(
            "This generator needs Markdown.\n"
            "  pip install -e \".[dev]\"     (or: pip install markdown)"
        )
    return __import__("markdown")


def favicon_href() -> str:
    """The landing page's inline favicon — same mark, one definition."""
    src = (SITE / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<link rel="icon" href="([^"]+)"', src)
    return m.group(1) if m else ""


def theme_css() -> str:
    """The token blocks from index.html — one source for the palette."""
    src = (SITE / "index.html").read_text(encoding="utf-8")
    style = re.search(r"<style>(.*?)</style>", src, re.S)
    if not style:
        sys.exit("site/index.html has no <style> block to take the theme from.")
    body = style.group(1)
    # Everything from the first :root through the last theme block — the
    # token definitions, and nothing about the landing page's own layout.
    start = body.index(":root{")
    end = body.index('}', body.index(':root[data-theme="dark"]{'))
    # Walk to the true close of the dark block (it contains nested braces).
    depth, i = 0, body.index(':root[data-theme="dark"]{')
    while i < len(body):
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1
    return body[start:end].strip()


def title_of(path: Path) -> str:
    """The page's first H1, with any inline-code backticks stripped."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip().replace("`", "")
    return path.stem.replace("-", " ").capitalize()


def collect() -> tuple[dict, dict, Path | None]:
    """Return ``(tree, by_stem, home)`` over the vault."""
    tree: dict[str, list] = {key: [] for key, _, _ in SECTIONS}
    by_stem: dict[str, tuple[str, str]] = {}
    home = None
    for md in sorted(DOCS.rglob("*.md")):
        rel = md.relative_to(DOCS)
        stem = md.stem
        if stem in HIDDEN:
            continue
        section = rel.parts[0] if len(rel.parts) > 1 else ""
        if not section:
            if stem == "index":
                home = md
                by_stem[stem] = ("", stem)
            continue
        if section in tree:
            tree[section].append(md)
            by_stem[stem] = (section, stem)
    for key, items in tree.items():
        lead = FIRST.get(key)
        items.sort(key=lambda p: (p.stem != lead, title_of(p).lower()))
    return tree, by_stem, home


def href_for(section: str, stem: str, *, depth: int) -> str:
    """A relative link from a page nested *depth* directories under docs/."""
    up = "../" * depth
    return f"{up}{section}/{stem}.html" if section else f"{up}{stem}.html"


def resolve_wiki(md_text: str, by_stem: dict, depth: int) -> str:
    """Rewrite ``[[target|alias]]`` into a real markdown link.

    An unresolved target becomes bold text rather than a dead link — the
    vault deliberately allows a link to a page that does not exist yet.
    """
    def one(m):
        inner = m.group(1)
        target, _, alias = inner.partition("|")
        page, _, anchor = target.partition("#")
        page = page.strip()
        label = (alias or page or anchor).strip().replace("`", "")
        if not page:                      # [[#Anchor]] — same page
            slug = re.sub(r"[^a-z0-9]+", "-", anchor.lower()).strip("-")
            return f"[{label}](#{slug})"
        hit = by_stem.get(page)
        if not hit:
            return f"**{label}**"
        sec, stem = hit
        if not alias:
            label = _TITLES.get(page, label)
        return f"[{label}]({href_for(sec, stem, depth=depth)})"
    return re.sub(r"\[\[([^\]]+)\]\]", one, md_text)


_TITLES: dict[str, str] = {}

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} — TraceBi docs</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{desc}">
<link rel="icon" href="{favicon}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Source+Code+Pro:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{up}docs.css">
</head>
<body>
<nav class="topbar"><div class="bar">
  <a class="brand" href="{up}../index.html">TraceBi</a>
  <div class="nav-links">
    <a href="{up}index.html">Docs</a>
    <a href="https://github.com/saltyscott0521/tracebi">GitHub</a>
    <button class="tgl" id="themeToggle" type="button" aria-label="Toggle light / dark">◐</button>
  </div>
</div></nav>
<div class="shell">
  <details class="side" id="nav" open>
    <summary>Docs menu</summary>
    <div class="links">{nav}</div>
  </details>
  <main class="doc">
{body}
  </main>
</div>
<script>
(function(){{
  // The nav ships open so the page needs no JS to work; on a phone that
  // means scrolling past every link to reach the first word, so collapse it
  // here and re-collapse after a tap.
  var nav=document.getElementById('nav'), narrow=matchMedia('(max-width:900px)');
  function fit(){{ nav.open = !narrow.matches; }}
  fit(); narrow.addEventListener('change', fit);
  nav.querySelector('.links').addEventListener('click', function(e){{
    if(e.target.tagName==='A' && narrow.matches) nav.open=false;
  }});

  var r=document.documentElement, k='tb-theme', s=localStorage.getItem(k);
  if(s) r.setAttribute('data-theme',s);
  document.getElementById('themeToggle').addEventListener('click',function(){{
    var cur=r.getAttribute('data-theme');
    if(!cur) cur=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';
    var next=cur==='dark'?'light':'dark';
    r.setAttribute('data-theme',next); localStorage.setItem(k,next);
  }});
}})();
</script>
</body>
</html>
"""


def build_nav(tree, home, *, current: Path, depth: int) -> str:
    out = []
    if home is not None:
        cls = ' class="on"' if current == home else ""
        out.append(f'<a{cls} href="{href_for("", "index", depth=depth)}">Overview</a>')
    for key, label, blurb in SECTIONS:
        items = tree.get(key) or []
        if not items:
            continue
        out.append(f'<div class="sec"><span>{label}</span><em>{html.escape(blurb)}</em></div>')
        for md in items:
            cls = ' class="on"' if md == current else ""
            out.append(
                f'<a{cls} href="{href_for(key, md.stem, depth=depth)}">'
                f"{html.escape(title_of(md))}</a>"
            )
    return "\n".join(out)


def main() -> int:
    markdown = _require_markdown()
    if not DOCS.is_dir():
        sys.exit(f"no docs directory at {DOCS}")

    icon = favicon_href()
    tree, by_stem, home = collect()
    _TITLES.clear()
    for stem, (sec, _) in by_stem.items():
        src = (DOCS / sec / f"{stem}.md") if sec else (DOCS / f"{stem}.md")
        if src.is_file():
            _TITLES[stem] = title_of(src)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "docs.css").write_text(PAGE_CSS.replace("/*TOKENS*/", theme_css()),
                                  encoding="utf-8")

    pages = [(home, "", 0)] if home else []
    for key, _, _ in SECTIONS:
        pages += [(md, key, 1) for md in (tree.get(key) or [])]

    written = 0
    for md, section, depth in pages:
        text = resolve_wiki(md.read_text(encoding="utf-8"), by_stem, depth)
        body = markdown.Markdown(
            extensions=["extra", "sane_lists", "toc"], output_format="html5"
        ).convert(text)
        title = title_of(md)
        first_para = next(
            (ln.strip() for ln in md.read_text(encoding="utf-8").splitlines()
             if ln.strip().startswith("**") and ln.strip().endswith("**")), title)
        page = SHELL.format(
            favicon=icon,
            title=html.escape(title),
            desc=html.escape(re.sub(r"[*`\[\]]", "", first_para))[:180],
            up="../" * depth,
            nav=build_nav(tree, home, current=md, depth=depth),
            body=body,
        )
        target = (OUT / section / f"{md.stem}.html") if section else (OUT / "index.html")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page, encoding="utf-8")
        written += 1

    print(f"docs site → {OUT.relative_to(REPO)}  ({written} pages)")
    return 0


PAGE_CSS = """/* GENERATED by site/build_docs.py — do not edit.
   The tokens below are extracted from site/index.html so the docs site and
   the landing page can never drift into two different blues. */
/*TOKENS*/

*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:15.5px;line-height:1.62;-webkit-font-smoothing:antialiased}

.topbar{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--paper) 88%,transparent);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--rule)}
.bar{max-width:1180px;margin:0 auto;padding:.7rem 1.2rem;display:flex;
  align-items:center;justify-content:space-between;gap:1rem}
.brand{font-weight:600;font-size:1.02rem;color:var(--ink);text-decoration:none;letter-spacing:-.01em}
.nav-links{display:flex;align-items:center;gap:1.1rem}
.nav-links a{color:var(--ink-2);text-decoration:none;font-size:.88rem;font-weight:500}
.nav-links a:hover{color:var(--accent)}
.tgl{background:none;border:1px solid var(--rule);color:var(--ink-2);border-radius:6px;
  width:30px;height:26px;cursor:pointer;font-size:.8rem;line-height:1}

.shell{max-width:1180px;margin:0 auto;padding:1.6rem 1.2rem 5rem;
  display:grid;grid-template-columns:236px minmax(0,1fr);gap:2.6rem;align-items:start}
.side{position:sticky;top:4.2rem;max-height:calc(100vh - 5.4rem);overflow-y:auto;padding-bottom:2rem}
.side a{display:block;padding:.28rem .5rem;border-radius:5px;color:var(--ink-2);
  text-decoration:none;font-size:.845rem;line-height:1.38}
.side a:hover{background:var(--paper-2);color:var(--ink)}
.side a.on{background:var(--brand);color:#fff;font-weight:500}
:root[data-theme="dark"] .side a.on{color:#0d1623}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]) .side a.on{color:#0d1623}}
.side .sec{margin:1.15rem 0 .3rem}
.side .sec span{display:block;font-size:.68rem;font-weight:700;letter-spacing:.07em;
  text-transform:uppercase;color:var(--ink-soft)}
.side .sec em{display:block;font-style:normal;font-size:.72rem;color:var(--ink-soft);opacity:.82}

.doc{max-width:44rem;background:var(--card);border:1px solid var(--rule);
  border-radius:var(--r);padding:1.9rem 2.2rem 2.6rem;min-width:0}
.doc h1{font-size:1.9rem;line-height:1.2;letter-spacing:-.02em;margin:0 0 .9rem;text-wrap:balance}
.doc h2{font-size:1.22rem;margin:2.1rem 0 .7rem;letter-spacing:-.01em;
  padding-top:1.1rem;border-top:1px solid var(--rule-2);text-wrap:balance}
.doc h3{font-size:1.02rem;margin:1.5rem 0 .5rem}
.doc h1+blockquote,.doc h1+p strong:first-child{font-size:1.02rem}
.doc p,.doc li{color:var(--ink-2)}
.doc a{color:var(--accent);text-decoration:none;border-bottom:1px solid color-mix(in srgb,var(--accent) 35%,transparent)}
.doc a:hover{border-bottom-color:var(--accent)}
.doc strong{color:var(--ink);font-weight:600}
.doc ul,.doc ol{padding-left:1.25rem}
.doc li{margin:.28rem 0}
.doc code{font-family:var(--mono);font-size:.855em;background:var(--paper-2);
  padding:.12em .38em;border-radius:4px;color:var(--ink)}
.doc pre{background:var(--paper-2);border:1px solid var(--rule);border-radius:var(--r);
  padding:.9rem 1rem;overflow-x:auto;margin:1rem 0}
.doc pre code{background:none;padding:0;font-size:.83rem;line-height:1.55}
.doc blockquote{margin:1.1rem 0;padding:.7rem 1rem;border-left:3px solid var(--accent);
  background:var(--paper-2);border-radius:0 var(--r) var(--r) 0}
.doc blockquote p{margin:.3rem 0}
.doc table{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.9rem;display:block;overflow-x:auto}
.doc th,.doc td{text-align:left;padding:.5rem .7rem;border-bottom:1px solid var(--rule-2);vertical-align:top}
.doc th{font-size:.74rem;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-soft);font-weight:700}
.doc hr{border:0;border-top:1px solid var(--rule);margin:2rem 0}

/* The nav is a <details>: a disclosure on a phone, always-open on a desktop
   where the summary is hidden and it behaves as a plain sidebar. */
.side>summary{display:none}
.side>summary::-webkit-details-marker{display:none}

@media(max-width:900px){
  .shell{grid-template-columns:1fr;gap:.9rem;padding:1rem .9rem 3rem}
  .side{position:sticky;top:3.3rem;z-index:10;max-height:none;padding-bottom:0;
    background:var(--card);border:1px solid var(--rule);border-radius:var(--r)}
  .side>summary{display:flex;align-items:center;justify-content:space-between;
    padding:.62rem .85rem;font-size:.86rem;font-weight:600;color:var(--ink);
    cursor:pointer;list-style:none;user-select:none}
  .side>summary::after{content:"▾";color:var(--ink-soft);font-size:.8rem;
    transition:transform .15s ease}
  .side[open]>summary{border-bottom:1px solid var(--rule-2)}
  .side[open]>summary::after{transform:rotate(180deg)}
  .side>.links{max-height:62vh;overflow-y:auto;padding:.5rem .55rem .8rem}
  .side .sec{margin:.85rem 0 .25rem}
  .side a{padding:.42rem .55rem;font-size:.9rem}
  .doc{padding:1.25rem 1.05rem 2rem;border-radius:var(--r)}
  .doc h1{font-size:1.6rem}
  .doc h2{font-size:1.12rem;margin-top:1.7rem}
  .bar{padding:.6rem .9rem}
  .nav-links{gap:.85rem}
  /* Long unbroken tokens — a URL, a fingerprint — must not widen the page. */
  .doc p,.doc li,.doc td{overflow-wrap:anywhere}
  .doc pre{padding:.75rem .8rem}
  .doc pre code{font-size:.78rem}
}
"""

if __name__ == "__main__":
    raise SystemExit(main())
