"""Audit the served web app the way a picky reviewer would, every time.

    python scripts/ui_audit.py --out /tmp/audit            # build a project, serve it, audit
    python scripts/ui_audit.py --base http://127.0.0.1:8011 --out /tmp/audit
    python scripts/ui_audit.py --out /tmp/a2 --baseline /tmp/audit/findings.json

It crawls every page it can reach (seeded from the API, then following in-app
links), in desktop and phone widths and in light and dark, and records what a
user would trip on: browser errors, failed API calls, sideways scrolling,
accessibility violations (axe-core), clickable things a keyboard can't reach,
missing focus rings, touch targets too small to hit, text cut off with no way
to read it, pages that say "not found", and slow pages. Every page is
screenshotted into a contact sheet (``index.html``) for the judgement calls a
script can't make.

``--baseline`` compares against an earlier run: a finding that is new fails
the run (exit 1), so a fix can never quietly undo another. That ratchet is the
point. When the loop fixes a class of bug, it adds the check here, and the
class stays fixed.

Needs the ``e2e`` extra (Playwright + Chromium) and the built UI
(``cd web/ui && npm run build``). axe-core comes from ``web/ui/node_modules``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AXE = REPO / "web" / "ui" / "node_modules" / "axe-core" / "axe.min.js"
DIST = REPO / "tracebi" / "web" / "ui" / "dist"

VIEWPORTS = {"desktop": (1440, 900), "phone": (390, 844)}
THEMES = ("light", "dark")
SEVERITY_WEIGHT = {"critical": 10, "serious": 4, "moderate": 1, "minor": 0}
SLOW_MS = 2500
MAX_PAGES = 120
PER_PATTERN = 3          # e.g. at most 3 different ?r=<report> pages
PER_MODEL_PAGE = 8       # every model's pages, up to 8 models


# ── serving a project ─────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def serve(project: Path, app: str, workdir: Path):
    """Copy *project*, build its data, serve it. Returns (base_url, process)."""
    dest = workdir / project.name
    shutil.copytree(project, dest, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".DS_Store", "data", "output", ".tracebi"))
    if (dest / "run_workflow.py").is_file():
        subprocess.run([sys.executable, "run_workflow.py"], cwd=dest, check=True,
                       capture_output=True, text=True)
    # Audit what a reader sees: every report built once, so receipts, verdicts
    # and "latest builds" render, not only the never-built empty states.
    reports = dest / "reports"
    names = [p.parent.relative_to(reports).as_posix() for p in reports.rglob("report.json")]
    names += [p.relative_to(reports).with_suffix("").as_posix() for p in reports.rglob("*.json")
              if p.name != "report.json" and not (p.with_suffix("") / "report.json").exists()]
    for name in sorted(set(names)):
        subprocess.run([sys.executable, "-c", "import sys; from tracebi.cli import main; sys.exit(main())",
                        "report", "build", name],
                       cwd=dest, capture_output=True, text=True)
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRACEBI_")}
    env["TRACEBI_APP"] = app
    env["TRACEBI_UPDATE_CHECK"] = "0"
    log = (workdir / "server.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tracebi.web.run", "--port", str(port), "--no-reload"],
        cwd=dest, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            sys.exit(f"server exited:\n{(workdir / 'server.log').read_text()[-3000:]}")
        try:
            _get(base + "/api/health", 1)
            return base, proc
        except Exception:
            time.sleep(0.3)
    proc.kill()
    sys.exit("server never became healthy")


# ── what to visit ─────────────────────────────────────────────────────────────

def seeds(base: str) -> list[str]:
    """Start pages: the app's own pages, plus one per model / tab / report.

    The crawl follows links from here, so a page reachable by a link is found
    without being listed. Pages reached only by a button (tabs) need a seed.
    """
    pages = ["reports", "model", "transform", "explore", "runs"]
    paths = ["/", "/models", "/explore", "/refresh", "/reports", "/sources", "/runs",
             "/verify", "/getting-started", "/handbook", "/workflow"]
    models = _safe(base, "/api/models")
    for m in models:
        name = urllib.parse.quote(m["name"])
        paths += [f"/m/{name}"] + [f"/m/{name}/{p}" for p in pages]
    for r in _safe(base, "/api/reports")[:PER_PATTERN]:
        paths.append("/reports?" + urllib.parse.urlencode({"r": r["name"]}))
    # Old addresses must still land somewhere real.
    if models:
        name = urllib.parse.quote(models[0]["name"])
        paths += [f"/models/{name}?tab=explore", f"/m/{name}/refresh", f"/m/{name}/sources",
                  "/connectors", "/pipelines"]
    paths.append("/no-such-page")
    return list(dict.fromkeys(paths))


def _safe(base: str, path: str) -> list:
    try:
        return _get(base + path) or []
    except Exception:
        return []


def _pattern(path: str) -> str:
    """Pages that differ only by which item is open share a pattern."""
    u = urllib.parse.urlsplit(path)
    q = urllib.parse.parse_qs(u.query)
    # A tab is a different page; which report is open is not.
    keys = sorted(f"{k}={q[k][0]}" if k == "tab" else k for k in q)
    p = re.sub(r"/models/[^/?]+", "/models/<m>", u.path)
    p = re.sub(r"/m/[^/?]+", "/m/<m>", p)
    return p + ("?" + "&".join(keys) if keys else "")


# ── checks run inside the page ────────────────────────────────────────────────

PAGE_CHECKS_JS = r"""
() => {
  const out = [];
  const vw = window.innerWidth;
  const label = el => {
    const t = (el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim().replace(/\s+/g, ' ');
    const tag = el.tagName.toLowerCase();
    const cls = (el.className && typeof el.className === 'string') ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.') : '';
    return `${tag}${cls}${t ? ' "' + t.slice(0, 40) + '"' : ''}`;
  };
  const visible = el => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const inNav = el => !!el.closest('nav.app-nav');
  const navOpen = !!document.querySelector('nav.app-nav.nav-open');

  // Sideways scroll: the page is wider than the screen.
  const sw = document.documentElement.scrollWidth;
  if (sw > vw + 1) {
    let worst = null, right = vw;
    for (const el of document.querySelectorAll('main *')) {
      const r = el.getBoundingClientRect();
      if (r.right > right + 1 && visible(el)) { right = r.right; worst = el; }
    }
    out.push({rule: 'h-overflow', severity: 'serious',
              target: worst ? label(worst) : 'document',
              detail: `page is ${sw}px wide on a ${vw}px screen`});
  }

  // The face is Hanken Grotesk, and it actually loaded (a failed font falls back silently).
  if (!document.fonts.check('14px "Hanken Grotesk Variable"'))
    out.push({rule: 'theme-font-loaded', severity: 'serious', target: 'body',
              detail: 'Hanken Grotesk did not load; the page is in the fallback face'});

  // One h1 per page: it is the page's name for screen readers and tabs.
  const h1s = [...document.querySelectorAll('main h1')].filter(visible);
  if (h1s.length !== 1)
    out.push({rule: 'one-h1', severity: 'moderate', target: 'main',
              detail: `${h1s.length} visible h1 elements`});

  // Clickable but not focusable: a div with an onClick a keyboard can't reach.
  for (const el of document.querySelectorAll('main *')) {
    if (!visible(el)) continue;
    const s = getComputedStyle(el);
    if (s.cursor !== 'pointer') continue;
    const tag = el.tagName.toLowerCase();
    if (['a', 'button', 'input', 'select', 'textarea', 'label', 'summary', 'option'].includes(tag)) continue;
    if (el.closest('a,button,label,summary,[role=button],[role=tab],[role=link],[tabindex]')) continue;
    if (el.closest('.react-flow')) continue;
    if (el.parentElement && getComputedStyle(el.parentElement).cursor === 'pointer') continue;
    out.push({rule: 'click-not-keyboard', severity: 'serious', target: label(el),
              detail: 'looks clickable (cursor:pointer) but is not a link/button and has no tabindex'});
  }

  // Text cut off with an ellipsis and no title to read the rest.
  for (const el of document.querySelectorAll('main *')) {
    if (!visible(el) || el.children.length) continue;
    const s = getComputedStyle(el);
    const clamped = s.webkitLineClamp && s.webkitLineClamp !== 'none';
    if (s.textOverflow !== 'ellipsis' && !clamped) continue;
    if (s.overflow === 'visible' && s.overflowX === 'visible') continue;
    if (el.scrollWidth <= el.clientWidth + 1 && el.scrollHeight <= el.clientHeight + 1) continue;
    if (el.title || el.closest('[title]')) continue;
    out.push({rule: 'truncated-no-title', severity: 'moderate', target: label(el),
              detail: 'text is cut off and there is no tooltip to read it'});
  }

  // Phone: things you tap must be at least 24x24 (WCAG 2.2 target size).
  if (vw < 600) {
    for (const el of document.querySelectorAll('a,button,input,select,[role=button],[role=tab]')) {
      if (!visible(el) || (inNav(el) && !navOpen)) continue;
      // A control inside a label is tapped through the label: measure that.
      const r = (el.tagName === 'INPUT' && el.closest('label') || el).getBoundingClientRect();
      if (r.width >= 24 && r.height >= 24) continue;
      if (el.tagName === 'A' && getComputedStyle(el).display === 'inline') continue; // a link in a sentence
      out.push({rule: 'small-target', severity: 'moderate', target: label(el),
                detail: `${Math.round(r.width)}x${Math.round(r.height)}px tap target`});
    }
  }

  // Desktop: a flat row of three or more action buttons (Btn, or a download link)
  // that wraps onto a second line has more actions than room. Group them: a menu,
  // a quieter secondary set. (A picker of chips may wrap; it is not a row of actions.)
  if (vw >= 1024) {
    for (const row of document.querySelectorAll('main *')) {
      if (!visible(row)) continue;
      const s = getComputedStyle(row);
      if (s.display !== 'flex' || s.flexWrap !== 'wrap') continue;
      const acts = [...row.children].filter(k => visible(k) && k.matches('button[class^="btn-"],a.dl-link'));
      const lines = new Set(acts.map(k => Math.round(k.getBoundingClientRect().top)));
      if (acts.length >= 3 && lines.size > 1)
        out.push({rule: 'action-row-wraps', severity: 'moderate', target: label(row),
                  detail: `${acts.length} actions wrap onto ${lines.size} lines at ${vw}px`});
    }
  }

  // "Where am I": exactly one sidebar entry marks the page you are on.
  const current = document.querySelectorAll('nav.app-nav [aria-current="page"]').length;
  if (location.pathname !== '/no-such-page' && current !== 1)
    out.push({rule: 'nav-you-are-here', severity: 'serious', target: 'nav.app-nav',
              detail: `${current} sidebar entries are marked as the current page`});

  // A list beside an empty "select something" pane: half the page is blank.
  for (const el of document.querySelectorAll('main .split-detail')) {
    const t = (el.innerText || '').trim();
    if (visible(el) && t.length < 200 && /(^|\n)\s*(select|pick|choose) (a|an|one)\b/i.test(t))
      out.push({rule: 'empty-detail-pane', severity: 'moderate', target: 'main .split-detail',
                detail: `the page opens on a blank pane: "${t.replace(/\s+/g, ' ').slice(0, 60)}"`});
  }

  // Dead ends and errors a reader would see.
  const body = (document.querySelector('main') || document.body).innerText || '';
  if (/Page not found|No (model|report) named/.test(body))
    out.push({rule: 'not-found', severity: 'serious', target: 'main',
              detail: body.trim().slice(0, 120)});
  if (/Traceback \(most recent call last\)|undefined|NaN|\[object Object\]/.test(body))
    out.push({rule: 'raw-garbage', severity: 'serious', target: 'main',
              detail: (body.match(/.{0,40}(Traceback|undefined|NaN|\[object Object\]).{0,40}/) || [''])[0]});
  if (!body.trim())
    out.push({rule: 'blank-page', severity: 'critical', target: 'main', detail: 'nothing rendered'});
  return out;
}
"""

FOCUS_JS = r"""
() => {
  let el = document.activeElement;
  // Focus that is inside a report's frame belongs to the element inside it: judge
  // that one, since the frame itself draws no ring.
  while (el && el.tagName === 'IFRAME') {
    const inner = el.contentDocument && el.contentDocument.activeElement;
    if (!inner || inner === el.contentDocument.body) break;
    el = inner;
  }
  if (!el || el === document.body) return null;
  const s = getComputedStyle(el);
  const ring = (s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0) ||
               (s.boxShadow && s.boxShadow !== 'none');
  const t = (el.innerText || el.getAttribute('aria-label') || el.tagName).trim().replace(/\s+/g, ' ').slice(0, 40);
  return {ring, target: el.tagName.toLowerCase() + ' "' + t + '"'};
}
"""


def _links(page, base: str) -> list[str]:
    hrefs = page.eval_on_selector_all(
        "a[href]", "els => els.map(e => e.getAttribute('href'))")
    out = []
    for h in hrefs:
        if not h or h.startswith(("#", "mailto:", "javascript:")):
            continue
        u = urllib.parse.urlsplit(urllib.parse.urljoin(base + "/", h))
        if f"{u.scheme}://{u.netloc}" != base:
            continue
        if u.path.startswith(("/api/", "/r/", "/assets/", "/docs/")):
            continue
        out.append(u.path + (f"?{u.query}" if u.query else ""))
    return out


def _settle(page) -> int:
    """Wait for the page to finish loading. Returns how long that took (ms)."""
    t0 = time.monotonic()
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
        page.wait_for_function(
            "() => !document.querySelector('.skeleton') && !!document.querySelector('main')",
            timeout=15_000)
    except Exception:
        pass
    page.wait_for_timeout(250)
    return int((time.monotonic() - t0) * 1000)


# ── the crawl ─────────────────────────────────────────────────────────────────

def audit(base: str, out: Path, deep: bool = True) -> dict:
    from playwright.sync_api import sync_playwright

    shots = out / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    axe_src = AXE.read_text() if AXE.is_file() else None
    findings: list[dict] = []
    pages: list[dict] = []

    queue = seeds(base)
    at_start = {r["name"] for r in _safe(base, "/api/reports")}
    seen: set[str] = set()
    per_pattern: dict[str, int] = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        while queue and len(seen) < MAX_PAGES:
            path = queue.pop(0)
            if path in seen:
                continue
            pat = _pattern(path)
            cap = PER_MODEL_PAGE if pat.startswith("/m/<m>") and "?" not in pat else PER_PATTERN
            if per_pattern.get(pat, 0) >= cap:
                continue
            per_pattern[pat] = per_pattern.get(pat, 0) + 1
            seen.add(path)
            record = {"path": path, "pattern": pat, "shots": {}, "ms": {}}
            combos = [(vp, theme) for vp in VIEWPORTS for theme in THEMES
                      if deep or not (vp == "phone" and theme == "dark")]
            for vp, theme in combos:
                    w, h = VIEWPORTS[vp]
                    ctx = browser.new_context(viewport={"width": w, "height": h},
                                              is_mobile=(vp == "phone"), has_touch=(vp == "phone"))
                    ctx.add_init_script(
                        f"try{{localStorage.setItem('tracebi-theme','{theme}')}}catch(e){{}}")
                    page = ctx.new_page()
                    errors: list[str] = []
                    page.on("pageerror", lambda e, errors=errors: errors.append(f"pageerror: {e}"))
                    page.on("console", lambda m, errors=errors: errors.append(f"console: {m.text}")
                            if m.type == "error" else None)
                    page.on("response", lambda r, errors=errors: errors.append(
                        f"HTTP {r.status} {urllib.parse.urlsplit(r.url).path}")
                        if r.status >= 400 and "/api/" in r.url else None)
                    page.goto(base + path, wait_until="domcontentloaded")
                    ms = _settle(page)
                    # An old address that redirects is checked once, where it lands.
                    u = urllib.parse.urlsplit(page.url)
                    final = u.path + (f"?{u.query}" if u.query else "")
                    if final != path:
                        record["redirect"] = final
                        if final not in seen and final not in queue:
                            queue.insert(0, final)
                        ctx.close()
                        break
                    record["ms"][f"{vp}-{theme}"] = ms
                    where = {"page": path, "pattern": pat, "viewport": vp, "theme": theme}

                    expected_404 = path == "/no-such-page"
                    for e in errors:
                        findings.append({**where, "rule": "browser-error", "severity": "critical",
                                         "target": re.sub(r"\d{3,}", "#", e)[:120], "detail": e[:400]})
                    for f in page.evaluate(PAGE_CHECKS_JS):
                        if expected_404 and f["rule"] == "not-found":
                            continue
                        findings.append({**where, **f})

                    if axe_src and (vp == "desktop" or theme == "light"):
                        page.add_script_tag(content=axe_src)
                        res = page.evaluate(
                            "async () => (await axe.run(document, {runOnly: ['wcag2a','wcag2aa','wcag21aa','best-practice'],"
                            " resultTypes: ['violations']})).violations"
                            ".map(v => ({id: v.id, impact: v.impact, help: v.help,"
                            " nodes: v.nodes.slice(0, 3).map(n => {"
                            "   const el = document.querySelector(n.target[0]);"
                            "   const shell = el && el.closest('nav.app-nav, .mobile-header') ? 'nav.app-nav ' : '';"
                            "   return shell + n.target.join(' ') + ' :: ' + (n.failureSummary||'').split('\\n')[1] })}))")
                        for v in res:
                            sev = v["impact"] or "moderate"
                            findings.append({**where, "rule": f"axe:{v['id']}", "severity": sev,
                                             "target": v["nodes"][0].split(" :: ")[0][:120] if v["nodes"] else "",
                                             "detail": v["help"] + " — " + " | ".join(v["nodes"])[:400]})

                    if vp == "desktop" and theme == "light":
                        # Keyboard: the first stops in the page must show where focus is.
                        page.mouse.click(w - 5, h - 5)
                        for _ in range(12):
                            page.keyboard.press("Tab")
                            f = page.evaluate(FOCUS_JS)
                            if f and not f["ring"]:
                                findings.append({**where, "rule": "no-focus-ring", "severity": "serious",
                                                 "target": f["target"], "detail": "keyboard focus is invisible"})
                        for link in _links(page, base):
                            if link not in seen and link not in queue:
                                queue.append(link)

                    shot = shots / f"{_slug(path)}__{vp}-{theme}.png"
                    page.screenshot(path=str(shot), full_page=True)
                    record["shots"][f"{vp}-{theme}"] = str(shot.relative_to(out))
                    ctx.close()
            # One slow rendering out of four is the machine; the median is the page.
            times = sorted(record["ms"].values())
            if times and times[len(times) // 2] > SLOW_MS:
                findings.append({"page": path, "pattern": pat, "viewport": "-", "theme": "-",
                                 "rule": "slow-page", "severity": "moderate", "target": "page",
                                 "detail": f"settles in a median {times[len(times) // 2]}ms"})
            pages.append(record)

        findings += scope_findings(browser, base)
        browser.close()

    # Nothing on disk changed during the crawl, so nothing should come or go.
    # (Live discovery once forgot an app module's own reports 5s after start.)
    at_end = {r["name"] for r in _safe(base, "/api/reports")}
    for name in sorted(at_start ^ at_end):
        findings.append({"page": "/reports", "pattern": "/reports", "viewport": "-", "theme": "-",
                         "rule": "registry-drift", "severity": "critical", "target": name,
                         "detail": f"{name} {'vanished' if name in at_start else 'appeared'} during the audit"})

    findings = _dedupe(findings + token_findings() + report_style_findings()
                       + theme_findings() + source_findings())
    report = {"base": base, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
              "pages": pages, "findings": findings, "metrics": static_metrics(),
              "score": score(findings)}
    (out / "findings.json").write_text(json.dumps(report, indent=1))
    (out / "index.html").write_text(contact_sheet(report))
    return report


def scope_findings(browser, base: str) -> list[dict]:
    """Picking a model must show everything that belongs to it, and only that.

    The API says which models each report reads and which models each source
    feeds; a model's Reports and Sources pages must list exactly those. This
    caught reports filed in a folder named unlike their model vanishing from
    it.
    """
    out = []
    reports = _safe(base, "/api/reports")
    models = _safe(base, "/api/models")
    listed = {m["name"] for m in models}
    for r in reports:
        stray = [m for m in r.get("models") or [] if m not in listed]
        if stray:
            out.append({"page": "/reports", "pattern": "/reports", "viewport": "desktop",
                        "theme": "light", "rule": "report-model-unlisted", "severity": "critical",
                        "target": r["name"],
                        "detail": f"reads {stray}, which the model switcher does not list"})
    # A model nobody can read a report from is a dead end in the app.
    for m in models:
        if not any(m["name"] in (r.get("models") or []) for r in reports):
            out.append({"page": f"/m/{m['name']}/reports", "pattern": "/m/<m>/reports",
                        "viewport": "desktop", "theme": "light", "rule": "model-without-reports",
                        "severity": "serious", "target": m["name"],
                        "detail": f"{m['name']} has no report; its Reports page is empty"})
    # ...and one nothing can refresh. A pipeline belongs to the models it names
    # (`models`), else the one it stamped, else its own name (as the app reads it).
    pipelines = _safe(base, "/api/pipelines")

    def pipeline_models(p):
        return p.get("models") or ([p["model"]] if p.get("model") else [p["pipeline"]])

    for m in models:
        if not any(m["name"] in pipeline_models(p) for p in pipelines):
            out.append({"page": f"/m/{m['name']}/transform", "pattern": "/m/<m>/transform",
                        "viewport": "desktop", "theme": "light", "rule": "model-without-pipeline",
                        "severity": "serious", "target": m["name"],
                        "detail": f"nothing refreshes {m['name']}; its Transform stage is empty"})
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    for m in models:
        name = m["name"]
        expect = {
            "reports": {r["name"].rsplit("/", 1)[-1] for r in reports if name in (r.get("models") or [])},
        }
        for key, want in expect.items():
            page.goto(f"{base}/m/{urllib.parse.quote(name)}/{key}")
            _settle(page)
            try:   # the list can land after the page settles; give it a moment
                page.wait_for_function(
                    "n => document.querySelectorAll('main .list-item__name').length >= n",
                    arg=len(want), timeout=5000)
            except Exception:
                pass
            shown = set(page.eval_on_selector_all(
                "main .list-item__name", "els => els.map(e => e.innerText.trim())"))
            if shown != want:
                out.append({"page": f"/m/{name}/{key}", "pattern": f"/m/<m>/{key}", "viewport": "desktop",
                            "theme": "light", "rule": "scope-mismatch", "severity": "critical",
                            "target": f"{key} of {name}",
                            "detail": f"shows {sorted(shown)}; belongs: {sorted(want)}"})
    page.close()
    return out


def _slug(path: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_") or "root"
    return s[:60] + "_" + hashlib.sha1(path.encode()).hexdigest()[:6]


SHELL = (".mobile-header", ".app-nav", "nav.app-nav", "nav")


def _key(f: dict) -> str:
    """What makes two findings the same finding (across widths and themes).

    The sidebar and phone header are on every page; a bug there is one bug,
    not one per page.
    """
    where = "(app shell)" if f["target"].startswith(SHELL) else f["pattern"]
    # The element, not its words: ten report rows with the same bug are one bug.
    what = re.sub(r'"[^"]*"', '', f["target"]).strip()
    return f"{f['rule']}|{where}|{re.sub(r'[0-9]+', '#', what)}"


def _dedupe(findings: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for f in findings:
        k = _key(f)
        if k not in merged:
            merged[k] = {**f, "key": k, "seen_in": []}
        merged[k]["seen_in"].append(f"{f['viewport']}-{f['theme']}")
    return sorted(merged.values(), key=lambda f: (-SEVERITY_WEIGHT.get(f["severity"], 0), f["rule"]))


def score(findings: list[dict]) -> int:
    """100 is clean. Halves at 50 weighted points, so progress always shows."""
    weight = sum(SEVERITY_WEIGHT.get(f["severity"], 0) for f in findings)
    return round(100 * 50 / (50 + weight))


# ── things measured from the source, not the page ─────────────────────────────

TEXT_TOKENS = ("--text", "--text-2", "--muted", "--accent-text",
               "--green-text", "--amber-text", "--red-text",
               "--role-dimension", "--role-fact", "--role-bridge", "--role-table",
               "--phase-transform", "--phase-model", "--phase-report")
SURFACE_TOKENS = ("--bg", "--surface", "--surface-2", "--card", "--card-hl")


def _tokens(css: str, selector: str) -> dict:
    m = re.search(re.escape(selector) + r"\s*\{(.*?)\n\}", css, re.S)
    return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6}\b|rgba\([^)]*\))", m.group(1))) if m else {}


def _contrast(a: str, b: str) -> float:
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def token_findings() -> list[dict]:
    """Every text colour on every surface, in both themes, must read at 4.5:1.

    axe-core skips text whose background it cannot resolve (anything under a
    blur), which is how dark-mode grey on dark cards went unseen. The tokens
    are the source of truth, so check them directly.
    """
    css = (REPO / "web" / "ui" / "src" / "styles" / "global.css").read_text()
    light = _tokens(css, ":root")
    themes = {"light": light, "dark": {**light, **_tokens(css, '[data-theme="dark"]')}}
    out = []
    for theme, t in themes.items():
        # Each diagram tag colour is drawn on its own background.
        pairs = [(fg, bg) for fg in TEXT_TOKENS for bg in SURFACE_TOKENS]
        pairs += [(k, k[:-3] + "-bg") for k in t if k.startswith("--op-") and k.endswith("-tx")]
        for fg, bg in pairs:
            if fg in t and bg in t and (r := _contrast(t[fg], t[bg])) < 4.5:
                out.append({"page": "(tokens)", "pattern": "(tokens)", "viewport": "-",
                            "theme": theme, "rule": "token-contrast", "severity": "serious",
                            "target": f"{fg} on {bg}",
                            "detail": f"{t[fg]} on {t[bg]} is {r:.2f}:1 in {theme} (needs 4.5)"})
    # Text on its own wash: a badge or link-button is --x-text on --x-lt laid over a
    # surface. The wash is translucent, so blend it first.
    def blend(rgba: str, base: str) -> str:
        r, g, b, a = (float(x) for x in re.findall(r"[\d.]+", rgba)[:4])
        under = [int(base[k:k + 2], 16) for k in (1, 3, 5)]
        return "#" + "".join(f"{round(c * a + u * (1 - a)):02x}" for c, u in zip((r, g, b), under))

    for theme, t in themes.items():
        for name, wash in (("--accent-text", "--blue-lt"), ("--green-text", "--green-lt"),
                           ("--amber-text", "--amber-lt"), ("--red-text", "--red-lt")):
            for ground in ("--card", "--bg", "--surface-2"):
                if all(k in t for k in (name, wash, ground)) and t[wash].startswith("rgba") and t[ground].startswith("#"):
                    once = blend(t[wash], t[ground])
                    twice = blend(t[wash], once)         # a tinted button inside a tinted box
                    for layers, under in (("", once), (" twice", twice)):
                        if (r := _contrast(t[name], under)) < 4.5:
                            out.append({"page": "(tokens)", "pattern": "(tokens)", "viewport": "-",
                                        "theme": theme, "rule": "token-contrast", "severity": "serious",
                                        "target": f"{name} on {wash}{layers} over {ground}",
                                        "detail": f"{t[name]} on {under} is {r:.2f}:1 in {theme} (needs 4.5)"})
    return out


def report_style_findings() -> list[dict]:
    """The report stylesheet (tracebi.css), checked at the source: axe cannot see
    inside the iframes a report is shown in, and a report is the product.

    Every text token must read at 4.5:1 on every surface a report puts text on,
    and the stylesheet must name Hanken Grotesk first.
    """
    css = (REPO / "tracebi" / "reports" / "assets" / "tracebi.css").read_text()
    t = _tokens(css, ":root")
    out = []

    def finding(target, detail):
        out.append({"page": "(report stylesheet)", "pattern": "(report stylesheet)", "viewport": "-",
                    "theme": "-", "rule": "report-style", "severity": "serious",
                    "target": target, "detail": detail})

    if not re.search(r'--tb-font:\s*"Hanken Grotesk"', css):
        finding("--tb-font", "the report face must lead with Hanken Grotesk (inlined by stack.font_face_css)")
    for fg in ("--tb-ink", "--tb-muted", "--tb-accent-text", "--tb-good", "--tb-bad"):
        for bg in ("--tb-bg", "--tb-page", "--tb-surface", "--tb-surface-accent"):
            if fg in t and bg in t and (r := _contrast(t[fg], t[bg])) < 4.5:
                finding(f"{fg} on {bg}", f"{t[fg]} on {t[bg]} is {r:.2f}:1 (needs 4.5)")
    return out


def theme_findings() -> list[dict]:
    """The look, as rules ([[design-direction]]): Hanken Grotesk, a cobalt brand that stays in
    its blue family, no gradients or blur, shadows only on popovers. Other colour
    means something: the status tokens, the diagram roles.
    """
    out = []
    ui = REPO / "web" / "ui" / "src"
    css = (ui / "styles" / "global.css").read_text()

    def finding(rule, target, detail, sev="serious"):
        out.append({"page": "(theme)", "pattern": "(theme)", "viewport": "-", "theme": "-",
                    "rule": rule, "severity": sev, "target": target, "detail": detail})

    root = re.search(r":root\s*\{(.*?)\n\}", css, re.S)
    if not root or "Hanken Grotesk" not in re.search(r"--font-sans:\s*([^;]*);", root.group(1)).group(1):
        finding("theme-font", "--font-sans", "the sans face must be Hanken Grotesk")
    if "hanken-grotesk" not in (ui / "main.jsx").read_text().lower():
        finding("theme-font", "main.jsx", "Hanken Grotesk is not imported")

    light = _tokens(css, ":root")
    dark_own = _tokens(css, '[data-theme="dark"]')
    themes = {"light": light, "dark": {**light, **dark_own}}
    # An ink token the dark theme does not set is the light theme's near-black on
    # a near-black ground: --blue was, and every active tab and row vanished.
    for name in ("--blue", "--accent-text", "--ink", "--on-ink"):
        if name in light and name not in dark_own:
            finding("theme-dark-ink", name, f"{name} is not set for dark mode")
    # The brand is cobalt and its blues. Any saturated brand token outside that family
    # (a purple, a green) is decoration spending a colour that means something.
    import colorsys
    for theme, t in themes.items():
        for name in ("--blue", "--blue-md", "--accent-text", "--accent-text-2", "--ink", "--brand", "--brand-text"):
            v = t.get(name)
            if not v:
                continue
            r, g, b = (int(v[i:i + 2], 16) / 255 for i in (1, 3, 5))
            h, _, sat = colorsys.rgb_to_hls(r, g, b)
            if sat > 0.2 and not (195 <= h * 360 <= 250):
                finding("theme-brand-colour", f"{name} ({theme})",
                        f"{v} is hue {h * 360:.0f}; the brand is cobalt and its blues (195-250)")
    for m in re.finditer(r"\n\s*(--shadow[\w-]*):\s*([^;]*);", css):
        if m.group(1) != "--shadow-pop" and m.group(2).strip() != "none":
            finding("theme-shadow", m.group(1), f"{m.group(1)} is {m.group(2).strip()[:40]}; resting shadows are none")

    # Gradients and blur: decoration. A loading shimmer is not.
    lines = css.splitlines()
    for n, line in enumerate(lines):
        if re.search(r"(linear|radial)-gradient|backdrop-filter", line):
            context = " ".join(lines[max(0, n - 4):n + 2])
            if not re.search(r"skeleton|shimmer", context):
                finding("theme-decoration", f"global.css:{n + 1}", line.strip()[:80])
    for p in sorted(ui.rglob("*.jsx")):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"(linear|radial)-gradient|backdropFilter", line):
                finding("theme-decoration", f"{p.relative_to(ui).as_posix()}:{n}", line.strip()[:80])
    return out


def source_findings() -> list[dict]:
    """Bugs visible in the source before anything renders.

    `${color}22` glues a hex alpha onto a colour. It works only while the
    colour is a literal hex: the day it becomes a theme token (var(--x)) the
    result is invalid CSS and the border or wash silently disappears. Use
    color-mix(in srgb, <colour> 13%, transparent), which takes either.
    """
    out = []
    src = REPO / "web" / "ui" / "src"
    for p in sorted(src.rglob("*.jsx")):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            for m in re.finditer(r"\$\{[\w.]+\}[0-9a-fA-F]{2}\b", line):
                rel = p.relative_to(src).as_posix()
                out.append({"page": f"(source) {rel}", "pattern": f"(source) {rel}", "viewport": "-",
                            "theme": "-", "rule": "hex-alpha-concat", "severity": "moderate",
                            "target": f"{rel}:{n}", "detail": m.group(0)})
    return out


def static_metrics() -> dict:
    src = REPO / "web" / "ui" / "src"
    hex_colors = 0
    inline_styles = 0
    for p in src.rglob("*.jsx"):
        text = p.read_text()
        hex_colors += len(re.findall(r"['\"]#[0-9a-fA-F]{3,8}['\"]|rgba?\(\d", text))
        inline_styles += text.count("style={{")
    js = sum(f.stat().st_size for f in (DIST / "assets").glob("*.js")) if DIST.is_dir() else 0
    css = sum(f.stat().st_size for f in (DIST / "assets").glob("*.css")) if DIST.is_dir() else 0
    return {"hardcoded_colors_in_jsx": hex_colors, "inline_style_blocks": inline_styles,
            "bundle_js_kb": round(js / 1024), "bundle_css_kb": round(css / 1024)}


# ── the contact sheet ─────────────────────────────────────────────────────────

def contact_sheet(report: dict) -> str:
    esc = html.escape
    by_page: dict[str, list] = {}
    for f in report["findings"]:
        by_page.setdefault(f["page"], []).append(f)
    rows = []
    for p in report["pages"]:
        fs = by_page.get(p["path"], [])
        items = "".join(
            f"<li class='{esc(f['severity'])}'><b>{esc(f['rule'])}</b> {esc(f['target'])} "
            f"<span>{esc(f['detail'][:200])}</span> <i>{esc(', '.join(f['seen_in']))}</i></li>" for f in fs)
        imgs = "".join(
            f"<figure><a href='{esc(s)}'><img loading=lazy src='{esc(s)}'></a><figcaption>{esc(k)} · {p['ms'].get(k, '')}ms</figcaption></figure>"
            for k, s in p["shots"].items())
        if p.get("redirect"):
            rows.append(f"<section><h2>{esc(p['path'])} → {esc(p['redirect'])}</h2></section>")
            continue
        rows.append(f"<section><h2>{esc(p['path'])}</h2><ul>{items or '<li class=ok>clean</li>'}</ul>"
                    f"<div class=imgs>{imgs}</div></section>")
    counts = {}
    for f in report["findings"]:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    summary = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -SEVERITY_WEIGHT[kv[0]]))
    return f"""<!doctype html><meta charset=utf-8><title>UI audit {esc(report['when'])}</title>
<style>
:root{{color-scheme:light dark;--bg:#f6f7f9;--fg:#111;--mut:#667;--card:#fff;--bd:#dde}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--card:#161b22;--bd:#30363d}}}}
body{{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;background:var(--bg);color:var(--fg)}}
h1{{margin:0 0 4px}} .sum{{color:var(--mut);margin-bottom:24px}}
section{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;margin-bottom:18px}}
h2{{font-size:15px;margin:0 0 8px;font-family:ui-monospace,monospace}}
ul{{margin:0 0 12px;padding-left:18px}} li span{{color:var(--mut)}} li i{{color:var(--mut);font-size:12px}}
li.critical b{{color:#e5484d}} li.serious b{{color:#f76b15}} li.moderate b{{color:#c9a400}} li.ok{{color:#30a46c}}
.imgs{{display:flex;gap:10px;overflow-x:auto}} figure{{margin:0;flex:none}}
img{{height:260px;border:1px solid var(--bd);border-radius:6px;display:block}} figcaption{{font-size:11px;color:var(--mut)}}
</style>
<h1>UI audit — score {report['score']}</h1>
<div class=sum>{esc(report['when'])} · {len(report['pages'])} pages · {summary or 'no findings'} · {esc(json.dumps(report['metrics']))}</div>
{''.join(rows)}"""


# ── ratchet ───────────────────────────────────────────────────────────────────

def compare(report: dict, baseline_path: Path) -> list[dict]:
    old = json.loads(baseline_path.read_text())
    before = {f["key"] for f in old["findings"]}
    after = {f["key"] for f in report["findings"]}
    fixed = before - after
    new = [f for f in report["findings"] if f["key"] not in before]
    print(f"\nvs baseline: score {old['score']} → {report['score']} · fixed {len(fixed)} · new {len(new)}")
    # Counts that only go down: a new hard-coded colour is a dark-mode bug
    # waiting, and the bundle is what every visitor downloads.
    for key in ("hardcoded_colors_in_jsx", "bundle_js_kb"):
        before, after = old.get("metrics", {}).get(key), report["metrics"].get(key)
        if before is not None and after is not None and after > before * 1.02:
            new.append({"severity": "serious", "rule": f"metric:{key}", "page": "(metrics)",
                        "target": f"{before} → {after}"})
    for f in new:
        print(f"  NEW {f['severity']:9} {f['rule']:24} {f['page']}  {f['target']}")
    return new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", help="audit an already-running server")
    ap.add_argument("--project", default=str(REPO / "examples" / "portfolio_project"))
    ap.add_argument("--app", default="", help="TRACEBI_APP for the served project")
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "tracebi-ui-audit"))
    ap.add_argument("--baseline", help="an earlier findings.json; new findings fail the run")
    ap.add_argument("--quick", action="store_true", help="skip phone/dark combinations")
    args = ap.parse_args(argv)

    if not args.base and not (DIST / "index.html").is_file():
        sys.exit("The UI is not built: cd web/ui && npm run build")
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    proc = None
    base = args.base
    if not base:
        base, proc = serve(Path(args.project), args.app, Path(tempfile.mkdtemp(prefix="ui-audit-")))
    try:
        report = audit(base.rstrip("/"), out, deep=not args.quick)
    finally:
        if proc:
            proc.terminate()

    print(f"score {report['score']} · {len(report['pages'])} pages · {len(report['findings'])} findings · {report['metrics']}")
    for f in report["findings"]:
        print(f"  {f['severity']:9} {f['rule']:24} {f['page'][:40]:40} {f['target'][:60]}")
    print(f"contact sheet: {out / 'index.html'}")
    if args.baseline:
        return 1 if compare(report, Path(args.baseline)) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
