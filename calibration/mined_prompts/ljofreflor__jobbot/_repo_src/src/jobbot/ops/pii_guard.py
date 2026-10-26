"""Block real PII from entering git (pre-commit guard). Local only, no network."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Paths that must never be tracked, even with `git add -f`.
_BLOCKED_PATHS: tuple[tuple[str, str], ...] = (
    (r"^data/profile\.yaml$", "real candidate source of truth"),
    (r"^data/profile\.generated\.yaml$", "derived from private LaTeX CV"),
    (r"^data/profile\.suggested\.yaml$", "derived from market feedback"),
    (r"^data/portals\.yaml$", "learned portals may include real applications"),
    (
        r"^data/companies\.yaml$",
        "learned company portals reflect your own search (share via companies export)",
    ),
    (
        r"^data/form_knowledge\.yaml$",
        "observed application forms (local knowledge, not shared)",
    ),
    (
        r"^data/recruiters\.yaml$",
        "hiring-practice sources you read (share via recruiters export)",
    ),
    (
        r"^data/application_answers\.yaml$",
        "your answers to application questions (language level, contract, pay)",
    ),
    (
        r"^data/(profile|portals|companies|recruiters|form_knowledge|application_answers)"
        r"\.yaml\..+$",
        "backup or variant of a local data file (same data as the original)",
    ),
    (r"^data/.*\.(bak|orig)([.\-]|$)", "backup of a local data file"),
    (r"^latex/cv\.tex$", "private LaTeX CV (only cv.tex.demo is tracked)"),
    (r"^\.jobbot\.toml$", "local config with private paths"),
    (r"^sandboxes/", "another candidate's workspace (test CVs are their PII)"),
    (r"(^|/)consent\.yaml$", "client consent and retention record"),
    (r"^browser-data/", "browser session data"),
    (
        r"(^|/)\.vault\.ya?ml$",
        "local portal passwords (opt-in vault; never commit)",
    ),
    (
        r"(^|/)\.vault\.ya?ml\..+$",
        "backup of the local password vault",
    ),
    (r"^output/", "generated artefacts with PII"),
    (r"\.sqlite3?$", "local database"),
    (r"\.db$", "local database"),
    (r"^(?!latex/).*\.pdf$", "PDF CV / attachment"),
    (r"^\.cursor/", "local editor/agent config"),
)

# Binary / non-text suffixes: never decode as UTF-8 for content scanning.
_BINARY_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".zip",
        ".gz",
        ".bz2",
        ".xz",
        ".pyc",
        ".so",
        ".dylib",
        ".sqlite",
        ".sqlite3",
        ".db",
    }
)

# Files allowed to contain example-looking PII (fixtures, templates, this guard).
_ALLOWED_PATHS: tuple[str, ...] = (
    r"^tests/",
    r"^data/.*\.example\.yaml$",
    r"^src/jobbot/resources/.*\.example\.yaml$",
    r"^latex/cv\.tex\.demo$",
    r"^src/jobbot/ops/pii_guard\.py$",
    r"^src/jobbot/portals/email_apply\.py$",
    r"^\.gitignore$",
    r"^uv\.lock$",
)

# Domains that are obviously placeholders.
_ALLOWED_EMAIL_DOMAINS: tuple[str, ...] = (
    "example.com",
    "example.org",
    "example.net",
    "example.cl",
    "empresa.cl",
    "acme.com",
    "localhost",
)

_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b")
_CL_PHONE_RE = re.compile(r"\+\s?56[\s\-.,]*9[\s\-.,]*\d{4}[\s\-.,]*\d{4}")
_RUT_RE = re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}\s*-\s*[\dkK]\b")
_HOME_PATH_RE = re.compile(r"/Users/[A-Za-z0-9._\-]+/")
_IMPORT_HEADER_RE = re.compile(r"#\s*Source:\s*/", re.I)


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str
    detail: str = ""


def is_blocked_path(path: str) -> str | None:
    """Reason why ``path`` must not be committed, or None."""
    norm = str(PurePosixPath(path))
    for pattern, reason in _BLOCKED_PATHS:
        if re.search(pattern, norm):
            return reason
    return None


def is_allowed_content_path(path: str) -> bool:
    norm = str(PurePosixPath(path))
    return any(re.search(pattern, norm) for pattern in _ALLOWED_PATHS)


def scan_text(text: str, path: str) -> list[Finding]:
    """Find likely-real PII in file content (skips fixtures/templates)."""
    if is_allowed_content_path(path):
        return []
    findings: list[Finding] = []
    for match in _EMAIL_RE.finditer(text):
        domain = match.group(1).casefold()
        if domain in _ALLOWED_EMAIL_DOMAINS:
            continue
        findings.append(Finding(path, "real-looking email", match.group(0)))
    if _CL_PHONE_RE.search(text):
        findings.append(Finding(path, "phone number", "+56 9 …"))
    if _RUT_RE.search(text):
        findings.append(Finding(path, "RUT", "nn.nnn.nnn-n"))
    for match in _HOME_PATH_RE.finditer(text):
        findings.append(Finding(path, "absolute home path", match.group(0)))
    if _IMPORT_HEADER_RE.search(text):
        findings.append(Finding(path, "import-latex header with local source path", ""))
    return findings


def redact(text: str) -> str:
    """Replace contact data with placeholders, keeping surrounding text.

    Used wherever text leaves the candidate's own files (failure payloads that
    may become GitHub issues, learned form labels, LLM prompts). Same patterns
    as the commit guard, so a change to what counts as PII lands in both places.
    """
    if not text:
        return text
    out = _EMAIL_RE.sub("[email]", text)
    out = _CL_PHONE_RE.sub("[phone]", out)
    out = _RUT_RE.sub("[id]", out)
    return _HOME_PATH_RE.sub("[path]/", out)


def _git(args: list[str]) -> str:
    proc = subprocess.run(  # noqa: S603
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def _git_bytes(args: list[str]) -> bytes:
    proc = subprocess.run(  # noqa: S603
        ["git", *args],
        check=False,
        capture_output=True,
    )
    return proc.stdout


def is_binary_path(path: str) -> bool:
    """True when the path suffix is a known non-text artefact."""
    suffix = PurePosixPath(path).suffix.casefold()
    return suffix in _BINARY_SUFFIXES


def looks_binary(data: bytes) -> bool:
    """Heuristic: NUL in the first chunk means not text for the PII scanner."""
    if not data:
        return False
    sample = data[:8192]
    return b"\0" in sample


def staged_paths() -> list[str]:
    out = _git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def staged_content(path: str) -> str | None:
    """Return staged text, or None when the blob is binary / undecodable.

    Architecture PNGs and similar must not crash the guard with UnicodeDecodeError.
    """
    if is_binary_path(path):
        return None
    raw = _git_bytes(["show", f":{path}"])
    if looks_binary(raw):
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_staged() -> list[Finding]:
    """Scan the git index for blocked paths and PII content."""
    findings: list[Finding] = []
    for path in staged_paths():
        reason = is_blocked_path(path)
        if reason is not None:
            findings.append(Finding(path, f"blocked path ({reason})"))
            continue
        text = staged_content(path)
        if text is None:
            continue
        findings.extend(scan_text(text, path))
    return findings


def tracked_paths() -> list[str]:
    out = _git(["ls-files"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def scan_tracked() -> list[Finding]:
    """Scan every tracked file, for commits that never ran the pre-commit hook."""
    top = _git(["rev-parse", "--show-toplevel"]).strip()
    if not top:
        return []
    root = Path(top)
    findings: list[Finding] = []
    for path in tracked_paths():
        reason = is_blocked_path(path)
        if reason is not None:
            findings.append(Finding(path, f"blocked path ({reason})"))
            continue
        if is_binary_path(path):
            continue
        try:
            raw = (root / path).read_bytes()
        except OSError:
            continue
        if looks_binary(raw):
            continue
        findings.extend(scan_text(raw.decode("utf-8", errors="replace"), path))
    return findings


def format_report(findings: list[Finding]) -> str:
    lines = ["PII guard blocked this commit:", ""]
    for f in findings:
        detail = f" → {f.detail}" if f.detail else ""
        lines.append(f"  {f.path}: {f.reason}{detail}")
    lines.extend(
        [
            "",
            "Fix: unstage the file (git restore --staged PATH) or replace real data",
            "with an *.example.yaml / fixture value. Never commit real PII.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    findings = scan_tracked() if "--all" in args else scan_staged()
    if not findings:
        return 0
    print(format_report(findings), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
