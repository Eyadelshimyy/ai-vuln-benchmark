#!/usr/bin/env python3
"""Scripted steps of the business-rules-catalogue skill.

Subcommands, in run order:
  init          pin the target repository's commit, create the work directory, render the briefs
  survey        map the repository and draft scope.json; once scope.json exists, validate it
  record        record the user's destination, audience and save-location choices
  partition     propose reader assignments and render one prompt per reader
  check         validate reader outputs: shape, vocabulary, duplicates, file coverage
  batches       sample rules for the second read and split them into checker batches
  build         merge outputs and verdicts into catalogue.json, site/ and bundle.html
  note          write the research note
  publish-prep  split site/ into hash-checked upload groups for a paste-only host

Every subcommand after init takes --work. Only the standard library is used; `node` is optional.
"""

import argparse
import ast
import fnmatch
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SKILL = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
CHUNK_BYTES = 62000   # one data file; keeps a host paste well inside one helper call
GROUP_BYTES = 64000   # one upload group: small data files share a group, large ones go alone
SAVED_DIR = ".business-rules"

KINDS = {"validation", "default", "derivation", "matching", "precedence", "state-transition",
         "threshold", "ordering", "guarantee", "naming", "authorization"}
ROLES = {"decides", "executes", "calls"}
ROLE_ALIASES = {"uses": "calls"}
TIERS = {"business", "platform", "skip"}
LANES = {"main", "side", "band"}
REQUIRED = ["key", "title", "stage", "kind", "statement", "applies_when", "exceptions", "example",
            "engineering", "surfaces", "flags", "confidence"]

CODE_EXT = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".java", ".kt", ".kts", ".scala",
    ".groovy", ".cs", ".fs", ".vb", ".rb", ".php", ".rs", ".swift", ".m", ".mm", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".ex", ".exs", ".erl", ".clj", ".cljs", ".dart", ".lua", ".r", ".jl", ".hs", ".ml",
    ".elm", ".vue", ".svelte", ".sql", ".pls", ".plsql", ".prc", ".drl", ".dmn", ".bpmn", ".sh", ".ps1",
}
SKIP_PARTS = {
    "test", "tests", "__tests__", "spec", "specs", "testdata", "test_data", "fixtures", "__mocks__",
    "mocks", "e2e", "node_modules", "vendor", "third_party", "dist", "build", "out", "target", "obj",
    ".next", ".nuxt", "coverage", "generated", "__generated__", ".venv", "venv", "site-packages",
    "__pycache__", "examples", "benchmarks", "docs", "doc",
}
SKIP_FILE = re.compile(
    r"(^test_.*\.py$|_test\.(py|go|exs?)$|\.(test|spec)\.[cm]?[jt]sx?$|Tests?\.(java|kt|cs|scala|groovy)$"
    r"|Spec\.(scala|groovy|rb)$|_spec\.rb$|^conftest\.py$|\.d\.ts$|\.min\.js$|\.pb\.go$|_pb2(_grpc)?\.py$"
    r"|\.generated\.\w+$|^testing(_\w+)?\.py$)")
MANIFESTS = {"package.json", "pyproject.toml", "setup.py", "setup.cfg", "go.mod", "Cargo.toml", "pom.xml",
             "build.gradle", "build.gradle.kts", "composer.json", "Gemfile", "mix.exs", "pubspec.yaml",
             "Package.swift", "project.clj", "deps.edn"}
MANIFEST_SUFFIX = (".csproj", ".fsproj", ".gemspec")
ENTRY_HINTS = {
    "api": {"routes", "router", "routers", "controllers", "controller", "endpoints", "handlers", "api", "resolvers",
            "rest", "grpc", "rpc", "views"},
    "job": {"jobs", "workers", "worker", "tasks", "dags", "cron", "schedulers", "scheduler", "consumers",
            "listeners", "subscribers", "queues", "workflows", "pipelines", "sagas"},
    "cli": {"cli", "commands", "cmd", "bin", "management"},
    "screen": {"pages", "screens", "app", "routes"},
}
SCREEN_EXT = {".tsx", ".jsx", ".vue", ".svelte", ".html", ".erb", ".cshtml", ".razor", ".dart"}
DECISION = re.compile(r"\b(if|elif|else\s+if|switch|case|when|match|raise|throw|assert|require|unless|"
                      r"WHERE|CASE|COALESCE|HAVING|QUALIFY)\b")
GLOSSARY_NAMES = re.compile(r"^(CONTEXT|GLOSSARY|TERMS|TERMINOLOGY|UBIQUITOUS[-_]LANGUAGE|DOMAIN)\.md$", re.I)
DECISION_DIRS = ("docs/decisions", "docs/adr", "docs/adrs", "docs/architecture/decisions", "adr", "adrs",
                 "decisions", "doc/adr", "architecture/decisions")
NOTE_DIRS = ("docs/history/research", "docs/research", "research", "docs/notes", "docs/reports", "notes")

DEFAULT_FLAG_TYPES = [
    {"id": "contradicts-docs", "label": "Contradicts docs",
     "description": "The code does something different from the glossary, a decision record, or its own documentation."},
    {"id": "inconsistent", "label": "Inconsistent",
     "description": "Two places in the codebase enforce this same rule differently."},
    {"id": "magic-value", "label": "Hard-coded value",
     "description": "A fixed constant with business meaning that you may want to confirm or change."},
    {"id": "silent", "label": "Silent",
     "description": "A failure, skip or drop that happens without telling the user."},
    {"id": "unreachable", "label": "Unreachable",
     "description": "Nothing in the product calls this rule, or it can never fire."},
    {"id": "ambiguous", "label": "Ambiguous",
     "description": "The outcome depends on order or an unstated tie-break; a business owner needs to choose."},
]
DEFAULT_SURFACES = {
    "screen": {"id": "screen", "label": "Screen", "icon": "[ ]"},
    "api": {"id": "api", "label": "API", "icon": "{ }"},
    "job": {"id": "job", "label": "Scheduled job", "icon": "≋"},
    "cli": {"id": "cli", "label": "CLI", "icon": ">_"},
    "internal": {"id": "internal", "label": "Internal", "icon": "•"},
}


# ---------- shared helpers ----------

def sh(*cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.PIPE).strip()


def js(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def fnv(text):
    """FNV-1a 32-bit over code points. Mirrored in the upload and verify snippets."""
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "project"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def work_dir(args):
    if not args.work:
        sys.exit("Pass --work <dir>; `init` printed it.")
    work = Path(args.work).expanduser().resolve()
    if not (work / "run.json").exists():
        sys.exit(f"{work}/run.json is missing. Run `br.py init` first.")
    if args.cmd in {"survey", "partition", "check", "batches", "build"}:
        run = run_info(work)
        repo = repo_of(run)
        if (sh("git", "rev-parse", "HEAD", cwd=repo) != run["commit"] or
                sh("git", "status", "--porcelain", "--untracked-files=no", cwd=repo)):
            sys.exit("Source changed since init. Use a clean checkout at the pinned commit, or start a new run.")
    return work


def run_info(work):
    return load_json(work / "run.json")


def repo_of(run):
    return Path(run["repo"])


def scope_of(work):
    path = work / "scope.json"
    if not path.exists():
        sys.exit(f"{path} is missing. Run `survey`, then write scope.json from scope.draft.json.")
    return load_json(path)


def commit_files(repo, commit):
    """Every path at the pinned commit, so a run reads exactly what it cites."""
    return sh("git", "ls-tree", "-r", "--name-only", commit, cwd=repo).splitlines()


def is_source(path, extensions, include=()):
    if excluded(path, include):
        return True
    p = PurePosixPath(path)
    if p.suffix.lower() not in extensions:
        return False
    if any(part.lower() in SKIP_PARTS for part in p.parts[:-1]):
        return False
    return not SKIP_FILE.search(p.name)


def line_count(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def under(path, root):
    root = root.rstrip("/")
    return root in ("", ".") or path == root or path.startswith(root + "/")


def area_files(files, area, extensions, exclude, include=()):
    out = []
    for f in files:
        if any(under(f, p) for p in area["paths"]) and is_source(f, extensions, include) and not excluded(f, exclude):
            out.append(f)
    return out


def excluded(path, patterns):
    """Whole-path globs where `*` also crosses `/`, such as `*/generated/*`."""
    return any(fnmatch.fnmatch(path, p) for p in patterns or [])


def in_scope_files(work, run, sc):
    """{file: area_id} for business and platform areas. A file belongs to the area with the longest path."""
    files = commit_files(repo_of(run), run["commit"])
    ext = set(sc.get("extensions") or CODE_EXT)
    owner = {}
    for area in sc["areas"]:
        for f in area_files(files, area, ext, sc.get("exclude"), sc.get("include", [])):
            best = max(len(p) for p in area["paths"] if under(f, p))
            if f not in owner or best > owner[f][1]:
                owner[f] = (area["id"], best, area["tier"])
    return {f: a for f, (a, _, tier) in owner.items() if tier != "skip"}


def is_reexport(path):
    """True for a Python module holding only imports, a docstring and dunder assignments."""
    if Path(path).suffix != ".py":
        return False
    try:
        tree = ast.parse(Path(path).read_text(errors="replace"))
    except (SyntaxError, OSError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if targets and all(isinstance(t, ast.Name) and t.id.startswith("__") for t in targets):
            continue
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            continue
        return False
    return True


def code_links(repo):
    """URL templates for the code host behind `origin`, or None when the host is unknown."""
    try:
        url = sh("git", "remote", "get-url", "origin", cwd=repo)
    except subprocess.CalledProcessError:
        return None
    m = re.match(r"^(?:git@|ssh://git@)([^:/]+)[:/](.+?)(?:\.git)?/?$", url) or \
        re.match(r"^https?://(?:[^@/]+@)?([^/]+)/(.+?)(?:\.git)?/?$", url)
    if not m:
        return None
    host, path = m.group(1).lower(), m.group(2)
    base = f"https://{host}/{path}"
    if host == "github.com":
        return {"file": base + "/blob/{commit}/{path}", "line": "#L{start}-L{end}", "commit": base + "/commit/{commit}"}
    if host == "gitlab.com" or host.startswith("gitlab."):
        return {"file": base + "/-/blob/{commit}/{path}", "line": "#L{start}-{end}", "commit": base + "/-/commit/{commit}"}
    if host == "bitbucket.org":
        return {"file": base + "/src/{commit}/{path}", "line": "#lines-{start}:{end}", "commit": base + "/commits/{commit}"}
    return None


def fill(template, **values):
    return re.sub(r"\{(\w+)\}", lambda m: str(values.get(m.group(1), "")), template)


def code_url(links, commit, ref):
    """`path:line` or `path:start-end` to a URL at the commit, or None."""
    if not links:
        return None
    m = re.match(r"^(.*?):(\d+)(?:-(\d+))?$", ref.strip())
    path, a, b = (m.group(1), m.group(2), m.group(3)) if m else (ref.strip(), None, None)
    url = fill(links["file"], commit=commit, path=path)
    return url + (fill(links["line"], start=a, end=b or a) if a else "")


def load_outputs(work):
    files = sorted(glob.glob(str(work / "out" / "W*.json")), key=lambda p: int(re.search(r"W(\d+)", p).group(1)))
    if not files:
        sys.exit(f"No reader outputs in {work / 'out'}.")
    outs = []
    for f in files:
        d = load_json(f)
        d["areas"] = d.get("areas") or d.get("packages") or []  # `packages` is the version-0 key
        for a in d["areas"]:
            a["area"] = a.get("area") or a.get("package")
        outs.append((f, d))
    return outs


def module_path(module, area):
    """A reader's module string as a repository-relative path.

    Readers write repository-relative paths. A path relative to the area's only root is accepted too.
    """
    m = module.strip()
    head = re.split(r"\s*[,(]", m, maxsplit=1)[0].strip().rstrip("/")
    if any(under(head, p) for p in area["paths"]) or len(area["paths"]) != 1:
        return m
    return f"{area['paths'][0].rstrip('/')}/{m}" if area["paths"][0] not in ("", ".") else m


def covers(module, rel):
    """Whether a module string covers one file: a file, a directory, `dir/a.py, b.py` or `dir (x, y)`."""
    head = re.split(r"\s*[,(]", module, maxsplit=1)[0].strip().rstrip("/")
    if rel == head or rel.startswith(head + "/"):
        return True
    base = head.rsplit("/", 1)[0] if "/" in head else ""
    for extra in re.split(r"[,()]", module)[1:]:
        name = extra.strip().rstrip("/")
        if not name:
            continue
        for cand in (f"{base}/{name}" if base else name, f"{head}/{name}"):
            if rel == cand or rel.startswith(cand + "/"):
                return True
    return False


# ---------- init ----------

def cmd_init(args):
    try:
        repo = Path(sh("git", "rev-parse", "--show-toplevel", cwd=Path(args.repo).expanduser().resolve()))
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit(f"{args.repo} is not inside a Git repository.")
    dirty = sh("git", "status", "--porcelain", "--untracked-files=no", cwd=repo)
    if dirty:
        sys.exit("The repository has uncommitted changes to tracked files. Use a clean checkout so every rule "
                 "cites the code the readers see.\n" + dirty)
    commit = sh("git", "rev-parse", "HEAD", cwd=repo)
    name = Path(sh("git", "remote", "get-url", "origin", cwd=repo)).stem if has_origin(repo) else repo.name
    state = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    work = Path(args.work).expanduser().resolve() if args.work else state / "business-rules" / f"{slugify(name)}-{commit[:9]}"
    if under(str(work), str(repo)):
        sys.exit(f"{work} is inside the repository. Choose a work directory outside it.")
    run_path = work / "run.json"
    if run_path.exists():
        run = load_json(run_path)
        if run["commit"] != commit or Path(run["repo"]).resolve() != repo:
            sys.exit(f"{work} belongs to another repository or commit. Use another --work.")
    else:
        run = {"schema_version": SCHEMA_VERSION, "repo": str(repo), "project": name, "commit": commit,
               "short": commit[:9], "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
               "links": code_links(repo), "dirty": bool(dirty), "choices": {}}
    for d in ("out", "verify", "build", "site", "publish", "shots", "prompts", "files"):
        (work / d).mkdir(parents=True, exist_ok=True)

    saved = Path(args.scope_dir).expanduser() if args.scope_dir else repo / SAVED_DIR
    loaded = []
    for f in ("scope.json", "partition.json"):
        if (saved / f).exists() and not (work / f).exists():
            shutil.copy(saved / f, work / f)
            loaded.append(f)
    if loaded:
        run["scope_source"] = str(saved)
    save_json(run_path, run)

    subs = {"{WORK}": str(work), "{REPO}": str(repo)}
    for src, dst in (("verify-brief.md", "verify/VERIFY.md"), ("upload-brief.md", "publish/UPLOAD.md")):
        text = (SKILL / "references" / src).read_text()
        for k, v in subs.items():
            text = text.replace(k, v)
        (work / dst).write_text(text)
    print(f"repo:   {repo}\ncommit: {commit}\nwork:   {work}")
    print(f"links:  {'from origin' if run['links'] else 'none (unknown code host; citations stay plain text)'}")
    print(f"scope:  {'loaded ' + ', '.join(loaded) + ' from ' + str(saved) if loaded else 'none saved; run survey'}")


def has_origin(repo):
    try:
        sh("git", "remote", "get-url", "origin", cwd=repo)
        return True
    except subprocess.CalledProcessError:
        return False


# ---------- survey ----------

def manifest_roots(files):
    roots = set()
    for f in files:
        p = PurePosixPath(f)
        if p.name in MANIFESTS or p.name.endswith(MANIFEST_SUFFIX):
            if not any(part.lower() in SKIP_PARTS for part in p.parts[:-1]):
                roots.add(str(p.parent) if str(p.parent) != "." else "")
    return roots


def candidate_units(sources):
    """Group source files into candidate areas: nearest package root, else the top-level directory."""
    roots = sorted(manifest_roots(sources["_all"]) - {""}, key=len, reverse=True)
    units = defaultdict(list)
    for f in sources["files"]:
        root = next((r for r in roots if under(f, r)), None)
        if root is None:
            parts = PurePosixPath(f).parts
            root = parts[0] if len(parts) > 1 else "."
            if root == "src" and len(parts) > 2:
                root = f"src/{parts[1]}"
        units[root].append(f)
    return units


def entry_kinds(files):
    hits = defaultdict(list)
    for f in files:
        p = PurePosixPath(f)
        names = {x.lower() for x in p.parts[:-1]} | {p.stem.lower()}
        for kind, words in ENTRY_HINTS.items():
            if names & words and (kind != "screen" or p.suffix in SCREEN_EXT):
                hits[kind].append(f)
    return hits


def glossary_format(path):
    text = Path(path).read_text(errors="replace")
    if len(re.findall(r"^\*\*.+?\*\*:\s*$", text, re.M)) >= 3:
        return "bold-colon"
    if len(re.findall(r"^#{2,4} \S", text, re.M)) >= 3:
        return "headings"
    return None


def cmd_survey(args):
    work = work_dir(args)
    run = run_info(work)
    repo = repo_of(run)
    if (work / "scope.json").exists():
        return validate_scope(work, run)

    all_files = commit_files(repo, run["commit"])
    ext_counts = Counter(PurePosixPath(f).suffix.lower() for f in all_files if PurePosixPath(f).suffix.lower() in CODE_EXT)
    files = [f for f in all_files if is_source(f, CODE_EXT)]
    kept = set(files)
    skipped = [f for f in all_files if PurePosixPath(f).suffix.lower() in CODE_EXT and f not in kept]
    units = candidate_units({"files": files, "_all": all_files})
    entries = entry_kinds(files)

    print(f"{len(all_files)} files at {run['short']}; {len(files)} source files; "
          f"{len(skipped)} test, generated or vendored files left out")
    print("languages:", ", ".join(f"{e} {n}" for e, n in ext_counts.most_common(12)))
    print(f"\n{'candidate area':48} {'files':>5} {'lines':>7} {'dec/kloc':>8}  languages")
    rows = []
    for root, fs in sorted(units.items()):
        lines = dec = 0
        for f in fs:
            text = (repo / f).read_text(errors="replace") if (repo / f).exists() else ""
            lines += text.count("\n") + 1
            dec += len(DECISION.findall(text))
        langs = ", ".join(e for e, _ in Counter(PurePosixPath(f).suffix for f in fs).most_common(3))
        rows.append((root, len(fs), lines, round(1000 * dec / max(lines, 1)), langs))
        print(f"{root or '.':48} {len(fs):5} {lines:7} {rows[-1][3]:8}  {langs}")
    for root, nfiles, lines, _, _ in rows:
        if lines > 30000:
            by = Counter()
            for f in units[root]:
                rel = PurePosixPath(f).relative_to(root) if root not in ("", ".") else PurePosixPath(f)
                by["/".join(rel.parts[:2][:-1]) or "."] += line_count(repo / f)
            print(f"\n{root} by folder:")
            for k, v in sorted(by.items()):
                print(f"  {k:56} {v:7}")

    print("\nentry-point candidates (surfaces):")
    for kind, fs in sorted(entries.items()):
        dirs = Counter(str(PurePosixPath(f).parent) for f in fs)
        print(f"  {kind:7} {len(fs):4} files, e.g. " + ", ".join(d for d, _ in dirs.most_common(4)))

    glossaries = [f for f in all_files if GLOSSARY_NAMES.match(PurePosixPath(f).name) and f.count("/") <= 3]
    decisions = [d for d in DECISION_DIRS if any(under(f, d) for f in all_files)]
    notes = [d for d in NOTE_DIRS if any(under(f, d) for f in all_files)]
    guidance = [f for f in all_files if PurePosixPath(f).name in ("AGENTS.md", "CLAUDE.md", "ARCHITECTURE.md")]
    print("\nglossary candidates:", ", ".join(glossaries) or "none")
    print("decision records:", ", ".join(decisions) or "none")
    print("research-note folders:", ", ".join(notes) or "none")
    print("agent and architecture guidance:", ", ".join(guidance[:6]) or "none")

    gloss = None
    if glossaries:
        fmt = glossary_format(repo / glossaries[0])
        gloss = {"path": glossaries[0], "format": fmt} if fmt else None
    codes = set()
    areas = []
    for root, nfiles, lines, dec, langs in sorted(rows, key=lambda r: -r[2]):
        aid = slugify(root.replace("/", "-")) if root not in ("", ".") else "root"
        areas.append({"id": aid, "code": unique_code(aid, codes), "title": PurePosixPath(root).name or run["project"],
                      "tier": "unclassified", "paths": [root if root else "."], "summary": None, "focus": None,
                      "_survey": {"files": nfiles, "lines": lines, "decisions_per_kloc": dec, "languages": langs}})
    surfaces = [DEFAULT_SURFACES[k] | {"where": sorted({str(PurePosixPath(f).parent) for f in entries[k]})[:6]}
                for k in ("screen", "api", "job", "cli") if entries.get(k)]
    draft = {
        "schema_version": SCHEMA_VERSION,
        "project": {"name": run["project"], "slug": slugify(run["project"])},
        "title": f"{run['project']} business rules",
        "glossary": gloss,
        "docs": ([gloss["path"]] if gloss else []) + decisions,
        "extensions": sorted(e for e in ext_counts),
        "include": [],
        "exclude": [],
        "areas": areas,
        "surfaceKinds": surfaces + [DEFAULT_SURFACES["internal"]],
        "stages": [],
        "flagTypes": DEFAULT_FLAG_TYPES,
        "noteDirs": notes,
    }
    save_json(work / "scope.draft.json", draft)
    print(f"\nwrote {work / 'scope.draft.json'}: classify every area, add stages, save it as scope.json, "
          f"then run survey again to validate.")


def unique_code(aid, taken):
    """A short code for rule IDs, favouring the last word: `services-orders` gives `OR`."""
    words = [w for w in re.split(r"[^a-z0-9]+", aid.lower()) if w] or ["area"]
    last = re.sub(r"[^a-z]", "", words[-1]).upper() or "AR"
    initials = "".join(w[0] for w in words).upper()
    for cand in (last[:2], initials[:3] if len(words) > 1 else "", last[:3], last[0] + last[-1]):
        if len(cand) >= 2 and cand not in taken:
            taken.add(cand)
            return cand
    n = 1
    while f"{last[0]}{n}" in taken:
        n += 1
    taken.add(f"{last[0]}{n}")
    return f"{last[0]}{n}"


def validate_scope(work, run):
    sc = scope_of(work)
    repo = repo_of(run)
    errors, warnings = [], []
    ids, codes = set(), set()
    for a in sc.get("areas", []):
        if not re.match(r"^[a-z0-9][a-z0-9-]*$", a.get("id", "")):
            errors.append(f"area id {a.get('id')!r} must be a lowercase slug")
        if a.get("id") in ids:
            errors.append(f"area id {a['id']} is used twice")
        ids.add(a.get("id"))
        if a.get("tier") not in TIERS:
            errors.append(f"area {a.get('id')}: tier {a.get('tier')!r} must be business, platform or skip")
        if a.get("tier") != "skip":
            if not re.match(r"^[A-Z][A-Z0-9]{1,3}$", a.get("code", "")):
                errors.append(f"area {a.get('id')}: code {a.get('code')!r} must be 2-4 capitals")
            if a.get("code") in codes:
                errors.append(f"area {a.get('id')}: code {a['code']} is used twice")
            codes.add(a.get("code"))
        for p in a.get("paths", []):
            if p not in ("", ".") and not (repo / p).exists():
                errors.append(f"area {a.get('id')}: path {p} does not exist")
        if not a.get("paths"):
            errors.append(f"area {a.get('id')}: no paths")
    if not any(a.get("tier") == "business" for a in sc.get("areas", [])):
        errors.append("no business area; the catalogue would be empty")
    stages = sc.get("stages", [])
    sids = [s.get("id") for s in stages]
    if not stages:
        errors.append("no stages; add the lifecycle (see references/discovery.md)")
    if len(set(sids)) != len(sids):
        errors.append("stage ids repeat")
    main = [s["id"] for s in stages if s.get("lane", "main") == "main"]
    for s in stages:
        if s.get("lane", "main") not in LANES:
            errors.append(f"stage {s.get('id')}: lane must be main, side or band")
        if s.get("lane") == "side" and s.get("laneFrom") not in main:
            errors.append(f"stage {s.get('id')}: laneFrom must name a main-lane stage")
        if not s.get("label") or not s.get("description"):
            errors.append(f"stage {s.get('id')}: needs a label and a description")
    if not main:
        errors.append("at least one stage must be in the main lane")
    if "internal" not in {k.get("id") for k in sc.get("surfaceKinds", [])}:
        errors.append("surfaceKinds must include `internal`")
    if not sc.get("flagTypes"):
        errors.append("flagTypes is empty")
    g = sc.get("glossary")
    if g and not (repo / g.get("path", "")).exists():
        errors.append(f"glossary {g.get('path')} does not exist")
    if g and g.get("format") not in ("bold-colon", "headings", "other"):
        errors.append("glossary format must be bold-colon, headings or other")

    if not errors:
        owner = in_scope_files(work, run, sc)
        all_src = [f for f in commit_files(repo, run["commit"])
                   if is_source(f, set(sc.get("extensions") or CODE_EXT), sc.get("include", []))
                   and not excluded(f, sc.get("exclude"))]
        covered = {f for f in all_src if any(under(f, p) for a in sc["areas"] for p in a["paths"])}
        stray = Counter(PurePosixPath(f).parts[0] for f in all_src if f not in covered)
        for top, n in stray.most_common():
            warnings.append(f"{n} source files under {top}/ are in no area (add an area or a skip area)")
        print(f"\n{'area':28} {'code':5} {'tier':9} {'files':>5} {'lines':>7}  paths")
        for a in sc["areas"]:
            fs = [f for f, aid in owner.items() if aid == a["id"]]
            lines = sum(line_count(repo / f) for f in fs)
            print(f"{a['id']:28} {a.get('code', ''):5} {a['tier']:9} {len(fs):5} {lines:7}  {', '.join(a['paths'])}")
        print("\nstages:", " → ".join(s["id"] for s in stages if s.get("lane", "main") == "main"),
              "| side:", ", ".join(s["id"] for s in stages if s.get("lane") == "side") or "none",
              "| band:", ", ".join(s["id"] for s in stages if s.get("lane") == "band") or "none")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"scope: {len(errors)} errors, {len(warnings)} warnings")
    sys.exit(1 if errors else 0)


# ---------- record ----------

def cmd_record(args):
    work = work_dir(args)
    run = run_info(work)
    choices = run.setdefault("choices", {})
    for key in ("destination", "audience", "note", "save_scope", "share_url"):
        value = getattr(args, key)
        if value is not None:
            choices[key] = value
    choices["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_json(work / "run.json", run)
    print(json.dumps(choices, indent=1))


# ---------- partition ----------

def cmd_partition(args):
    work = work_dir(args)
    run = run_info(work)
    sc = scope_of(work)
    repo = repo_of(run)
    owner = in_scope_files(work, run, sc)
    areas = {a["id"]: a for a in sc["areas"]}
    lines = {f: line_count(repo / f) for f in owner}
    part_path = work / "partition.json"

    if not part_path.exists() or args.force:
        readers = propose_readers(sc, owner, lines, args.target_lines)
        save_json(part_path, {"schema_version": SCHEMA_VERSION, "target_lines": args.target_lines, "readers": readers})
        print(f"proposed {len(readers)} readers in {part_path}")
    part = load_json(part_path)

    errors = []
    assigned = {}
    for r in part["readers"]:
        for p in r["owns"]:
            p = p.rstrip("/")
            for f in owner:
                if under(f, p):
                    prev = assigned.get(f)
                    if prev is None or len(p) > prev[1]:
                        assigned[f] = (r["id"], len(p))
                    elif len(p) == prev[1] and prev[0] != r["id"]:
                        errors.append(f"{f} is owned by both {prev[0]} and {r['id']} through {p}")
    missing = sorted(f for f in owner if f not in assigned)
    for f in missing[:20]:
        errors.append(f"{f} ({areas[owner[f]]['id']}) has no reader")
    if len(missing) > 20:
        errors.append(f"... and {len(missing) - 20} more files without a reader")

    by_reader = defaultdict(list)
    for f, (rid, _) in assigned.items():
        by_reader[rid].append(f)
    for r in part["readers"]:
        fs = sorted(by_reader.get(r["id"], []))
        (work / "files" / f"{r['id']}.txt").write_text("\n".join(fs) + "\n")
        r["_files"], r["_lines"] = len(fs), sum(lines[f] for f in fs)
        r["_areas"] = sorted({owner[f] for f in fs})
    for r in part["readers"]:
        r["_neighbours"] = [o["id"] for o in part["readers"] if o is not r and set(o["_areas"]) & set(r["_areas"])]
        (work / "prompts" / f"{r['id']}.md").write_text(reader_prompt(work, run, sc, r, part["readers"], areas))
    render_brief(work, run, sc)

    print(f"{'reader':7} {'tier':9} {'files':>5} {'lines':>7}  owns")
    for r in part["readers"]:
        print(f"{r['id']:7} {r['tier']:9} {r['_files']:5} {r['_lines']:7}  {', '.join(r['owns'])[:110]}")
    for e in errors:
        print("ERROR", e)
    print(f"partition: {len(errors)} errors; prompts in {work / 'prompts'}")
    sys.exit(1 if errors else 0)


def propose_readers(sc, owner, lines, target):
    """Pack areas into readers of about `target` lines, splitting a large area by sub-directory."""
    chunks = []  # (area_id, tier, path, lines)
    for a in sc["areas"]:
        if a["tier"] == "skip":
            continue
        fs = [f for f, aid in owner.items() if aid == a["id"]]
        total = sum(lines[f] for f in fs)
        if a["tier"] == "platform" or total <= target:
            chunks.append((a["id"], a["tier"], a["paths"], total))
            continue
        chunks.extend(split_area(a, fs, lines, target))
    readers, n = [], 0

    def new(tier):
        nonlocal n
        n += 1
        readers.append({"id": f"W{n}", "tier": tier, "owns": [], "focus": "", "lines": 0})
        return readers[-1]

    platform = [c for c in chunks if c[1] == "platform"]
    business = sorted((c for c in chunks if c[1] == "business"), key=lambda c: -c[3])
    for c in business:
        home = next((r for r in readers if r["lines"] + c[3] <= target), None) or new("business")
        home["owns"] += list(c[2])
        home["lines"] += c[3]
    if platform:
        r = new("platform")
        for c in platform:
            r["owns"] += list(c[2])
            r["lines"] += c[3]
    for r in readers:
        r.pop("lines")
    return readers


def split_area(area, files, lines, target):
    def group(fs, root, depth):
        by = defaultdict(list)
        for f in fs:
            rel = PurePosixPath(f).relative_to(root) if root not in ("", ".") else PurePosixPath(f)
            key = str(PurePosixPath(root) / rel.parts[0]) if len(rel.parts) > 1 else f
            by[key].append(f)
        out = []
        for key, g in by.items():
            total = sum(lines[f] for f in g)
            if total > target and depth < 6 and g != [key]:  # a directory over budget splits further
                out += group(g, key, depth + 1)
            else:
                out.append((key, total))
        return out

    root = area["paths"][0] if len(area["paths"]) == 1 else ""
    pieces = sorted(group(files, root, 0), key=lambda x: x[0])
    chunks, cur, size = [], [], 0
    for key, total in pieces:
        if cur and size + total > target:
            chunks.append((area["id"], "business", cur, size))
            cur, size = [], 0
        cur.append(key)
        size += total
    if cur:
        chunks.append((area["id"], "business", cur, size))
    return chunks


def reader_prompt(work, run, sc, reader, readers, areas):
    rid = reader["id"]
    lines = [f"You are reader {rid}, a researcher. Read `{work}/BRIEF.md` first and follow it exactly. "
             f"It defines a rule, the output shape, the style, the flags and how to trace surfaces. "
             f"Write one file, `{work}/out/{rid}.json`. Leave the repository unchanged.", "",
             "## Your files", "",
             f"Repository root: `{run['repo']}`. You own the rules whose deciding code is in these paths; "
             f"`{work}/files/{rid}.txt` lists all {reader['_files']} files ({reader['_lines']} lines):", ""]
    lines += [f"- `{p}`" for p in reader["owns"]]
    if reader["tier"] == "platform":
        lines += ["", "This is a **summary branch**: write a summary for each area and module, and record only the "
                      "3 to 12 rules per area a business owner would care about."]
    lines += ["", "## Areas", ""]
    for aid in reader["_areas"]:
        a = areas[aid]
        lines.append(f"- `{aid}` ({a['title']}, paths {', '.join(a['paths'])}). "
                     + (f"{a['summary']} " if a.get("summary") else "")
                     + (f"Check: {a['focus']}" if a.get("focus") else ""))
    if reader.get("focus"):
        lines += ["", "## Focus", "", reader["focus"]]
    neighbours = [r for r in readers if r["id"] in reader["_neighbours"]]
    if neighbours:
        lines += ["", "## Neighbours", "", "These readers own the rest of your areas. Write rules only for your own "
                  "files and record overlaps in `notes`:", ""]
        lines += [f"- {r['id']}: {', '.join(r['owns'])[:300]}" for r in neighbours]
    lines += ["", "Reply with the summary BRIEF.md asks for."]
    return "\n".join(lines) + "\n"


def render_brief(work, run, sc):
    text = (SKILL / "references" / "reader-brief.md").read_text()
    gloss = sc.get("glossary")
    docs = sc.get("docs") or []
    stages = "\n".join(f"  - `{s['id']}`: {s['description']}" for s in sc["stages"])
    surfaces = "\n".join(f"  - `{k['id']}`: {k['label']}" + (f", found under {', '.join(k['where'])}" if k.get("where") else "")
                         for k in sc["surfaceKinds"])
    flags = "\n".join(f"  - `{f['id']}`: {f['description']}" for f in sc["flagTypes"])
    subs = {
        "{WORK}": str(work), "{REPO}": run["repo"], "{PROJECT}": sc["project"]["name"],
        "{GLOSSARY}": (f"`{gloss['path']}` is the project's glossary. Read it first and write rules in its terms; "
                       f"where it names a word to avoid, use its preferred term." if gloss else
                       "The project has no glossary. Use the names the code itself uses for domain concepts."),
        "{DOCS}": ", ".join(f"`{d}`" for d in docs) or "the code's own docstrings and comments",
        "{STAGES}": stages, "{SURFACES}": surfaces, "{FLAGS}": flags,
    }
    for k, v in subs.items():
        text = text.replace(k, v)
    (work / "BRIEF.md").write_text(text)


# ---------- check ----------

def cmd_check(args):
    work = work_dir(args)
    run = run_info(work)
    sc = scope_of(work)
    repo = repo_of(run)
    stages = {s["id"] for s in sc["stages"]}
    flag_types = {f["id"] for f in sc["flagTypes"]}
    surface_kinds = {k["id"] for k in sc["surfaceKinds"]}
    areas = {a["id"]: a for a in sc["areas"] if a["tier"] != "skip"}
    errors, warnings = [], []
    tracked = set(commit_files(repo, run["commit"]))
    lengths = {}

    def check_ref(tag, ref):
        m = re.fullmatch(r"(.+):(\d+)(?:-(\d+))?", str(ref))
        if not m or m[1] not in tracked:
            errors.append(f"{tag}: citation {ref!r} must name a tracked file and line range")
            return
        path, start, end = m[1], int(m[2]), int(m[3] or m[2])
        if path not in lengths:
            lengths[path] = line_count(repo / path)
        if not 1 <= start <= end <= lengths[path]:
            errors.append(f"{tag}: citation {ref!r} is outside the file's {lengths[path]} lines")

    seen_mod, seen_key = {}, {}
    rules, flags, by_area = 0, Counter(), Counter()
    modules_by_area = defaultdict(list)
    for path, d in load_outputs(work):
        w = d.get("worker") or Path(path).stem
        for p in d["areas"]:
            aid = p.get("area")
            if aid not in areas:
                errors.append(f"{w}: area {aid!r} is not in scope.json")
                continue
            for m in p.get("modules", []):
                mod = module_path(m.get("module", ""), areas[aid])
                if (aid, mod) in seen_mod:
                    errors.append(f"{w}: module {mod} also written by {seen_mod[(aid, mod)]}")
                seen_mod[(aid, mod)] = w
                modules_by_area[aid].append(mod)
                if m.get("stage") not in stages:
                    errors.append(f"{w}: module {mod} has stage {m.get('stage')!r}")
                for r in m.get("rules", []):
                    rules += 1
                    by_area[aid] += 1
                    tag = f"{w}:{aid}:{r.get('key')}"
                    prior = seen_key.get((aid, r.get("key")))
                    if prior and prior[1] == mod:
                        errors.append(f"{tag}: duplicate key in one module")
                    elif prior:
                        errors.append(f"{tag}: key also used by {prior[0]} in {prior[1]}; use an area-unique key")
                    seen_key[(aid, r.get("key"))] = (w, mod)
                    missing = [k for k in REQUIRED if k not in r]
                    if missing:
                        errors.append(f"{tag}: missing {', '.join(missing)}")
                    if r.get("stage") not in stages:
                        errors.append(f"{tag}: stage {r.get('stage')!r}")
                    if r.get("kind") not in KINDS:
                        errors.append(f"{tag}: kind {r.get('kind')!r}")
                    if r.get("confidence") not in ("high", "medium"):
                        errors.append(f"{tag}: confidence {r.get('confidence')!r}")
                    if not r.get("surfaces"):
                        errors.append(f"{tag}: no surfaces")
                    for s in r.get("surfaces", []):
                        if s.get("kind") not in surface_kinds:
                            errors.append(f"{tag}: surface kind {s.get('kind')!r} is not in scope.surfaceKinds")
                        if s.get("ref"):
                            check_ref(tag, s["ref"])
                    if not r.get("engineering"):
                        errors.append(f"{tag}: no engineering")
                    elif not any(e.get("role") == "decides" for e in r["engineering"]):
                        errors.append(f"{tag}: no deciding code citation")
                    for e in r.get("engineering", []):
                        role = e.get("role")
                        if role in ROLE_ALIASES:
                            warnings.append(f"{tag}: engineering role {role!r} is read as {ROLE_ALIASES[role]!r}")
                        elif role not in ROLES:
                            errors.append(f"{tag}: engineering role {role!r}")
                        if not e.get("file"):
                            errors.append(f"{tag}: engineering entry without a file")
                        check_ref(tag, f"{e.get('file', '')}:{e.get('lines', '')}")
                    for f in r.get("flags", []):
                        flags[f.get("type")] += 1
                        if f.get("type") not in flag_types:
                            errors.append(f"{tag}: flag type {f.get('type')!r}")
                        if not f.get("evidence"):
                            errors.append(f"{tag}: flag {f.get('type')} has no evidence")
                        for ref in f.get("evidence", []):
                            check_ref(tag, ref)
                    if r.get("statement", "").count("`") % 2 or r.get("title", "").count("`") % 2:
                        errors.append(f"{tag}: unbalanced backticks in title or statement")

    owner = in_scope_files(work, run, sc)
    for aid, a in areas.items():
        files = [f for f, x in owner.items() if x == aid]
        if files and not modules_by_area.get(aid):
            errors.append(f"{aid}: no reader covered this area")
            continue
        gaps = [f for f in files if not any(covers(m, f) for m in modules_by_area[aid]) and not is_reexport(repo / f)]
        gaps = sorted(((f, line_count(repo / f)) for f in gaps), key=lambda x: -x[1])
        if a["tier"] == "platform" and gaps:
            warnings.append(f"coverage: {aid} (summary tier) leaves {len(gaps)} files, {sum(n for _, n in gaps)} "
                            f"lines, in no module; largest {gaps[0][0]}")
            continue
        for f, n in gaps:
            warnings.append(f"coverage: {f} ({n} lines) is in no module")

    print(f"rules {rules}  flags {sum(flags.values())} {dict(flags)}")
    print("per area:", dict(by_area))
    for w_ in warnings:
        print("WARN ", w_)
    for e in errors:
        print("ERROR", e)
    print(f"{len(errors)} errors, {len(warnings)} warnings")
    if errors:
        sys.exit(1)


# ---------- batches ----------

def cmd_batches(args):
    work = work_dir(args)
    run = run_info(work)
    cmd_check(args)
    rng = random.Random(int(run["commit"][:12], 16))
    hard = {"contradicts-docs", "inconsistent"}
    flagged, sample = [], []
    for _, d in load_outputs(work):
        pool = []
        for p in d["areas"]:
            for m in p["modules"]:
                for r in m["rules"]:
                    ref = {"worker": d.get("worker"), "area": p["area"], "module": m["module"], "key": r["key"], "rule": r}
                    picked = [i for i, f in enumerate(r["flags"]) if f["type"] in hard]
                    if picked:
                        flagged.append({**ref, "check_flags": picked})
                    elif not r["flags"]:
                        pool.append(ref)
        sample += [{**x, "check_flags": [], "sample": True} for x in rng.sample(pool, min(args.per_reader, len(pool)))]
    n = max(1, args.batches)
    batches = [[] for _ in range(n)]
    flagged.sort(key=lambda x: str(x["worker"]))
    for i, item in enumerate(flagged):
        batches[i * n // len(flagged)].append(item)
    for i, item in enumerate(sample):
        batches[i % n].append(item)
    for f in glob.glob(str(work / "verify" / "in*.json")) + glob.glob(str(work / "verify" / "out*.json")):
        Path(f).unlink()
    for i, b in enumerate(batches, 1):
        save_json(work / "verify" / f"in{i}.json", b)
        print(f"batch {i}: {len(b)} items, {sum(1 for x in b if not x.get('sample'))} flagged, "
              f"{sum(len(x['check_flags']) for x in b)} flags to check")


# ---------- build ----------

def apply_verification(work, mods):
    inputs = sorted((work / "verify").glob("in*.json"))
    if not inputs:
        sys.exit("No second-read batches. Run `br.py batches`, then complete every checker batch.")
    verdicts = []
    rules = {(aid, r["key"]): r for (aid, _), m in mods.items() for r in m["rules"]}
    for source in inputs:
        output = source.with_name(source.name.replace("in", "out", 1))
        if not output.exists():
            sys.exit(f"Missing checker output {output.name}; write [] for an empty batch.")
        expected = load_json(source)
        actual = load_json(output)
        if not isinstance(actual, list):
            sys.exit(f"{output.name}: expected a JSON list of checker verdicts.")
        key = lambda v: (v.get("worker"), v.get("area"), v.get("key"))
        if Counter(map(key, actual)) != Counter(map(key, expected)):
            sys.exit(f"{output.name}: checker verdicts must cover each input exactly once.")
        by_key = {key(v): v for v in actual}
        for item in expected:
            if rules.get((item["area"], item["key"])) != item["rule"]:
                sys.exit("Reader output changed after sampling. Re-run batches and the second read.")
            v = by_key[key(item)]
            revision = v.get("rule_revision")
            if v.get("rule_verdict") not in {"correct", "needs-revision", "wrong"}:
                sys.exit(f"{output.name}: invalid rule verdict for {item['key']}.")
            if v["rule_verdict"] == "needs-revision" and not revision:
                sys.exit(f"{output.name}: needs-revision requires corrected fields.")
            if v["rule_verdict"] == "correct" and revision:
                sys.exit(f"{output.name}: corrected fields require needs-revision or wrong.")
            if revision and (not isinstance(revision, dict) or set(revision) -
                             {"title", "statement", "applies_when", "exceptions", "example", "on_violation",
                              "engineering", "surfaces", "terms", "confidence"}):
                sys.exit(f"{output.name}: rule_revision contains unsupported fields.")
            flags = v.get("flags", [])
            if Counter(f.get("index") for f in flags) != Counter(item["check_flags"]):
                sys.exit(f"{output.name}: flag verdicts must cover check_flags exactly once.")
            for f in flags:
                if f.get("verdict") not in {"confirmed", "revised", "refuted"}:
                    sys.exit(f"{output.name}: invalid flag verdict.")
                if f["verdict"] == "revised" and not f.get("revised_detail"):
                    sys.exit(f"{output.name}: revised flag requires revised_detail.")
        verdicts += actual
    index = {(v.get("area") or v.get("package"), v["key"]): v for v in verdicts}
    stats = {"rules_checked": 0, "rules_correct": 0, "rules_revised": 0, "rules_removed": 0,
             "flags_checked": 0, "flags_confirmed": 0, "flags_revised": 0, "flags_removed": 0}
    for v in verdicts:
        stats["rules_correct"] += v.get("rule_verdict") == "correct"
        for fv in v.get("flags", []):
            stats["flags_checked"] += 1
            stats["flags_confirmed"] += fv["verdict"] == "confirmed"
    for (aid, _), m in mods.items():
        keep = []
        for r in m["rules"]:
            v = index.get((aid, r["key"]))
            if v:
                stats["rules_checked"] += 1
                r["verified"] = True
                if v.get("rule_verdict") == "wrong" and not v.get("rule_revision"):
                    stats["rules_removed"] += 1
                    continue
                if v.get("rule_revision"):
                    r.update(v["rule_revision"])
                    stats["rules_revised"] += 1
                new_flags = []
                for i, fl in enumerate(r["flags"]):
                    fv = next((x for x in v.get("flags", []) if x["index"] == i), None)
                    if fv and fv["verdict"] == "refuted":
                        stats["flags_removed"] += 1
                        continue
                    if fv and fv.get("revised_detail"):
                        fl = {**fl, "detail": fv["revised_detail"]}
                        if fv.get("revised_evidence"):
                            fl["evidence"] = fv["revised_evidence"]
                        stats["flags_revised"] += 1
                    if fv:
                        fl["verified"] = True
                    new_flags.append(fl)
                r["flags"] = new_flags
            keep.append(r)
        m["rules"] = keep
    return stats


def glossary(work, repo, sc):
    if (work / "glossary.json").exists():
        return load_json(work / "glossary.json")
    g = sc.get("glossary")
    if not g:
        return {}
    if g["format"] == "other":
        return {}
    text = (repo / g["path"]).read_text(errors="replace")
    out = {}
    if g["format"] == "bold-colon":
        for m in re.finditer(r"^\*\*(.+?)\*\*:\n(.*?)(?=\n\n|\Z)", text, re.S | re.M):
            out[m.group(1)] = " ".join(re.sub(r"\n_Avoid_:.*", "", m.group(2), flags=re.S).split())
    else:
        for m in re.finditer(r"^#{2,4} (.+?)\n+(.*?)(?=\n\n|\n#|\Z)", text, re.S | re.M):
            out[m.group(1).strip()] = " ".join(m.group(2).split())
    return out


def folder_and_name(common, module):
    """Split an area-relative module string into its tree folder (below `common`) and its leaf name."""
    head = re.split(r"\s*[,(]", module, maxsplit=1)[0].strip().rstrip("/")
    parent = str(PurePosixPath(head).parent)
    parent = "" if parent == "." else parent
    rel_parent = parent[len(common):].lstrip("/") if common and under(parent, common) else parent
    name = module.rstrip("/")[len(parent) + 1:] if parent else module.rstrip("/")
    return rel_parent, name


def cmd_build(args):
    work = work_dir(args)
    run = run_info(work)
    sc = scope_of(work)
    repo = repo_of(run)
    cmd_check(args)
    areas_by_id = {a["id"]: a for a in sc["areas"] if a["tier"] != "skip"}
    mods, area_summary = {}, {}
    for _, d in load_outputs(work):
        for p in d["areas"]:
            aid = p["area"]
            if aid not in areas_by_id:
                sys.exit(f"outputs name area {aid!r}, which scope.json lacks; run `br.py check`")
            if p.get("summary"):
                area_summary[aid] = p["summary"]
            for m in p["modules"]:
                key = (aid, module_path(m["module"], areas_by_id[aid]))
                if key in mods:
                    sys.exit(f"duplicate module {key}; run `br.py check`")
                mods[key] = m
    verification = apply_verification(work, mods)

    rules, modules, areas, key_to_id = [], [], [], {}
    for a in sc["areas"]:
        if a["tier"] == "skip":
            continue
        aid = a["id"]
        mine = sorted(k for k in mods if k[0] == aid)
        rel = {}
        for k in mine:
            root = next((p for p in a["paths"] if under(k[1], p)), "")
            rel[k] = k[1][len(root):].lstrip("/") if root not in ("", ".") else k[1]
        parents = [str(PurePosixPath(re.split(r"\s*[,(]", rel[k], maxsplit=1)[0].rstrip("/")).parent) for k in mine]
        parents = ["" if p == "." else p for p in parents]
        common = os.path.commonpath(parents) if parents and all(parents) else ""
        n, mod_ids = 0, []
        for k in mine:
            m = mods[k]
            rids = []
            for r in m["rules"]:
                n += 1
                rid = f"BR-{a['code']}-{n:03d}"
                key_to_id.setdefault((aid, r["key"]), rid)
                rids.append(rid)
                rules.append({**r, "id": rid, "area": aid, "module": k[1]})
            folder, name = folder_and_name(common, rel[k])
            modules.append({"id": k[1], "package": aid, "path": k[1], "folder": folder, "name": name,
                            "title": m["title"], "stage": m["stage"], "summary": m.get("summary") or "", "rules": rids})
            mod_ids.append(k[1])
        areas.append({"id": aid, "code": a["code"], "title": a["title"], "tier": a["tier"], "paths": a["paths"],
                      "summary": a.get("summary") or area_summary.get(aid, ""), "modules": mod_ids})

    by_key = defaultdict(list)
    for (aid, key), rid in key_to_id.items():
        by_key[key].append(rid)
    unresolved = 0
    for r in rules:
        out = set()
        for ref in r.get("related") or []:
            if ref.startswith("BR-"):
                out.add(ref)
                continue
            rid = key_to_id.get(tuple(ref.split(":", 1))) if ":" in ref else \
                key_to_id.get((r["area"], ref)) or (by_key.get(ref) or [None])[0]
            if rid and rid != r["id"]:
                out.add(rid)
            else:
                unresolved += 1
        r["related"] = sorted(out)

    gl = glossary(work, repo, sc)
    catalogue = {
        "meta": {"title": sc["title"], "project": sc["project"], "generated": run["generated"], "commit": run["commit"],
                 "links": sc.get("links", run.get("links")),
                 "counts": {"rules": len(rules), "modules": len(modules), "areas": len(areas),
                            "flags": sum(len(r["flags"]) for r in rules)},
                 "verification": verification, "unresolved_related": unresolved},
        "stages": sc["stages"], "flagTypes": sc["flagTypes"], "surfaceKinds": sc["surfaceKinds"],
        "glossary": {t: gl[t] for t in sorted({t for r in rules for t in r.get("terms") or [] if t in gl})},
        "areas": areas, "modules": modules, "rules": rules,
    }
    save_json(work / "build" / "catalogue.json", catalogue)
    write_site(work, catalogue)


def write_site(work, catalogue):
    surfaces, surface_ix, files, file_ix = [], {}, [], {}

    def surface_index(s):
        t = (s.get("kind"), s.get("name") or "", s.get("ref") or "")
        if t not in surface_ix:
            surface_ix[t] = len(surfaces)
            surfaces.append(list(t))
        return surface_ix[t]

    def file_index(path):
        if path not in file_ix:
            file_ix[path] = len(files)
            files.append(path)
        return file_ix[path]

    compact = []
    for r in catalogue["rules"]:
        c = {"id": r["id"], "m": r["module"], "t": r["title"], "st": r["stage"], "k": r["kind"],
             "s": r["statement"], "aw": r.get("applies_when") or "", "ex": r.get("exceptions") or [],
             "eg": r.get("example") or "", "ov": r.get("on_violation") or "", "tm": r.get("terms") or [],
             "su": [surface_index(s) for s in r.get("surfaces") or []],
             "en": [[ROLE_ALIASES.get(e.get("role"), e.get("role")), file_index(e.get("file", "")),
                     e.get("symbol") or "", str(e.get("lines") or "")] for e in r.get("engineering") or []],
             "fl": [[f["type"], f.get("detail", ""), f.get("evidence") or [], 1 if f.get("verified") else 0]
                    for f in r.get("flags") or []],
             "re": r["related"], "c": r.get("confidence", "high")}
        if r.get("verified"):
            c["v"] = 1
        compact.append(c)

    # Each data file calls BR_DATA.put(chunkNo, kind, startIndex, items). The app counts loaded chunks
    # against `expected` and shows a banner when one is missing, so a lost upload cannot hide rules.
    chunks = []
    for kind, items in (("modules", catalogue["modules"]), ("surfaces", surfaces), ("files", files), ("rules", compact)):
        start, cur, size = 0, [], 0
        for i, it in enumerate(items):
            b = len(js(it))
            if cur and size + b > CHUNK_BYTES:
                chunks.append((kind, start, cur))
                start, cur, size = i, [], 0
            cur.append(it)
            size += b
        if cur:
            chunks.append((kind, start, cur))

    site = work / "site"
    shutil.rmtree(site, ignore_errors=True)
    (site / "data").mkdir(parents=True)
    core = {k: catalogue[k] for k in ("meta", "stages", "flagTypes", "surfaceKinds", "glossary")}
    core.update({"packages": catalogue["areas"], "modules": [], "surfaces": [], "files": [], "rules": [],
                 "loaded": [], "expected": len(chunks)})
    data = {"core.js": "window.BR_DATA=" + js(core) + ";\n"
            "BR_DATA.put=function(n,k,s,a){for(var i=0;i<a.length;i++)this[k][s+i]=a[i];this.loaded.push(n)};\n"}
    for n, (kind, start, items) in enumerate(chunks, 1):
        data[f"d{n:02d}.js"] = f"BR_DATA.put({n},{js(kind)},{start},{js(items)});\n"
    for name, text in data.items():
        (site / "data" / name).write_text(text)

    template = (SKILL / "assets" / "app" / "index.html").read_text()
    if "<!-- BR:DATA-SCRIPTS -->" not in template:
        sys.exit("assets/app/index.html lost its <!-- BR:DATA-SCRIPTS --> marker")
    tags = "\n".join(f'<script src="data/{f}"></script>' for f in data)
    (site / "index.html").write_text(template.replace("<!-- BR:DATA-SCRIPTS -->", tags))
    app = (SKILL / "assets" / "app" / "app.js").read_text()
    css = (SKILL / "assets" / "app" / "styles.css").read_text()
    (site / "app.js").write_text(app)
    (site / "styles.css").write_text(css)

    # One self-contained page for hosts that take a single HTML file or for sending as an attachment.
    def inline(text):
        return re.sub(r"</(script)", r"<\\/\1", text, flags=re.I)
    single = template.replace('<link rel="stylesheet" href="styles.css">', f"<style>\n{css}</style>")
    single = single.replace("<!-- BR:DATA-SCRIPTS -->", "\n".join(f"<script>{inline(t)}</script>" for t in data.values()))
    single = single.replace('<script src="app.js"></script>', f"<script>{inline(app)}</script>")
    (work / "bundle.html").write_text(single)

    print(json.dumps(catalogue["meta"], indent=1))
    for f in ["index.html", "app.js", "styles.css"] + [f"data/{x}" for x in data]:
        text = (site / f).read_text()
        print(f"{f:16} {len(text):7} {fnv(text)}")
    print(f"bundle.html      {len(single):7}")


# ---------- note ----------

def cell(text):
    """One line of prose. Escape `<` outside code spans so a docs site does not read it as HTML."""
    parts = " ".join(str(text).split()).split("`")
    return "`".join(p if i % 2 else p.replace("<", "&lt;") for i, p in enumerate(parts))


def listing(names):
    names = [f"`{n}`" for n in names]
    return " and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + " and " + names[-1]


def cmd_note(args):
    work = work_dir(args)
    run = run_info(work)
    sc = scope_of(work)
    d = load_json(work / "build" / "catalogue.json")
    meta, rules = d["meta"], d["rules"]
    mods = {m["id"]: m for m in d["modules"]}
    by_id = {r["id"]: r for r in rules}
    stage_label = {s["id"]: s["label"] for s in d["stages"]}
    flag_label = {f["id"]: f["label"] for f in d["flagTypes"]}
    commit, short, v, links = meta["commit"], meta["commit"][:9], meta["verification"], meta.get("links")
    project = meta["project"]["name"]
    choice = run.get("choices", {}).get("note")
    out_path = Path(args.out) if args.out else Path(run["repo"]) / choice if choice and choice != "work" else work / "note.md"
    share = args.share_url or run.get("choices", {}).get("share_url")

    def ref(ev, link=True):
        url = code_url(links, commit, ev) if link else None
        return f"[`{ev.strip()}`]({url})" if url else f"`{ev.strip()}`"

    commit_md = f"[`{short}`]({fill(links['commit'], commit=commit)})" if links else f"`{short}`"
    counts = Counter(f["type"] for r in rules for f in r["flags"])
    business = [a for a in d["areas"] if a["tier"] == "business"]
    platform = [a for a in d["areas"] if a["tier"] == "platform"]
    skipped = [a for a in sc["areas"] if a["tier"] == "skip"]
    docs = sc.get("docs") or []
    lines = []
    w = lines.append
    w(f'---\ntitle: "What business rules does {project} enforce?"\n---\n')
    w(f"# What business rules does {project} enforce?\n")
    w(f"Primary-source research deriving the business rules that {project}'s code enforces, read from function "
      f"bodies, queries, guards, defaults and constants at commit {commit_md} ({meta['generated']}). Comments and "
      f"documentation{' (' + listing(docs) + ')' if docs else ''} were used only to detect disagreement: where they "
      f"differ from the code, the code is recorded as the rule and the disagreement is flagged.\n")
    if share:
        w(f"The same catalogue is published as a reviewable explorer at [{share}]({share}). Business owners mark each "
          f"rule Correct, Wrong or Unsure there and export corrections for engineering. This note is the citable "
          f"record behind it.\n")
    w("## 1. Scope and method\n")
    w(f"- **{meta['counts']['rules']} rules** across **{meta['counts']['modules']} modules** in "
      f"**{len(d['areas'])} areas**.")
    w("- Read at full depth: " + "; ".join(f"{a['title']} ({listing(a['paths'])})" for a in business) + ".")
    if platform:
        w("- Read as a summary branch, recording only rules a business owner would care about: "
          + "; ".join(f"{a['title']} ({listing(a['paths'])})" for a in platform) + ".")
    if skipped:
        w("- Left out on purpose: " + "; ".join(f"{a['title']} ({listing(a['paths'])})"
                                                + (f", {cell(a['summary'])}" if a.get("summary") else "")
                                                for a in skipped) + ".")
    w("- A rule is one decision that changes a data outcome or rejects input (validation, default, derivation, "
      "matching, precedence, state transition, threshold, ordering, naming, authorization), or an operational "
      "guarantee a user feels (idempotency, re-run behaviour, partial failure, ordering). Plumbing with no business "
      "effect is left out unless it silently changes an outcome.")
    w("- Each rule lives with the module holding the condition that decides it. When that condition is query or "
      "template text, the rule lives with the text, and the code that runs it is cited as `executes`.")
    w("- Readers each owned a disjoint file set. Each rule cites its deciding code as `path:line`, and each "
      "\"where you see this\" surface was traced from the enforcing code's callers.")
    w("- *Inferred* marks rules that needed inference across calls the reader did not fully trace.\n")
    w("## 2. How far to trust a rule\n")
    w(f"A second, independent reader re-checked **{v['rules_checked']} rule{'' if v['rules_checked'] == 1 else 's'}** "
      f"against the code: every rule carrying "
      f"a *contradicts docs* or *inconsistent* flag, plus a random sample of unflagged rules from each reader.\n")
    w(f"- Rules: {v['rules_correct']} correct as written, {v['rules_revised']} needed a correction, "
      f"{v['rules_removed']} wrong. The corrections are applied in this note.")
    w(f"- Flags: of {v['flags_checked']} re-checked, {v['flags_confirmed']} held, {v['flags_revised']} were reworded "
      f"and {v['flags_removed']} were refuted and removed.")
    if v["rules_checked"]:
        rate = round(100 * v["rules_revised"] / v["rules_checked"])
        w(f"- Rules not re-checked carry the first reader's wording. In the re-checked set {rate}% needed a "
          "correction. The set combines targeted flagged rules with sampled unflagged rules; it does not "
          "estimate the error rate among unchecked rules.\n")
    w("## 3. Flags at a glance\n")
    w("| Flag | Meaning | Count |\n|---|---|---|")
    for f in d["flagTypes"]:
        w(f"| {f['label']} | {f['description']} | {counts.get(f['id'], 0)} |")
    w("")
    w("## 4. Where the code disagrees with the docs or with itself\n")
    w("Every rule carrying a *contradicts docs* or *inconsistent* flag, all re-checked by a second reader. These are "
      "the decisions a business owner most needs to make: which side is right.\n")
    for i, t in enumerate(("contradicts-docs", "inconsistent"), 1):
        w(f"### 4.{i} {flag_label.get(t, t)}\n")
        hits = [(r, f) for r in rules for f in r["flags"] if f["type"] == t]
        for r, f in hits:
            ev = ", ".join(ref(e) for e in f.get("evidence") or [])
            w(f"- **{r['id']}** {cell(r['title'])}: {cell(f['detail'])}" + (f" ({ev})" if ev else ""))
        w("" if hits else "None.\n")
    w("## 5. Rule catalogue\n")
    w(f"Grouped by area, then module. Paths are repository-relative `path:line` citations at commit `{short}`.\n")
    for a in d["areas"]:
        w(f"### {a['title']} ({a['code']})\n")
        if a["summary"]:
            w(cell(a["summary"]) + "\n")
        for mid in a["modules"]:
            m = mods[mid]
            w(f"#### {cell(m['title'])} · `{m['path']}`\n")
            w(f"*Stage: {stage_label.get(m['stage'], m['stage'])}.* {cell(m['summary'])}\n")
            if not m["rules"]:
                w("No business rules: this module decides nothing a business owner reviews.\n")
            for rid in m["rules"]:
                r = by_id[rid]
                tags = [stage_label.get(r["stage"], r["stage"]), r["kind"]]
                tags += ["inferred"] if r.get("confidence") == "medium" else []
                tags += ["second read"] if r.get("verified") else []
                w(f"##### {r['id']} {cell(r['title'])}\n")
                w(f"*{' · '.join(tags)}*\n")
                w(cell(r["statement"]) + "\n")
                w(f"- **Applies when:** {cell(r.get('applies_when') or '')}")
                ex = r.get("exceptions") or []
                w(f"- **Exceptions:** {'; '.join(cell(x) for x in ex) if ex else 'none'}")
                if r.get("example"):
                    w(f"- **Example:** {cell(r['example'])}")
                if r.get("on_violation"):
                    w(f"- **If violated:** {cell(r['on_violation'])}")
                for f in r["flags"]:
                    ev = ", ".join(ref(e, link=False) for e in f.get("evidence") or [])
                    w(f"- **Flag, {flag_label.get(f['type'], f['type'])}:** {cell(f['detail'])}" + (f" ({ev})" if ev else ""))
                surf = [s for s in r.get("surfaces") or [] if s.get("kind") != "internal"]
                if surf:
                    w("- **Where you see this:** " + "; ".join(f"{s['kind']} {cell(s['name'])}" for s in surf))
                eng = []
                for e in r.get("engineering") or []:
                    loc = f"{e['file']}:{e['lines']}" if e.get("lines") else e["file"]
                    sym = f" `{e['symbol']}`" if e.get("symbol") else ""
                    eng.append(f"{ROLE_ALIASES.get(e.get('role'), e.get('role'))}{sym} `{loc}`")
                w("- **Enforced at:** " + "; ".join(eng))
                w("")
    w("## 6. Provenance\n")
    w(f"- Commit: {commit_md}.")
    w("- Produced with the `business-rules-catalogue` skill: parallel reader agents over disjoint file sets, then "
      "independent checker agents. Rule IDs (`BR-<area code>-<n>`) hold for this commit only; a run against a later "
      "commit renumbers them.")
    text = "\n".join(lines) + "\n"
    odd = [i for i, line in enumerate(text.split("\n"), 1) if line.count("`") % 2]
    if odd:
        sys.exit(f"unbalanced backticks on lines {odd[:10]}; fix the rule text in the reader output")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(f"{out_path} ({len(text)} bytes, {len(rules)} rules)")


# ---------- publish-prep ----------

def record_lines(value, depth):
    """JSON with one record per line down to `depth` levels, compact below. Helpers paste it verbatim."""
    if depth == 0 or not isinstance(value, (list, dict)) or not value:
        return js(value)
    if isinstance(value, list):
        return "[\n" + ",\n".join(record_lines(x, depth - 1) for x in value) + "\n]"
    return "{\n" + ",\n".join(js(k) + ":" + record_lines(x, depth - 1) for k, x in value.items()) + "\n}"


def cmd_publish_prep(args):
    work = work_dir(args)
    run = run_info(work)
    dest = run.get("choices", {}).get("destination")
    audience = run.get("choices", {}).get("audience")
    if not dest or dest == "local" or audience not in {"private", "shared", "public"}:
        sys.exit("No hosted destination and audience are recorded. Ask the user (references/destinations.md), then run "
                 "`br.py record --destination <host> --audience <audience>`.")
    site, pub = work / "site", work / "publish"
    for f in pub.glob("*.json"):
        f.unlink()
    data = [f"data/{p.name}" for p in sorted((site / "data").glob("*.js"))]
    sizes = {f: len((site / f).read_text()) for f in data}
    groups = []  # first-fit decreasing: large files alone, small ones share a group
    for f in sorted(data, key=lambda x: -sizes[x]):
        for g in groups:
            if sum(sizes[x] for x in g) + sizes[f] <= GROUP_BYTES:
                g.append(f)
                break
        else:
            groups.append([f])

    plan, placeholders = [], {}
    for gi, g in enumerate(groups, 1):
        items = []
        for f in g:
            text = (site / f).read_text()
            name = Path(f).name
            if name == "core.js":
                m = re.match(r"(window\.BR_DATA=)(.*)(;\nBR_DATA\.put=function.*\n)$", text, re.S)
            else:
                m = re.match(r"(BR_DATA\.put\(\d+,\"\w+\",\d+,)(.*)(\);\n)$", text, re.S)
            if not m or "".join(m.groups()) != text:
                sys.exit(f"{f} does not have the shape build writes")
            prefix, payload, suffix = m.groups()
            if js(json.loads(payload)) != payload:
                sys.exit(f"{f}: payload does not round-trip through JSON")
            payload_path = pub / f"{name}.payload.json"
            payload_path.write_text(record_lines(json.loads(payload), 2 if name == "core.js" else 1) + "\n")
            placeholders[f] = f"/*PENDING {name}*/"
            items.append({"file": f, "payload": str(payload_path), "prefix": prefix, "suffix": suffix,
                          "placeholder": placeholders[f], "fnv": fnv(text), "bytes": len(text)})
        plan.append({"group": gi, "items": items})
    save_json(pub / "plan.json", plan)

    shell = []
    for f in ("index.html", "styles.css", "app.js"):
        text = (site / f).read_text()
        lines_path = pub / f"{f}.lines.json"
        lines_path.write_text(json.dumps(text.split("\n"), ensure_ascii=False, indent=0) + "\n")
        shell.append({"file": f, "lines": str(lines_path), "fnv": fnv(text), "bytes": len(text)})
    save_json(pub / "shell.json", {"files": shell, "placeholders": placeholders})

    expected = {x["file"]: x["fnv"] for x in shell}
    expected.update({it["file"]: it["fnv"] for g in plan for it in g["items"]})
    (pub / "verify.js").write_text(
        'import { read_document } from "docstash"\n'
        'function fnv(s) { let h = 0x811c9dc5; for (const ch of s) { h ^= ch.codePointAt(0); '
        'h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16).padStart(8, "0"); }\n'
        'const slug = "SLUG_HERE";\n'
        f"const expected = {json.dumps(expected)};\n"
        "const bad = [];\n"
        "for (const [file, want] of Object.entries(expected)) {\n"
        "  const d = await read_document({ slug, file });\n"
        '  if (fnv(d.content || "") !== want) bad.push(file);\n'
        "}\n"
        "text({ allOk: bad.length === 0, checked: Object.keys(expected).length, bad });\n")

    if shutil.which("node"):  # prove JavaScript rebuilds every file byte-for-byte from what helpers will paste
        script = (
            "const fs=require('fs');function fnv(s){let h=0x811c9dc5;for(const ch of s){h^=ch.codePointAt(0);"
            "h=Math.imul(h,0x01000193)>>>0;}return h.toString(16).padStart(8,'0');}"
            f"const plan=JSON.parse(fs.readFileSync({json.dumps(str(pub / 'plan.json'))}));"
            f"const shell=JSON.parse(fs.readFileSync({json.dumps(str(pub / 'shell.json'))}));let bad=[];"
            "for(const g of plan)for(const it of g.items){const c=it.prefix+JSON.stringify(JSON.parse("
            "fs.readFileSync(it.payload,'utf8')))+it.suffix;if(fnv(c)!==it.fnv)bad.push(it.file);}"
            "for(const f of shell.files){if(fnv(JSON.parse(fs.readFileSync(f.lines,'utf8')).join('\\n'))!==f.fnv)"
            "bad.push(f.file);}console.log(JSON.stringify(bad));")
        bad = json.loads(sh("node", "-e", script))
        if bad:
            sys.exit(f"JavaScript does not rebuild these files byte-for-byte: {bad}")
        print("node round-trip: every file rebuilds with the expected hash")
    for g in plan:
        print(f"group {g['group']:2}: " + ", ".join(f"{it['file']} ({it['bytes']})" for it in g["items"]))
    print(f"audience: {run['choices'].get('audience', 'NOT RECORDED')}; verify snippet: {pub / 'verify.js'}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--work", help="the work directory `init` printed")
        p.set_defaults(fn=fn)
        return p

    p = add("init", cmd_init, "pin the commit and create the work directory")
    p.add_argument("--repo", default=".", help="any path inside the target repository (default: the current directory)")
    p.add_argument("--scope-dir", help=f"a saved scope to load (default: <repo>/{SAVED_DIR})")
    add("survey", cmd_survey, "map the repository and draft scope.json, or validate it")
    p = add("record", cmd_record, "record the user's delivery choices")
    p.add_argument("--destination", help="`local`, or the host the user chose, such as `docstash`")
    p.add_argument("--audience", choices=["private", "shared", "public", "local"])
    p.add_argument("--note", help="repository-relative path for the research note, or `work`")
    p.add_argument("--save-scope", dest="save_scope", help="repository-relative directory for the saved scope, or `none`")
    p.add_argument("--share-url", dest="share_url", help="the link the destination returned")
    p = add("partition", cmd_partition, "propose reader assignments and render prompts")
    p.add_argument("--target-lines", type=int, default=10000)
    p.add_argument("--force", action="store_true", help="replace partition.json with a fresh proposal")
    add("check", cmd_check, "validate reader outputs")
    p = add("batches", cmd_batches, "write second-read batches")
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--per-reader", type=int, default=5, help="random unflagged rules sampled per reader")
    add("build", cmd_build, "write catalogue.json, site/ and bundle.html")
    p = add("note", cmd_note, "write the research note")
    p.add_argument("--out", help="output path (default: the recorded note choice, else <work>/note.md)")
    p.add_argument("--share-url", help="the explorer's link, once delivered")
    add("publish-prep", cmd_publish_prep, "split site/ into upload groups for a paste-only host")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
