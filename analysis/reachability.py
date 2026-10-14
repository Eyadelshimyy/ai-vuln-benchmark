#!/usr/bin/env python3
"""VULCAN entry-point REACHABILITY analysis (scope- & import-aware call graph).

The disclosure question is not "is this function vulnerable" (Tier-2 proved that)
but "can attacker input get here through the program's REAL interface?" -- a CLI
command, an HTTP route, an exported API function. This builds a call graph and
reports a path from such an entry point to a confirmed sink, so a human can trace
it by hand.

PRECISION (why this is not a naive name match): a bare name-based graph that keys
edges by function name and unions callees across every same-named function wildly
over-connects on large OOP codebases -- e.g. it will "reach" an RCE sink from a
`/health` route through unrelated `get`/`run`/`_execute` methods that merely share
a name. To avoid that, each function is a QUALIFIED node (file::Class.method), and
a call is resolved only when scope makes it unambiguous:
  * `name()`            -> a def of `name` in the SAME file, else a name explicitly
                           imported into this file (`from m import name`)
  * `self.m()`/`cls.m()`-> a method `m` of the SAME class
  * `mod.f()`           -> `f`, only if `mod` is a module imported in this file
  * anything else (an arbitrary `obj.attr()` with an unknown receiver type) is
    DROPPED -- conservative: it loses some real edges (noted as a recall caveat)
    but it does not fabricate the collision chains that make a worklist unusable.

It remains a static over-approximation; a reported path is a lead to verify by
hand (does attacker input actually reach the sink parameter -- the taint-source
question reachability does not answer), and "unreachable" strongly suggests the
sink is internal/dead. Both are the right conservative behaviours for disclosure.

USAGE
  python3 analysis/reachability.py --root <repo-or-_repo_src> --function do_thing
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
from pathlib import Path

_ROUTE_HINTS = ("route", "get", "post", "put", "delete", "patch", "websocket",
                "api_route", "on_event", "endpoint", "view", "add_url_rule")
_CLI_HINTS = ("command", "cli", "group")
_PRIO = {"http": 0, "cli": 1, "api": 2, "main": 3}


def _is_test_path(rel: str) -> bool:
    low = rel.replace("\\", "/").lower()
    parts = low.split("/")
    base = parts[-1]
    return (any(p in ("test", "tests", "testing") for p in parts[:-1])
            or base.startswith("test_") or base.endswith("_test.py")
            or base == "conftest.py")


def _decorator_names(node):
    for d in getattr(node, "decorator_list", []):
        cur = d.func if isinstance(d, ast.Call) else d
        while isinstance(cur, ast.Attribute):
            yield cur.attr.lower()
            cur = cur.value
        if isinstance(cur, ast.Name):
            yield cur.id.lower()


def _entry_kind(node, is_module_level: bool):
    decs = list(_decorator_names(node))
    if any(h in _ROUTE_HINTS for h in decs):
        return "http"
    if any(h in _CLI_HINTS for h in decs):
        return "cli"
    if node.name == "main":
        return "main"
    if is_module_level and not node.name.startswith("_"):
        return "api"
    return None


class Graph:
    def __init__(self):
        self.edges: dict[str, set] = defaultdict(set)   # qid -> set(qid)
        self.entries: dict[str, str] = {}               # qid -> kind
        self.name_of: dict[str, str] = {}               # qid -> bare name
        self.by_name: dict[str, list] = defaultdict(list)  # name -> [qid]
        self.class_method: dict[tuple, list] = defaultdict(list)  # (ClassName, method) -> [qid]
        self.classes: set = set()                       # class names defined anywhere


def analyze_repo(root: Path) -> Graph:
    root = Path(root)
    files = [p for p in root.rglob("*.py") if not _is_test_path(str(p.relative_to(root)))]

    # ---- pass 1: index every function as a qualified node ----
    g = Graph()
    parsed = {}
    # per-file: module-level def name -> qid ; (class,method) -> qid ; imported names ; imported module aliases
    file_funcs: dict[str, dict] = {}
    file_methods: dict[str, dict] = {}
    file_imports: dict[str, set] = {}
    file_modaliases: dict[str, set] = {}
    func_nodes = []  # (rel, qid, node, class_name)

    for p in files:
        rel = str(p.relative_to(root))
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        parsed[rel] = tree
        file_funcs[rel] = {}
        file_methods[rel] = {}
        imports, modaliases = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    imports.add(a.asname or a.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    modaliases.add((a.asname or a.name).split(".")[0])
        file_imports[rel] = imports
        file_modaliases[rel] = modaliases

        def visit(node, cls):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    g.classes.add(child.name)
                    visit(child, child.name)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qid = f"{rel}::{cls + '.' if cls else ''}{child.name}"
                    g.name_of[qid] = child.name
                    g.by_name[child.name].append(qid)
                    if cls is None:
                        file_funcs[rel][child.name] = qid
                    else:
                        file_methods[rel][(cls, child.name)] = qid
                        g.class_method[(cls, child.name)].append(qid)
                    k = _entry_kind(child, cls is None)
                    if k and g.entries.get(qid) not in ("http", "cli"):
                        g.entries[qid] = k
                    func_nodes.append((rel, qid, child, cls))
                    visit(child, cls)
                else:
                    visit(child, cls)
        visit(tree, None)

    # ---- pass 2: resolve each function's calls to qualified targets ----
    for rel, qid, node, cls in func_nodes:
        funcs = file_funcs.get(rel, {})
        methods = file_methods.get(rel, {})
        imports = file_imports.get(rel, set())
        modaliases = file_modaliases.get(rel, set())
        for n in ast.walk(node):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            if isinstance(f, ast.Name):
                nm = f.id
                if nm in funcs:
                    g.edges[qid].add(funcs[nm])
                elif nm in imports:
                    for t in g.by_name.get(nm, []):
                        g.edges[qid].add(t)
            elif isinstance(f, ast.Attribute):
                attr = f.attr
                v = f.value
                if isinstance(v, ast.Name) and v.id in ("self", "cls") and cls is not None:
                    if (cls, attr) in methods:
                        g.edges[qid].add(methods[(cls, attr)])
                elif isinstance(v, ast.Name) and v.id in modaliases:
                    for t in g.by_name.get(attr, []):
                        g.edges[qid].add(t)
                else:
                    # ClassName().method() / ClassName.method() -> resolve to that
                    # class's methods by name (constructor receiver is a known class).
                    recv_cls = None
                    if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in g.classes:
                        recv_cls = v.func.id
                    elif isinstance(v, ast.Name) and v.id in g.classes:
                        recv_cls = v.id
                    if recv_cls and (recv_cls, attr) in g.class_method:
                        for t in g.class_method[(recv_cls, attr)]:
                            g.edges[qid].add(t)
                    # else: unknown-receiver obj.attr() -> dropped (conservative)
    return g


def reaches(root: Path, target_func: str, max_depth: int = 15):
    return reaches_with_graph(analyze_repo(root), target_func, max_depth=max_depth)


def reaches_with_graph(g: Graph, target_func: str, max_depth: int = 15):
    """(reachable, path_of_names, entry_kind) for the best (http>cli>api>main) path."""
    targets = set(g.by_name.get(target_func, []))
    if not targets:
        return False, [], None

    best = None  # (prio, length, [qids], kind)
    for tq in targets:
        if tq in g.entries:
            k = g.entries[tq]
            cand = (_PRIO.get(k, 9), 1, [tq], k)
            if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                best = cand

    for entry, kind in g.entries.items():
        prio = _PRIO.get(kind, 9)
        if best and prio > best[0]:
            continue
        seen = {entry}
        q = deque([(entry, [entry])])
        found = None
        while q:
            cur, path = q.popleft()
            if len(path) > max_depth:
                continue
            for callee in g.edges.get(cur, ()):
                if callee in targets:
                    found = path + [callee]
                    break
                if callee not in seen:
                    seen.add(callee)
                    q.append((callee, path + [callee]))
            if found:
                break
        if found:
            cand = (prio, len(found), found, kind)
            if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                best = cand

    if best:
        return True, [g.name_of.get(q, q) for q in best[2]], best[3]
    return False, [], None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--function", required=True)
    ap.add_argument("--file", default=None)
    args = ap.parse_args()
    ok, path, kind = reaches(args.root, args.function)
    if ok:
        print(f"REACHABLE via {kind} entry point:\n    " + " -> ".join(path))
    else:
        print(f"NOT reachable from any detected entry point: {args.function!r}")
        print("    (internal / dead / test-only, OR reached only via an unknown-receiver "
              "call this scope-aware graph conservatively dropped -- verify by hand)")


if __name__ == "__main__":
    main()
