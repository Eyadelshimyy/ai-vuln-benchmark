#!/usr/bin/env python3
"""Regenerate docs/api/inventory-<date>.csv.

Inventories every public callable exported by ``python/fast_mlsirm``
(top-level package surface, ``_legacy_init.py`` re-exports, and every
submodule) plus the PyO3 entry points compiled from
``crates/fast-mlsirm-py``. Static analysis only: does not require a built
Rust extension (a stub ``fast_mlsirm._core`` module is injected so the
Python package still imports) and does not invoke cargo/maturin.

Usage:
    python tools/inventory_public_api.py [--date YYYYMMDD]
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import pkgutil
import re
import sys
import types
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY_ROOT = REPO_ROOT / "python"
RUST_SRC = REPO_ROOT / "crates" / "fast-mlsirm-py" / "src"
FIELDS = ["source", "current_name", "module", "kind", "parameters"]


def _install_core_stub() -> None:
    """Inject a fake ``fast_mlsirm._core`` so the package imports without cargo."""

    class _FakeCore(types.ModuleType):
        def __getattr__(self, name: str):  # noqa: ANN001 - dynamic stub
            def _stub(*_args, **_kwargs):
                raise RuntimeError("fast_mlsirm._core stub: Rust extension not built")

            return _stub

    sys.modules.setdefault("fast_mlsirm._core", _FakeCore("fast_mlsirm._core"))


def _format_default(value: object) -> str:
    if value is inspect.Parameter.empty:
        return ""
    try:
        return repr(value)
    except Exception:
        return "<unrepr-able>"


def _format_params(sig: inspect.Signature) -> str:
    parts = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{name}")
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{name}")
        else:
            default = _format_default(p.default)
            parts.append(f"{name}={default}" if default else name)
    return ", ".join(parts)


def collect_python_rows() -> list[dict]:
    sys.path.insert(0, str(PY_ROOT))
    _install_core_stub()
    import fast_mlsirm  # noqa: F401  (import after path/stub setup)

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def visit(obj_name: str, obj: object) -> None:
        if obj_name.startswith("_"):
            return
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            return
        module = getattr(obj, "__module__", "") or ""
        if not module.startswith("fast_mlsirm"):
            return
        qualname = getattr(obj, "__qualname__", obj_name)
        key = (module, qualname)
        if key in seen:
            return
        seen.add(key)
        kind = "function" if inspect.isfunction(obj) else "class"
        try:
            params = _format_params(inspect.signature(obj))
        except (ValueError, TypeError):
            params = "<no-signature>"
        rows.append(
            {
                "source": "python",
                "current_name": obj_name,
                "module": module,
                "kind": kind,
                "parameters": params,
            }
        )

    # Top-level package surface (includes _legacy_init re-exports pulled
    # into fast_mlsirm/__init__.py).
    for name in sorted(dir(fast_mlsirm)):
        visit(name, getattr(fast_mlsirm, name))

    # Every submodule's own public names, since not everything reaches the
    # top-level namespace (e.g. polytomous.py helpers used internally by
    # other public functions but still importable/public by convention).
    for _finder, modname, _ispkg in pkgutil.walk_packages(
        fast_mlsirm.__path__, prefix="fast_mlsirm."
    ):
        if any(part.startswith("_") for part in modname.split(".")):
            continue
        try:
            # modname comes from pkgutil.walk_packages over fast_mlsirm.__path__,
            # so the only importable values are this package's own installed
            # submodules. There is no caller-supplied input on this path, and
            # this file is a repository tool that is not shipped in the wheel.
            # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name in sorted(vars(mod)):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if getattr(obj, "__module__", None) != modname:
                continue  # skip names imported from elsewhere; own module only
            visit(name, obj)

    rows.sort(key=lambda r: (r["module"], r["current_name"]))
    return rows


_PYMODULE_RE = re.compile(
    r'#\[pymodule\]\s*(?:#\[pyo3\(name\s*=\s*"([^"]+)"\)\]\s*)?'
    r"(?:pub\s+)?fn\s+(\w+)\s*\([^)]*\)\s*->\s*PyResult<\(\)>\s*\{"
)
_WRAP_PYFUNCTION_RE = re.compile(r"wrap_pyfunction!\((\w+)")
_SIGNATURE_ATTR_RE = re.compile(r"#\[pyo3\(signature\s*=\s*\((.*?)\)\)\]", re.DOTALL)
_FN_DEF_RE = re.compile(r"fn\s+{name}\s*(?:<[^>]*>)?\s*\((.*?)\)\s*(?:->|\{{)", re.DOTALL)


def _extract_braced_block(text: str, open_brace_index: int) -> str:
    depth = 0
    for i in range(open_brace_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index : i + 1]
    return text[open_brace_index:]


def _rust_params_from_signature_attr(params_blob: str) -> str:
    parts = [p.strip() for p in params_blob.strip().split(",") if p.strip()]
    return ", ".join(parts)


def _rust_params_from_fn_decl(params_blob: str) -> str:
    parts = []
    depth = 0
    current = ""
    for ch in params_blob:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    names = []
    for p in parts:
        p = p.strip()
        if not p or p in ("&self", "self"):
            continue
        name = p.split(":", 1)[0].strip()
        names.append(name)
    return ", ".join(names)


def collect_rust_rows() -> list[dict]:
    rows: list[dict] = []
    for rs_file in sorted(RUST_SRC.glob("*.rs")):
        text = rs_file.read_text()
        for match in _PYMODULE_RE.finditer(text):
            pymodule_name = match.group(1) or match.group(2)
            body = _extract_braced_block(text, match.end() - 1)
            fn_names = _WRAP_PYFUNCTION_RE.findall(body)
            for fn_name in fn_names:
                fn_def_match = re.search(
                    _FN_DEF_RE.pattern.format(name=re.escape(fn_name)), text, re.DOTALL
                )
                if fn_def_match is None:
                    params = "<not-found>"
                else:
                    preceding = text[: fn_def_match.start()]
                    sig_match = None
                    for sig_match in _SIGNATURE_ATTR_RE.finditer(preceding):
                        pass  # take the last (closest) match before fn decl
                    # Nothing but whitespace/other attrs sits between the closest
                    # #[pyo3(signature = (...))] and the "fn <name>(" it governs.
                    if sig_match is not None and preceding[sig_match.end() :].strip() in (
                        "",
                        "#[allow(clippy::too_many_arguments)]",
                    ):
                        params = _rust_params_from_signature_attr(sig_match.group(1))
                    else:
                        params = _rust_params_from_fn_decl(fn_def_match.group(1))
                rows.append(
                    {
                        "source": "pyo3",
                        "current_name": fn_name,
                        "module": f"fast_mlsirm.{pymodule_name}",
                        "kind": "function",
                        "parameters": params,
                    }
                )
    rows.sort(key=lambda r: (r["module"], r["current_name"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    args = parser.parse_args()

    rows = collect_python_rows() + collect_rust_rows()

    out_dir = REPO_ROOT / "docs" / "api"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inventory-{args.date}.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
