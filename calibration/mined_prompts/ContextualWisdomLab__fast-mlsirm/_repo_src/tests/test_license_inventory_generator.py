"""Negative regression tests for tools/license_inventory.py (fail-closed behaviour)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "license_inventory.py"
MIT_TEXT = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
MIT_WITH_HEADER = MIT_TEXT.replace(
    "MIT License\n\n", "MIT License\n\nCopyright (c) 2026 Test Author\n\n", 1
)
LGPL_TEXT = "This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License"


def _module():
    """Load the generator from its file path, as the other tools tests do."""
    spec = importlib.util.spec_from_file_location("license_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _module()

REVIEWED = json.loads((Path(__file__).parent / "fixtures/license_inventory_reviewed_texts.json").read_text())


@pytest.mark.parametrize("row", REVIEWED, ids=lambda r: r["package"])
def test_reviewed_whole_license_bytes_and_mutations(row):
    text = row["text"]
    identifiers = row.get("identifiers", [row.get("identifier")])
    assert hashlib.sha256(text.encode()).hexdigest() == row["raw_sha256"]
    normalized = L.re.sub(r"[ \t\r\n]+", " ", text).strip(" \t\r\n")
    assert hashlib.sha256(normalized.encode()).hexdigest() == row["normalized_sha256"]
    assert L.verified_standard_text(text) == identifiers
    assert L.verified_standard_text(text.replace("\n", "\r\n")) == identifiers
    word = next(w for w in ("License", "software", "Redistribution") if w in text)
    for changed in (text.replace(word, "Restriction", 1),
                    text + "\nCommercial use is prohibited.", "extra\n" + text, text + "\nextra"):
        assert changed != text
        assert L.verified_standard_text(changed) == []


def test_bsd_non_endorsement_clause_is_detected_case_insensitively():
    colorama = next(row["text"] for row in REVIEWED if row["package"] == "colorama@0.4.6")
    assert L.detect(colorama) == ["BSD-3-Clause"]
    assert L.detect(colorama.replace("Neither the name", "NEITHER THE NAME")) == ["BSD-3-Clause"]


@pytest.mark.parametrize("extra", [None, "COPYRIGHT", "missing-mit", "restricted", "LGPL"])
def test_reviewed_alternative_preserves_all_candidate_validation(tmp_path, extra):
    args = _rust_args(tmp_path)
    apache = next(r["text"] for r in REVIEWED if r["identifier"] == "Apache-2.0")
    files = {"LICENSE-MIT": L.MIT_CANONICAL_BODY, "LICENSE-APACHE": apache}
    if extra == "COPYRIGHT":
        files["COPYRIGHT"] = "Except as otherwise noted in individual files; some code derives from libstd."
    elif extra == "missing-mit":
        del files["LICENSE-MIT"]
    elif extra == "restricted":
        files["LICENSE-APACHE"] += "\nCommercial use is prohibited."
    elif extra == "LGPL":
        files["vendor/COPYING"] = LGPL_TEXT
    digest = _crate(Path(args.cargo_registry_cache), "dep", "1.0", files)
    lock = Path(args.cargo_lock_workspace)
    content = lock.read_text()
    old = L.tomllib.loads(content)["package"][0]["checksum"]
    lock.write_text(content.replace(old, digest))
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == ("PERMISSIVE" if extra is None else "HOLD")
    if extra == "missing-mit":
        assert row["elected_text_present_in_artifact"] is False


@pytest.mark.parametrize("expr", ["LicenseRef-unreviewed", "NOASSERTION", "NONE", "", "   ", "MIT AND", "(MIT", "MIT WITH Bogus-exception"])
def test_unreviewed_or_malformed_expressions_are_unknown(expr):
    """Unregistered ids, sentinels and grammar errors never classify as permissive."""
    assert L.classify(expr) == "UNKNOWN"
    elected, rationale = L.elect(expr)
    assert elected is None and rationale


def test_failed_disjunction_election_propagates_unknown():
    """A disjunction without a fully permissive branch cannot yield 'None AND MIT'."""
    expr = "(GPL-3.0-only OR LGPL-2.1-only) AND MIT"
    assert L.classify(expr) == "COPYLEFT"
    assert L.elect(expr) == (None, L.ELECTION_FAILED)


def test_permissive_election_keeps_conjunct_and_copyleft_alternative_is_declared():
    """Valid elections still work; a copyleft alternative stays visible as declared class."""
    assert L.elect("(MIT OR Apache-2.0) AND Unicode-3.0")[0] == "MIT AND Unicode-3.0"
    assert L.elect("MIT OR Apache-2.0 OR LGPL-2.1-or-later")[0] == "MIT"
    assert L.classify("MIT OR Apache-2.0 OR LGPL-2.1-or-later") == "COPYLEFT"
    assert L.classify("GPL-3.0-or-later WITH GCC-exception-3.1") == "COPYLEFT"


def _wheel(path: Path, name: str, version: str, license_text: str | None) -> str:
    """Write a minimal wheel and return its sha256."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{name}-{version}.dist-info/METADATA", f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n")
        if license_text is not None:
            zf.writestr(f"{name}-{version}.dist-info/licenses/LICENSE", license_text)
    path.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _native_wheel(
    path: Path,
    *,
    native_license: str | None,
    bind_notice: bool = True,
    license_body: str = "",
) -> str:
    """Write the minimal shape used by real wheels with vendored native notices."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg.libs/libexample.so", b"synthetic native bytes")
        if bind_notice:
            zf.writestr(
                "pkg-1.0.dist-info/licenses/NOTICE",
                "Name: native-component\nFiles: pkg.libs/libexample.so\n"
                f"License: {native_license or ''}\n"
                + "\n".join(f" {line}" if line else " ." for line in license_body.splitlines())
                + "\n",
            )
    path.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _python_args(tmp_path: Path, *, expression, license_text, lock_hash=None, scope=True, requirements="") -> Namespace:
    """Build python_inventory inputs for one package 'pkg 1.0'."""
    art = tmp_path / "art"
    meta = tmp_path / "meta"
    art.mkdir()
    meta.mkdir()
    digest = None
    if license_text is not False:
        digest = _wheel(art / "pkg-1.0-py3-none-any.whl", "pkg", "1.0", license_text)
    (meta / "pkg-1.0.json").write_text(json.dumps({"info": {"license_expression": expression, "classifiers": []}, "urls": []}))
    uv = tmp_path / "uv.lock"
    uv.write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\nsource = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{lock_hash or digest or "0" * 64}" }}]\n'
    )
    req = tmp_path / "requirements.txt"
    req.write_text(requirements)
    scope_file = tmp_path / "scope.json"
    scope_file.write_text(json.dumps({"pkg": {"scope": "dev", "in_published_artifact_scope": False}} if scope else {}))
    return Namespace(uv_lock=str(uv), requirements=[str(req)], extra_python=[], python_scope=str(scope_file),
                     pypi_meta_dir=str(meta), pypi_artifact_dir=str(art))


def test_declared_license_without_artifact_text_is_hold(tmp_path):
    """Metadata MIT with zero examined artifacts must not read as PERMISSIVE."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=False), gaps)
    assert row["license_class"] == "HOLD"
    assert row["license_determination"]["text_agrees"] is False


def test_copyleft_text_is_kept_when_metadata_is_absent(tmp_path):
    """No metadata plus MIT and LGPL text holds, keeping LGPL instead of dropping it."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression=None, license_text=MIT_TEXT + "\n" + LGPL_TEXT), gaps)
    assert row["license_class"] == "UNKNOWN"
    assert row["elected_license"] is None
    assert "LGPL" in row["license_determination"]["copyleft_text_not_in_declared_expression"]
    assert row["hold_reasons"]


def test_copyleft_text_outside_declared_expression_is_hold(tmp_path):
    """Declared MIT with LGPL grant text in the artifact is HOLD, not PERMISSIVE."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=MIT_TEXT + "\n" + LGPL_TEXT), gaps)
    assert row["license_class"] == "HOLD"
    assert row["license_determination"]["copyleft_text_not_in_declared_expression"] == ["LGPL"]


def test_artifact_hash_mismatch_is_hold_and_text_unused(tmp_path):
    """An artifact whose sha256 is not the lock's expected hash is not evidence."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=MIT_TEXT, lock_hash="f" * 64), gaps)
    assert row["artifacts_examined"][0]["hash_binding"]["match"] is False
    assert row["license_text_detected_in_artifact"] == []
    assert row["license_class"] == "HOLD"


def test_bound_artifact_with_matching_text_is_permissive(tmp_path):
    """Control: hash-bound artifact whose text matches the declared expression passes."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=MIT_TEXT), gaps)
    assert (row["license_class"], row["hold_reasons"], gaps) == ("PERMISSIVE", [], [])


@pytest.mark.parametrize(
    "native_license",
    [
        "LGPL-2.1-or-later",
        "GPL-3.0-or-later WITH GCC-exception-3.1",
        "AGPL-3.0-only",
        "LicenseRef-unreviewed",
        None,
    ],
)
def test_vendored_native_nonpermissive_or_unknown_license_holds(tmp_path, native_license):
    """Permissive package metadata cannot hide a bound native component policy failure."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(wheel, native_license=native_license)
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["vendored_native_license_holds"]


def test_vendored_native_without_bound_notice_holds(tmp_path):
    """A native wheel member with no matching notice stanza is not silently accepted."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(wheel, native_license=None, bind_notice=False)
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "no bound notice stanza" in " ".join(row["vendored_native_license_holds"])


def test_vendored_native_verified_permissive_election_remains_allowed(tmp_path):
    """A hash-bound native component may elect its verified permissive alternative."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(
        wheel,
        native_license="MIT OR LGPL-2.1-or-later",
        license_body=MIT_TEXT,
    )
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["vendored_native_license_holds"] == []
    assert row["license_class"] == "PERMISSIVE"


@pytest.mark.parametrize("native_license", ["MIT", "MIT OR LGPL-2.1-or-later"])
def test_vendored_native_label_without_verified_grant_holds(tmp_path, native_license):
    """A hash-bound SPDX label alone does not verify a native component grant."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(wheel, native_license=native_license)
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "no verified elected grant" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize("native_license", ["MIT", "MIT OR LGPL-2.1-or-later"])
def test_vendored_native_additional_condition_holds(tmp_path, native_license):
    """Canonical MIT text plus a commercial-use restriction is not permissive evidence."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(
        wheel,
        native_license=native_license,
        license_body=MIT_TEXT + "\nAdditional restriction: commercial use prohibited.",
    )
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "no verified elected grant" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize("native_license", ["MIT", "MIT OR LGPL-2.1-or-later"])
def test_vendored_native_pre_license_condition_holds(tmp_path, native_license):
    """A condition before License is part of the notice and invalidates verification."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    _native_wheel(wheel, native_license=native_license, license_body=MIT_TEXT)
    result = io.BytesIO()
    with zipfile.ZipFile(wheel) as old, zipfile.ZipFile(result, "w") as new:
        for entry in old.infolist():
            data = old.read(entry.filename)
            if entry.filename.endswith("/NOTICE"):
                data = data.replace(
                    b"License: ",
                    b"Description: Additional restriction: commercial use prohibited\nLicense: ",
                    1,
                )
            new.writestr(entry.filename, data)
    wheel.write_bytes(result.getvalue())
    digest = hashlib.sha256(result.getvalue()).hexdigest()
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "no verified elected grant" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize("native_license", ["0BSD", "CC0-1.0", "BSD-3-Clause-Open-MPI"])
def test_vendored_native_unfingerprinted_label_without_grant_holds(tmp_path, native_license):
    """A native component needs scope-bound permission evidence even for these ids."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(wheel, native_license=native_license)
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "no verified elected grant" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize(
    "needle,replacement",
    [
        (b"Name: native-component", b"Unscoped permission statement\nName: native-component"),
        (b"Files: pkg.libs/libexample.so", b"Files: pkg.libs/libexample.so\n continuation"),
        (b"License: MIT", b"Intermediate statement\nLicense: MIT"),
        (b"License: MIT", b"Name: duplicate\nFiles: pkg.libs/libexample.so\nLicense: MIT"),
        (b" MIT License", b" MIT License\nUnindented middle statement"),
        (b" SOFTWARE.\n", b" SOFTWARE.\nTrailing statement\n"),
    ],
)
def test_vendored_native_notice_rejects_unparsed_nonblank_bytes(tmp_path, needle, replacement):
    """Every nonblank notice byte must belong to the verified header/body grammar."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    _native_wheel(wheel, native_license="MIT", license_body=MIT_TEXT)
    result = io.BytesIO()
    with zipfile.ZipFile(wheel) as old, zipfile.ZipFile(result, "w") as new:
        for entry in old.infolist():
            data = old.read(entry.filename)
            if entry.filename.endswith("/NOTICE"):
                assert needle in data
                data = data.replace(needle, replacement, 1)
            new.writestr(entry.filename, data)
    wheel.write_bytes(result.getvalue())
    digest = hashlib.sha256(result.getvalue()).hexdigest()
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["vendored_native_license_holds"]


@pytest.mark.parametrize("case", ["separate-condition", "unmatched-prefix", "invalid-second-match"])
def test_vendored_native_all_candidate_files_and_stanzas_must_verify(tmp_path, case):
    """Package acceptance consumes every candidate file and every native stanza."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    _native_wheel(
        wheel,
        native_license="MIT OR LGPL-2.1-or-later",
        license_body=MIT_TEXT,
    )
    result = io.BytesIO()
    with zipfile.ZipFile(wheel) as old, zipfile.ZipFile(result, "w") as new:
        for entry in old.infolist():
            data = old.read(entry.filename)
            if entry.filename.endswith("/NOTICE") and case == "unmatched-prefix":
                data = (
                    b"Unparsed package-wide condition\n"
                    b"Name: unrelated\nFiles: unused.so\nLicense: MIT\n .\n"
                    + data
                )
            elif entry.filename.endswith("/NOTICE") and case == "invalid-second-match":
                data += (
                    b"Name: extra\nFiles: pkg.libs/libexample.so\nLicense: MIT\n"
                    b" Unverified condition\n"
                )
            new.writestr(entry.filename, data)
        if case == "separate-condition":
            new.writestr(
                "pkg-1.0.dist-info/licenses/NOTICE-CONDITIONS",
                "Unparsed package-wide condition\n",
            )
    wheel.write_bytes(result.getvalue())
    digest = hashlib.sha256(result.getvalue()).hexdigest()
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["vendored_native_license_holds"]
    assert "not fully verified" in " ".join(row["hold_reasons"])


def test_vendored_native_unmatched_verified_stanza_holds_package(tmp_path):
    """A valid grant for a nonexistent binary is still unconsumed artifact evidence."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    _native_wheel(wheel, native_license="MIT", license_body=MIT_TEXT)
    result = io.BytesIO()
    with zipfile.ZipFile(wheel) as old, zipfile.ZipFile(result, "w") as new:
        for entry in old.infolist():
            data = old.read(entry.filename)
            if entry.filename.endswith("/NOTICE"):
                body = "\n".join(f" {line}" if line else " ." for line in MIT_TEXT.splitlines())
                data += (
                    b"Name: absent-component\nFiles: absent.so\nLicense: MIT\n"
                    + body.encode()
                    + b"\n"
                )
            new.writestr(entry.filename, data)
    wheel.write_bytes(result.getvalue())
    digest = hashlib.sha256(result.getvalue()).hexdigest()
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "not fully verified" in " ".join(row["hold_reasons"])


def _replace_wheel(args, writer):
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        writer(zf)
    wheel.write_bytes(buf.getvalue())
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )


def test_metadata_declared_custom_license_file_is_consumed(tmp_path):
    """A nonstandard License-File cannot hide a package-wide restriction."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\nLicense-File: terms.txt\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg-1.0.dist-info/licenses/terms.txt", "Commercial use is prohibited.\n")

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "terms.txt" in " ".join(row["hold_reasons"])


@pytest.mark.parametrize(
    "header",
    [
        "License-File:terms.txt",
        "license-file: terms.txt",
        "License-File:\tterms.txt",
        "License-File:\n terms.txt",
    ],
)
def test_metadata_license_file_uses_standard_header_parsing(tmp_path, header):
    """Whitespace, case, and folding cannot make a declared file disappear."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr(
            "pkg-1.0.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n{header}\n\n",
        )
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg-1.0.dist-info/licenses/terms.txt", "Commercial use is prohibited.\n")

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "terms.txt" in " ".join(row["hold_reasons"])


def test_metadata_parser_defect_holds(tmp_path):
    """Malformed metadata cannot fall back to a permissive partial parse."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr(
            "pkg-1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: pkg\nMalformed header\nLicense-File: terms.txt\n\n",
        )
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg-1.0.dist-info/licenses/terms.txt", MIT_TEXT)

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "parser reported defects" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize("declared", ["missing.txt", "..", "../LICENSE", "/LICENSE", r"dir\LICENSE"])
def test_invalid_or_missing_metadata_license_file_holds(tmp_path, declared):
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr("pkg-1.0.dist-info/METADATA", f"Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\nLicense-File: {declared}\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["vendored_native_license_holds"]


def test_duplicate_wheel_member_holds_before_last_entry_can_mask_first(tmp_path):
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg-1.0.dist-info/licenses/NOTICE", "Commercial use is prohibited.\n")
        zf.writestr("pkg-1.0.dist-info/licenses/NOTICE", MIT_TEXT)

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "duplicate wheel member" in " ".join(row["vendored_native_license_holds"])


@pytest.mark.parametrize("pattern,expected", [("pkg.libs/libexample.so", "HOLD"), ("pkg.libs/libexample*.so", "PERMISSIVE")])
def test_native_files_scope_is_exact_unless_notice_declares_glob(tmp_path, pattern, expected):
    args = _python_args(tmp_path, expression="MIT", license_text=False)

    def writer(zf):
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr("pkg.libs/libexample_unrelated.so", b"synthetic")
        body = "\n".join(f" {line}" if line else " ." for line in MIT_TEXT.splitlines())
        zf.writestr("pkg-1.0.dist-info/licenses/NOTICE", f"Name: component\nFiles: {pattern}\nLicense: MIT OR LGPL-2.1-or-later\n{body}\n")

    _replace_wheel(args, writer)
    (row,) = L.python_inventory(args, [])
    assert row["license_class"] == expected
    assert bool(row["vendored_native_license_holds"]) is (expected == "HOLD")


def test_missing_scope_is_unclassified_gap_not_exclusion(tmp_path):
    """A package without a scope entry is UNCLASSIFIED with unknown scope, and a gap."""
    gaps = []
    (row,) = L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=MIT_TEXT, scope=False), gaps)
    assert row["scope"] == "UNCLASSIFIED"
    assert row["in_published_artifact_scope"] is None
    assert any("no scope entry" in g for g in gaps)


@pytest.mark.parametrize("line", ["other==2.0\n", "other>=2.0\n", "-e .\n", "other @ https://x/other.whl\n"])
def test_requirements_lines_that_do_not_parse_are_gaps(tmp_path, line):
    """Unhashed pins and non-pin requirement lines are reported, not silently skipped."""
    gaps = []
    L.python_inventory(_python_args(tmp_path, expression="MIT", license_text=MIT_TEXT, requirements=line), gaps)
    assert any("requirements.txt" in g for g in gaps)


def _crate(cache: Path, name: str, version: str, files: dict[str, str]) -> str:
    """Write a .crate (tar.gz) into a registry-cache layout and return its sha256."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for fname, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{name}-{version}/{fname}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    (cache / "index.crates.io").mkdir(parents=True, exist_ok=True)
    (cache / "index.crates.io" / f"{name}-{version}.crate").write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _rust_args(tmp_path: Path, *, lock_checksum_override=None, drop_from_metadata=False) -> Namespace:
    """Build rust_inventory inputs with one registry crate 'dep 1.0'."""
    cache = tmp_path / "cache"
    digest = _crate(cache, "dep", "1.0", {"LICENSE-MIT": MIT_TEXT, "Cargo.toml": ""})
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        f'source = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "{lock_checksum_override or digest}"\n'
    )
    packages = [] if drop_from_metadata else [{
        "name": "dep", "version": "1.0", "source": "registry+https://github.com/rust-lang/crates.io-index",
        "license": "MIT OR Apache-2.0", "license_file": None, "manifest_path": "/nonexistent/Cargo.toml",
    }]
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"packages": packages}))
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"components": []}))
    tree = tmp_path / "tree"
    tree.mkdir()
    return Namespace(cargo_metadata_workspace=str(meta), cargo_metadata_binding=str(meta), cargo_lock_workspace=str(lock),
                     cargo_lock_binding=str(lock), cargo_registry_cache=str(cache), wheel_sbom=str(sbom), tree_dir=str(tree))


def test_crate_license_text_is_read_only_from_hash_bound_crate(tmp_path):
    """Matching .crate bytes: text comes from the archive and the row passes."""
    gaps = []
    (row,) = L.rust_inventory(_rust_args(tmp_path), gaps)
    assert row["source_hash"]["match"] is True
    assert [f["path"] for f in row["license_files_in_artifact"]] == ["LICENSE-MIT"]
    assert row["license_files_in_artifact"][0]["artifact_sha256"] == row["source_hash"]["measured"]
    assert (row["license_class"], gaps) == ("PERMISSIVE", [])


def test_crate_hash_mismatch_is_hold_and_text_unread(tmp_path):
    """A .crate that does not match the Cargo.lock checksum is not evidence."""
    gaps = []
    (row,) = L.rust_inventory(_rust_args(tmp_path, lock_checksum_override="0" * 64), gaps)
    assert row["source_hash"]["match"] is False
    assert row["license_files_in_artifact"] == []
    assert row["license_class"] == "HOLD"
    assert row["value_origin"].startswith("UNVERIFIED")


def test_lock_entry_missing_from_metadata_is_a_gap_and_unknown(tmp_path):
    """Every lock package yields a row; a metadata/lock mismatch is reported."""
    gaps = []
    (row,) = L.rust_inventory(_rust_args(tmp_path, drop_from_metadata=True), gaps)
    assert row["license_class"] == "UNKNOWN"
    assert any("not in cargo metadata" in g for g in gaps)


def test_registry_crate_without_license_file_is_hold(tmp_path):
    """Cargo metadata alone cannot replace license text from the exact archive."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"Cargo.toml": ""})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_files_in_artifact"] == []
    assert row["license_class"] == "HOLD"
    assert "contains no license file" in " ".join(row["hold_reasons"])


def test_rust_source_named_copying_is_not_a_license_unless_declared(tmp_path):
    cache = tmp_path / "cache"
    _crate(cache, "dep", "1.0", {"src/copying.rs": "fn copying() {}"})
    archive = next(cache.glob("*/dep-1.0.crate")).read_bytes()
    assert L.read_crate_license_files(archive, hashlib.sha256(archive).hexdigest(), None)[0] == []
    files, errors = L.read_crate_license_files(archive, hashlib.sha256(archive).hexdigest(), "src/copying.rs")
    assert not errors
    assert [f["path"] for f in files] == ["src/copying.rs"]


def test_cargo_metadata_mit_does_not_hide_separate_lgpl_text(tmp_path):
    """A permissive declaration cannot erase an additional copyleft grant."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE-MIT": MIT_TEXT, "COPYING.LESSER": LGPL_TEXT})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert "LGPL" in " ".join(row["hold_reasons"])


def test_crate_archive_is_read_once_for_hash_and_license_text(tmp_path, monkeypatch):
    """The bytes hashed are the same bytes used to extract license evidence."""
    args = _rust_args(tmp_path)
    original = L.read_stable_bytes
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(L, "read_stable_bytes", counted)
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "PERMISSIVE"
    assert len(calls) == 1


def test_crate_path_replacement_during_read_is_hold(tmp_path, monkeypatch):
    """Replacing the source path after open cannot retain a trusted hash binding."""
    args = _rust_args(tmp_path)
    crate = next(Path(args.cargo_registry_cache).glob("*/dep-1.0.crate"))
    original_open = Path.open

    class ReplacingReader:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return self.handle.__exit__(*exc)
        def fileno(self):
            return self.handle.fileno()
        def read(self):
            data = self.handle.read()
            replacement = crate.with_suffix(".replacement")
            replacement.write_bytes(data)
            replacement.replace(crate)
            return data

    def replacing_open(path, *open_args, **open_kwargs):
        handle = original_open(path, *open_args, **open_kwargs)
        return ReplacingReader(handle) if path == crate else handle

    monkeypatch.setattr(Path, "open", replacing_open)
    (row,) = L.rust_inventory(args, [])
    assert row["source_hash"]["match"] is False
    assert row["license_class"] == "HOLD"
    assert "source path was not stable" in " ".join(row["hold_reasons"])


@pytest.mark.parametrize(
    "license_text",
    [
        "UNKNOWN",
        "Commercial redistribution is prohibited. Permission is not granted.",
        "Permission is hereby granted, free of charge, but this grant does not apply. "
        'THE SOFTWARE IS PROVIDED "AS IS". IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE.',
    ],
)
def test_cargo_permissive_metadata_requires_verified_unqualified_text(tmp_path, license_text):
    """Names, restrictions, and negated standard phrases are not permissive evidence."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": license_text})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


def test_nested_uppercase_lgpl_grant_is_collected_and_holds(tmp_path):
    """Nested candidate paths and case variants cannot hide copyleft text."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    uppercase_lgpl = LGPL_TEXT.upper()
    digest = _crate(
        cache,
        "dep",
        "1.0",
        {"LICENSE-MIT": MIT_TEXT, "vendor/other/COPYING": uppercase_lgpl},
    )
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert {f["path"] for f in row["license_files_in_artifact"]} == {
        "LICENSE-MIT", "vendor/other/COPYING"
    }
    assert row["license_class"] == "HOLD"
    assert "LGPL" in " ".join(row["hold_reasons"])


def test_declared_nested_license_file_is_collected(tmp_path):
    """Cargo metadata's LicenseFile path is evidence even without a license-like basename."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"legal/grant.txt": MIT_TEXT})
    meta_path = Path(args.cargo_metadata_workspace)
    metadata = json.loads(meta_path.read_text())
    metadata["packages"][0]["license_file"] = "legal/grant.txt"
    meta_path.write_text(json.dumps(metadata))
    Path(args.cargo_metadata_binding).write_text(meta_path.read_text())
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "PERMISSIVE"
    assert row["license_files_in_artifact"][0]["path"] == "legal/grant.txt"
    assert row["license_files_in_artifact"][0]["declared_license_file"] is True


def test_unknown_notice_alongside_mit_is_hold(tmp_path):
    """An unparsed candidate file remains UNKNOWN even when MIT text is present."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE-MIT": MIT_TEXT, "NOTICE": "UNKNOWN"})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["elected_text_present_in_artifact"] is True
    assert row["license_class"] == "HOLD"
    assert "unrecognized" in " ".join(row["hold_reasons"])


def test_full_mit_text_with_additional_condition_is_hold(tmp_path):
    """A canonical grant plus an extra restriction is not the canonical MIT license."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    restricted = MIT_TEXT + "\nAdditional condition: Use is permitted solely for academic research.\n"
    digest = _crate(cache, "dep", "1.0", {"LICENSE": restricted})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


def test_missing_declared_license_file_is_hold_even_with_root_mit(tmp_path):
    """A different root license cannot substitute for Cargo metadata's missing path."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": MIT_TEXT})
    meta_path = Path(args.cargo_metadata_workspace)
    metadata = json.loads(meta_path.read_text())
    metadata["packages"][0]["license_file"] = "legal/terms.txt"
    meta_path.write_text(json.dumps(metadata))
    Path(args.cargo_metadata_binding).write_text(meta_path.read_text())
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["elected_text_present_in_artifact"] is True
    assert row["license_class"] == "HOLD"
    assert "license_file is absent" in " ".join(row["hold_reasons"])


def test_positive_phrase_only_is_not_verified_for_other_supported_license(tmp_path):
    """A known Apache phrase is detection evidence, not a complete verified grant."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": "Apache License Version 2.0"})
    meta_path = Path(args.cargo_metadata_workspace)
    metadata = json.loads(meta_path.read_text())
    metadata["packages"][0]["license"] = "Apache-2.0"
    meta_path.write_text(json.dumps(metadata))
    Path(args.cargo_metadata_binding).write_text(meta_path.read_text())
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


def test_copyright_header_cannot_hide_an_additional_condition(tmp_path):
    """A copyright-looking line is unverified content, not removable syntax."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    tainted = MIT_TEXT.replace(
        "MIT License\n\n",
        "MIT License\n\nCopyright (c) 2026 Test Author Use is permitted solely for academic research\n\n",
        1,
    )
    digest = _crate(cache, "dep", "1.0", {"LICENSE": tainted})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


@pytest.mark.parametrize(
    "header",
    [
        "Copyright (c) 2026 Test Author",
        "Copyright (c) 2026 Test Author Commercial use is prohibited",
    ],
)
def test_unreviewed_copyright_header_is_hold(tmp_path, header):
    """No inferred author-name grammar may erase an unreviewed header line."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    text = MIT_TEXT.replace("MIT License\n\n", f"MIT License\n\n{header}\n\n", 1)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": text})
    lock = (
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_workspace).write_text(lock)
    Path(args.cargo_lock_binding).write_text(lock)
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


def test_canonical_mit_body_without_unreviewed_header_remains_permissive(tmp_path):
    """The exact canonical body remains the bounded positive control."""
    args = _rust_args(tmp_path)
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "PERMISSIVE"
    assert row["elected_text_present_in_artifact"] is True


def test_prior_copyright_header_fixture_now_holds(tmp_path):
    """Record the intentional PERMISSIVE-to-HOLD change for the old fixture."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": MIT_WITH_HEADER})
    lock = (
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_workspace).write_text(lock)
    Path(args.cargo_lock_binding).write_text(lock)
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert row["elected_text_present_in_artifact"] is False


@pytest.mark.parametrize(
    "other_name,other_text",
    [
        ("COPYING", MIT_TEXT + "\nAdditional condition: Use is permitted solely for academic research.\n"),
        ("NOTICE", "Apache License Version 2.0"),
    ],
)
def test_verified_mit_cannot_hide_an_unverified_candidate_file(tmp_path, other_name, other_text):
    """Every collected candidate is independently verified or the row is held."""
    args = _rust_args(tmp_path)
    cache = Path(args.cargo_registry_cache)
    digest = _crate(cache, "dep", "1.0", {"LICENSE": MIT_TEXT, other_name: other_text})
    Path(args.cargo_lock_workspace).write_text(
        'version = 4\n[[package]]\nname = "dep"\nversion = "1.0"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        f'checksum = "{digest}"\n'
    )
    Path(args.cargo_lock_binding).write_text(Path(args.cargo_lock_workspace).read_text())
    (row,) = L.rust_inventory(args, [])
    assert row["elected_text_present_in_artifact"] is True
    assert row["license_class"] == "HOLD"
    assert other_name in " ".join(row["hold_reasons"])


LGPL_NOTICE_BODY = (
    "This library is free software; you can redistribute it and/or\n"
    "modify it under the terms of the GNU Lesser General Public\n"
    "License as published by the Free Software Foundation; either\n"
    "version 2.1 of the License, or (at your option) any later version."
)


def _lgpl_native_wheel(args, member: str, files_pattern: str) -> None:
    """A NumPy-shaped wheel: vendored libquadmath described by a generic ``*.so`` notice."""
    def writer(zf):
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr(member, b"synthetic native bytes")
        body = "\n".join(f" {line}" for line in LGPL_NOTICE_BODY.splitlines())
        zf.writestr(
            "pkg-1.0.dist-info/licenses/NOTICE",
            f"Name: libquadmath\nFiles: {files_pattern}\nLicense: LGPL-2.1-or-later\n{body}\n",
        )

    _replace_wheel(args, writer)


@pytest.mark.parametrize(
    "member,pattern",
    [
        ("pkg.libs/libquadmath-2284e583-a9307bba.so.0.0.0", "pkg.libs/libquadmath*.so"),
        ("pkg/.dylibs/libquadmath.0.dylib", "pkg/.dylibs/libquadmath*.so"),
    ],
)
def test_versioned_soname_and_dylib_bind_lgpl_notice_and_hold(tmp_path, member, pattern):
    """Real wheels ship versioned/dylib names under a generic *.so notice; LGPL must stay visible."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    _lgpl_native_wheel(args, member, pattern)
    (row,) = L.python_inventory(args, [])
    (native,) = [v for a in row["artifacts_examined"] for v in a["vendored_native"] if v["path"] == member]
    assert [s["license"] for s in native["notice_stanza"]] == ["LGPL-2.1-or-later"]
    holds = " ".join(row["vendored_native_license_holds"])
    assert f"{member}: no bound notice stanza" not in holds
    assert f"{member}: vendored native license is LGPL-2.1-or-later" in holds
    assert row["license_class"] == "HOLD"


@pytest.mark.parametrize(
    "member,pattern",
    [
        ("pkg.libs/libquadmath.so.0", "pkg.libs/libquadmath.so"),  # exact pattern stays exact
        ("pkg.libs/libgfortran-83c28eba.so.5.0.0", "pkg.libs/libquadmath*.so"),  # other stem
        ("other.libs/libquadmath-2284e583.so.0.0.0", "pkg.libs/libquadmath*.so"),  # other directory
        ("pkg.libs/libquadmath.0.dylib", "pkg.libs/libquadmath*.so"),  # dylib outside .dylibs/
        ("pkg.libs/libquadmath-2284e583.so.0.x", "pkg.libs/libquadmath*.so"),  # not a numeric soname
    ],
)
def test_soname_binding_never_widens_to_unrelated_members(tmp_path, member, pattern):
    """An unmatched native member keeps failing closed instead of borrowing a notice."""
    assert L.notice_files_pattern_matches(member, pattern) is False
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    _lgpl_native_wheel(args, member, pattern)
    (row,) = L.python_inventory(args, [])
    if any(v["path"] == member for a in row["artifacts_examined"] for v in a["vendored_native"]):
        assert f"{member}: no bound notice stanza" in " ".join(row["vendored_native_license_holds"])
    assert row["license_class"] == "HOLD"


def test_notice_for_absent_file_binds_nothing_and_holds(tmp_path):
    """A notice naming a library that is not in the wheel cannot vouch for the one that is."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    _lgpl_native_wheel(args, "pkg.libs/libexample-1a2b.so.1", "pkg.libs/libquadmath*.so")
    (row,) = L.python_inventory(args, [])
    holds = " ".join(row["vendored_native_license_holds"])
    assert "pkg.libs/libexample-1a2b.so.1: no bound notice stanza" in holds
    assert row["license_class"] == "HOLD"


def test_negating_phrase_never_hides_copyleft_text_labels():
    """GPL-3.0 says a clause 'does not apply'; that must not erase GPL/LGPL evidence."""
    text = (
        "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. "
        "This requirement does not apply if neither you nor any third party retains the ability. "
        "Libquadmath is free software; you can redistribute it and/or modify it "
        "under the terms of the GNU Library General Public License."
    )
    labels = L.detect(text)
    assert "GPL" in labels and "LGPL" in labels
    # The same phrase still voids a permissive-looking grant.
    assert "MIT" not in L.detect(
        "Permission is hereby granted, free of charge, but this grant does not apply. "
        'THE SOFTWARE IS PROVIDED "AS IS". IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE.'
    )


def test_affero_mention_in_gpl3_is_not_an_agpl_grant():
    """GPL-3.0 section 13 names the Affero license; only an Affero title or grant is AGPL."""
    gpl3_section_13 = (
        "GNU GENERAL PUBLIC LICENSE Version 3. 13. Use with the GNU Affero General Public License. "
        "You have permission to link or combine any covered work with a work licensed under "
        "version 3 of the GNU Affero General Public License into a single combined work."
    )
    assert "AGPL" not in L.detect(gpl3_section_13)
    assert "GPL" in L.detect(gpl3_section_13)
    assert "AGPL" in L.detect("GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007")
    assert "AGPL" in L.detect("you can redistribute it under the terms of the GNU Affero General Public License")


GCC_RUNTIME_NOTICE_BODY = (
    "Libgfortran is free software; you can redistribute it and/or modify\n"
    "it under the terms of the GNU General Public License as published by\n"
    "the Free Software Foundation; either version 3, or (at your option)\n"
    "any later version.\n"
    "\n"
    "Under Section 7 of GPL version 3, you are granted additional\n"
    "permissions described in the GCC Runtime Library Exception, version\n"
    "3.1, as published by the Free Software Foundation."
)
WINDOWS_OPENBLAS_DLL = "pkg.libs/libscipy_openblas64_-ed4f167a5330424524f45258e7ca2c8d.dll"


def _windows_notice_wheel(args, member: str, files_pattern: str, *, with_permissive_stanza: bool = False) -> None:
    """A NumPy win_amd64-shaped wheel: the notice names its DLL with a backslash path."""
    def stanza(name, license_id, body):
        lines = "\n".join(f" {line}" if line else " ." for line in body.splitlines())
        return f"Name: {name}\nFiles: {files_pattern}\nLicense: {license_id}\n{lines}\n"

    def writer(zf):
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr(member, b"synthetic native bytes")
        notice = stanza("GCC runtime library", "GPL-3.0-or-later WITH GCC-exception-3.1", GCC_RUNTIME_NOTICE_BODY)
        if with_permissive_stanza:
            notice = stanza("OpenBLAS", "MIT", MIT_TEXT) + "\n" + notice
        zf.writestr("pkg-1.0.dist-info/licenses/NOTICE", notice)

    _replace_wheel(args, writer)


@pytest.mark.parametrize("with_permissive_stanza", [False, True])
def test_windows_backslash_notice_binds_gcc_runtime_label_and_holds(tmp_path, with_permissive_stanza):
    """numpy win_amd64 writes 'numpy.libs\\libscipy_openblas*.dll'; the GCC-exception label must bind."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    _windows_notice_wheel(
        args, WINDOWS_OPENBLAS_DLL, "pkg.libs\\libscipy_openblas*.dll", with_permissive_stanza=with_permissive_stanza
    )
    (row,) = L.python_inventory(args, [])
    (native,) = [v for a in row["artifacts_examined"] for v in a["vendored_native"] if v["path"] == WINDOWS_OPENBLAS_DLL]
    licenses = [s["license"] for s in native["notice_stanza"]]
    assert "GPL-3.0-or-later WITH GCC-exception-3.1" in licenses
    holds = " ".join(row["vendored_native_license_holds"])
    assert f"{WINDOWS_OPENBLAS_DLL}: no bound notice stanza" not in holds
    assert f"{WINDOWS_OPENBLAS_DLL}: vendored native license is GPL-3.0-or-later WITH GCC-exception-3.1" in holds
    # A runtime exception never approves the component, even beside a verified permissive stanza.
    assert row["license_class"] == "HOLD"


@pytest.mark.parametrize(
    "pattern",
    [
        "..\\pkg.libs\\libscipy_openblas*.dll",  # traversal
        "pkg.libs\\..\\..\\libscipy_openblas*.dll",  # traversal after a component
        "\\pkg.libs\\libscipy_openblas*.dll",  # absolute root
        "\\\\server\\share\\libscipy_openblas*.dll",  # UNC root
        "C:\\pkg.libs\\libscipy_openblas*.dll",  # drive letter
        "other.libs\\libscipy_openblas*.dll",  # other directory
        "pkg.libs\\msvcp140*.dll",  # other DLL
    ],
)
def test_windows_backslash_notice_never_binds_unsafe_or_unrelated(tmp_path, pattern):
    """Backslash conversion keeps every canonical-path rejection and never widens binding."""
    assert L.notice_files_pattern_matches(WINDOWS_OPENBLAS_DLL, pattern) is False
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    _windows_notice_wheel(args, WINDOWS_OPENBLAS_DLL, pattern)
    (row,) = L.python_inventory(args, [])
    assert f"{WINDOWS_OPENBLAS_DLL}: no bound notice stanza" in " ".join(row["vendored_native_license_holds"])
    assert row["license_class"] == "HOLD"


def test_windows_notice_pattern_normalization_is_notice_side_only():
    """Only notice patterns are converted; the canonical-path gate itself is unchanged."""
    assert L.notice_files_pattern_to_posix("pkg.libs\\libscipy_openblas*.dll") == "pkg.libs/libscipy_openblas*.dll"
    assert L.normalized_archive_path("pkg.libs\\libscipy_openblas64_.dll") is None


def _target_meta(tmp_path: Path, packages: list[tuple[str, str]], nodes: list[str]) -> str:
    path = tmp_path / "target-meta.json"
    path.write_text(json.dumps({
        "packages": [{"id": f"{n}@{v}", "name": n, "version": v} for n, v in packages],
        "resolve": {"nodes": [{"id": i} for i in nodes]},
    }))
    return str(path)


def test_binding_target_graph_marks_rows_and_absent_filter_is_unchanged(tmp_path):
    """Target-filtered metadata marks graph membership; without it rows carry no target field."""
    args = _rust_args(tmp_path)
    (plain,) = L.rust_inventory(args, [])
    assert "in_binding_target_graph" not in plain
    for nodes, expected in ((["dep@1.0"], True), ([], False)):
        args.cargo_metadata_binding_target = _target_meta(tmp_path, [("dep", "1.0")], nodes)
        args.binding_target = "x86_64-unknown-linux-gnu"
        gaps = []
        (row,) = L.rust_inventory(args, gaps)
        assert (row["in_binding_target_graph"], gaps) == (expected, [])
        assert {k: v for k, v in row.items() if k != "in_binding_target_graph"} == plain


def test_binding_target_package_outside_binding_lock_is_a_gap(tmp_path):
    args = _rust_args(tmp_path)
    args.cargo_metadata_binding_target = _target_meta(tmp_path, [("dep", "1.0"), ("ghost", "9.9")], ["dep@1.0", "ghost@9.9"])
    args.binding_target = "x86_64-unknown-linux-gnu"
    gaps = []
    L.rust_inventory(args, gaps)
    assert gaps == ["cargo binding target x86_64-unknown-linux-gnu: ghost@9.9 is not in the binding Cargo.lock"]


def test_binding_target_flags_must_be_paired(tmp_path):
    with pytest.raises(SystemExit):
        L.main(["--cargo-metadata-workspace", "x", "--cargo-metadata-binding", "x", "--cargo-lock-workspace", "x",
                "--cargo-lock-binding", "x", "--cargo-registry-cache", "x", "--wheel-sbom", "x", "--tree-dir", "x",
                "--uv-lock", "x", "--requirements", "x", "--python-scope", "x", "--pypi-meta-dir", "x",
                "--pypi-artifact-dir", "x", "--out", str(tmp_path / "o.json"), "--binding-target", "x86_64-unknown-linux-gnu"])


def _upstream_args(tmp_path: Path, *, crate_sha1="a" * 40, recorded_sha1="a" * 40, text=None, recorded_text=None):
    """dep 1.0 whose hash-bound .crate has no license file, plus an upstream evidence record."""
    args = _rust_args(tmp_path)
    vcs = json.dumps({"git": {"sha1": crate_sha1}, "path_in_vcs": "dep"})
    digest = _crate(Path(args.cargo_registry_cache), "dep", "1.0", {"Cargo.toml": "", ".cargo_vcs_info.json": vcs})
    lock = Path(args.cargo_lock_workspace)
    content = lock.read_text()
    lock.write_text(content.replace(L.tomllib.loads(content)["package"][0]["checksum"], digest))
    text = L.MIT_CANONICAL_BODY if text is None else text
    (tmp_path / "LICENSE-MIT").write_text(text)
    evidence = tmp_path / "upstream.json"
    evidence.write_text(json.dumps({"dep@1.0": {
        "repository": "https://example.invalid/dep", "vcs_sha1": recorded_sha1, "path_in_vcs": "dep",
        "files": [{"path": "LICENSE-MIT", "url": "https://example.invalid/LICENSE-MIT", "local_path": "LICENSE-MIT",
                   "sha256": hashlib.sha256((recorded_text if recorded_text is not None else text).encode()).hexdigest()}],
    }}))
    return args, str(evidence)


def test_upstream_license_evidence_binds_only_with_matching_commit_and_hash(tmp_path):
    args, evidence = _upstream_args(tmp_path)
    (plain,) = L.rust_inventory(args, [])
    assert plain["license_class"] == "HOLD" and "license_text_origin" not in plain
    assert "the hash-bound .crate contains no license file" in plain["hold_reasons"]
    args.cargo_upstream_license_evidence = evidence
    (row,) = L.rust_inventory(args, [])
    assert (row["license_class"], row["hold_reasons"], row["license_text_origin"]) == ("PERMISSIVE", [], "upstream-vcs")
    assert row["license_files_in_artifact"][0]["vcs_sha1"] == "a" * 40


@pytest.mark.parametrize("case", ["sha1-mismatch", "file-hash-mismatch", "extra-condition"])
def test_upstream_license_evidence_mismatch_stays_hold(tmp_path, case):
    kwargs = {
        "sha1-mismatch": {"crate_sha1": "b" * 40},
        "file-hash-mismatch": {"recorded_text": L.MIT_CANONICAL_BODY + " "},
        "extra-condition": {"text": L.MIT_CANONICAL_BODY + "\nCommercial use is prohibited.\n"},
    }[case]
    args, evidence = _upstream_args(tmp_path, **kwargs)
    args.cargo_upstream_license_evidence = evidence
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    if case != "extra-condition":
        assert row["license_files_in_artifact"] == [] and "license_text_origin" not in row


def _own_crate_args(tmp_path: Path, *, wheel_version="1.0", license_member=True, license_text=None,
                    extra_license_files=None):
    """One path crate 'own 1.0' with no LICENSE beside Cargo.toml, plus a published wheel."""
    crate_dir = tmp_path / "own"
    crate_dir.mkdir()
    (crate_dir / "Cargo.toml").write_text("")
    lock = tmp_path / "Cargo.lock"
    lock.write_text('version = 4\n[[package]]\nname = "own"\nversion = "1.0"\n')
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"packages": [{"name": "own", "version": "1.0", "source": None, "license": "MIT",
                                              "license_file": None, "manifest_path": str(crate_dir / "Cargo.toml")}]}))
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({"components": []}))
    (tmp_path / "tree").mkdir()
    wheel = tmp_path / "own-1.0-py3-none-any.whl"
    extra_license_files = extra_license_files or {}
    (tmp_path / "LICENSE").write_text(license_text or L.MIT_CANONICAL_BODY)
    for name, text in extra_license_files.items():
        (tmp_path / name).write_text(text)
    with zipfile.ZipFile(wheel, "w") as zf:
        declarations = "".join(f"License-File: {name}\n" for name in extra_license_files)
        zf.writestr("own-1.0.dist-info/METADATA", f"Metadata-Version: 2.4\nName: own\nVersion: {wheel_version}\nLicense-File: LICENSE\n{declarations}")
        if license_member:
            zf.writestr("own-1.0.dist-info/licenses/LICENSE", license_text or L.MIT_CANONICAL_BODY)
        for name, text in extra_license_files.items():
            zf.writestr(f"own-1.0.dist-info/licenses/{name}", text)
    args = Namespace(cargo_metadata_workspace=str(meta), cargo_metadata_binding=str(meta), cargo_lock_workspace=str(lock),
                     cargo_lock_binding=str(lock), cargo_registry_cache=str(tmp_path / "cache"), wheel_sbom=str(sbom),
                     tree_dir=str(tmp_path / "tree"), own_crate_wheel_source_root=str(tmp_path))
    return args, str(wheel), hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_own_crate_binds_to_published_wheel_license_and_absent_flag_is_unchanged(tmp_path):
    args, wheel, digest = _own_crate_args(tmp_path)
    (plain,) = L.rust_inventory(args, [])
    assert plain["license_class"] == "HOLD" and "license_text_origin" not in plain
    args.own_crate_wheel, args.own_crate_wheel_sha256 = wheel, digest
    (row,) = L.rust_inventory(args, [])
    assert (row["license_class"], row["hold_reasons"], row["license_text_origin"]) == ("PERMISSIVE", [], "published-wheel")
    assert row["license_files_in_artifact"][0]["artifact_sha256"] == digest


def test_own_crate_wheel_notices_are_recorded_but_not_own_license_candidates(tmp_path):
    args, wheel, digest = _own_crate_args(tmp_path, extra_license_files={
        "NOTICE": "Attribution notice; see LICENSE.",
    })
    args.own_crate_wheel, args.own_crate_wheel_sha256 = wheel, digest
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "PERMISSIVE" and row["hold_reasons"] == []
    assert {f["path"].rsplit("/", 1)[-1]: f["wheel_role"] for f in row["license_files_in_artifact"]} == {
        "LICENSE": "own-license", "NOTICE": "distribution-notice",
    }
    assert all(not f["verified_standard_text"] for f in row["license_files_in_artifact"]
               if f["wheel_role"] != "own-license")


def test_own_crate_wheel_unknown_notice_owner_stays_hold(tmp_path):
    args, wheel, digest = _own_crate_args(tmp_path, extra_license_files={"NOTICE-unknown-1.0.txt": "Notice"})
    args.own_crate_wheel, args.own_crate_wheel_sha256 = wheel, digest
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert any("no unique component" in reason for reason in row["hold_reasons"])


def test_own_crate_wheel_notice_source_mismatch_stays_hold(tmp_path):
    args, wheel, digest = _own_crate_args(tmp_path, extra_license_files={"NOTICE": "Original notice"})
    (tmp_path / "NOTICE").write_text("Changed notice")
    args.own_crate_wheel, args.own_crate_wheel_sha256 = wheel, digest
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert any("differs from source" in reason for reason in row["hold_reasons"])


def verify_actual_a3_wheel_license_roles(wheel_path: Path, source_root: Path):
    """Explicit artifact check; run this file with wheel and exact source paths."""
    expected = ("d8ec1d497763abd9" "43dfc5ba0defa93a"
                "67f141b8bab9adf0" "4a02c8d09a9d43bb")
    version, files, errors = L.own_crate_wheel_license_files(
        wheel_path, expected, {("cfg_aliases", "0.2.2"), ("libm", "0.2.16")}, source_root)
    assert (version, errors, len(files)) == ("0.11.5", [], 6)
    expected_files = {
        "LICENSE": "08f1fd81fb120bc468b69dc3e58ea0dc23c216305c766e45e107f56c76559e3f",
        "LICENSE-THIRD-PARTY": "d46f307a2e8a49e2d638ee4e0b768c6c908c7786cb9106e74948561bf8cf0af0",
        "NOTICE": "7192b2614bfee6e95283ef9db9f5fe41c2acb579f5cd30e6482445e42fca0ae5",
        "NOTICE-cfg_aliases-0.2.2-NOTICES.md": "1e2b7ade3fb228130408b9990cae6a7618eb314c75aa0b164bfe485d9d9756ee",
        "NOTICE-libm-0.2.16-LICENSE.txt": "3823dda7cf046602f4b4e77ec8e227863dc4736037cc85bb33d9f19febe16bb7",
        "NOTICE-libm-0.2.16-source-notices.txt": "9e949a13f66c0f9b60b73b54e8ab2940ccff92d704c46c53103b1028e2cc75ba",
    }
    assert {f["path"].split("/licenses/", 1)[1]: f["sha256"] for f in files} == expected_files
    assert all(hashlib.sha256((source_root / name).read_bytes()).hexdigest() == digest
               for name, digest in expected_files.items())
    assert all(f["artifact_sha256"] == expected for f in files)
    assert [f["wheel_role"] for f in files].count("own-license") == 1
    assert {f["wheel_component"] for f in files if f["wheel_role"] == "third-party-notice"} == {
        "cfg_aliases@0.2.2", "libm@0.2.16"}
    assert all(not f["verified_standard_text"] for f in files if f["wheel_role"] != "own-license")


@pytest.mark.parametrize("case", ["wheel-sha-mismatch", "version-mismatch", "license-member-missing"])
def test_own_crate_wheel_mismatch_stays_hold(tmp_path, case):
    args, wheel, digest = _own_crate_args(
        tmp_path, wheel_version="9.9" if case == "version-mismatch" else "1.0",
        license_member=case != "license-member-missing")
    args.own_crate_wheel = wheel
    args.own_crate_wheel_sha256 = "0" * 64 if case == "wheel-sha-mismatch" else digest
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD" and "license_text_origin" not in row
    assert row["license_files_in_artifact"] == []


# Exact reviewed pointer-notice bytes (s1 text-review-20260926 texts/23860c2a..., 01c266bc...).
UNICODE_WIDTH_COPYRIGHT = 'Licensed under the Apache License, Version 2.0\n<LICENSE-APACHE or\nhttp://www.apache.org/licenses/LICENSE-2.0> or the MIT\nlicense <LICENSE-MIT or http://opensource.org/licenses/MIT>,\nat your option. All files in the project carrying such\nnotice may not be copied, modified, or distributed except\naccording to those terms.\n'
MEMCHR_COPYING = 'This project is dual-licensed under the Unlicense and MIT licenses.\n\nYou may use this code under the terms of either license.\n'



def test_pointer_notice_map_holds_exactly_the_reviewed_texts():
    for text, names in ((UNICODE_WIDTH_COPYRIGHT, ("Apache-2.0", "MIT")), (MEMCHR_COPYING, ("Unlicense", "MIT")),
                        ("MIT OR Apache-2.0", ("MIT", "Apache-2.0"))):
        normalized = L.re.sub(r"[ \t\r\n]+", " ", text).strip(" \t\r\n")
        assert L.POINTER_NOTICES[hashlib.sha256(normalized.encode()).hexdigest()] == names
        assert L.verified_standard_text(text) == []


def _pointer_args(tmp_path: Path, *, pointer="MIT OR Apache-2.0", with_apache=True, declared="MIT OR Apache-2.0"):
    args = _rust_args(tmp_path)
    apache = next(r["text"] for r in REVIEWED if r["identifier"] == "Apache-2.0")
    files = {"LICENSE": pointer, "LICENSE-MIT": L.MIT_CANONICAL_BODY}
    if with_apache:
        files["LICENSE-APACHE"] = apache
    digest = _crate(Path(args.cargo_registry_cache), "dep", "1.0", files)
    lock = Path(args.cargo_lock_workspace)
    content = lock.read_text()
    lock.write_text(content.replace(L.tomllib.loads(content)["package"][0]["checksum"], digest))
    meta = Path(args.cargo_metadata_workspace)
    meta.write_text(meta.read_text().replace('"MIT OR Apache-2.0"', json.dumps(declared)))
    return args


def test_reviewed_pointer_is_satisfied_only_with_the_flag(tmp_path):
    args = _pointer_args(tmp_path)
    (plain,) = L.rust_inventory(args, [])
    assert plain["license_class"] == "HOLD"
    assert all("pointer_notice" not in f for f in plain["license_files_in_artifact"])
    args.reviewed_pointer_notices = True
    (row,) = L.rust_inventory(args, [])
    assert (row["license_class"], row["hold_reasons"]) == ("PERMISSIVE", [])
    (pointer,) = [f for f in row["license_files_in_artifact"] if f["path"] == "LICENSE"]
    assert pointer["pointer_notice"] == {"names": ["MIT", "Apache-2.0"], "satisfied": True}


@pytest.mark.parametrize("case", ["a-unreviewed-pointer", "b-named-text-missing", "c-declared-mismatch"])
def test_reviewed_pointer_violation_stays_hold(tmp_path, case):
    args = _pointer_args(tmp_path, **{
        "a-unreviewed-pointer": {"pointer": "MIT OR Apache-2.0 OR BSD-3-Clause"},
        "b-named-text-missing": {"with_apache": False},
        "c-declared-mismatch": {"declared": "MIT"},
    }[case])
    args.reviewed_pointer_notices = True
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert not any(f.get("pointer_notice", {}).get("satisfied") for f in row["license_files_in_artifact"])


def test_reviewed_unicode_license_satisfies_and_conjunct(tmp_path):
    """MIT AND Unicode-3.0 passes only with the exact reviewed Unicode text in the crate."""
    unicode = next(r["text"] for r in REVIEWED if r["identifier"] == "Unicode-3.0")
    for text, expected in ((unicode, "PERMISSIVE"), (unicode + "\nNo commercial use.", "HOLD")):
        (tmp_path / expected).mkdir()
        args = _pointer_args(tmp_path / expected, pointer=L.MIT_CANONICAL_BODY, with_apache=False,
                             declared="MIT AND Unicode-3.0")
        digest = _crate(Path(args.cargo_registry_cache), "dep", "1.0",
                        {"LICENSE-MIT": L.MIT_CANONICAL_BODY, "LICENSE-UNICODE": text})
        lock = Path(args.cargo_lock_workspace)
        content = lock.read_text()
        lock.write_text(content.replace(L.tomllib.loads(content)["package"][0]["checksum"], digest))
        (row,) = L.rust_inventory(args, [])
        assert row["license_class"] == expected


def test_documented_exception_is_pinned_to_crate_and_member_hash(tmp_path, monkeypatch):
    """An exception applies only to the exact (crate sha, member sha) pair it names."""
    text = "Combined notice: MIT for the crate; third-party parts under notice-preserving terms.\n"
    args = _rust_args(tmp_path)
    digest = _crate(Path(args.cargo_registry_cache), "dep", "1.0", {"LICENSE.txt": text})
    lock = Path(args.cargo_lock_workspace)
    content = lock.read_text()
    lock.write_text(content.replace(L.tomllib.loads(content)["package"][0]["checksum"], digest))
    member = hashlib.sha256(text.encode()).hexdigest()
    (plain,) = L.rust_inventory(args, [])
    assert plain["license_class"] == "HOLD"
    monkeypatch.setitem(L.DOCUMENTED_EXCEPTIONS, (digest, member), {"identifier": "MIT"})
    (row,) = L.rust_inventory(args, [])
    assert (row["license_class"], row["license_files_in_artifact"][0]["documented_exception"]) == ("PERMISSIVE", {"identifier": "MIT"})
    monkeypatch.delitem(L.DOCUMENTED_EXCEPTIONS, (digest, member))
    monkeypatch.setitem(L.DOCUMENTED_EXCEPTIONS, ("0" * 64, member), {"identifier": "MIT"})
    (other,) = L.rust_inventory(args, [])
    assert other["license_class"] == "HOLD"


CFG_ALIASES_NOTICES = '# 3rd Party Notices\n\nThe `cfg_aliases!` macro uses a lot of the code from [`tectonic_cfg_support::target_cfg!`] macro which is under the following license:\n\n[`tectonic_cfg_support::target_cfg!`]: https://github.com/tectonic-typesetting/tectonic/blob/f2439b936470ad27bdf92882064bc4702ee01899/cfg_support/src/lib.rs#L166\n\n    tectonic_cfg_support is licensed under the MIT License.\n\n    Permission is hereby granted, free of charge, to any person obtaining a copy\n    of this software and associated documentation files (the “Software”), to deal\n    in the Software without restriction, including without limitation the rights\n    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n    copies of the Software, and to permit persons to whom the Software is\n    furnished to do so, subject to the following conditions:\n\n    The above copyright notice and this permission notice shall be included in all\n    copies or substantial portions of the Software.\n\n    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n    SOFTWARE.\n---\n'


def test_cfg_aliases_notice_exception_is_pinned_to_the_0_2_2_crate(tmp_path):
    """The same NOTICES.md bytes in any crate other than the pinned cfg_aliases 0.2.2 stay HOLD."""
    member = hashlib.sha256(CFG_ALIASES_NOTICES.encode()).hexdigest()
    pinned = [k for k, v in L.DOCUMENTED_EXCEPTIONS.items() if v.get("package") == "cfg_aliases@0.2.2"]
    assert pinned == [("f079e83a288787bcd14a6aea84cee5c87a67c5a3e660c30f557a3d24761b3527", member)]
    assert L.DOCUMENTED_EXCEPTIONS[pinned[0]]["classification"] == "third-party MIT attribution notice"
    args = _rust_args(tmp_path)
    digest = _crate(Path(args.cargo_registry_cache), "dep", "1.0", {"LICENSE-MIT": L.MIT_CANONICAL_BODY, "NOTICES.md": CFG_ALIASES_NOTICES})
    lock = Path(args.cargo_lock_workspace)
    content = lock.read_text()
    lock.write_text(content.replace(L.tomllib.loads(content)["package"][0]["checksum"], digest))
    (row,) = L.rust_inventory(args, [])
    assert row["license_class"] == "HOLD"
    assert not any(f.get("documented_exception") for f in row["license_files_in_artifact"])


def test_wheel_directory_entry_is_not_a_license_candidate(tmp_path):
    """A zip directory entry such as pkg-1.0.dist-info/licenses/ is not an (empty) license file."""
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n\n")
        zf.writestr(zipfile.ZipInfo("pkg-1.0.dist-info/licenses/"), "")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
    paths = [f["path"] for f in L.python_artifact_evidence(wheel)["license_files"]]
    assert paths == ["pkg-1.0.dist-info/licenses/LICENSE"]


def test_reviewed_python_companion_requires_exact_member_and_sibling_grant(tmp_path, monkeypatch):
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    companion = "Names of contributors, without license terms."
    path = "pkg-1.0.dist-info/licenses/AUTHORS"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("pkg-1.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: pkg\nVersion: 1.0\n\n")
        zf.writestr("pkg-1.0.dist-info/licenses/LICENSE", MIT_TEXT)
        zf.writestr(path, companion)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    key = (digest, path, hashlib.sha256(companion.encode()).hexdigest())
    monkeypatch.setitem(L.REVIEWED_PYTHON_COMPANIONS, key, ("MIT",))
    files = {f["path"]: f for f in L.python_artifact_evidence(wheel)["license_files"]}
    assert files[path]["candidate_verified"] is True
    assert files[path]["reviewed_companion_grants"] == ["MIT"]
    monkeypatch.setitem(L.REVIEWED_PYTHON_COMPANIONS, key, ("BSD-2-Clause",))
    files = {f["path"]: f for f in L.python_artifact_evidence(wheel)["license_files"]}
    assert files[path]["candidate_verified"] is False
    monkeypatch.delitem(L.REVIEWED_PYTHON_COMPANIONS, key)
    files = {f["path"]: f for f in L.python_artifact_evidence(wheel)["license_files"]}
    assert files[path]["candidate_verified"] is False


def test_locked_sortedcontainers_source_wheel_has_complete_bound_license():
    root = Path(__file__).resolve().parents[1]
    wheel = root / "third_party/wheels/sortedcontainers-2.4.0-py3-none-any.whl"
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    locked = next(p for p in L.tomllib.loads((root / "uv.lock").read_text())["package"]
                  if p["name"] == "sortedcontainers")
    assert locked["wheels"] == [{"filename": wheel.name, "hash": f"sha256:{digest}"}]
    assert locked["source"] == {"path": wheel.relative_to(root).as_posix()}
    evidence = L.python_artifact_evidence(wheel)
    assert evidence["sha256"] == digest
    files = {f["path"].rsplit("/", 1)[-1]: f for f in evidence["license_files"]}
    assert set(files) == {"LICENSE", "LICENSE-APACHE-2.0.txt"}
    assert files["LICENSE"]["reviewed_companion_grants"] == ["Apache-2.0"]
    assert all(f["candidate_verified"] for f in files.values())


@pytest.mark.parametrize("pinned", [True, False])
def test_external_runtime_dependency_is_pinned_to_its_wheel(tmp_path, monkeypatch, pinned):
    """Vendored-native findings become notes only for the exact pinned wheel of an external dependency."""
    args = _python_args(tmp_path, expression="MIT", license_text=False)
    wheel = Path(args.pypi_artifact_dir) / "pkg-1.0-py3-none-any.whl"
    digest = _native_wheel(wheel, native_license=None, bind_notice=False)
    Path(args.uv_lock).write_text(
        'version = 1\n[[package]]\nname = "pkg"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://x/pkg.whl", hash = "sha256:{digest}" }}]\n'
    )
    monkeypatch.setitem(L.EXTERNAL_RUNTIME_DEPENDENCIES, ("pkg", "1.0"),
                        {"wheel_sha256": digest if pinned else "0" * 64, "classification": "external"})
    (row,) = L.python_inventory(args, [])
    if pinned:
        assert (row["license_class"], row["vendored_native_license_holds"]) == ("PERMISSIVE", [])
        assert "no bound notice stanza" in " ".join(row["external_runtime_notes"])
    else:
        assert row["license_class"] == "HOLD" and "external_runtime_dependency" not in row


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_license_inventory_generator.py WHEEL EXACT_SOURCE_ROOT")
    verify_actual_a3_wheel_license_roles(Path(sys.argv[1]), Path(sys.argv[2]))
    print("actual A3 wheel license roles and six source hashes verified")
