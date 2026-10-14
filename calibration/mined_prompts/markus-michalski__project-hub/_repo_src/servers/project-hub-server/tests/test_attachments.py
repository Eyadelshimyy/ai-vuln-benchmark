"""Tests for file attachment operations on notes."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
from tools import attachments as attachments_module
from tools.attachments import (
    _convert_to_markdown_sibling,
    attach_file,
    attach_files,
    list_attachments,
    remove_attachment,
)
from tools.notes import add_note, get_note
from tools.projects import create_project


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    return create_project("Attach Project")


@pytest.fixture
def note(project):
    return add_note(project["id"], "Test Note", "Content here")


# pytest's tmp_path fixture always lives under the OS temp dir (/tmp on POSIX,
# %TEMP% on Windows) — use the same OS-correct root here instead of hardcoding
# the POSIX path, or sample_file (built on tmp_path) fails relative_to() on Windows.
# .resolve() matters on Windows: tempfile.gettempdir() can return an 8.3 short
# path (e.g. RUNNER~1) while attach_file() resolves the source file to its long
# canonical form — relative_to() needs both sides in the same form to match.
_HOME = Path(tempfile.gettempdir()).resolve()


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "report.pdf"
    f.write_bytes(b"PDF content" * 100)
    return f


# --- attach_file ---

def test_attach_file_copies_to_attachments_folder(project, note, sample_file):
    result = attach_file(note["id"], str(sample_file), home_override=_HOME)

    assert result["name"] == "report.pdf"
    assert Path(result["path"]).exists()
    assert Path(result["path"]).name == "report.pdf"
    assert "attachments" in str(result["path"])


def test_attach_file_original_untouched(project, note, sample_file):
    attach_file(note["id"], str(sample_file), home_override=_HOME)

    assert sample_file.exists()


def test_attach_file_stores_reference_in_db(project, note, sample_file):
    attach_file(note["id"], str(sample_file), home_override=_HOME)

    updated = get_note(note["id"])
    attachments = json.loads(updated["attachments"])
    assert len(attachments) == 1
    assert attachments[0]["name"] == "report.pdf"


def test_attach_file_returns_size(project, note, sample_file):
    result = attach_file(note["id"], str(sample_file), home_override=_HOME)

    assert result["size"] == sample_file.stat().st_size


def test_attach_file_multiple_files(project, note, tmp_path):
    f1 = tmp_path / "doc1.txt"
    f2 = tmp_path / "doc2.txt"
    f1.write_text("first")
    f2.write_text("second")

    attach_file(note["id"], str(f1), home_override=_HOME)
    attach_file(note["id"], str(f2), home_override=_HOME)

    updated = get_note(note["id"])
    attachments = json.loads(updated["attachments"])
    assert len(attachments) == 2
    names = {a["name"] for a in attachments}
    assert names == {"doc1.txt", "doc2.txt"}


def test_attach_file_same_basename_different_sources_does_not_collide(project, note, tmp_path):
    # Two different physical files sharing a basename (e.g. "invoice.pdf" from
    # Downloads/ and Desktop/) must not silently overwrite each other on disk
    # while the DB ends up with two entries pointing at the same, now-wrong file.
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    f1 = dir_a / "invoice.pdf"
    f2 = dir_b / "invoice.pdf"
    f1.write_text("first invoice")
    f2.write_text("second invoice")

    result1 = attach_file(note["id"], str(f1), home_override=_HOME)
    result2 = attach_file(note["id"], str(f2), home_override=_HOME)

    assert result1["path"] != result2["path"]
    assert result1["name"] != result2["name"]
    assert Path(result1["path"]).read_text() == "first invoice"
    assert Path(result2["path"]).read_text() == "second invoice"

    attachments = list_attachments(note["id"])
    assert len(attachments) == 2
    assert {a["name"] for a in attachments} == {result1["name"], result2["name"]}


def test_attach_file_same_basename_disambiguates_with_counter(project, note, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_c = tmp_path / "c"
    dir_a.mkdir()
    dir_b.mkdir()
    dir_c.mkdir()
    for d, content in [(dir_a, "one"), (dir_b, "two"), (dir_c, "three")]:
        (d / "same.txt").write_text(content)

    r1 = attach_file(note["id"], str(dir_a / "same.txt"), home_override=_HOME)
    r2 = attach_file(note["id"], str(dir_b / "same.txt"), home_override=_HOME)
    r3 = attach_file(note["id"], str(dir_c / "same.txt"), home_override=_HOME)

    assert r1["name"] == "same.txt"
    assert r2["name"] == "same-2.txt"
    assert r3["name"] == "same-3.txt"


def test_attach_file_note_not_found(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")

    with pytest.raises(ValueError, match="Note not found"):
        attach_file(99999, str(f))


def test_attach_file_source_not_found(project, note):
    with pytest.raises(FileNotFoundError):
        attach_file(note["id"], "/nonexistent/file.pdf")


def test_attach_file_path_traversal_blocked(project, note, tmp_path):
    evil = tmp_path / "evil.sh"
    evil.write_text("rm -rf /")

    with pytest.raises(ValueError, match="[Pp]ath"):
        attach_file(note["id"], str(evil), home_override=Path("/different/root"))


def test_attach_file_warns_large_file(project, note, tmp_path, capsys):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (11 * 1024 * 1024))  # 11 MB

    attach_file(note["id"], str(big), home_override=_HOME)

    captured = capsys.readouterr()
    assert "10" in captured.err or "10" in captured.out or "MB" in captured.err or "MB" in captured.out


# --- attach_files (batch, project-hub#134) ---

def test_attach_files_copies_all(project, note, tmp_path):
    f1 = tmp_path / "doc1.txt"
    f2 = tmp_path / "doc2.txt"
    f1.write_text("first")
    f2.write_text("second")

    result = attach_files(note["id"], [str(f1), str(f2)], home_override=_HOME)

    assert result["failed"] == []
    assert {a["name"] for a in result["attached"]} == {"doc1.txt", "doc2.txt"}
    for a in result["attached"]:
        assert Path(a["path"]).exists()

    updated = get_note(note["id"])
    assert len(json.loads(updated["attachments"])) == 2


def test_attach_files_creates_markdown_siblings(project, note, tmp_path):
    f1 = tmp_path / "report.txt"
    f1.write_text("important content")

    result = attach_files(note["id"], [str(f1)], home_override=_HOME)

    sibling = Path(result["attached"][0]["path"]).with_name("report.txt.md")
    assert sibling.exists()


def test_attach_files_disambiguates_same_basename_within_batch(project, note, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "same.txt").write_text("first")
    (dir_b / "same.txt").write_text("second")

    result = attach_files(
        note["id"], [str(dir_a / "same.txt"), str(dir_b / "same.txt")], home_override=_HOME
    )

    assert result["failed"] == []
    names = {a["name"] for a in result["attached"]}
    assert names == {"same.txt", "same-2.txt"}


def test_attach_files_one_missing_path_does_not_abort_batch(project, note, tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("ok")

    result = attach_files(
        note["id"], [str(good), "/nonexistent/missing.txt"], home_override=_HOME
    )

    assert len(result["attached"]) == 1
    assert result["attached"][0]["name"] == "good.txt"
    assert len(result["failed"]) == 1
    assert result["failed"][0]["path"] == "/nonexistent/missing.txt"
    assert "not found" in result["failed"][0]["error"].lower()

    # the good file must still be reflected in the DB despite the other failure
    updated = get_note(note["id"])
    assert len(json.loads(updated["attachments"])) == 1


def test_attach_files_path_traversal_on_one_file_does_not_abort_batch(project, note, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    good = home / "good.txt"
    good.write_text("ok")
    evil = tmp_path / "evil.sh"  # outside `home`
    evil.write_text("rm -rf /")

    result = attach_files(note["id"], [str(good), str(evil)], home_override=home)

    assert len(result["attached"]) == 1
    assert result["attached"][0]["name"] == "good.txt"
    assert len(result["failed"]) == 1
    assert result["failed"][0]["path"] == str(evil)
    assert "path" in result["failed"][0]["error"].lower()


def test_attach_files_empty_list(project, note):
    result = attach_files(note["id"], [], home_override=_HOME)

    assert result == {"attached": [], "failed": []}
    updated = get_note(note["id"])
    assert json.loads(updated["attachments"]) == []


def test_attach_files_note_not_found(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")

    with pytest.raises(ValueError, match="Note not found"):
        attach_files(99999, [str(f)], home_override=_HOME)


def test_attach_files_rejects_directory_input(project, note, tmp_path):
    # Regression: exists() is true for directories too, and #134 was specifically about
    # someone handing over a *folder*. Without an is_file() check, shutil.copy2() would
    # raise an ugly "[Errno 21] Is a directory" — or, on a FIFO, hang forever.
    a_dir = tmp_path / "some_folder"
    a_dir.mkdir()

    result = attach_files(note["id"], [str(a_dir)], home_override=_HOME)

    assert result["attached"] == []
    assert len(result["failed"]) == 1
    assert result["failed"][0]["path"] == str(a_dir)


def test_attach_files_deduplicates_same_resolved_path(project, note, tmp_path):
    f = tmp_path / "dup.txt"
    f.write_text("once")

    result = attach_files(note["id"], [str(f), str(f)], home_override=_HOME)

    assert result["failed"] == []
    assert len(result["attached"]) == 1
    assert result["attached"][0]["name"] == "dup.txt"


def test_attach_files_deduplicates_path_equivalent_but_different_strings(project, note, tmp_path):
    # Dedup must compare *resolved* paths, not raw input strings — prove it with two
    # differently-spelled paths that resolve to the same file.
    sub = tmp_path / "sub"
    sub.mkdir()
    f = sub / "dup.txt"
    f.write_text("once")

    literal = str(f)
    via_dotdot = str(tmp_path / "sub" / ".." / "sub" / "dup.txt")
    assert literal != via_dotdot  # sanity: genuinely different strings

    result = attach_files(note["id"], [literal, via_dotdot], home_override=_HOME)

    assert result["failed"] == []
    assert len(result["attached"]) == 1


def test_attach_files_rejects_hidden_dotfile_path(project, note, tmp_path):
    # Defense in depth: preservation is now mandatory and folder-shaped (project-hub#134
    # follow-up), so guard against sweeping up dotfiles like ~/.ssh/id_rsa or
    # ~/.aws/credentials when a broad path gets passed through.
    hidden_dir = tmp_path / ".secret"
    hidden_dir.mkdir()
    secret = hidden_dir / "id_rsa"
    secret.write_text("private key material")

    result = attach_files(note["id"], [str(secret)], home_override=_HOME)

    assert result["attached"] == []
    assert len(result["failed"]) == 1
    assert result["failed"][0]["path"] == str(secret)


def test_attach_files_rejects_hidden_target_via_visible_symlink(project, note, tmp_path):
    # Symlink-laundering: an innocuous-looking, non-hidden path must not bypass the
    # hidden-dotfile guard just because the *symlink itself* isn't hidden — the check
    # must apply to the resolved target, not the input string.
    hidden_dir = tmp_path / ".secret"
    hidden_dir.mkdir()
    target = hidden_dir / "id_rsa"
    target.write_text("private key material")
    link = tmp_path / "innocuous_link.txt"
    link.symlink_to(target)

    result = attach_files(note["id"], [str(link)], home_override=_HOME)

    assert result["attached"] == []
    assert len(result["failed"]) == 1


def test_attach_files_rejects_symlink_escaping_home(project, note, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not under home")
    link = home / "link.txt"
    link.symlink_to(outside)

    result = attach_files(note["id"], [str(link)], home_override=home)

    assert result["attached"] == []
    assert len(result["failed"]) == 1


def test_attach_file_rejects_directory_input(project, note, tmp_path):
    a_dir = tmp_path / "some_folder"
    a_dir.mkdir()

    with pytest.raises(ValueError, match="[Dd]irectory|regular file"):
        attach_file(note["id"], str(a_dir), home_override=_HOME)


# --- list_attachments ---

def test_list_attachments_empty(note):
    result = list_attachments(note["id"])
    assert result == []


def test_list_attachments_returns_file_info(project, note, sample_file):
    attach_file(note["id"], str(sample_file), home_override=_HOME)

    result = list_attachments(note["id"])
    assert len(result) == 1
    assert result[0]["name"] == "report.pdf"
    assert "path" in result[0]
    assert "size" in result[0]


def test_list_attachments_note_not_found():
    with pytest.raises(ValueError, match="Note not found"):
        list_attachments(99999)


# --- remove_attachment ---

def test_remove_attachment_deletes_file(project, note, sample_file):
    info = attach_file(note["id"], str(sample_file), home_override=_HOME)
    copied_path = Path(info["path"])

    remove_attachment(note["id"], "report.pdf")

    assert not copied_path.exists()


def test_remove_attachment_updates_db(project, note, sample_file):
    attach_file(note["id"], str(sample_file), home_override=_HOME)

    remove_attachment(note["id"], "report.pdf")

    updated = get_note(note["id"])
    attachments = json.loads(updated["attachments"])
    assert attachments == []


def test_remove_attachment_removes_only_the_matching_disambiguated_entry(project, note, tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "same.txt").write_text("first")
    (dir_b / "same.txt").write_text("second")

    r1 = attach_file(note["id"], str(dir_a / "same.txt"), home_override=_HOME)
    r2 = attach_file(note["id"], str(dir_b / "same.txt"), home_override=_HOME)

    remove_attachment(note["id"], r1["name"])

    assert not Path(r1["path"]).exists()
    assert Path(r2["path"]).exists()
    assert Path(r2["path"]).read_text() == "second"
    remaining = list_attachments(note["id"])
    assert len(remaining) == 1
    assert remaining[0]["name"] == r2["name"]


def test_remove_attachment_deletes_markdown_sibling(project, note, tmp_path):
    f = tmp_path / "report.txt"
    f.write_text("important content")
    info = attach_file(note["id"], str(f), home_override=_HOME)
    sibling = Path(info["path"]).with_name("report.txt.md")
    assert sibling.exists()

    remove_attachment(note["id"], "report.txt")

    assert not sibling.exists()


def test_remove_attachment_no_sibling_present_does_not_raise(project, note, sample_file):
    # sample_file's fake "PDF" content produces no markdown sibling (not real PDF
    # bytes) — removal must still succeed without erroring on a missing sibling.
    attach_file(note["id"], str(sample_file), home_override=_HOME)

    remove_attachment(note["id"], "report.pdf")  # must not raise


def test_remove_attachment_not_found_raises(project, note):
    with pytest.raises(ValueError, match="[Nn]ot found"):
        remove_attachment(note["id"], "missing.pdf")


def test_remove_attachment_note_not_found():
    with pytest.raises(ValueError, match="Note not found"):
        remove_attachment(99999, "x.pdf")


# --- get_note includes attachments ---

def test_get_note_includes_attachments_field(note):
    result = get_note(note["id"])
    assert "attachments" in result
    assert result["attachments"] == "[]"


# --- _get_markitdown ---

def test_get_markitdown_swallows_non_import_error(monkeypatch):
    # Regression: a broken native dependency (e.g. onnxruntime failing to load
    # its platform binary) raises OSError/RuntimeError from MarkItDown(), not
    # ImportError — _get_markitdown() must swallow that too, not just ImportError,
    # or attach_file() crashes after the file is already copied but before the
    # DB write, leaving an orphaned attachment with no DB reference.
    monkeypatch.setattr(attachments_module, "_markitdown", None)
    monkeypatch.setattr(attachments_module, "_markitdown_load_attempted", False)

    class _BrokenMarkItDown:
        def __init__(self):
            raise OSError("simulated broken onnxruntime native lib")

    fake_module = type(sys)("markitdown")
    fake_module.MarkItDown = _BrokenMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    result = attachments_module._get_markitdown()

    assert result is None


def test_attach_file_survives_broken_markitdown_constructor(project, note, tmp_path, monkeypatch):
    # End-to-end version of the above: attach_file must still succeed (file copied,
    # DB updated) even if the markitdown backend is broken at instantiation time.
    monkeypatch.setattr(attachments_module, "_markitdown", None)
    monkeypatch.setattr(attachments_module, "_markitdown_load_attempted", False)

    class _BrokenMarkItDown:
        def __init__(self):
            raise OSError("simulated broken onnxruntime native lib")

    fake_module = type(sys)("markitdown")
    fake_module.MarkItDown = _BrokenMarkItDown
    monkeypatch.setitem(sys.modules, "markitdown", fake_module)

    f = tmp_path / "report.txt"
    f.write_text("important content")

    result = attach_file(note["id"], str(f), home_override=_HOME)

    assert Path(result["path"]).exists()
    assert list_attachments(note["id"]) == [result]


# --- _convert_to_markdown_sibling ---

def test_markdown_sibling_never_overwrites_existing_file(tmp_path):
    # Regression: a sibling write must not clobber a file that already occupies
    # its exact target name — e.g. a real attachment literally named "notes.md"
    # that was attached before some other attachment named "notes" whose
    # generated sibling would otherwise land on the exact same path.
    dest_dir = tmp_path / "attachments"
    dest_dir.mkdir()
    existing_md = dest_dir / "notes.md"
    existing_md.write_text("pre-existing real attachment, do not touch")

    dest = dest_dir / "notes"
    dest.write_text("some other convertible content")

    _convert_to_markdown_sibling(dest)

    assert existing_md.read_text() == "pre-existing real attachment, do not touch"


def test_attach_file_does_not_overwrite_attachment_via_sibling_collision(project, note, tmp_path):
    # End-to-end version: attach a real "notes.md" file first, then attach an
    # unrelated file named "notes" whose markdown sibling would collide with it.
    md_source = tmp_path / "notes.md"
    md_source.write_text("original markdown attachment")
    plain_source = tmp_path / "notes"
    plain_source.write_text("unrelated convertible content")

    attach_file(note["id"], str(md_source), home_override=_HOME)
    attach_file(note["id"], str(plain_source), home_override=_HOME)

    attachments = list_attachments(note["id"])
    md_attachment = next(a for a in attachments if a["name"] == "notes.md")
    assert Path(md_attachment["path"]).read_text() == "original markdown attachment"


def test_markdown_sibling_created_for_convertible_file(tmp_path):
    source = tmp_path / "notiz.txt"
    source.write_text("Hallo Welt")
    dest = tmp_path / "attachments" / "notiz.txt"
    dest.parent.mkdir()
    import shutil
    shutil.copy2(source, dest)

    _convert_to_markdown_sibling(dest)

    sibling = dest.with_name("notiz.txt.md")
    assert sibling.exists()
    assert sibling.read_text().strip() == "Hallo Welt"


def test_markdown_sibling_skipped_for_images(tmp_path):
    dest = tmp_path / "foto.png"
    dest.write_bytes(b"\x89PNG\r\n\x1a\nnot a real png")

    _convert_to_markdown_sibling(dest)

    assert not dest.with_name("foto.png.md").exists()


def test_markdown_sibling_skipped_for_already_markdown_file(tmp_path):
    dest = tmp_path / "bereits.md"
    dest.write_text("# Schon Markdown")

    _convert_to_markdown_sibling(dest)

    assert not dest.with_name("bereits.md.md").exists()


def test_markdown_conversion_failure_does_not_abort_copy(tmp_path, monkeypatch):
    # Regression: _convert_to_markdown_sibling promises "never raises" — a broken
    # markitdown install or a conversion error must not abort the attachment copy.
    def _boom(_path):
        raise RuntimeError("simulated conversion failure")

    monkeypatch.setattr(attachments_module, "_markitdown", type("M", (), {"convert": staticmethod(_boom)})())

    dest = tmp_path / "doc.pdf"
    dest.write_text("not really a pdf")

    _convert_to_markdown_sibling(dest)

    assert dest.exists()
    assert not dest.with_name("doc.pdf.md").exists()


def test_markdown_sibling_skipped_for_empty_conversion_result(tmp_path, monkeypatch):
    class _EmptyResult:
        markdown = "   \n  "

    monkeypatch.setattr(
        attachments_module, "_markitdown",
        type("M", (), {"convert": staticmethod(lambda _p: _EmptyResult())})(),
    )

    dest = tmp_path / "leer.pdf"
    dest.write_text("irrelevant")

    _convert_to_markdown_sibling(dest)

    assert not dest.with_name("leer.pdf.md").exists()


def test_attach_file_creates_markdown_sibling(project, note, tmp_path):
    f = tmp_path / "report.txt"
    f.write_text("important content")

    attach_file(note["id"], str(f), home_override=_HOME)

    attachments = list_attachments(note["id"])
    copied_path = Path(attachments[0]["path"])
    sibling = copied_path.with_name("report.txt.md")
    assert sibling.exists()
    assert sibling.read_text().strip() == "important content"
