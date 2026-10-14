"""
Patch validator — ensures AI-generated patches are safe before application.

Generated patches are UNTRUSTED AI output. Never apply without validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Modules that must not appear in patch additions
_DANGEROUS_MODULE_PATTERN = re.compile(
    r"^\+.*\b(import\s+(os|subprocess|socket|sys|shutil|pickle|ctypes|importlib"
    r"|multiprocessing|threading|urllib|http|requests|httpx|pty|signal|mmap)"
    r"|from\s+(os|subprocess|socket|sys|shutil|pickle|ctypes|importlib"
    r"|multiprocessing|threading|urllib|http|requests|httpx|pty|signal|mmap)"
    r"\s+import)",
    re.MULTILINE,
)

# Dangerous shell/exec operations in added lines
_DANGEROUS_CALL_PATTERN = re.compile(
    r"^\+.*\b(os\.system|os\.popen|subprocess\.(run|Popen|call|check_output)|"
    r"eval\s*\(|exec\s*\(|__import__\s*\(|compile\s*\()",
    re.MULTILINE,
)

# Path traversal in file references
_PATH_TRAVERSAL_PATTERN = re.compile(r"\.\./|/\.\.")


@dataclass
class PatchValidationResult:
    valid: bool
    error: str | None = None
    changed_files: list[str] | None = None


def validate_patch(patch_diff: str, changed_files: list[str]) -> PatchValidationResult:
    """Validate an AI-generated unified diff before application.

    Checks:
      1. Diff is non-empty
      2. Changed file paths are repository-relative (no absolute paths, no traversal)
      3. Changed file paths match what the diff actually touches
      4. No dangerous imports added in diff lines
      5. No dangerous function calls added in diff lines
    """
    if not patch_diff or not patch_diff.strip():
        return PatchValidationResult(valid=False, error="Patch diff is empty")

    if not changed_files:
        return PatchValidationResult(valid=False, error="changed_files list is empty")

    # 1. Validate all changed file paths
    for path_str in changed_files:
        err = _validate_file_path(path_str)
        if err:
            return PatchValidationResult(valid=False, error=err)

    # 2. Extract paths actually referenced in the diff header lines
    diff_paths = _extract_diff_paths(patch_diff)

    # 3. Each path in diff must be in changed_files (both canonicalized)
    declared = {_normalize_path(p) for p in changed_files}
    for dp in diff_paths:
        norm = _normalize_path(dp)
        if norm and norm not in declared:
            return PatchValidationResult(
                valid=False,
                error=f"Diff references file '{dp}' not listed in changed_files",
            )

    # 4. No dangerous imports in added lines
    if _DANGEROUS_MODULE_PATTERN.search(patch_diff):
        return PatchValidationResult(
            valid=False,
            error="Patch adds imports of dangerous modules; rejecting for safety",
        )

    # 5. No dangerous function calls in added lines
    if _DANGEROUS_CALL_PATTERN.search(patch_diff):
        return PatchValidationResult(
            valid=False,
            error="Patch adds dangerous function calls; rejecting for safety",
        )

    return PatchValidationResult(valid=True, changed_files=changed_files)


def _validate_file_path(path_str: str) -> str | None:
    """Return an error string if the path is unsafe, else None."""
    if not path_str:
        return "Empty file path in changed_files"

    # Must not be absolute
    if Path(path_str).is_absolute():
        return f"Patch path '{path_str}' is absolute; only repository-relative paths are allowed"

    # Must not traverse upward
    if _PATH_TRAVERSAL_PATTERN.search(path_str):
        return f"Patch path '{path_str}' contains path traversal sequence"

    # Must have a file extension (rough sanity check)
    if not PurePosixPath(path_str).suffix:
        return f"Patch path '{path_str}' has no file extension"

    return None


def _extract_diff_paths(diff: str) -> list[str]:
    """Extract file paths from '--- a/...' and '+++ b/...' lines."""
    paths: list[str] = []
    for line in diff.splitlines():
        for prefix in ("--- a/", "+++ b/", "--- ", "+++ "):
            if line.startswith(prefix):
                raw = line[len(prefix) :].strip()
                # Strip git sentinel /dev/null
                if raw and raw != "/dev/null":
                    paths.append(raw)
                break
    return paths


def _normalize_path(path: str) -> str:
    """Strip leading a/ or b/ prefixes used by git diff."""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path
