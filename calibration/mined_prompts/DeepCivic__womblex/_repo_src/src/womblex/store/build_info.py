"""Which build of Womblex is running: its version, and the commit it came from.

The version was already carried — ``store/run_stamp.py`` reads ``__version__``
and writes it into every pipeline Parquet's footer. What it could not say is
*which source* that version was built from, and a version alone does not
identify a build: the same ``0.5.12`` is cut from every commit between two
releases.

Two sources answer it, and neither answers it everywhere:

- **A git work tree.** True in a development checkout and in CI, where the
  package sits inside the repository it was built from. Never true of an
  installed wheel.
- **A build stamp.** ``_build_stamp.py``, generated at build time beside the
  package and never committed. True of a wheel the release workflow built,
  false of one built by hand.

So the work tree is asked first — it is the source actually running, and a
stamp left behind by an earlier local build would be stale against it — and the
stamp answers for the wheel. Where neither answers, the commit is
``unavailable`` **with a reason**, which is a value rather than an absence: an
empty string reads as "no commit" and a default reads as a wrong one, and the
whole point of the record is that it does not claim what it cannot establish.
A container is that third case only when its build was given no commit: the
image build takes the commit as an argument and writes this stamp into the
installed package, so an image built by CI or by any build that supplies one
answers here, and a local ``docker compose build`` — which supplies none —
reports ``unavailable`` with its reason.

**A dirty work tree gets its sha suffixed ``-dirty``.** The bytes that ran are
not the bytes at that commit, and reporting the sha bare would claim exactly
the kind of thing this module exists not to claim. The suffix keeps the sha
readable and the caveat unmissable.

Attribution, not reproduction: the commit says what a run was built from, never
that the run can be recomputed from it.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from womblex import __version__

#: The commit value when neither source can answer, and the prefix its reason
#: is appended to. A first-class value, not an empty string.
UNAVAILABLE = "unavailable"

#: Where the running image's own reference is read from. It has to be *given* —
#: a digest is content-addressed after the push, so it cannot be baked into the
#: image it names, and a container cannot read its own labels from inside. The
#: compose stack supplies it from the same value R3 pins the service to.
IMAGE_REF_ENV = "WOMBLEX_IMAGE_REF"

#: Files a container runtime leaves behind. Docker writes the first, Podman the
#: second; both are the runtime's own marker rather than something inferred.
_CONTAINER_MARKERS = (Path("/.dockerenv"), Path("/run/.containerenv"))

#: The separator that makes a reference a digest rather than a tag, and the
#: shape of the digest itself. Checked rather than assumed: this module exists
#: not to claim what it cannot establish, and a string that cannot be a digest
#: recorded as one would be exactly that. The algorithm is not pinned to
#: sha256 — a registry may serve another — but the form is.
_DIGEST_SEP = "@"
_DIGEST_RE = re.compile(r"^[a-z0-9]+(?:[.+_-][a-z0-9]+)*:[0-9a-f]{32,}$")

# Every git call is bounded: a resolver that hangs would hang the run that
# asked it, for a fact the run can honestly do without.
_GIT_TIMEOUT = 5

# The package directory — inside the work tree for a checkout or an editable
# install, inside site-packages for a wheel. Asking git about *this* path is
# what makes the two cases distinguish themselves.
_PACKAGE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BuildInfo:
    """The version and commit of the running Womblex, and where each came from.

    ``commit`` is a sha or :data:`UNAVAILABLE`; ``source`` is one of ``git``,
    ``git-dirty``, ``stamp`` or ``unavailable``; ``reason`` is empty unless the
    commit is unavailable, in which case it says which of the two sources was
    missing. The structured form is kept for the run record P9 will assemble —
    the stamp carries the rendered string.
    """

    version: str
    commit: str
    source: str
    reason: str

    @property
    def commit_value(self) -> str:
        """The single string the run stamp carries.

        ``<sha>`` from a clean work tree or a stamped wheel, ``<sha>-dirty``
        from a modified one, ``unavailable:<reason>`` where neither source
        answered.
        """
        if self.commit == UNAVAILABLE:
            return f"{UNAVAILABLE}:{self.reason}"
        return f"{self.commit}-dirty" if self.source == "git-dirty" else self.commit


def _git(*args: str) -> tuple[str | None, str]:
    """Run ``git`` against the package directory: its output, or ``None`` and why.

    The two failures are worth telling apart in the record — no git binary is a
    different situation from a directory that is not a work tree — so they carry
    different reasons rather than one shrug.
    """
    try:
        # Fixed argv, no shell, and bounded by a timeout: the only variable
        # part is the subcommand this module itself supplies.
        completed = subprocess.run(
            ["git", "-C", str(_PACKAGE_DIR), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=True,
        )
    except FileNotFoundError:
        return None, "git-not-installed"
    except (OSError, subprocess.SubprocessError):
        return None, "not-a-git-work-tree"
    return completed.stdout.strip(), ""


def _git_commit() -> tuple[str | None, str]:
    """The HEAD sha of the work tree the package sits in, or ``None`` and why."""
    head, reason = _git("rev-parse", "HEAD")
    if not head:
        return None, reason or "not-a-git-work-tree"
    return head, ""


def _work_tree_is_dirty() -> bool:
    """True if the work tree has uncommitted changes.

    An unanswerable question is answered ``False``: the sha is already known at
    this point, and suffixing it on a failed check would be inventing the very
    caveat the suffix exists to report honestly.
    """
    status, _ = _git("status", "--porcelain")
    return bool(status)


def _stamped_commit() -> str | None:
    """The commit a build stamp names, or ``None`` if the package carries none.

    ``_build_stamp.py`` is generated at build time and is not in the repository,
    so its absence is the normal case rather than an error.
    """
    try:
        from womblex._build_stamp import COMMIT  # type: ignore[import-not-found]
    except ImportError:
        return None
    return str(COMMIT).strip() or None


@lru_cache(maxsize=1)
def build_info() -> BuildInfo:
    """Resolve the running build once per process.

    Cached because a run asks for the stamp on every file it writes, and this
    shells out. :func:`build_info.cache_clear` resets it, which only a test
    changing the environment under it needs.
    """
    head, reason = _git_commit()
    if head:
        return BuildInfo(__version__, head, "git-dirty" if _work_tree_is_dirty() else "git", "")
    if stamped := _stamped_commit():
        return BuildInfo(__version__, stamped, "stamp", "")
    return BuildInfo(__version__, UNAVAILABLE, UNAVAILABLE, reason)


def resolve_commit() -> str:
    """The commit string for the run stamp — never empty, never invented."""
    return build_info().commit_value


# ---------------------------------------------------------------------------
# Which image, if any, the run is executing inside
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageInfo:
    """The container image a run is executing inside, as far as it can be known.

    ``containerised`` is the only fact this module establishes on its own.
    ``reference`` and ``digest`` are read from the environment, because neither
    is discoverable from inside: a digest is content-addressed *after* the push,
    so it cannot be baked into the image it names, and a container cannot read
    its own labels. ``reason`` is empty only when a digest was established.
    """

    containerised: bool
    reference: str
    digest: str
    reason: str

    def as_record(self) -> dict:
        """The block the run record carries. Every field present, none invented."""
        return {
            "containerised": self.containerised,
            "reference": self.reference or None,
            "digest": self.digest or None,
            "reason": self.reason or None,
        }


def _is_containerised() -> bool:
    """Whether the process is running inside a container runtime.

    Asked of the runtime's own marker files rather than inferred from cgroups,
    which differ by kernel, runtime and orchestrator and would turn a plain
    fact into a guess. A false negative here is honest — it reports "not
    containerised" and records no digest, which is what an unrecognised runtime
    should produce rather than a claim about an image nobody named.
    """
    return any(marker.exists() for marker in _CONTAINER_MARKERS)


@lru_cache(maxsize=1)
def image_info() -> ImageInfo:
    """Resolve the running image once per process.

    Four outcomes, each a recorded fact rather than an absence:

    - **Not containerised.** No digest, and the reason says so — a local run is
      not a deployment that failed to identify itself.
    - **Containerised, nothing injected.** The deployment did not supply the
      reference; a locally built image is the ordinary case.
    - **Containerised, a tag.** The deployment named a tag, which does not
      identify bytes, so there is no digest to record.
    - **Containerised, a digest.** Recorded, with the full reference beside it.

    :func:`image_info.cache_clear` resets it, which only a test changing the
    environment under it needs.
    """
    containerised = _is_containerised()
    reference = os.environ.get(IMAGE_REF_ENV, "").strip()
    if not containerised:
        return ImageInfo(False, reference, "", "not running in a container")
    if not reference:
        return ImageInfo(
            True, "", "", f"{IMAGE_REF_ENV} was not supplied to the container",
        )
    name, separator, digest = reference.partition(_DIGEST_SEP)
    if not separator:
        return ImageInfo(
            True, reference, "",
            "the deployment names a tag, which does not identify an image",
        )
    if not name or not _DIGEST_RE.match(digest):
        return ImageInfo(
            True, reference, "",
            "the reference is not a name and a well-formed digest",
        )
    return ImageInfo(True, reference, digest, "")


__all__ = [
    "IMAGE_REF_ENV",
    "UNAVAILABLE",
    "BuildInfo",
    "ImageInfo",
    "build_info",
    "image_info",
    "resolve_commit",
]
