"""
CWE-22 (Path Traversal) IAST tests, mirroring test_against_real_misses.py and
test_cwe89.py's structure: sanity-check cases SAST already catches, plus one
real case SAST is STRUCTURALLY UNABLE to catch at all.

TEST 1: khoj's unauthenticated path traversal in /home/{file_path} (GHSA-62mm-
xwmv-crhg, real, disclosed). taint_scan.py catches this NOW -- it already has
"FileResponse" in PATH_SINK_NAMES (added from this exact case in an earlier
session) and file_path is a plain function parameter, no cross-method taint
needed. Verified directly: running taint_scan.py on this reconstruction
produces one finding. Included as a sanity check.

TEST 2: nltk's StreamBackedCorpusView._open() (GHSA-x5ph-mj9p-rfr8, CVE-2026-
63312, real, disclosed) -- "Bypasses pathsec.ENFORCE". The real bug: nltk has
its own security wrapper (nltk.pathsec.open) used almost everywhere, but this
one method still called the bare builtin open(self._fileid, "rb") directly.
taint_scan.py catches this too -- self._fileid is set from a constructor
param and read in a different method, the same same-class self.attr pattern
already fixed for PraisonAI. Verified directly: 2 findings. Also a sanity
check, not a miss -- both TEST 1 and TEST 2 exist to confirm IAST agrees with
SAST wherever SAST already gets it right, before looking at where it doesn't.

TEST 3: GitPython's clone()/clone_from() missing "--separate-git-dir" from
its unsafe_git_clone_options denylist (GHSA-8mcc-hrx5-hvxc, CVE-2026-78677,
real, disclosed). THIS IS THE REAL HEADLINE MISS for CWE-22.

Why SAST cannot catch this one, verified directly (0 findings from
taint_scan.py on the reconstruction below): this isn't a missing-sink-name
gap like TEST 1/2 were -- there is no open()/write_text()/write_bytes()/
FileResponse() call anywhere in the vulnerable code path at all. The real
"write" is git itself creating a .git directory structure at an attacker-
chosen path, as a SIDE EFFECT of a subprocess call that git interprets. The
actual Python-level call is `subprocess.Popen(["git", "clone", ...,
"--separate-git-dir=<path>", ...])` -- no shell=True, so taint_scan.py's
CWE-78 gate (which only fires on bare os.system/popen or a LITERAL shell=True
constant) doesn't fire either. The vulnerability is a missing entry in a
security DENYLIST (a list of git CLI flags considered dangerous), not a
dataflow/sink-naming problem -- there is structurally no sink name a static
scanner could add to catch "this specific command-line flag was omitted from
that specific list two files away."

Why IAST catches it anyway: IAST doesn't know or care about denylists either
-- it just watches the REAL subprocess.Popen() call that actually runs, and
checks whether the taint marker shows up anywhere in its real argv. Since the
denylist bug lets the tainted path straight through into the real git
command line, it shows up in the Popen args and the EXISTING CWE-78 sink
(no new sink code needed) fires on it.

Honest note on the CWE label: the hit below is logged as CWE-78 (that's
literally which sink fired -- the generic subprocess.Popen patch), even
though the real, disclosed vulnerability is filed as CWE-22 (arbitrary
directory write). This is worth stating plainly rather than relabeling the
hit to make the CWE match: IAST's sinks are organized by what dangerous
REAL OPERATION happens (a process gets spawned, a file gets opened, a query
gets prepared), not by which CWE number a human later assigned to the bug.
One real sink -- "a subprocess actually ran with this argv" -- can be the
detection mechanism for vulnerabilities nominally filed under different CWEs,
because from the machine's point of view they're the same kind of event:
attacker-controlled data reached a real, dangerous system call.

All three reconstructions below are built from the real pre-fix source
(fetched this session via GitHub's API/web for GHSA-62mm-xwmv-crhg and
GHSA-x5ph-mj9p-rfr8's JSON, and via direct GitHub commit/raw-file fetches for
GHSA-8mcc-hrx5-hvxc's denylist, check_unsafe_options, _option_candidates, and
transform_kwarg/transform_kwargs implementations), trimmed to what's needed
to run standalone -- same standard as every other test file in this project.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from harness import run_with_taint  # noqa: E402
from tainted import new_marker  # noqa: E402


# --- TEST 1: khoj's home_static_files (real source, this session) ---
# Real source: src/khoj/routers/web_client.py, pre-fix.
import tempfile  # noqa: E402

_fake_home_dir = Path(tempfile.mkdtemp(prefix="khoj_fake_home_"))


class _FakeConstants:
    # Stands in for khoj.utils.constants -- only home_directory matters here.
    home_directory = _fake_home_dir


def home_static_files(file_path: str):
    """Serve static files from the home landing page directory"""
    from fastapi.responses import FileResponse

    return FileResponse(_FakeConstants.home_directory / file_path)


def _serve_home_static_file(file_path: str):
    """FileResponse only actually opens the file when the ASGI response is
    SENT, not when it's constructed (verified directly: Starlette's
    FileResponse.__call__ -> _handle_simple -> anyio.open_file(self.path) ->
    a worker-thread call to the real builtins.open -- confirmed by reading
    starlette's source this session). So testing this sink for real means
    actually awaiting the response through a minimal ASGI scope/receive/send,
    the same way a real web server would -- just calling the route function
    and getting a FileResponse object back proves nothing on its own."""
    import asyncio

    response = home_static_files(file_path)

    async def _run():
        scope = {"type": "http", "method": "GET", "headers": []}

        async def receive():
            return {"type": "http.request"}

        sent = []

        async def send(message):
            sent.append(message)

        await response(scope, receive, send)

    asyncio.run(_run())


# --- TEST 2: nltk's StreamBackedCorpusView._open (real source, this session) ---
# Real source: nltk/corpus/reader/util.py, pre-fix, trimmed to __init__ + _open
# (the rest of the class -- block readers, caching, iteration -- doesn't
# affect whether the tainted fileid reaches the real open() sink).
class StreamBackedCorpusView:
    def __init__(self, fileid, block_reader=None, startpos=0, encoding="utf8"):
        if block_reader:
            self.read_block = block_reader
        self._toknum = [0]
        self._filepos = [startpos]
        self._encoding = encoding
        self._len = None
        self._fileid = fileid
        self._stream = None

    def _open(self):
        """Open the file stream associated with this corpus view."""
        if self._encoding:
            # Real source wraps this in SeekableUnicodeStreamReader; irrelevant
            # to whether the tainted path reaches the real open() sink.
            self._stream = open(self._fileid, "rb")
        else:
            self._stream = open(self._fileid, "rb")


# --- TEST 3: GitPython's clone()/clone_from() denylist bypass (real source,
# fetched this session from git/cmd.py and git/repo/base.py at commit
# 9729ed3b948f2bde09f1f188c5311e172212b67e, the parent of the fix commit). ---
class UnsafeOptionError(Exception):
    pass


def _dashify(string):
    return string.replace("_", "-")


class _Git:
    """Reconstruction of GitPython's git.cmd.Git classmethods actually
    involved in the denylist check -- not the full Git command-wrapper
    class."""

    @classmethod
    def _canonicalize_option_name(cls, option):
        return option.lstrip("-").replace("-", "").lower()

    @classmethod
    def check_unsafe_options(cls, options, unsafe_options, clusterable_short_options="46flnqsv"):
        canonical_unsafe_options = {cls._canonicalize_option_name(o): o for o in unsafe_options}
        unsafe_short_options = {
            canonical: option
            for canonical, option in canonical_unsafe_options.items()
            if option.startswith("-") and not option.startswith("--") and len(canonical) == 1
        }
        clusterable_short_options_set = frozenset(clusterable_short_options)
        options_are_kwargs = all(not option.startswith("-") for option in options)
        for option in options:
            candidate = cls._canonicalize_option_name(option)
            if not candidate:
                continue
            unsafe_option = canonical_unsafe_options.get(candidate)
            if unsafe_option is not None:
                raise UnsafeOptionError(f"{unsafe_option} is not allowed, use `allow_unsafe_options=True` to allow it.")
            option_token = option.split("=", 1)[0].split(None, 1)[0]
            if option_token.startswith("-") and not option_token.startswith("--"):
                for option_char in option_token[1:]:
                    unsafe_option = unsafe_short_options.get(option_char)
                    if unsafe_option is not None:
                        raise UnsafeOptionError(f"{unsafe_option} is not allowed, use `allow_unsafe_options=True` to allow it.")
                    if option_char not in clusterable_short_options_set:
                        break
            if not (option.startswith("--") or (options_are_kwargs and len(candidate) > 1)):
                continue
            for canonical, unsafe_option in canonical_unsafe_options.items():
                if canonical.startswith(candidate):
                    raise UnsafeOptionError(f"{unsafe_option} is not allowed, use `allow_unsafe_options=True` to allow it.")

    @classmethod
    def _option_candidates(cls, args=(), kwargs=None):
        options = [o for o in args if isinstance(o, str) and o.startswith("-")]
        if kwargs:
            for key, value in kwargs.items():
                values = value if isinstance(value, (list, tuple)) else (value,)
                if any(v is True or (v is not False and v is not None) for v in values):
                    options.append(f"--{_dashify(str(key))}")
        return options


# PRE-FIX denylist, exactly as it was -- "--separate-git-dir" is MISSING.
# The real fix commit's only change to this list was adding that one entry.
unsafe_git_clone_options = [
    "--upload-pack",
    "-u",
    "--config",
    "-c",
    "--template",
    "--bundle-uri",
]


def _transform_kwarg(name, value, split_single_char_options=True):
    if len(name) == 1:
        if value is True:
            return ["-%s" % name]
        elif value not in (False, None):
            if split_single_char_options:
                return ["-%s" % name, "%s" % value]
            return ["-%s%s" % (name, value)]
    else:
        if value is True:
            return ["--%s" % _dashify(name)]
        elif value is not False and value is not None:
            return ["--%s=%s" % (_dashify(name), value)]
    return []


def _transform_kwargs(**kwargs):
    args = []
    for k, v in kwargs.items():
        args += _transform_kwarg(k, v)
    return args


def clone_from(url, path, allow_unsafe_options=False, **kwargs):
    """Faithful, minimal reconstruction of GitPython's Repo._clone()
    (git/repo/base.py): validate caller kwargs against the unsafe-options
    denylist, then build and run the real git subprocess command. The real
    method also handles progress callbacks, stdout/stderr pipes, and actual
    Repo object construction -- none of that changes whether the tainted
    value reaches the real subprocess call, so it's left out."""
    import subprocess

    if not allow_unsafe_options:
        _Git.check_unsafe_options(
            options=_Git._option_candidates([], kwargs),
            unsafe_options=unsafe_git_clone_options,
        )
    # Real code: `git.clone(multi, "--", clone_url, clone_path, ..., **kwargs)`,
    # which internally (Git._call_process -> transform_kwargs) turns kwargs
    # into real command-line flags before invoking the subprocess.
    argv = ["git", "clone"] + _transform_kwargs(**kwargs) + ["--", url, path]
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    print("=" * 70)
    print("TEST 1: khoj home_static_files -- SAST catches this now")
    print("        (sanity check)")
    print("=" * 70)
    marker1 = new_marker()
    # The target file has to actually EXIST on disk: Starlette's FileResponse
    # calls os.stat() (unpatched -- it's not a sink, just a metadata check)
    # before it ever calls open(), and raises if the file isn't there. This
    # mirrors the real attack: the attacker picks a file_path that resolves
    # to a real file somewhere on the server (e.g. ../../../etc/passwd),
    # which obviously exists -- it's not the existence check that's the bug,
    # it's the missing containment check on WHERE that path is allowed to be.
    (_fake_home_dir / marker1).write_text("dummy content for the IAST sink test")
    result1 = run_with_taint(_serve_home_static_file, args=[marker1], marker=marker1)
    print(f"  triggered: {result1.triggered}")
    for h in result1.hits:
        print(f"  HIT: {h.cwe} via {h.sink} -- {h.detail[:100]}")
    if result1.error:
        print(f"  error: {result1.error}")

    print()
    print("=" * 70)
    print("TEST 2: nltk StreamBackedCorpusView._open -- SAST catches this now")
    print("        (sanity check)")
    print("=" * 70)
    marker2 = new_marker()

    def build_and_open():
        view = StreamBackedCorpusView(marker2)
        view._open()

    result2 = run_with_taint(build_and_open, marker=marker2)
    print(f"  triggered: {result2.triggered}")
    for h in result2.hits:
        print(f"  HIT: {h.cwe} via {h.sink} -- {h.detail[:100]}")
    if result2.error:
        print(f"  error: {result2.error}")

    print()
    print("=" * 70)
    print("TEST 3: GitPython clone_from() -- SAST finds NOTHING here (0")
    print("        findings, verified) -- the real headline CWE-22 miss")
    print("=" * 70)
    marker3 = new_marker()
    result3 = run_with_taint(
        clone_from,
        args=["https://example.com/repo.git", "/tmp/fake-clone-dest"],
        kwargs={"separate_git_dir": f"/tmp/{marker3}-attacker-dir"},
        marker=marker3,
    )
    print(f"  triggered: {result3.triggered}")
    for h in result3.hits:
        print(f"  HIT: {h.cwe} via {h.sink} -- {h.detail[:150]}")
    if result3.error:
        print(f"  error: {result3.error}")

    print()
    print("=" * 70)
    print("TEST 4: GitPython clone_from(), NEGATIVE CONTROL -- no dangerous")
    print("        kwarg at all, marker not present anywhere in this call")
    print("=" * 70)
    marker4 = new_marker()
    result4 = run_with_taint(
        clone_from,
        args=["https://example.com/safe-repo.git", "/tmp/fake-clone-dest-2"],
        marker=marker4,
    )
    print(f"  triggered: {result4.triggered}  (expected: False)")
    for h in result4.hits:
        print(f"  HIT: {h.cwe} via {h.sink} -- {h.detail[:150]}")
    if result4.error:
        print(f"  error: {result4.error}")


if __name__ == "__main__":
    main()
