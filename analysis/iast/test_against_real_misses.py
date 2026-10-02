"""
Runs the IAST harness against reconstructions of the two real invoke/fabric
functions from this session's SAST baseline run:

  - start(): taint_scan.py CORRECTLY flagged this one (literal shell=True
    reaching Popen directly) -- included here as a sanity check that IAST
    agrees with the static scanner where the static scanner already works.

  - local(): taint_scan.py could NOT scan this one at all -- its signature
    is `def local(self, *args, **kwargs)`, and the scanner's seeding logic
    only reads named positional/keyword params, silently skipping *args/
    **kwargs. This is the real test: does IAST catch what SAST structurally
    couldn't even attempt?

Both function bodies below are reconstructed from the real invoke/fabric
source pasted into this session (not invented) -- trimmed to what's needed
to run standalone (no network/real subprocess dependencies beyond what the
patched sinks intercept).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from harness import run_with_taint  # noqa: E402


# --- Reconstruction of invoke's Runner.start() (non-pty branch) ---
# Real source (pasted this session, invoke/runners.py):
#   self.process = Popen(command, shell=True, executable=shell, env=env,
#                         stdout=PIPE, stderr=PIPE, stdin=PIPE)
class FakeRunner:
    def __init__(self):
        self.using_pty = False

    def start(self, command, shell, env):
        from subprocess import Popen, PIPE
        if self.using_pty:
            raise NotImplementedError("pty branch not needed for this test")
        else:
            self.process = Popen(
                command, shell=True, executable=shell, env=env,
                stdout=PIPE, stderr=PIPE, stdin=PIPE,
            )


# --- Reconstruction of invoke's Context.local() / Runner chain ---
# Real source (pasted this session, invoke/__init__.py + context.py):
#   def local(self, *args, **kwargs):
#       return super().run(*args, **kwargs)
# `run()` ultimately reaches the same Popen-based execution path as start()
# above -- that's the real invoke architecture (local() -> run() -> a
# Runner.start()-shaped call). Reconstructed minimally here: local()
# forwards whatever it's given, several frames deep, to a real sink.
class FakeContext:
    def local(self, *args, **kwargs):
        return self._run_impl(*args, **kwargs)

    def _run_impl(self, command, **kwargs):
        # Stands in for the real multi-layer Runner machinery invoke uses
        # between Context.run() and the actual Popen call.
        runner = FakeRunner()
        runner.start(command, shell="/bin/sh", env={})


def main():
    print("=" * 70)
    print("TEST 1: start() -- SAST already caught this one (sanity check)")
    print("=" * 70)
    from tainted import new_marker
    marker = new_marker()
    runner = FakeRunner()
    result = run_with_taint(runner.start, args=[marker, "/bin/sh", {}], marker=marker)
    print(f"  triggered: {result.triggered}")
    for h in result.hits:
        print(f"  HIT: {h.cwe} via {h.sink} -- {h.detail[:100]}")
    if result.error:
        print(f"  error: {result.error}")

    print()
    print("=" * 70)
    print("TEST 2: local(self, *args, **kwargs) -- SAST could NOT scan this")
    print("         (no named params for *args/**kwargs to seed as tainted)")
    print("=" * 70)
    ctx = FakeContext()
    marker2 = new_marker()
    # The taint goes into the FIRST POSITIONAL ARG passed through *args --
    # this is the point: we don't need to know/declare a parameter NAME at
    # all, we just call it the way a real caller (or attacker) would.
    result2 = run_with_taint(ctx.local, args=[marker2], marker=marker2)
    print(f"  triggered: {result2.triggered}")
    for h in result2.hits:
        print(f"  HIT: {h.cwe} via {h.sink} -- {h.detail[:100]}")
    if result2.error:
        print(f"  error: {result2.error}")


if __name__ == "__main__":
    main()
