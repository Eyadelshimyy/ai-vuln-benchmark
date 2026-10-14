"""The running log's capture: what the semantics of "this thread's output, and
no one else's" have to be, which a journey can only show by accident."""

import io
import sys
import threading

from tracebi.pipeline import runlog


def test_capture_sees_this_threads_prints_and_not_another_threads(capsys):
    seen = []
    other_done = threading.Event()

    def elsewhere():
        print("from another thread")
        other_done.set()

    with runlog.capture(seen.append):
        print("mine")
        threading.Thread(target=elsewhere).start()
        other_done.wait(5)
    assert "".join(seen) == "mine\n"
    # ...and it still reached the terminal, which is where it was going.
    assert "mine" in capsys.readouterr().out


def test_an_inner_silent_capture_is_still_seen_by_the_outer_one(capsys):
    outer, inner = [], io.StringIO()
    with runlog.capture(outer.append):
        with runlog.capture(inner.write, echo=False):
            print("step output")
        print("after")
    assert inner.getvalue() == "step output\n"
    assert "".join(outer) == "step output\nafter\n"
    shown = capsys.readouterr().out
    assert "after" in shown and "step output" not in shown


def test_streams_are_put_back(capsys):
    before = sys.stdout
    with runlog.capture(lambda text: None):
        assert sys.stdout is not before
    assert sys.stdout is before


def test_a_log_stamps_whole_lines_holds_a_partial_one_and_keeps_a_progress_bars_last_frame(tmp_path):
    log = runlog.RunLog(str(tmp_path / "1.log"))
    log.write("loading")
    assert runlog.read_from(log.path) == ("", 0)          # no half lines handed out
    log.write(" rows\n10%\r50%\r100%\ndone")
    log.close()
    text, nxt = runlog.read_from(log.path)
    assert [line.split("  ", 1)[1] for line in text.splitlines()] == ["loading rows", "100%", "done"]
    assert all(line[2] == ":" for line in text.splitlines())     # mm:ss.s
    assert runlog.read_from(log.path, nxt) == ("", nxt)           # nothing new from where it left off


def test_a_chatty_script_cannot_fill_the_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog.RunLog, "MAX_BYTES", 200)
    log = runlog.RunLog(str(tmp_path / "1.log"))
    for i in range(100):
        log.line(f"row {i}")
    log.close()
    text, _ = runlog.read_from(log.path)
    assert len(text) < 400 and text.rstrip().endswith("[log full: output stops here]")
