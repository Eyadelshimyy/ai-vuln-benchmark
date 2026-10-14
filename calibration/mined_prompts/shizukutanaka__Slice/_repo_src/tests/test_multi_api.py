"""Tests for multi-person API: REST /analyze?multi=1 and CLI --multi."""
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from http.server import ThreadingHTTPServer

from slice import __main__, bitmap, evaluate, knowledge, rest
from slice.bitmap import Bitmap

BG = (235, 235, 235, 255)


def _two_people_png() -> bytes:
    person, _ = evaluate.draw_case(120, 260)
    wide = Bitmap.new(320, 300, BG)
    for dx in (10, 190):
        for y in range(person.height):
            for x in range(person.width):
                p = person.get(x, y)
                if p != BG:
                    wide.set(dx + x, 20 + y, p)
    return bitmap.encode_png(wide)


def _serve(tmpdir):
    rest.Handler.store = knowledge.KnowledgeStore(tmpdir)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), rest.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=body, method="POST",
        headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


class TestMultiApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.httpd, cls.port = _serve(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls._tmp.cleanup()

    def test_rest_multi(self):
        res = _post(self.port, "/analyze?multi=1", _two_people_png())
        self.assertEqual(res["count"], 2)
        self.assertEqual(len(res["people"]), 2)
        self.assertEqual(res["people"][1]["people"]["index"], 1)

    def test_rest_multi_save(self):
        res = _post(self.port, "/analyze?multi=1&save=1",
                    _two_people_png())
        self.assertEqual(res["count"], 2)
        for p in res["people"]:
            self.assertIn("overlay_url", p)

    def test_rest_single_untouched(self):
        res = _post(self.port, "/analyze", _two_people_png())
        self.assertIn("skeleton", res)
        self.assertNotIn("people", res)

    def test_cli_multi(self):
        with tempfile.TemporaryDirectory() as td:
            img = os.path.join(td, "two.png")
            with open(img, "wb") as f:
                f.write(_two_people_png())
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = __main__.main(["analyze", img, "--multi"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(json.loads(out.getvalue())), 2)
            self.assertIn("people: 2", err.getvalue())


if __name__ == "__main__":
    unittest.main()
