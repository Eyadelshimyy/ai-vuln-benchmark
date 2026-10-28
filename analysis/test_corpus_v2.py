#!/usr/bin/env python3
"""Tests for the corpus-v2 application-level screen (static; no network).

Run:  python3 -m pytest analysis/test_corpus_v2.py -q
 or:  python3 analysis/test_corpus_v2.py
"""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app_screen as a  # noqa: E402


def _tree(files: dict) -> tempfile.TemporaryDirectory:
    td = tempfile.TemporaryDirectory()
    for rel, txt in files.items():
        p = Path(td.name) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
    return td


def _filler(n):
    return {f"pkg/m{i}.py": "x = 1\n" for i in range(n)}


def test_fastapi_app_is_app_level():
    with _tree({**_filler(10), "app/main.py": "from fastapi import FastAPI\napp = FastAPI()\n"
                "@app.get('/a')\ndef a(): ...\n@app.post('/b')\ndef b(): ...\n",
                "app/r.py": "@router.delete('/c')\ndef c(): ...\n"}) as d:
        r = a.screen_tree(Path(d))
        assert r["decision"] == "APP_LEVEL" and r["frameworks"] == ["fastapi"] and r["route_handlers"] == 3, r


def test_routes_in_tests_do_not_count():
    with _tree({**_filler(10), "app.py": "from flask import Flask\napp = Flask(__name__)\n@app.route('/')\ndef i(): ...\n",
                "tests/test_app.py": "@app.route('/x')\ndef x(): ...\n@app.route('/y')\ndef y(): ...\n"}) as d:
        r = a.screen_tree(Path(d))
        assert r["decision"] == "REJECT" and "route handlers" in r["reason"], r


def test_app_object_only_in_tests_is_rejected():
    with _tree({**_filler(10), "tests/conftest.py": "app = Flask(__name__)\n",
                "views.py": "@bp.route('/a')\ndef a(): ...\n@bp.route('/b')\ndef b(): ...\n@bp.route('/c')\ndef c(): ...\n"}) as d:
        r = a.screen_tree(Path(d))
        assert r["reason"] == "no web application object constructed", r


def test_django_project():
    with _tree({**_filler(10), "proj/settings.py": "ROOT_URLCONF = 'proj.urls'\n",
                "proj/urls.py": "urlpatterns = [path('a/', v), path('b/', v), re_path('c', v)]\n"}) as d:
        r = a.screen_tree(Path(d))
        assert r["decision"] == "APP_LEVEL" and r["frameworks"] == ["django"], r


def test_small_repo_rejected():
    with _tree({"app.py": "app = FastAPI()\n@app.get('/a')\n@app.get('/b')\n@app.get('/c')\ndef f(): ...\n"}) as d:
        assert "non-test .py files" in a.screen_tree(Path(d))["reason"]


def test_venv_and_vendor_ignored():
    with _tree({**_filler(10), ".venv/lib/flask/app.py": "app = Flask(__name__)\n@app.route('/')\n@app.route('/a')\n@app.route('/b')\n"}) as d:
        assert a.screen_tree(Path(d))["decision"] == "REJECT"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
