"""Drafts: the security edges. The happy path is the journey in
tests/e2e/test_web_journey.py; these pin what a draft must never do."""

import json

import pytest

from tracebi import drafts

_REPORT_JSON = json.dumps({"name": "r", "title": "R"})


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("TRACEBI_DRAFTS_DIR", "TRACEBI_REPORTS_DIR", "TRACEBI_MODELS_DIR",
                "TRACEBI_LIBRARY_MOUNTS"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


class TestWhatADraftMayHold:
    @pytest.mark.parametrize("path", ["../x", "a/../../x", "/etc/x", "a\\b", "_hidden", ".git"])
    def test_a_report_path_cannot_climb_out(self, project, path):
        with pytest.raises(drafts.DraftError):
            drafts.start_draft("alice", "reports", path)
        assert not (project / "drafts").exists()

    @pytest.mark.parametrize("name", ["../x", "Bad-Name", "a/b", ""])
    def test_a_model_name_is_a_plain_word(self, project, name):
        with pytest.raises(drafts.DraftError):
            drafts.start_draft("alice", "models", name)

    @pytest.mark.parametrize("owner", ["..", "../x", "a/b", "Alice", ""])
    def test_an_owner_is_a_slug(self, project, owner):
        with pytest.raises(drafts.DraftError):
            drafts.draft_dir(owner, "reports", "r")
        assert not drafts.slug_owner(owner).startswith(".")

    @pytest.mark.parametrize("file", ["report.py", "script.js", "../x", "sub/report.json", "notes.txt"])
    def test_only_the_three_report_files_are_writable(self, project, file):
        drafts.start_draft("alice", "reports", "r")
        with pytest.raises(drafts.DraftError):
            drafts.write_draft_file("alice", "reports", "r", file, "x")
        assert [p.name for p in (project / "drafts/alice/reports/r").iterdir()] == []

    def test_a_model_draft_holds_only_its_own_json(self, project):
        drafts.start_draft("alice", "models", "m")
        with pytest.raises(drafts.DraftError):
            drafts.write_draft_file("alice", "models", "m", "other.json", "{}")

    def test_a_file_is_capped_at_512_kb(self, project):
        drafts.start_draft("alice", "reports", "r")
        drafts.write_draft_file("alice", "reports", "r", "style.css", "a" * drafts.MAX_FILE_BYTES)
        with pytest.raises(drafts.DraftError, match="512 KB"):
            drafts.write_draft_file("alice", "reports", "r", "style.css",
                                    "a" * (drafts.MAX_FILE_BYTES + 1))
        # Bytes, not characters.
        with pytest.raises(drafts.DraftError, match="512 KB"):
            drafts.write_draft_file("alice", "reports", "r", "style.css",
                                    "é" * drafts.MAX_FILE_BYTES)

    @pytest.mark.parametrize("extra", ["report.py", "script.js"])
    def test_a_report_that_runs_code_cannot_be_drafted(self, project, extra):
        pkg = project / "reports" / "sales"
        pkg.mkdir(parents=True)
        (pkg / "report.json").write_text(_REPORT_JSON)
        (pkg / "template.html").write_text("<p>x</p>")
        (pkg / extra).write_text("1")
        with pytest.raises(drafts.DraftError, match="laptop"):
            drafts.start_draft("alice", "reports", "sales", from_published=True)
        assert not (project / "drafts").exists()


class TestPublishing:
    def test_a_draft_that_does_not_render_publishes_nothing(self, project):
        pkg = project / "reports" / "sales"
        pkg.mkdir(parents=True)
        (pkg / "report.json").write_text(_REPORT_JSON)
        (pkg / "template.html").write_text("<p>old</p>")
        drafts.start_draft("alice", "reports", "sales")
        drafts.write_draft_file("alice", "reports", "sales", "report.json", "{not json")
        drafts.write_draft_file("alice", "reports", "sales", "template.html", "<p>new</p>")
        with pytest.raises(drafts.DraftError, match="does not render"):
            drafts.publish_draft("alice", "reports", "sales", "alice", models={})
        assert (pkg / "template.html").read_text() == "<p>old</p>"
        assert not (project / ".tracebi").exists()

    def test_a_draft_cannot_replace_a_published_report_py(self, project):
        pkg = project / "reports" / "sales"
        pkg.mkdir(parents=True)
        (pkg / "report.py").write_text("print('server code')")
        drafts.start_draft("alice", "reports", "sales")
        with pytest.raises(drafts.DraftError, match="report.py"):
            drafts.publish_draft("alice", "reports", "sales", "alice", models={})

    def test_a_model_draft_must_name_itself(self, project):
        drafts.start_draft("alice", "models", "m")
        drafts.write_draft_file("alice", "models", "m", "m.json", json.dumps({"name": "other"}))
        with pytest.raises(drafts.DraftError, match="name"):
            drafts.publish_draft("alice", "models", "m", "alice")
        assert not (project / "models").exists()


class TestWhoMayTouchADraft:
    """Over HTTP with identities from a proxy: an analyst has their own drafts,
    not each other's; an admin has everyone's; a viewer publishes nothing."""

    @pytest.fixture
    def client(self, project, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tracebi.web.api.auth import ProxyHeaderAuthMiddleware
        from tracebi.web.api.routers import drafts as drafts_router

        monkeypatch.setenv("TRACEBI_AUTH_ROLE_HEADER", "X-Role")
        monkeypatch.delenv("TRACEBI_AUTH_ROLE_MAP", raising=False)
        monkeypatch.delenv("TRACEBI_AUTH_DEFAULT_ROLE", raising=False)
        app = FastAPI()
        app.include_router(drafts_router.router, prefix="/api")
        app.add_middleware(ProxyHeaderAuthMiddleware, header="X-User")
        drafts.start_draft("alice", "reports", "finance/weekly")
        drafts.write_draft_file("alice", "reports", "finance/weekly", "template.html", "<p>secret</p>")
        return TestClient(app)

    @staticmethod
    def _as(user, role):
        return {"X-User": user, "X-Role": role}

    @pytest.mark.parametrize("tail", ["version", "files", "preview"])
    def test_another_analyst_cannot_read_it(self, client, tail):
        url = f"/api/drafts/alice/reports/finance/weekly/{tail}"
        assert client.get(url, headers=self._as("alice", "analyst")).status_code != 403
        denied = client.get(url, headers=self._as("bob", "analyst"))
        assert denied.status_code == 403
        assert "secret" not in denied.text

    def test_another_analyst_cannot_publish_or_delete_it(self, client, project):
        url = "/api/drafts/alice/reports/finance/weekly"
        assert client.post(f"{url}/publish", headers=self._as("bob", "analyst")).status_code == 403
        assert client.delete(url, headers=self._as("bob", "analyst")).status_code == 403
        assert not (project / "reports").exists()
        assert (project / "drafts/alice/reports/finance/weekly").is_dir()

    def test_a_viewer_cannot_publish_or_delete_even_their_own(self, client):
        drafts.start_draft("carol", "reports", "mine")
        url = "/api/drafts/carol/reports/mine"
        assert client.get(f"{url}/files", headers=self._as("carol", "viewer")).status_code == 200
        assert client.post(f"{url}/publish", headers=self._as("carol", "viewer")).status_code == 403
        assert client.delete(url, headers=self._as("carol", "viewer")).status_code == 403

    def test_only_the_owner_at_analyst_or_above_can_start_or_write(self, client, project):
        """Starting and writing are writes: a viewer gets 403 on both, even on
        their own draft, and another analyst cannot write into someone else's."""
        url = "/api/drafts/alice/reports/finance/weekly/files/template.html"
        body = {"content": "<p>changed</p>"}
        start = {"kind": "reports", "path": "finance/weekly"}
        assert client.put(url, json=body, headers=self._as("alice", "viewer")).status_code == 403
        assert client.put(url, json=body, headers=self._as("bob", "analyst")).status_code == 403
        assert client.post("/api/drafts", json=start, headers=self._as("carol", "viewer")).status_code == 403
        assert not (project / "drafts/carol").exists()
        assert (project / "drafts/alice/reports/finance/weekly/template.html").read_text() == "<p>secret</p>"

        assert client.put(url, json=body, headers=self._as("alice", "analyst")).status_code == 200
        assert client.put(url, json={"content": "<p>root</p>"},
                          headers=self._as("root", "admin")).status_code == 200

    def test_the_list_is_your_own_and_an_admin_sees_all(self, client):
        drafts.start_draft("bob", "reports", "other")
        mine = client.get("/api/drafts", headers=self._as("bob", "analyst")).json()["drafts"]
        assert [(d["owner"], d["path"]) for d in mine] == [("bob", "other")]
        everyone = client.get("/api/drafts", headers=self._as("root", "admin")).json()["drafts"]
        assert sorted(d["owner"] for d in everyone) == ["alice", "bob"]
        assert client.get("/api/drafts/alice/reports/finance/weekly/files",
                          headers=self._as("root", "admin")).status_code == 200


def test_two_people_never_share_a_drafts_folder():
    """Owner folders come from sign-in identities. A replaced character must not
    let two people land in one folder and read each other's drafts."""
    from tracebi.drafts import slug_owner

    people = ["a/b@x.com", "a-b@x.com", "a b@x.com", "DOMAIN\\ann", "domain-ann",
              "ann+bi@x.com", "ann-bi@x.com"]
    folders = [slug_owner(p) for p in people]
    assert len(set(folders)) == len(people), folders
    assert all(slug_owner(f) == f for f in folders), "a folder name is its own slug"
