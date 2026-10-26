#!/usr/bin/env python3
"""Render JobBot architecture diagrams for the README.

Requires Graphviz (``dot`` on PATH) and the optional ``docs`` dependency group::

    brew install graphviz   # or apt install graphviz
    make architecture

Outputs PNG under ``docs/images/``. Commit the rendered PNGs so the README
stays readable without regenerating. (Do not commit SVG: Graphviz embeds
absolute icon paths from the local diagrams install.)
"""

from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.generic.blank import Blank
from diagrams.generic.storage import Storage
from diagrams.onprem.client import Client, User
from diagrams.onprem.vcs import Github
from diagrams.programming.language import Bash, Python

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "images"


def _write(filename: str, title: str, *, direction: str, build) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = str(OUT_DIR / filename)
    graph_attr = {
        "fontsize": "14",
        "bgcolor": "white",
        "pad": "0.4",
        "splines": "true",
        "nodesep": "0.5",
        "ranksep": "0.75",
    }
    node_attr = {"fontsize": "11", "height": "0.6", "width": "1.25"}
    edge_attr = {"fontsize": "10"}
    with Diagram(
        title,
        filename=out,
        outformat="png",
        show=False,
        direction=direction,
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
    ):
        build()
    png = Path(out + ".png")
    if not png.is_file():
        raise SystemExit(f"diagrams did not write {png}; is Graphviz (dot) on PATH?")
    return png


def render_overview() -> Path:
    """Landscape overview: SoT, two loops, map vs traveller, autopoiesis."""

    def build() -> None:
        human = User("you\n(HITL)")
        cli = Python("jobbot CLI")

        with Cluster("Source of truth (local, never invent)"):
            profile = Storage("profile.yaml\n→ Candidate")
            latex = Storage("LaTeX / PDF\n(presentation)")

        with Cluster("Standing presence (loop 1)"):
            advise = Blank("cv advise")
            sync = Blank("cv sync #43")
            signup = Blank("signup #44")
            recon = Blank("recon #45")
            status = Blank("status")

        with Cluster("Apply / cargo (loop 2)"):
            discover = Blank("search / sweep\nget / torre")
            job = Blank("JobPosting\nJxxxx")
            match = Blank("match")
            cv_job = Blank("cv build --job")
            apply = Blank("application apply")

        with Cluster("Job sources"):
            sources = Client("Indeed · LinkedIn\nGoB · Torre · fixture")

        with Cluster("Application adapters (Domain → adapters)"):
            adapters = Blank("Greenhouse · Lever\nAshby · Workday\nIndeed · email · GoB")

        with Cluster("Collaborative map (export, no PII)"):
            companies = Storage("companies.yaml")
            portals = Storage("portals.yaml")
            recruiters = Storage("recruiters.yaml")

        with Cluster("Local only (traveller)"):
            db = Storage("SQLite")
            browser = Client("browser / CDP")
            output = Storage("output/")
            forms = Storage("form_knowledge")

        with Cluster("Autopoiesis"):
            fail = Blank("ops_failures")
            work = Bash("failure work")
            gh = Github("issue → PR → re-run")

        human >> Edge(label="confirm · submit\nCAPTCHA · password") >> cli
        cli >> profile
        profile >> latex

        profile >> advise >> sync >> status
        sync >> signup
        recon >> Edge(label="ATS + forms") >> companies
        recon >> forms

        sources >> discover >> job >> match >> cv_job >> apply
        profile >> Edge(label="facts only") >> match
        profile >> cv_job
        apply >> Edge(label="prefill;\nyou submit") >> adapters
        job >> db
        cv_job >> output
        apply >> forms
        status >> browser

        discover >> Edge(label="candidates") >> companies
        sync >> Edge(label="active portals") >> adapters
        companies >> portals
        portals >> recruiters

        cli >> Edge(label="exit ≠ 0\nclass+message", style="dashed") >> fail
        fail >> work >> gh
        gh >> Edge(label="goto 0", style="dashed", constraint="false") >> cli

    return _write("architecture", "JobBot architecture", direction="LR", build=build)


def render_detail() -> Path:
    """Detail of adapters, knowledge, and the failure lane."""

    def build() -> None:
        profile = Storage("profile.yaml\n(Candidate)")

        with Cluster("JobSourceAdapter"):
            indeed = Client("IndeedJobSource")
            linkedin = Client("LinkedInPostJobSource")
            gob_src = Client("GetOnBoardJobSource")
            torre = Client("Torre search")
            career = Blank("career HTML\n(--fixture)")
            get = Blank("jobbot get URL\n(+ park / parked)")

        with Cluster("Cargo pipeline"):
            repo = Storage("JobRepository\n(SQLite)")
            match = Blank("jobs match /\nshortlist")
            package = Blank("application\nprepare")
            apply = Blank("application apply\n--apply")

        with Cluster("ApplicationPortalAdapter"):
            gh_ats = Blank("Greenhouse")
            lever = Blank("Lever")
            ashby = Blank("Ashby")
            workday = Blank("Workday")
            indeed_app = Blank("Indeed Apply\n/ external ATS")
            email = Blank("Email → Gmail\nattach PDF")
            gob_app = Blank("GetOnBoard\nprofile + CV")

        with Cluster("Standing presence"):
            advise = Blank("cv advise")
            sync = Blank("cv sync #43")
            signup = Blank("signup #44")
            recon = Blank("recon #45")
            status = Blank("status\n(evidence)")

        with Cluster("Shared knowledge (export)"):
            companies = Storage("companies.yaml")
            portals = Storage("portals.yaml")
            recruiters = Storage("recruiters.yaml")
            forms = Storage("form_knowledge\n(local only)")

        with Cluster("Autopoiesis (URL gaps, portal crashes, …)"):
            recorded = Blank("_RecordedExit\nclass + message")
            failures = Storage("ops_failures\nFxxxx")
            work = Bash("ops failure work")
            issue = Github("gh issue (HITL)")
            pr = Github("branch → PR\ntriage → re-run")

        indeed >> get
        linkedin >> get
        gob_src >> get
        career >> get
        torre >> repo
        get >> repo
        repo >> match >> package >> apply
        profile >> Edge(label="facts") >> match
        profile >> package

        apply >> gh_ats
        apply >> lever
        apply >> ashby
        apply >> workday
        apply >> indeed_app
        apply >> email
        sync >> gob_app
        signup >> gob_app

        profile >> advise >> sync >> status
        recon >> companies
        recon >> forms
        apply >> forms
        get >> Edge(label="candidates") >> companies
        companies >> portals
        portals >> recruiters

        get >> Edge(label="UnsupportedPortal\nFetchError", style="dashed") >> recorded
        recorded >> failures >> work >> issue >> pr

    return _write(
        "architecture-detail",
        "JobBot — adapters, knowledge, autopoiesis",
        direction="TB",
        build=build,
    )


def main() -> None:
    overview = render_overview()
    detail = render_detail()
    for path in (overview, detail):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
