"""Run with python3 -B -m unittest discover -s <this directory> -v."""

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parent.parent
SETUP = RUNNER.parent / "orchestration-setup"
DISCOVERY = RUNNER.parent / "harness-discovery"
sys.path.insert(0, str(RUNNER / "scripts"))
from render_dashboard import render, skill_version


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.ids = []
        self.rows = {}
        self.row_text = {}
        self.work_states = {}
        self.groups = {}
        self.group = None
        self.contexts = {}
        self.row = None
        self.refresh = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("http-equiv") == "refresh":
            self.refresh = attrs["content"]
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if "data-context" in attrs:
            packet = json.loads(attrs["data-context"])
            key = packet["decision"]["id"] if "decision" in packet else packet["tasks"][0]["id"]
            self.contexts[key] = packet
        if tag == "tbody":
            self.group = attrs.get("class", "")
            self.groups[self.group] = []
        if tag == "tr" and attrs.get("id", "").startswith("row-"):
            self.row = attrs["id"]
            self.rows[self.row] = []
            self.row_text[self.row] = []
            self.work_states[self.row] = attrs.get("data-work-state")
            self.groups[self.group].append(self.row)
        if self.row and attrs.get("role") == "img":
            self.rows[self.row].append(attrs["aria-label"])

    def handle_data(self, text):
        if self.row:
            self.row_text[self.row].append(text)

    def handle_endtag(self, tag):
        if tag == "tr":
            self.row = None


STAGE_LABELS = ("Defined", "Researched", "Implemented", "Submitted", "Reviewed", "Verified", "Deployed", "Accepted")


def stage_row(*states):
    return [f"{label}: {state}" for label, state in zip(STAGE_LABELS, states)]


OUT = "Not in scope"


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((RUNNER / "evals" / "dashboard-preview.json").read_text())
        self.data["skill_version"] = skill_version()

    def test_scoped_readiness_and_delivery_are_visibly_distinct(self):
        page = Page(render(self.data))
        self.assertEqual(page.rows["row-UI-101"], stage_row("Done", OUT, "Done", OUT, OUT, OUT, OUT, OUT))
        self.assertEqual(page.rows["row-SHIP-107"], stage_row(*["Done"] * 8))
        self.assertEqual(page.rows["row-OPS-105"], stage_row("Done", OUT, "Done", "Blocked", OUT, OUT, OUT, OUT))
        self.assertIn("Submitted — blocked", page.row_text["row-OPS-105"])
        self.assertIn("Target: Submitted", page.row_text["row-OPS-105"])
        self.assertIn("Implemented ✓", page.row_text["row-UI-101"])
        self.assertIn("Accepted ✓", page.row_text["row-SHIP-107"])
        self.assertEqual(len(page.rows), 11)
        self.assertEqual(len(page.ids), len(set(page.ids)))

    def test_grouped_ledger_distinguishes_resolution_from_activity_and_delivery(self):
        html = render(self.data)
        page = Page(html)
        self.assertEqual(page.groups, {
            "group-attention": ["row-REVIEW-103", "row-OPS-105", "row-ENV-106"],
            "group-moving": ["row-API-102", "row-API-104"],
            "group-resolved": ["row-UI-101", "row-SHIP-107", "row-TRIAGE-108", "row-UI-109", "row-UI-110", "row-UI-111"],
        })
        for row, label, stage in (("row-TRIAGE-108", "No change needed", "Researched ✓"),
                                  ("row-UI-109", "Cancelled", "Stopped after Defined"),
                                  ("row-UI-110", "Postponed", "Stopped after Defined")):
            with self.subTest(row=row):
                self.assertIn(label, page.row_text[row])
                self.assertIn(stage, page.row_text[row])
                self.assertEqual(page.rows[row][2:], stage_row(*[OUT] * 8)[2:])
        self.assertNotIn("Reached this run's target", page.row_text["row-UI-110"])
        self.assertIn("We'll revisit: After the authentication fix is accepted.", page.row_text["row-UI-110"])
        self.assertIn('href="#decision-Q1"', html)
        self.assertIn('<b>6 of 11</b> items wrapped up', html)
        self.assertIn('2 done (1 not submitted) · 2 in progress · 3 need attention', html)
        self.assertIn('<b>4</b> Wrapped up without delivery', html)
        self.assertIn('1 decision needs your input', html)
        self.assertIn('<b>2</b> Done', html)
        self.assertIn('<b>2</b> Postponed', html)
        self.assertEqual(page.contexts["UI-110"]["tasks"][0]["work"], {
            "state": "deferred", "summary": "Postponed until the sign-in fix is accepted.",
            "reason": "The user moved draft recovery out of this run.",
            "receipt": "Synthetic Q-auth answer: defer UI-110 until the authentication fix is accepted.",
            "revisit": "After the authentication fix is accepted.",
        })
        self.assertIn("E10 establish", html)

    def test_overview_is_plain_and_drill_in_keeps_technical_detail(self):
        html = render(self.data)
        page = Page(html)
        row = page.row_text["row-OPS-105"]
        self.assertIn("Ready to go, but releasing it would affect more than planned. It needs your approval.", row)
        for technical in ("Publication would trigger a broader deployment; approval is missing.",
                          "Resolve Q1. The old hash is not a separate behavioral hold.", "Reviewer owner"):
            with self.subTest(technical=technical):
                self.assertNotIn(technical, row)
                self.assertIn(technical, html.split('id="task-OPS-105"', 1)[1])
        self.assertIn("<code>needs_decision</code>", html)
        self.assertIn("<b>PR readiness</b> <code>pr_readiness</code><span><code>verified</code></span>", html)
        self.assertIn("<li><b>Submitted</b> The change is shared as a branch or pull request.</li>", html)
        task = self.data["tasks"][4]
        del task["work"]["summary"]
        self.assertIn("Publication would trigger a broader deployment; approval is missing.",
                      Page(render(self.data)).row_text["row-OPS-105"])
        for value in ("", "  ", None):
            with self.subTest(summary=value):
                task["work"]["summary"] = value
                with self.assertRaisesRegex(ValueError, "summary must be nonempty text"):
                    render(self.data)

    def test_legacy_closed_owners_need_classification_not_a_delivery_guess(self):
        for task in self.data["tasks"]:
            del task["work"]
        html = render(self.data)
        page = Page(html)
        self.assertEqual(page.work_states["row-UI-101"], "unknown")
        self.assertEqual(page.work_states["row-UI-109"], "unknown")
        self.assertEqual(page.work_states["row-API-102"], "working")
        self.assertEqual(page.work_states["row-ENV-106"], "blocked")
        self.assertEqual(page.work_states["row-REVIEW-103"], "waiting")
        self.assertIn("Confirming status", page.row_text["row-UI-101"])
        self.assertIn("We're confirming how this item ended.", page.row_text["row-UI-101"])
        self.assertIn('<b>6</b> Confirming status', html)
        self.assertIn('<b>0 of 11</b> items wrapped up', html)
        self.assertNotIn("group-resolved", page.groups)

    def test_business_stage_and_agent_status_are_separate(self):
        html = render(self.data)
        page = Page(html)
        self.assertIn('<b>2</b> Working</span>', html)
        self.assertIn('<b>1</b> Blocked</span>', html)
        self.assertIn('<b>5</b> Closed</span>', html)
        # The business target is reached while the agent is still working on its handoff.
        self.assertIn("Implemented ✓", page.row_text["row-API-102"])
        self.assertIn("Working", page.row_text["row-API-102"])
        self.assertIn("Two subagents are fixing the regressions", html.split('id="task-API-104"', 1)[1])
        task = self.data["tasks"][5]
        task["blocker"] = ""
        task["work"] = {"state": "waiting", "reason": "Lane is held."}
        with self.assertRaisesRegex(ValueError, "blocked owner needs an actionable blocker"):
            render(self.data)
        task["owner"]["state"] = "unreachable"
        self.assertIn("Not responding", Page(render(self.data)).row_text["row-ENV-106"])
        task["owner"]["activity"] = " "
        with self.assertRaisesRegex(ValueError, "owner activity must be nonempty text"):
            render(self.data)

    def test_stages_reject_unknown_keys_and_tolerate_older_records(self):
        task = self.data["tasks"][0]
        task["milestones"]["shipped"] = {"state": "verified", "note": "", "evidence": ["E1"]}
        with self.assertRaisesRegex(ValueError, "unknown stage shipped"):
            render(self.data)
        del task["milestones"]["shipped"]
        del task["milestones"]["deployment"]
        with self.assertRaisesRegex(ValueError, "missing stage deployment"):
            render(self.data)
        self.data = json.loads((RUNNER / "evals" / "dashboard-preview.json").read_text())
        self.data["skill_version"] = skill_version()
        for task in self.data["tasks"]:
            for key in ("defined", "researched", "review", "verification"):
                del task["milestones"][key]
        page = Page(render(self.data))
        self.assertEqual(page.rows["row-UI-101"], stage_row(OUT, OUT, "Done", OUT, OUT, OUT, OUT, OUT))
        self.assertIn("No stages requested", page.row_text["row-TRIAGE-108"])

    def test_resolutions_require_proof_or_decisions_and_settled_scope(self):
        baseline = deepcopy(self.data)
        cases = [
            ("UI-101", "work", "evidence", [], "accepted supporting proof"),
            ("TRIAGE-108", "work", "evidence", ["E4"], "accepted supporting proof"),
            ("TRIAGE-108", "work", "evidence", ["missing"], "retained proof"),
            ("UI-109", "work", "receipt", "", "decision receipt"),
            ("UI-110", "work", "receipt", "", "decision receipt"),
            ("UI-110", "work", "revisit", "", "revisit trigger"),
            ("UI-109", None, "obligations", ["Release runtime"], "settled obligations"),
            ("UI-110", None, "blocker", "Approval missing", "settled obligations"),
            ("UI-109", "work", "reason", "", "work needs a reason"),
        ]
        for ticket, section, field, value, message in cases:
            with self.subTest(ticket=ticket, field=field):
                self.data = deepcopy(baseline)
                task = next(task for task in self.data["tasks"] if task["id"] == ticket)
                (task[section] if section else task)[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    render(self.data)
        self.data = deepcopy(baseline)
        task = self.data["tasks"][8]
        task["owner"]["state"] = "active"
        task["milestones"]["pr_readiness"]["state"] = "pending"
        with self.assertRaisesRegex(ValueError, "unfinished stage prevents work resolution"):
            render(self.data)

    def test_unfinished_states_and_check_times_do_not_imply_success_or_stalls(self):
        task = self.data["tasks"][1]
        task["owner"]["state"] = "waiting"
        task["owner"]["last_checked_at"] = "2026-10-01T10:00:00-05:00"
        for state, label, group in (("ready", "Ready to start", "group-moving"),
                                    ("awaiting_acceptance", "Ready for sign-off", "group-moving"),
                                    ("waiting", "On hold", "group-attention"),
                                    ("unknown", "Confirming status", "group-attention")):
            with self.subTest(state=state):
                task["work"] = {"state": state, "reason": "Next inspection must confirm the result."}
                html = render(self.data)
                page = Page(html)
                self.assertIn("row-API-102", page.groups[group])
                self.assertIn(label, page.row_text["row-API-102"])
                self.assertIn("<dt>Last checked</dt><dd>2026-10-01T10:00:00-05:00</dd>", html.split('id="task-API-102"', 1)[1])
        task["work"]["state"] = "blocked"
        with self.assertRaisesRegex(ValueError, "actionable blocker"):
            render(self.data)
        task["work"]["state"] = "needs_decision"
        with self.assertRaisesRegex(ValueError, "pending decision"):
            render(self.data)
        task["work"]["state"] = "working"
        task["owner"].update(state="complete", handoff="Retained")
        with self.assertRaisesRegex(ValueError, "unfinished work prevents owner completion"):
            render(self.data)

    def test_auto_refresh_uses_run_interval_even_when_paused_or_finished(self):
        self.data["presentation"]["auto_refresh"] = True
        for interval, hold in ((1, None), (17, "Paused for user input"), (137, "Run complete")):
            with self.subTest(interval=interval, hold=hold):
                self.data["monitoring"].update(interval_seconds=interval, hold=hold, next_check_at=None)
                if hold == "Run complete":
                    self.data.update(tasks=[], decisions=[], evidence=[], history=[], approvals=[], follow_ups=[])
                html = render(self.data, hosted=True)
                self.assertEqual(Page(html).refresh, str(interval))
                unit = "second" if interval == 1 else "seconds"
                self.assertIn(f"This page reloads every {interval} {unit}", html)
                self.assertIn('<time datetime="2026-10-08T14:30:00-05:00" data-local>Oct 8, 2026, 2:30 PM (UTC-05:00)</time>', html)
        self.assertEqual(self.data["updated_at"], "2026-10-08T14:30:00-05:00")

    def test_absent_or_unavailable_auto_refresh_keeps_static_snapshot(self):
        self.data["monitoring"].update(interval_seconds=17)
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                if explicit:
                    self.data["presentation"]["auto_refresh"] = False
                html = render(self.data)
                self.assertIsNone(Page(html).refresh)
                self.assertIn("This page is a snapshot. It changes when the coordinator posts an update.", html)

    def test_auto_refresh_rejects_invalid_capability_flags_and_intervals(self):
        for value in (None, "true", 1):
            with self.subTest(flag=value):
                self.data["presentation"]["auto_refresh"] = value
                with self.assertRaisesRegex(ValueError, "presentation.auto_refresh must be boolean"):
                    render(self.data)
        self.data["presentation"]["auto_refresh"] = True
        del self.data["monitoring"]["interval_seconds"]
        with self.assertRaisesRegex(ValueError, "positive integer monitoring.interval_seconds"):
            render(self.data)
        for interval in (None, 0, -1, True, "17", 1.5):
            with self.subTest(interval=interval):
                self.data["monitoring"].update(interval_seconds=interval)
                with self.assertRaisesRegex(ValueError, "positive integer monitoring.interval_seconds"):
                    render(self.data)

    def test_green_job_cannot_be_used_as_verified_review_proof(self):
        self.data["tasks"][2]["milestones"]["pr_readiness"]["state"] = "verified"
        with self.assertRaisesRegex(ValueError, "accepted supporting proof"):
            render(self.data)

    def test_old_verdict_and_failed_interrupted_evidence_remain(self):
        html = render(self.data)
        self.assertIn('id="proof-E5"', html)
        self.assertIn("passed · superseded", html)
        self.assertIn("failed · limited", html)
        self.assertIn("interrupted · limited", html)
        self.assertIn("posted reviews: []", html)

    def test_pending_obligation_blocks_owner_closure_even_with_green_milestones(self):
        task = self.data["tasks"][0]
        task["obligations"] = ["Required CI still running"]
        with self.assertRaisesRegex(ValueError, "settled obligations"):
            render(self.data)
        task["obligations"] = []
        task["owner"]["handoff"] = ""
        with self.assertRaisesRegex(ValueError, "handoff"):
            render(self.data)

    def test_unfinished_milestone_blocks_complete_owner(self):
        task = self.data["tasks"][0]
        task["milestones"]["code_publication"]["state"] = "pending"
        with self.assertRaisesRegex(ValueError, "unfinished stage"):
            render(self.data)

    def test_one_owner_cannot_be_assigned_to_two_tickets_even_after_closure(self):
        first, second = self.data["tasks"][:2]
        second["owner"]["id"] = first["owner"]["id"]
        for state in ("active", "closed"):
            with self.subTest(first_owner_state=state):
                first["owner"]["state"] = state
                with self.assertRaisesRegex(ValueError, "API-102: owner ui-owner is already bound to another ticket"):
                    render(self.data)

    def test_completed_but_unclosed_owner_is_visible(self):
        self.data["tasks"] = self.data["tasks"][:7]
        self.data["tasks"][0]["owner"]["state"] = "complete"
        html = render(self.data)
        self.assertIn('<b>1</b> Finished</span>', html)
        self.assertIn('<b>1</b> Closed</span>', html)
        self.assertIn("Complete, closure pending / unavailable", html)

    def test_missing_and_superseded_proof_do_not_mark_verified(self):
        phase = self.data["tasks"][0]["milestones"]["pr_readiness"]
        for proofs, expected in (([], "accepted supporting proof"), (["missing"], "missing proof"), (["E5"], "accepted supporting proof")):
            with self.subTest(proofs=proofs):
                phase["evidence"] = proofs
                with self.assertRaisesRegex(ValueError, expected):
                    render(self.data)

    def test_hosted_links_reject_private_local_and_script_targets(self):
        for href in ("file:///tmp/receipt.txt", "./receipt.txt", "http://localhost:3000/receipt", "http://127.0.0.1/", "http://[::1]/", "http://10.2.3.4/", "javascript:alert(1)"):
            with self.subTest(href=href):
                self.data["evidence"][0]["href"] = href
                with self.assertRaises(ValueError):
                    render(self.data, hosted=True)
        self.data["evidence"][0]["href"] = "https://example.org/approved-receipt"
        self.assertIn('href="https://example.org/approved-receipt"', render(self.data, hosted=True))

    def test_copied_questions_name_run_task_and_retained_proof(self):
        html = render(self.data)
        self.assertIn("In orchestration run template-preview, task API-104: What does retained proof E11, E5, E6 establish", html)
        self.assertIn("In orchestration run template-preview, decision Q1:", html)
        self.assertIn('data-copy-id="UI-101"', html)
        self.assertIn("Copy does not send", html)

    def test_copied_context_is_scoped_and_keeps_work_owner_distinct_from_answer_route(self):
        self.data["tasks"][4]["owner"]["href"] = "https://example.org/reviewer-owner"
        contexts = Page(render(self.data)).contexts
        packet = contexts["Q1"]
        self.assertEqual(packet["run_id"], "template-preview")
        self.assertEqual(packet["snapshot_time"], "2026-10-08T14:30:00-05:00")
        self.assertEqual([task["id"] for task in packet["tasks"]], ["OPS-105"])
        task = packet["tasks"][0]
        self.assertEqual(task["owner"]["id"], "reviewer-owner")
        self.assertEqual(packet["answer_route"]["owner_id"], "fixture-coordinator")
        self.assertEqual(task["milestones"]["pr_readiness"]["evidence"], ["E7"])
        self.assertEqual(task["milestones"]["code_publication"]["state"], "blocked")
        self.assertEqual(task["milestones"]["deployment"]["state"], "not_requested")
        self.assertIn("unapproved", task["blocker"])
        self.assertIn("no approval", packet["approval_constraints"])
        self.assertEqual(packet["decision"]["task_ids"], ["OPS-105"])
        self.assertEqual(contexts["OPS-105"]["answer_route"], {
            "owner_id": "reviewer-owner", "label": "Reviewer owner",
            "href": "https://example.org/reviewer-owner",
        })
        self.assertNotIn("decision", contexts["OPS-105"])

    def test_decision_questions_route_to_their_owners_and_name_human_only_gates(self):
        self.data["decisions"] += [
            {"id": "Q-local", "text": "Choose the description format?", "recommendation": "Use full labels.",
             "task_ids": ["API-104"], "unblocks": "Description formatting",
             "discussion": {"owner_id": "list-owner", "label": "List owner",
                            "href": "https://example.org/discuss-list"}},
            {"id": "Q-gate", "text": "Complete the existing sign-in dialog.", "recommendation": "Answer in the runtime owner conversation.",
             "task_ids": ["ENV-106"], "unblocks": "Runtime sign-in", "requires_human": True,
             "discussion": {"owner_id": "runtime-owner", "label": "Runtime owner",
                            "href": "https://example.org/runtime-input"}},
        ]
        html = render(self.data, hosted=True)
        self.assertIn('href="https://example.org/discuss-list">Open discussion</a>', html)
        self.assertIn('href="https://example.org/runtime-input">Answer in conversation</a>', html)
        self.assertIn('href="#task-API-104">API-104</a>', html)
        self.assertIn("Answering unlocks: Description formatting", html)
        self.assertIn('data-discussion-owner="list-owner"', html)
        self.assertIn('data-requires-human="true"', html)
        self.assertIn("You need to answer this yourself in the linked conversation.", html)
        self.assertIn('data-copy-id="Q-gate"', html)

    def test_decision_route_validation_and_legacy_coordinator_fallback(self):
        self.data["tasks"][4]["work"]["state"] = "waiting"
        decision = {"id": "Q-old", "text": "Keep publication held?", "recommendation": "Keep the hold."}
        self.data["decisions"] = [decision]
        self.assertIn('data-discussion-owner=""', render(self.data))
        for fields, expected in (
            ({"requires_human": True}, "human-only gate needs a discussion owner"),
            ({"requires_human": "true"}, "requires_human must be boolean"),
            ({"task_ids": ["MISSING"]}, "unknown affected task"),
            ({"discussion": {"owner_id": "runtime-owner", "label": "Runtime owner", "href": "javascript:bad()"}}, "unsupported link scheme"),
            ({"discussion": {"owner_id": "runtime-owner", "label": "Runtime owner", "requires_human": True}},
             "requires_human belongs on the decision"),
        ):
            with self.subTest(fields=fields):
                self.data["decisions"] = [{**decision, **fields}]
                with self.assertRaisesRegex(ValueError, expected):
                    render(self.data, hosted=True)

    def test_text_is_escaped_and_not_reprocessed_as_a_template(self):
        self.data["title"] = '<script>alert("title")</script> {{ROWS}}'
        self.data["tasks"][0]["title"] = '</button><img src=x onerror="bad()">'
        html = render(self.data)
        self.assertIn("&lt;script&gt;alert(&quot;title&quot;)&lt;/script&gt; {{ROWS}}", html)
        self.assertNotIn('<img src=x', html)
        self.assertEqual(html.count('<script>'), 1)

    def test_empty_run_and_cli_failure_preserve_previous_dashboard(self):
        self.data.update(tasks=[], decisions=[], evidence=[], history=[], approvals=[], follow_ups=[])
        self.assertIn("No items are assigned yet.", render(self.data))
        with tempfile.TemporaryDirectory() as directory:
            status, output = Path(directory) / "status.json", Path(directory) / "dashboard.html"
            output.write_text("previous useful dashboard")
            self.data["schema_version"] = 99
            status.write_text(json.dumps(self.data))
            result = subprocess.run([sys.executable, str(RUNNER / "scripts" / "render_dashboard.py"), str(status), str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("unsupported dashboard schema_version", result.stderr)
            self.assertEqual(output.read_text(), "previous useful dashboard")


class LifecycleTests(unittest.TestCase):
    """Schema 2 rules: publication scope, durable code, runtime, approvals and idle scans."""

    def setUp(self):
        self.data = json.loads((RUNNER / "evals" / "dashboard-preview.json").read_text())
        self.data["skill_version"] = skill_version()
        self.tasks = {task["id"]: task for task in self.data["tasks"]}

    def test_schema_1_records_keep_rendering_without_lifecycle_fields(self):
        self.data["schema_version"] = 1
        self.data.pop("approvals")
        self.data["extra_notes"] = "legacy field"
        for task in self.data["tasks"]:
            task.pop("runtime")
        self.assertIn("Orchestration preview", render(self.data))

    def test_unknown_top_level_fields_are_rejected(self):
        self.data["feedback_coordination"] = {}
        with self.assertRaisesRegex(ValueError, "unknown top-level field feedback_coordination"):
            render(self.data)

    def test_run_choices_name_target_stage_publication_scope_and_idle_limit(self):
        for field, value, message in (("target_stage", "review-ready PRs", "target_stage must name a business stage"),
                                      ("publication_scope", "push", "publication_scope must be one of"),
                                      ("idle_scan_limit", 0, "idle_scan_limit must be a positive integer")):
            with self.subTest(field=field):
                data = deepcopy(self.data)
                data["run_choices"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    render(data)

    def test_publication_request_needs_a_publication_scope(self):
        self.data["run_choices"]["publication_scope"] = "none"
        with self.assertRaisesRegex(ValueError, "OPS-105: code publication is requested but this run's publication scope is none"):
            render(self.data)

    def test_closed_owner_with_unpublished_code_needs_durable_copy(self):
        owner = self.tasks["UI-101"]["owner"]
        owner.pop("retained_code")
        with self.assertRaisesRegex(ValueError, "UI-101: unpublished code must be kept outside the owner's workspace"):
            render(self.data)
        owner["retained_code"] = {"kind": "bundle", "ref": "proof/ui-101/pinning.bundle"}
        with self.assertRaisesRegex(ValueError, "retained_code.checksum"):
            render(self.data)
        owner["retained_code"] = {"kind": "pushed_branch", "ref": "origin/ui-101-pinning"}
        self.assertIn("Pushed branch <code>origin/ui-101-pinning</code>", render(self.data))

    def test_finishing_needs_runtime_released_or_retained_with_reason_and_date(self):
        runtime = self.tasks["SHIP-107"]["runtime"]
        runtime.append({"id": "orb", "kind": "sandbox", "ref": "orb-107", "state": "held"})
        with self.assertRaisesRegex(ValueError, "SHIP-107: release or retain runtime orb"):
            render(self.data)
        runtime[-1].update(state="retained")
        with self.assertRaisesRegex(ValueError, "runtime.orb.reason"):
            render(self.data)
        runtime[-1].update(reason="Browser session for the expiry check")
        with self.assertRaisesRegex(ValueError, "runtime.orb.until needs an ISO 8601 timestamp"):
            render(self.data)

    def test_retention_past_its_date_is_called_out_for_cleanup(self):
        html = render(self.data)
        self.assertNotIn("past its cleanup date", html)
        self.assertIn("Kept until 2026-10-15T12:00:00-05:00", html)
        self.tasks["SHIP-107"]["runtime"][0]["until"] = "2026-10-08T12:00:00-05:00"
        html = render(self.data)
        self.assertIn('href="#runtime">1 kept resource is past its cleanup date', html)
        self.assertIn("Cleanup overdue since 2026-10-08T12:00:00-05:00", html)

    def test_expired_approval_cannot_stay_active(self):
        approval = self.data["approvals"][1]
        approval["state"] = "active"
        with self.assertRaisesRegex(ValueError, "A2: approval expired; mark it expired and ask again"):
            render(self.data)
        approval["state"] = "expired"
        self.assertIn("<b>A2</b> Expired", render(self.data))
        approval["task_ids"] = ["NOPE-1"]
        with self.assertRaisesRegex(ValueError, "A2: approval names an unknown task"):
            render(self.data)

    def test_idle_scans_with_only_people_gated_work_must_stop_scheduling(self):
        for task in self.data["tasks"]:
            if task["work"]["state"] not in ("delivered", "no_change_needed", "cancelled", "deferred", "needs_decision"):
                task["work"]["state"] = "waiting"
                task["blocker"] = task["blocker"] or "Waiting on a person"
        self.data["monitoring"]["unchanged_scans"] = 3
        with self.assertRaisesRegex(ValueError, "3 unchanged scans with only people-gated work left"):
            render(self.data)
        self.data["monitoring"].update(next_check_at=None, hold="Waiting on Q1; resume when answered")
        render(self.data)
        self.data["monitoring"].update(next_check_at="2026-10-08T14:40:00-05:00", unchanged_scans=2)
        render(self.data)

    def test_implemented_but_unsubmitted_delivery_is_labelled(self):
        page = render(self.data)
        row = "".join(Page(page).row_text["row-UI-101"])
        self.assertIn("Done, not submitted", row)
        self.assertNotIn("Done, not submitted", "".join(Page(page).row_text["row-SHIP-107"]))
        self.assertIn("2 done (1 not submitted)", page)

    def test_follow_ups_need_a_source_task_and_unowned_ones_are_called_out(self):
        page = render(self.data)
        self.assertIn('href="#follow-ups">1 follow-up needs an owner', page)
        self.assertIn("Owned by OPS-112", page)
        self.data["follow_ups"][1]["from_task"] = "NOPE-1"
        with self.assertRaisesRegex(ValueError, "F2: follow-up needs the assigned task"):
            render(self.data)

    def test_skill_version_is_required_and_a_mismatch_is_visible(self):
        self.assertNotIn("The coordinator migrates it", render(self.data))
        self.data["skill_version"] = "0123abcd4567"
        self.assertIn("written for orchestration-run 0123abcd4567", render(self.data))
        del self.data["skill_version"]
        with self.assertRaisesRegex(ValueError, "skill_version must be nonempty"):
            render(self.data)
        result = subprocess.run([sys.executable, "-B", str(RUNNER / "scripts" / "render_dashboard.py"), "--skill-version"],
                                capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), skill_version())

    def test_history_entries_stay_short(self):
        self.data["history"].append("Scan inspected six owners; all idle and unchanged. " * 20)
        with self.assertRaisesRegex(ValueError, "history entries \\[5\\] exceed 800 characters"):
            render(self.data)

    def test_pending_business_acceptance_names_who_signs_off_and_when(self):
        acceptance = self.tasks["API-102"]["milestones"]["business_acceptance"]
        acceptance.update(state="pending", note="Product owner sign-off")
        with self.assertRaisesRegex(ValueError, "API-102.business_acceptance.accountable"):
            render(self.data)
        acceptance.update(accountable="Product owner (Sam)")
        with self.assertRaisesRegex(ValueError, "API-102.business_acceptance.revisit"):
            render(self.data)
        acceptance.update(revisit="Friday demo")
        self.assertIn("Sign-off by Product owner (Sam); revisit Friday demo.", render(self.data))

    def test_closed_run_has_nothing_open_or_scheduled(self):
        self.data["closed_at"] = "2026-10-08T15:00:00-05:00"
        with self.assertRaisesRegex(ValueError, "a closed run cannot keep open owner API-102.*held runtime on API-102"
                                                 ".*unowned follow-up F2.*pending decisions.*a scheduled check"):
            render(self.data)
        self.data["tasks"] = [task for task in self.data["tasks"] if task["owner"]["state"] in ("complete", "closed")]
        self.data["follow_ups"][1]["owner"] = "API-113"
        self.data["follow_ups"] = [item for item in self.data["follow_ups"]
                                   if item["from_task"] in {task["id"] for task in self.data["tasks"]}]
        self.data.update(decisions=[])
        self.data["monitoring"].update(next_check_at=None, hold="Run closed")
        self.assertIn("This run closed at 2026-10-08T15:00:00-05:00", render(self.data))

    def test_served_page_check_compares_bytes_without_a_browser(self):
        import http.server, threading, functools
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "index.html"
            page.write_text(render(self.data))
            handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
            handler.log_message = lambda *args: None
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/index.html"
                command = [sys.executable, str(RUNNER / "scripts" / "check_served.py"), str(page)]
                self.assertIn("PASS", subprocess.run(command + [url], capture_output=True, text=True).stdout)
                local = Path(directory) / "local.html"
                local.write_text("newer render")
                result = subprocess.run([command[0], command[1], str(local), url], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                result = subprocess.run(command + [url.replace("index", "missing")], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("LIMITED", result.stderr)
            finally:
                server.shutdown()


class InstallTests(unittest.TestCase):
    """The skills CLI and Workspace Skills install each skill folder alone."""

    def test_shared_files_match_their_owning_skill(self):
        for owner, copy, path in ((SETUP, RUNNER, "references/ticket-template.md"),
                                  (DISCOVERY, SETUP, "scripts/state.py"),
                                  (DISCOVERY, RUNNER, "scripts/state.py")):
            with self.subTest(path=path):
                self.assertEqual((copy / path).read_bytes(), (owner / path).read_bytes(),
                                 f"copy {copy.name}/{path} from {owner.name}/{path}")


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.saved = self.repository / ".nthplusio-skills"
        self.saved.mkdir(parents=True)
        self.cwd = self.root / "elsewhere"
        self.cwd.mkdir()
        self.config_path = self.saved / "orchestration.json"
        self.discovery_path = self.saved / "discovery.json"
        self.marker = self.root / "tool-invoked"
        self.discovery = json.loads((DISCOVERY / "assets" / "discovery.template.json").read_text())
        facts = self.discovery["harnesses"].pop("REPLACE_WITH_DISCOVERED_HARNESS_NAME")
        facts.update(context="Orb", observed_at="2026-10-09T12:00:00Z",
                     sources=["Inspected tool inventory"], owner_term="thread", helper_term="subagent")
        facts["owners"].update(launch=f"touch {self.marker}", inspect="Inspect by thread ID",
                               workspace_model="isolated")
        facts["monitoring"]["wait"] = "Wait with timeout_seconds; blocks caller"
        facts["skills"]["load"] = "Load by skill name"
        facts["artifacts"] = [{"id": "portal", "label": "Portal", "present": "Publish HTML",
                               "update": "Replace HTML", "audience": "Project reviewers", "interaction": None}]
        facts["limitations"] = ["No wake after yielding"]
        other = deepcopy(facts)
        other.update(context="Local terminal", owner_term="session", helper_term="bounded helper")
        other["owners"].update(launch=None, inspect=None, workspace_model="shared")
        other["monitoring"]["wait"] = None
        other["skills"]["load"] = None
        other["artifacts"] = []
        other["limitations"] = ["No independent owners"]
        self.discovery["harnesses"] = {"Amp": facts, "Claude Code": other}
        self.config = json.loads((SETUP / "assets" / "orchestration.template.json").read_text())
        profile = self.config["harnesses"].pop("REPLACE_WITH_DISCOVERED_HARNESS_NAME")
        profile.update(run_root="runs/amp", shared_skills=["speak-clearly", "amp-proof"],
                       confirmed_from="User confirmed Amp choices")
        profile["defaults"] = {"finish_line": "merged PRs", "destination": "portal", "interval_seconds": 15}
        other_profile = deepcopy(profile)
        other_profile.update(run_root="runs/claude", shared_skills=["terminal-proof"],
                             confirmed_from="User confirmed terminal choices")
        other_profile["defaults"] = {"finish_line": "review-ready PRs", "destination": None, "interval_seconds": 90}
        self.config["harnesses"] = {"Amp": profile, "Claude Code": other_profile}
        self.config["project"] = {
            "ticket_policy": {"source": "Linear", "location": "ENG team", "definition": "One shipped outcome",
                              "template": "docs/tickets.md", "id_convention": "ENG-N", "criteria_limit": 3,
                              "confirmed_from": "User confirmed project policy"},
            "shared_skills": ["speak-clearly", "test-behavior"],
        }
        self.save()

    def save(self):
        self.discovery_path.write_text(json.dumps(self.discovery))
        self.config_path.write_text(json.dumps(self.config))

    def cli(self, skill, *arguments):
        return subprocess.run([sys.executable, "-B", str(skill / "scripts" / "state.py"), *arguments,
                               "--root", str(self.repository)], cwd=self.cwd, capture_output=True, text=True)

    def selected(self, name):
        result = self.cli(SETUP, "resolve", "--harness", name, "--orchestration")
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_saved_discovery_and_setup_validate_without_selecting_a_profile(self):
        for skill, scope, path in ((DISCOVERY, "discovery", self.discovery_path),
                                   (SETUP, "orchestration", self.config_path)):
            with self.subTest(scope=scope):
                result = self.cli(skill, "check", scope)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, f"PASS: {scope} {path}\n")
        result = self.cli(RUNNER, "check")
        self.assertEqual(result.stdout, f"PASS: discovery {self.discovery_path}\nPASS: orchestration {self.config_path}\n")
        self.assertFalse(self.marker.exists())

    def test_setup_validates_when_installed_without_its_sibling_skills(self):
        alone = self.root / "installed" / SETUP.name
        shutil.copytree(SETUP, alone, ignore=shutil.ignore_patterns("__pycache__"))
        result = self.cli(alone, "check", "orchestration")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, f"PASS: orchestration {self.config_path}\n")

    def test_exact_names_select_asymmetric_capabilities_defaults_and_shared_skills(self):
        amp = self.selected("Amp")
        terminal = self.selected("Claude Code")
        self.assertEqual(amp["harness"], {"name": "Amp", "slug": "amp", "context": "Orb", "owner_term": "thread", "helper_term": "subagent"})
        self.assertEqual(terminal["harness"], {"name": "Claude Code", "slug": "claude-code", "context": "Local terminal",
                                               "owner_term": "session", "helper_term": "bounded helper"})
        self.assertEqual(amp["owners"]["inspect"], "Inspect by thread ID")
        self.assertEqual(amp["owners"]["workspace_model"], "isolated")
        self.assertEqual(terminal["owners"]["workspace_model"], "shared")
        self.assertEqual(amp["defaults"], {"finish_line": "merged PRs", "destination": "portal", "interval_seconds": 15})
        self.assertEqual(terminal["defaults"], {"finish_line": "review-ready PRs", "destination": None, "interval_seconds": 90})
        self.assertEqual(amp["skills"], {"load": "Load by skill name", "brief": "Pass applicable instructions explicitly; do not assume inheritance.",
                                        "locations": None, "reload": None,
                                        "shared": ["speak-clearly", "test-behavior", "amp-proof"], "stages": {}})
        self.assertEqual(terminal["skills"]["shared"], ["speak-clearly", "test-behavior", "terminal-proof"])
        self.assertFalse(self.marker.exists())

    def test_stage_skills_are_optional_validated_and_resolved(self):
        self.config["project"]["stage_skills"] = {"code_publication": ["start-work"], "review": ["pr-ready"]}
        self.config_path.write_text(json.dumps(self.config))
        self.assertEqual(self.selected("Amp")["skills"]["stages"], {"code_publication": ["start-work"], "review": ["pr-ready"]})
        for bad in ({"shipping": ["start-work"]}, {"review": [RUNNER.name]}):
            with self.subTest(stage_skills=bad):
                self.config["project"]["stage_skills"] = bad
                self.config_path.write_text(json.dumps(self.config))
                result = self.cli(SETUP, "check", "orchestration")
                self.assertEqual(result.returncode, 1)
        del self.config["project"]["stage_skills"]
        self.config_path.write_text(json.dumps(self.config))
        self.assertEqual(self.selected("Amp")["skills"]["stages"], {})

    def test_project_policy_is_retained_for_both_profiles(self):
        for name in ("Amp", "Claude Code"):
            with self.subTest(name=name):
                self.assertEqual(self.selected(name)["ticket_policy"], {
                    "source": "Linear", "location": "ENG team", "definition": "One shipped outcome",
                    "template": "docs/tickets.md", "id_convention": "ENG-N", "criteria_limit": 3,
                    "confirmed_from": "User confirmed project policy",
                })

    def test_portable_paths_resolve_against_config_directory_not_cwd(self):
        for name, directory in (("Amp", "amp"), ("Claude Code", "claude")):
            with self.subTest(name=name):
                selected = self.selected(name)
                self.assertEqual(selected["config_path"], str(self.config_path))
                self.assertEqual(selected["run_root"], str(self.saved / "runs" / directory))
                self.assertFalse((self.saved / "runs").exists())
        self.assertEqual(list(self.cwd.iterdir()), [])

    def test_a_slug_selects_its_key_and_other_names_refuse_fallback(self):
        self.assertEqual(self.selected("amp"), self.selected("Amp"))
        self.assertEqual(self.selected("claude-code"), self.selected("Claude Code"))
        for name in ("Claude", "Unknown", "amp code"):
            with self.subTest(name=name):
                result = self.cli(SETUP, "resolve", "--harness", name, "--orchestration")
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn("no saved discovery for this harness", result.stderr)
        del self.config["harnesses"]["Claude Code"]
        self.save()
        result = self.cli(SETUP, "resolve", "--harness", "Claude Code", "--orchestration")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no setup profile", result.stderr)
        self.assertEqual(json.loads(self.cli(DISCOVERY, "resolve", "--harness", "claude-code").stdout)["harness"]["name"],
                         "Claude Code")
        self.config["harnesses"]["Claude Code"] = {"run_root": "runs/claude", "shared_skills": [], "confirmed_from": "Test",
                                                   "defaults": {"finish_line": "review-ready PRs", "destination": None,
                                                                "interval_seconds": 90}}
        del self.discovery["harnesses"]["Claude Code"]
        self.save()
        result = self.cli(SETUP, "resolve", "--harness", "Amp", "--orchestration")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Claude Code: no saved discovery", result.stderr)

    def test_missing_state_names_the_skill_to_run(self):
        self.config_path.unlink()
        result = self.cli(RUNNER, "resolve", "--harness", "Amp", "--orchestration")
        self.assertEqual(result.returncode, 1)
        self.assertIn("run orchestration-setup", result.stderr)
        self.discovery_path.unlink()
        result = self.cli(DISCOVERY, "check")
        self.assertEqual(result.returncode, 1)
        self.assertIn("run harness-discovery", result.stderr)

    def test_unavailable_operations_remain_null(self):
        selected = self.selected("Claude Code")
        self.assertEqual(selected["owners"], {"launch": None, "inspect": None, "message": None,
                                             "collect": None, "close": None, "resume": None, "workspace_model": "shared"})
        self.assertEqual(selected["helpers"], {"invoke": None, "collect": None, "blocks_caller": None, "follow_up": None,
                                              "readonly": None})
        self.assertEqual(selected["monitoring"], {"notifications": None, "wait": None, "wake": None})
        self.assertEqual(selected["evidence"], {"transfer": None, "retain_before_close": None})
        self.assertEqual(selected["artifacts"], [])
        self.assertEqual(selected["limitations"], ["No independent owners"])
        self.assertIsNone(self.selected("Amp")["artifacts"][0]["interaction"])

    def test_artifact_auto_refresh_is_an_optional_capability_description(self):
        artifact = self.discovery["harnesses"]["Amp"]["artifacts"][0]
        self.assertNotIn("auto_refresh", self.selected("Amp")["artifacts"][0])
        for value in (None, "Supports HTML meta refresh at the updated page URL"):
            with self.subTest(capability=value):
                artifact["auto_refresh"] = value
                self.save()
                self.assertEqual(self.selected("Amp")["artifacts"][0]["auto_refresh"], value)
        for value in ("", True, 17):
            with self.subTest(invalid=value):
                artifact["auto_refresh"] = value
                self.save()
                result = self.cli(DISCOVERY, "check", "discovery")
                self.assertEqual(result.returncode, 1)
                self.assertIn("artifact.auto_refresh must be nonempty text", result.stderr)

    def test_all_orchestration_names_are_excluded_from_shared_worker_skills(self):
        original = deepcopy(self.config)
        for name in (DISCOVERY.name, SETUP.name, RUNNER.name):
            for scope in ("project", "Amp", "Claude Code"):
                with self.subTest(name=name, scope=scope):
                    self.config = deepcopy(original)
                    target = self.config["project"] if scope == "project" else self.config["harnesses"][scope]
                    target["shared_skills"].append(name)
                    self.save()
                    result = self.cli(SETUP, "resolve", "--harness", "Amp", "--orchestration")
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("coordinator", result.stderr)

    def test_malformed_discovery_fails_validation_and_resolution_without_tools(self):
        original = deepcopy(self.discovery)
        for field, value in (("schema_version", 2), ("harnesses", []), ("repository", None)):
            with self.subTest(field=field):
                self.discovery = deepcopy(original)
                self.discovery[field] = value
                self.save()
                for skill, args in ((DISCOVERY, ("check", "discovery")),
                                    (SETUP, ("resolve", "--harness", "Amp", "--orchestration"))):
                    result = self.cli(skill, *args)
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("FAIL:", result.stderr)
                self.assertFalse(self.marker.exists())
        self.discovery_path.write_text("{not JSON")
        result = self.cli(SETUP, "resolve", "--harness", "Amp", "--orchestration")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.marker.exists())

    def test_malformed_setup_fails_before_tools(self):
        original = deepcopy(self.config)
        for field, value in (("schema_version", 2), ("schema_version", 1), ("project", None), ("harnesses", [])):
            with self.subTest(field=field, value=value):
                self.config = deepcopy(original)
                self.config[field] = value
                self.save()
                result = self.cli(SETUP, "resolve", "--harness", "Amp", "--orchestration")
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn("FAIL:", result.stderr)
                self.assertFalse(self.marker.exists())
        self.config_path.write_text("{not JSON")
        result = self.cli(SETUP, "resolve", "--harness", "Amp", "--orchestration")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.marker.exists())

    def test_resolution_does_not_modify_saved_sources(self):
        before = {path: path.read_bytes() for path in (self.config_path, self.discovery_path)}
        self.selected("Amp")
        self.selected("Claude Code")
        self.selected("Amp")
        for path, expected in before.items():
            self.assertEqual(path.read_bytes(), expected)
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
