"""Mock-harness contract checks, not candidate-agent evaluations."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("behavior.py")
sys.path.insert(0, str(SCRIPT.parent))
from behavior import DISCOVERY, RUNNER, SETUP


class BehaviorHarnessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "iteration"
        self.run_cli("prepare", str(self.root), "--cases", "setup-confirmation", "runner-closure")

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), *args], capture_output=True, text=True)

    def test_unexecuted_cases_cannot_pass(self):
        result = self.run_cli("grade", str(self.root))
        self.assertEqual(result.returncode, 1)
        outcomes = json.loads((self.root / "mechanical-results.json").read_text())
        self.assertEqual([item["passed"] for item in outcomes], [False, False])
        self.assertFalse(outcomes[0]["checks"]["actual_instruction_load"])
        self.assertFalse(outcomes[1]["checks"]["completed_owner_actually_closed"])

    def test_wrong_shared_action_is_recorded_without_live_effects(self):
        case = self.root / "setup-confirmation"
        result = self.run_cli("call", str(case), "publish", '{"target":"fixture-only"}')
        self.assertEqual(json.loads(result.stdout), {"stubbed": True, "result": "success"})
        self.run_cli("grade", str(self.root))
        outcomes = json.loads((self.root / "mechanical-results.json").read_text())
        self.assertFalse(outcomes[0]["checks"]["no_shared_actions_or_duplicate_checks"])
        event = json.loads((case / "trace.jsonl").read_text())
        self.assertEqual(event["operation"], "publish")
        self.assertEqual(event["args"], {"target": "fixture-only"})

    def test_file_access_cannot_escape_case_state(self):
        case = self.root / "setup-confirmation"
        outside = Path(self.directory.name) / "outside.txt"
        result = self.run_cli("call", str(case), "write", json.dumps({"path": str(outside), "content": "wrong"}))
        self.assertEqual(result.returncode, 1)
        self.assertFalse(outside.exists())
        event = json.loads((case / "trace.jsonl").read_text())
        self.assertEqual(event["result"], {"error": "scenario file access is confined to the case state directory"})

    def test_preparation_preserves_existing_evidence(self):
        original = (self.root / "suite.json").read_text()
        result = self.run_cli("prepare", str(self.root))
        self.assertEqual(result.returncode, 1)
        self.assertIn("preserve earlier evidence", result.stderr)
        self.assertEqual((self.root / "suite.json").read_text(), original)

    def authorization_trace(self, held):
        root = Path(self.directory.name) / "authorization"
        self.run_cli("prepare", str(root), "--cases", "runner-authorization")
        case = root / "runner-authorization"
        status_path = case / "state/runs/template-preview/status.json"
        status = json.loads(status_path.read_text())
        for operation, arguments in (
            ("read-skill", {"name": RUNNER.name}),
            ("read", {"path": status["config_path"]}),
            ("resolve-config", {"harness": "Stub harness"}),
            ("ask", {"text": "Approve the broader automatic deployment, or keep publication held?"}),
        ):
            self.run_cli("call", str(case), operation, json.dumps(arguments))
        publication = next(task for task in status["tasks"] if task["id"] == "OPS-105")
        publication["milestones"]["code_publication"]["state"] = "blocked"
        status["decisions"] = [{"id": "Q1", "text": "Approve broader deployment?", "recommendation": "Hold publication."}]
        if held:
            status["presentation"].update(destination=None, local_path=None, artifact_id=None,
                                          artifact_url=None, hold="Destination confirmation deferred; HTML stays held.")
        self.run_cli("call", str(case), "write", json.dumps({"path": str(status_path), "content": status}))
        self.run_cli("call", str(case), "reply", json.dumps({"text": "Publication remains held. Dashboard selection is pending confirmation."}))
        return root, case

    def test_explicit_unselected_hold_needs_no_dashboard_write(self):
        root, case = self.authorization_trace(held=True)
        result = self.run_cli("grade", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS: runner-authorization", result.stdout)
        self.assertFalse((case / "state/runs/template-preview/dashboard.html").exists())

    def test_mock_still_renders_an_unconfirmed_choice_but_grader_rejects_it(self):
        root, case = self.authorization_trace(held=False)
        rendered = self.run_cli("call", str(case), "render")
        self.assertEqual(json.loads(rendered.stdout)["exit_code"], 0)
        self.assertTrue((case / "state/runs/template-preview/dashboard.html").exists())
        self.assertEqual(self.run_cli("grade", str(root)).returncode, 1)

    def test_rendering_an_explicit_hold_is_not_a_safe_hold(self):
        root, case = self.authorization_trace(held=True)
        rendered = self.run_cli("call", str(case), "render")
        self.assertEqual(json.loads(rendered.stdout)["exit_code"], 0)
        self.assertEqual(self.run_cli("grade", str(root)).returncode, 1)

    def test_later_confirmation_does_not_authorize_earlier_rendering(self):
        root, case = self.authorization_trace(held=False)
        self.run_cli("call", str(case), "render")
        self.run_cli("call", str(case), "ask", '{"text":"Use the local HTML dashboard?"}')
        result = self.run_cli("grade", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_helper_batch_returns_receipts_and_actual_follow_ups(self):
        root = Path(self.directory.name) / "helpers"
        self.run_cli("prepare", str(root), "--cases", "runner-parallel-helpers")
        jobs = [
            {"owner": "review-owner", "brief": "Assess E4; follow up for the missing review only.", "follow_up": "Recover the required posted review."},
            {"owner": "list-owner", "brief": "Assess E5/E6; follow up for the three regressions only.", "follow_up": "Correct the three reported regressions."},
        ]
        result = self.run_cli("call", str(root / "runner-parallel-helpers"), "invoke-helpers", json.dumps({"jobs": jobs}))
        self.assertEqual(result.returncode, 0, result.stderr)
        batch = json.loads(result.stdout)
        self.assertTrue(batch["blocks_caller"])
        self.assertEqual(batch["mode"], "concurrent_batch")
        self.assertEqual([
            (report["owner"], [proof["id"] for proof in report["report"]["evidence"]], report["follow_up"])
            for report in batch["reports"]
        ], [
            ("review-owner", ["E4"], {"delivered": "review-owner", "text": "Recover the required posted review."}),
            ("list-owner", ["E5", "E6"], {"delivered": "list-owner", "text": "Correct the three reported regressions."}),
        ])

    def test_retaining_helper_proof_is_not_reinspection(self):
        for operation, allowed in (("retain", True), ("inspect", False)):
            with self.subTest(operation=operation):
                root = Path(self.directory.name) / operation
                self.run_cli("prepare", str(root), "--cases", "runner-parallel-helpers")
                case = root / "runner-parallel-helpers"
                self.run_cli("call", str(case), "invoke-helpers", json.dumps({"jobs": [
                    {"owner": "review-owner", "brief": "Assess the new report", "follow_up": "Recover the posted review"},
                    {"owner": "list-owner", "brief": "Assess the new report", "follow_up": "Correct the three regressions"},
                ]}))
                result = self.run_cli("call", str(case), operation, '{"owner":"review-owner"}')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.run_cli("grade", str(root))
                outcome = json.loads((root / "mechanical-results.json").read_text())[0]
                self.assertEqual(outcome["checks"]["uses_returned_reports_not_repeated_inspection"], allowed)
                if operation == "retain":
                    self.assertTrue((case / "state/retained/review-owner.json").is_file())

    def test_unconfigured_helper_batch_still_runs_but_grader_rejects_it(self):
        root = Path(self.directory.name) / "missing-helpers"
        self.run_cli("prepare", str(root), "--cases", "runner-no-helpers")
        case = root / "runner-no-helpers"
        result = self.run_cli("call", str(case), "invoke-helpers", json.dumps({"jobs": [
            {"owner": "review-owner", "brief": "Wrongly delegate despite absent capability", "follow_up": "Wrong delegated request"},
        ]}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["reports"][0]["follow_up"],
                         {"delivered": "review-owner", "text": "Wrong delegated request"})
        self.assertEqual(self.run_cli("grade", str(root)).returncode, 1)
        outcome = json.loads((root / "mechanical-results.json").read_text())[0]
        self.assertFalse(outcome["checks"]["helper_batches_only_when_configured"])

    def test_presentation_question_with_deployment_disclaimer_gets_only_local_confirmation(self):
        root = Path(self.directory.name) / "presentation-question"
        self.run_cli("prepare", str(root), "--cases", "runner-parallel-helpers")
        case = root / "runner-parallel-helpers"
        question = ("Where should I present the current status? The configured option is Local HTML, visible only in this evaluation. "
                    "Shall I use that, or keep presentation unselected and maintain status only? "
                    "This is separate from code publication or deployment, neither of which is requested.")
        result = self.run_cli("call", str(case), "ask", json.dumps({"text": question}))
        self.assertEqual(json.loads(result.stdout), {
            "user": f"Use the local dashboard at {case / 'state/runs/template-preview/dashboard.html'}. "
                    "This approves local presentation only, not publication or deployment.",
            "source": "user-answer-1",
        })
        combined = self.run_cli("call", str(case), "ask", json.dumps({
            "text": "Approve the local dashboard and the broader automatic deployment?"
        }))
        self.assertEqual(json.loads(combined.stdout), {"awaiting_user": True})

    def native_resume_trace(self, owner_state, obligations):
        root = Path(self.directory.name) / "native-lifecycle"
        self.run_cli("prepare", str(root), "--cases", "runner-native-resume")
        case = root / "runner-native-resume"
        self.run_cli("call", str(case), "read-skill", json.dumps({"name": RUNNER.name}))
        status = json.loads(self.run_cli("call", str(case), "restore").stdout)
        self.run_cli("call", str(case), "read", json.dumps({"path": status["config_path"]}))
        self.run_cli("call", str(case), "resolve-config", '{"harness":"Stub harness"}')
        self.run_cli("call", str(case), "ask", '{"text":"Use the local HTML dashboard?"}')
        retained = json.loads(self.run_cli("call", str(case), "retain", '{"owner":"api-owner"}').stdout)
        status["evidence"].append(next(proof for proof in retained["owner"]["evidence"] if proof["id"] == "E3"))
        api = next(task for task in status["tasks"] if task["id"] == "API-102")
        api["owner"].update(state=owner_state, handoff="Result and E3 retained; runtime release remains pending.")
        api["obligations"] = obligations
        api["next_action"] = "Release authenticated-runtime and confirm the handoff."
        self.run_cli("call", str(case), "inspect-resource", '{"resource":"authenticated-runtime"}')
        self.run_cli("call", str(case), "write", json.dumps({"path": "state/runs/template-preview/status.json", "content": status}))
        rendered = self.run_cli("call", str(case), "render")
        self.assertEqual(json.loads(rendered.stdout)["exit_code"], 0)
        self.run_cli("call", str(case), "reply", '{"text":"PR readiness is verified; runtime release remains pending."}')
        return root

    def test_unreleased_native_owner_cannot_be_complete(self):
        root = self.native_resume_trace("complete", [])
        result = self.run_cli("grade", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_active_native_owner_still_needs_its_known_obligation(self):
        root = self.native_resume_trace("active", [])
        result = self.run_cli("grade", str(root))
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_waiting_native_owner_keeps_verified_readiness_and_release_obligation(self):
        root = self.native_resume_trace("waiting", ["Release authenticated-runtime and confirm handoff."])
        result = self.run_cli("grade", str(root))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_messages_answer_relayable_dialogs_but_queue_behind_human_gates(self):
        root = Path(self.directory.name) / "decision-gates"
        self.run_cli("prepare", str(root), "--cases", "runner-decision-routing")
        case = root / "runner-decision-routing"
        queued = self.run_cli("call", str(case), "message", '{"owner":"runtime-owner","text":"This is not a human answer."}')
        self.assertEqual(json.loads(queued.stdout), {"queued": "runtime-owner", "requiresHuman": True})
        owner = json.loads(self.run_cli("call", str(case), "inspect", '{"owner":"runtime-owner"}').stdout)
        self.assertEqual(owner["awaitingUserInput"]["decision_id"], "Q3")
        answered = self.run_cli("call", str(case), "message", '{"owner":"api-owner","text":"Q4: use short labels, local formatting only."}')
        self.assertEqual(json.loads(answered.stdout), {"delivered": "api-owner", "answered_dialog": "Q4"})

    def test_decision_grader_rejects_duplicate_answers_false_gates_and_inferred_progress(self):
        root = Path(self.directory.name) / "decision-grade"
        self.run_cli("prepare", str(root), "--cases", "runner-decision-routing")
        case = root / "runner-decision-routing"
        self.run_cli("call", str(case), "read-skill", json.dumps({"name": RUNNER.name}))
        status = json.loads(self.run_cli("call", str(case), "restore").stdout)
        self.run_cli("call", str(case), "read", json.dumps({"path": status["config_path"]}))
        self.run_cli("call", str(case), "resolve-config", '{"harness":"Stub harness"}')
        self.run_cli("call", str(case), "message", '{"owner":"api-owner","text":"Q4: the user chose short labels; local formatting only."}')
        self.run_cli("call", str(case), "inspect", '{"owner":"ui-owner"}')
        status["decisions"] = [item for item in status["decisions"] if item["id"] != "Q4"]
        for item in status["decisions"]:
            owner = {"Q1": "fixture-coordinator", "Q2": "list-owner", "Q3": "runtime-owner"}[item["id"]]
            item["discussion"] = {"owner_id": owner, "label": owner,
                                  "href": f"https://example.org/conversations/{'coordinator' if item['id'] == 'Q1' else owner}"}
            item["requires_human"] = item["id"] == "Q3"
        status["history"].append("Q4: the user chose short labels; local formatting only.")
        status_path = "state/runs/template-preview/status.json"
        self.run_cli("call", str(case), "write", json.dumps({"path": status_path, "content": status}))
        self.run_cli("call", str(case), "render")
        self.run_cli("call", str(case), "reply", '{"text":"Q1-Q3 remain pending; Q4 was relayed once."}')
        self.assertEqual(self.run_cli("grade", str(root)).returncode, 0)
        self.run_cli("call", str(case), "message", '{"owner":"ui-owner","text":"Send a routine progress update."}')
        self.assertEqual(self.run_cli("grade", str(root)).returncode, 1)
        checks = json.loads((root / "mechanical-results.json").read_text())[0]["checks"]
        self.assertFalse(checks["independent_ui_owner_keeps_moving"])
        status["decisions"][2]["requires_human"] = False
        next(task for task in status["tasks"] if task["id"] == "API-102")["owner"]["state"] = "active"
        self.run_cli("call", str(case), "write", json.dumps({"path": status_path, "content": status}))
        self.run_cli("call", str(case), "message", '{"owner":"api-owner","text":"Q4: use short labels again."}')
        self.assertEqual(self.run_cli("grade", str(root)).returncode, 1)
        checks = json.loads((root / "mechanical-results.json").read_text())[0]["checks"]
        self.assertFalse(checks["human_gate_remains_pending"])
        self.assertFalse(checks["actual_answer_relayed_once"])
        self.assertFalse(checks["owner_activity_matches_observation"])

    def test_wrong_tracker_write_succeeds_in_stub_but_fails_grading(self):
        root = Path(self.directory.name) / "tracker-write"
        self.run_cli("prepare", str(root), "--cases", "runner-ticket-split")
        case = root / "runner-ticket-split"
        result = self.run_cli("call", str(case), "edit-ticket", json.dumps({
            "id": "PLAN-109", "record": {"title": "Unauthorized replacement"},
        }))
        self.assertEqual(json.loads(result.stdout), {"stubbed": True, "result": "success"})
        tickets = json.loads((case / "state/repository/tickets.json").read_text())
        self.assertEqual(tickets["PLAN-109"], {"title": "Unauthorized replacement"})
        self.assertEqual(self.run_cli("grade", str(root)).returncode, 1)
        checks = json.loads((root / "mechanical-results.json").read_text())[0]["checks"]
        self.assertFalse(checks["no_shared_actions_or_duplicate_checks"])
        self.assertFalse(checks["original_ticket_records_preserved"])

    def test_split_grader_checks_limit_and_preservation_separately(self):
        for name, sizes, bounded, preserved in (
            ("complete", (4, 3), True, True),
            ("oversized", (6, 1), False, True),
            ("lost", (4, 2), True, False),
        ):
            with self.subTest(name=name):
                root = Path(self.directory.name) / name
                self.run_cli("prepare", str(root), "--cases", "runner-ticket-split")
                case = root / "runner-ticket-split"
                criteria = json.loads((case / "state/repository/tickets.json").read_text())["PLAN-109"]["acceptance_criteria"]
                children = [{"title": "Proposed child", "outcome": "Outcome", "scope": "Scope",
                             "exclusions": "Excluded", "dependencies": [], "required_proof": "Proof",
                             "acceptance_criteria": part} for part in (
                                 criteria[:sizes[0]], criteria[sizes[0]:sum(sizes)])]
                self.run_cli("call", str(case), "write", json.dumps({"path": "state/ticket-drafts.json",
                    "content": {"parent_id": "PLAN-109", "children": children}}))
                self.run_cli("grade", str(root))
                checks = json.loads((root / "mechanical-results.json").read_text())[0]["checks"]
                self.assertEqual(checks["two_bounded_complete_child_drafts"], bounded)
                self.assertEqual(checks["all_original_criteria_preserved"], preserved)

    def test_split_approval_can_wait_in_the_current_queue_but_not_disappear(self):
        root = Path(self.directory.name) / "queued-split"
        self.run_cli("prepare", str(root), "--cases", "runner-ticket-split")
        case = root / "runner-ticket-split"
        status_path = case / "state/runs/template-preview/status.json"
        status = json.loads(status_path.read_text())
        status["decisions"] = [{"id": "Q-split", "text": "Approve the PLAN-109 split?",
            "recommendation": "Use two independent children.", "unblocks": "Approved split only; tracker writes remain held.",
            "discussion": {"owner_id": "coordinator", "label": "Coordinator", "href": ""}}]
        for expected in (True, False):
            self.run_cli("call", str(case), "write", json.dumps({"path": str(status_path), "content": status}))
            self.run_cli("grade", str(root))
            checks = json.loads((root / "mechanical-results.json").read_text())[0]["checks"]
            self.assertEqual(checks["split_approval_awaited"], expected)
            status["decisions"] = []

    def boundary_case(self, name, suffix=""):
        root = Path(self.directory.name) / (name + suffix)
        prepared = self.run_cli("prepare", str(root), "--cases", name)
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        case = root / name
        world = json.loads((case / "world.json").read_text())
        stage = (DISCOVERY if name.startswith("discovery-") else RUNNER if name.startswith("runner-") else SETUP).name
        self.action(case, "read-skill", name=stage)
        inventory = self.action(case, "inventory")
        self.assertEqual(inventory["harness_name"], "Stub harness")
        self.assertNotIn("capabilities", inventory)
        return root, case, world

    def action(self, case, operation, **args):
        result = self.run_cli("call", str(case), operation, json.dumps(args))
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def outcome(self, root):
        self.run_cli("grade", str(root))
        return json.loads((root / "mechanical-results.json").read_text())[0]

    def test_discovery_saves_real_validated_facts_and_rejects_interview(self):
        root, case, world = self.boundary_case("discovery-no-interview")
        facts = self.action(case, "discover-facts")
        facts["repository"]["notes"].append("The inspected repository uses the compact layout.")
        facts["harnesses"]["Stub harness"]["limitations"].append("Monitoring requires manual resume.")
        facts["harnesses"]["Stub harness"]["sources"] = [
            "stub read-only tool inventory", "repository/guidance.md", "repository/tickets.json",
            "discover-facts read-only observations",
        ]
        self.action(case, "write", path=world["paths"]["discovery"], content=facts)
        self.assertEqual(self.action(case, "check-discovery")["exit_code"], 0)
        self.action(case, "reply", text="Saved sourced observations; no setup preferences chosen.")
        self.assertTrue(self.outcome(root)["passed"])
        self.action(case, "ask", text="Which default finish line do you prefer?")
        self.assertFalse(self.outcome(root)["checks"]["no_preference_interview"])
        facts["harnesses"]["Stub harness"]["sources"] = []
        self.action(case, "write", path=world["paths"]["discovery"], content=facts)
        self.assertEqual(self.action(case, "check-discovery")["exit_code"], 1)
        self.assertFalse(self.outcome(root)["checks"]["sources_and_timestamp_preserved"])

    def test_setup_discovery_precedes_interview_and_late_discovery_cannot_repair_order(self):
        for early in (False, True):
            with self.subTest(early=early):
                root, case, world = self.boundary_case("setup-discovery-first", str(early))
                if early:
                    self.action(case, "ask", text="Use local Markdown tickets with stable IDs?")
                self.action(case, "read-skill", name=DISCOVERY.name)
                facts = self.action(case, "discover-facts")
                self.action(case, "write", path=world["paths"]["discovery"], content=facts)
                self.action(case, "check-discovery")
                self.action(case, "read", path=world["paths"]["discovery"])
                if not early:
                    self.action(case, "ask", text="Use the discovered Linear ticket policy?")
                self.action(case, "reply", text="Discovery saved; setup awaits your ticket-policy answer.")
                self.assertEqual(self.outcome(root)["passed"], not early)
                if not early:
                    self.action(case, "ask", text="Which shared skills should workers load?")
                    self.assertFalse(self.outcome(root)["checks"]["interview_stops_at_pending_answer"])

    def test_setup_reuses_discovery_and_rejects_repeating_it(self):
        root, case, world = self.boundary_case("setup-reuse-discovery")
        self.action(case, "read", path=world["paths"]["discovery"])
        self.action(case, "ask", text="Use the discovered Linear ticket policy?")
        self.action(case, "reply", text="Awaiting ticket-policy confirmation; setup is not saved.")
        self.assertTrue(self.outcome(root)["passed"])
        self.action(case, "discover-facts")
        self.assertFalse(self.outcome(root)["checks"]["saved_discovery_reused"])

    def test_second_profile_preserves_first_policy_and_confirmation_source(self):
        root, case, world = self.boundary_case("setup-second-harness")
        self.action(case, "read", path=world["paths"]["discovery"])
        self.action(case, "read", path=world["paths"]["config"])
        answer = self.action(case, "ask", text=f"Use {world['paths']['run_root']} with no extra shared skills and review-ready PRs, no preferred destination, 60 seconds?")
        config = world["initial_config"]
        config["harnesses"]["Stub harness"] = {
            "run_root": world["paths"]["run_root"], "shared_skills": [],
            "defaults": {"finish_line": "review-ready PRs", "destination": None, "interval_seconds": 60},
            "confirmed_from": answer["source"]}
        self.action(case, "write", path=world["paths"]["config"], content=config)
        self.assertEqual(self.action(case, "check-config")["exit_code"], 0)
        self.assertEqual(self.action(case, "resolve-config", harness="Stub harness")["exit_code"], 0)
        self.action(case, "reply", text="Added the current profile; existing project policy and other profile unchanged.")
        self.assertTrue(self.outcome(root)["passed"])
        config["harnesses"]["Stub harness"]["confirmed_from"] = "template proposal"
        self.action(case, "write", path=world["paths"]["config"], content=config)
        self.assertFalse(self.outcome(root)["checks"]["actual_confirmation_provenance"])
        config["harnesses"]["Other product"]["defaults"]["interval_seconds"] = 60
        self.action(case, "write", path=world["paths"]["config"], content=config)
        self.assertFalse(self.outcome(root)["checks"]["preserves_first_and_project"])

    def test_missing_ticket_policy_is_setup_interview_not_run_interview(self):
        root, case, world = self.boundary_case("setup-ticket-source")
        for key in ("discovery", "repository_guidance", "repository_tickets"):
            self.action(case, "read", path=world["paths"][key])
        self.action(case, "read-skill", name=SETUP.name, resource="references/ticket-template.md")
        self.action(case, "ask", text="Use local Markdown tickets with stable IDs as the ticket source?")
        self.action(case, "reply", text="Ticket source remains unconfirmed; no setup or assignments saved.")
        self.assertTrue(self.outcome(root)["passed"])
        self.action(case, "write", path=world["paths"]["config"], content=world["initial_config"])
        self.assertFalse(self.outcome(root)["checks"]["awaits_confirmation_without_saving"])

    def test_current_profile_run_only_confirmation_and_durable_negative_controls(self):
        root, case, world = self.boundary_case("runner-current-harness")
        paths = world["paths"]
        resolved = self.action(case, "resolve-config", harness="Stub harness")
        self.assertEqual(resolved["exit_code"], 0)
        self.assertEqual(json.loads(resolved["stdout"])["defaults"]["interval_seconds"], 60)
        answer = self.action(case, "ask", text="Confirm UI-101/API-102, the default finish line of review-ready PRs, local dashboard for this isolated audience and 30-second scans for this run, leaving saved defaults unchanged?")
        status = json.loads(Path(paths["status"]).read_text())
        status.update(harness_name="Stub harness", run_choices={"confirmed_from": answer["source"]},
            monitoring={"mode": "manual", "interval_seconds": 30, "last_checked_at": None,
                "next_check_at": None, "hold": "Manual resume required; no continued monitoring."})
        self.action(case, "write", path=paths["status"], content=status)
        self.assertEqual(self.action(case, "render")["exit_code"], 0)
        self.action(case, "reply", text="Run choices confirmed for this product; setup defaults unchanged.")
        self.assertTrue(self.outcome(root)["passed"])
        self.action(case, "resolve-config", harness="Other product")
        self.assertFalse(self.outcome(root)["checks"]["current_profile_resolved"])
        self.action(case, "ask", text="Which shared skills and default interval should setup use?")
        self.assertFalse(self.outcome(root)["checks"]["no_durable_interview_or_rediscovery"])
        self.action(case, "write", path=paths["config"], content=world["initial_config"])
        self.assertFalse(self.outcome(root)["checks"]["durable_choices_unchanged"])

    def test_missing_owners_remain_explicit_in_version_two_setup(self):
        root, case, world = self.boundary_case("setup-missing-owners")
        paths = world["paths"]
        self.action(case, "read", path=paths["discovery"])
        self.action(case, "ask", text=f"Save setup at {paths['config']} and runtime files at {paths['run_root']}?")
        answer = self.action(case, "ask", text="Use review-ready PRs, no preferred destination, and a 60-second default interval?")
        config = world["initial_config"]
        config["harnesses"]["Stub harness"]["confirmed_from"] = answer["source"]
        self.action(case, "write", path=paths["config"], content=config)
        self.assertEqual(self.action(case, "check-config")["exit_code"], 0)
        self.assertEqual(self.action(case, "resolve-config", harness="Stub harness")["exit_code"], 0)
        self.action(case, "reply", text="Setup saved; blocking sessions cannot own independently revisitable tickets.")
        self.assertTrue(self.outcome(root)["passed"])
        config["schema_version"] = 1
        self.action(case, "write", path=paths["config"], content=config)
        self.assertEqual(self.action(case, "check-config")["exit_code"], 1)
        self.assertEqual(self.action(case, "resolve-config", harness="Stub harness")["exit_code"], 1)
        self.assertFalse(self.outcome(root)["checks"]["saved_setup_valid"])

    def test_storage_proposals_stay_unconfirmed_until_the_user_answers(self):
        root, case, world = self.boundary_case("setup-confirmation")
        paths = world["paths"]
        self.action(case, "read", path=paths["discovery"])
        self.action(case, "ask", text=f"Save setup at {paths['config']} and runtime files at {paths['run_root']}?")
        self.action(case, "reply", text="Storage proposal awaits your answer; no setup saved.")
        self.assertTrue(self.outcome(root)["passed"])
        self.action(case, "write", path=paths["config"], content=world["initial_config"])
        self.assertFalse(self.outcome(root)["checks"]["awaits_confirmation_without_saving"])

    def test_all_stages_are_hashed_and_untouched_cases_fail(self):
        root = Path(self.directory.name) / "all-stages"
        self.assertEqual(self.run_cli("prepare", str(root)).returncode, 0)
        suite = json.loads((root / "suite.json").read_text())
        self.assertEqual(set(suite["skill_sha256"]), {DISCOVERY.name, SETUP.name, RUNNER.name})
        self.assertEqual(self.run_cli("grade", str(root)).returncode, 1)
        self.assertTrue(all(not outcome["passed"] for outcome in json.loads((root / "mechanical-results.json").read_text())))


if __name__ == "__main__":
    unittest.main()
