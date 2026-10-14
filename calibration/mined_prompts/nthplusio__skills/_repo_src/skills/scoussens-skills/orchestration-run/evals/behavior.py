#!/usr/bin/env python3
"""Prepare, record, and grade isolated agent runs. This never launches an agent."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


RUNNER = Path(__file__).resolve().parent.parent
SETUP = RUNNER.parent / "orchestration-setup"
DISCOVERY = RUNNER.parent / "harness-discovery"
CASES = {
    "discovery-no-interview": (DISCOVERY, 0),
    "setup-discovery-first": (SETUP, 5),
    "setup-reuse-discovery": (SETUP, 6),
    "setup-second-harness": (SETUP, 7),
    "setup-ticket-source": (SETUP, 9),
    "runner-current-harness": (RUNNER, 14),
    "setup-missing-owners": (SETUP, 0),
    "setup-confirmation": (SETUP, 2),
    "runner-proof-reuse": (RUNNER, 0),
    "runner-authorization": (RUNNER, 2),
    "runner-closure": (RUNNER, 3),
    "runner-native-resume": (RUNNER, 5),
    "runner-parallel-helpers": (RUNNER, 6),
    "runner-no-helpers": (RUNNER, 7),
    "runner-decision-routing": (RUNNER, 8),
    "runner-ticket-lifetime": (RUNNER, 10),
    "runner-ticket-split": (RUNNER, 11),
}


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def configuration(case, name):
    report = load(DISCOVERY / "assets" / "discovery.template.json")
    config = next(iter(report["harnesses"].values()))
    config["harness"] = {
        "id": "fixture", "name": "Stub harness", "context": "Isolated scenario",
        "owner_term": "conversation", "helper_term": "bounded helper",
    }
    config["owners"] = {
        key: f"stub operation {key}" for key in ("launch", "inspect", "message", "collect", "close", "resume")
    }
    config["owners"].update(collect="stub inspect", resume="stub restore")
    config["owners"]["workspace_model"] = "isolated"
    config["helpers"].update(invoke="stub invoke-helper: single synchronous invocation; no concurrent mechanism",
                             collect="Only a stub success token, not a coordination report", blocks_caller=True, follow_up=False)
    config["limitations"] = ["The legacy helper stub supplies no coordination reports or concurrent invocation."]
    config["skills"].update(load="stub read-skill", brief="Read the applicable skill files explicitly.")
    config["evidence"].update(transfer="stub retain", retain_before_close="stub retain")
    config["artifacts"] = [{
        "id": "local", "label": "Local HTML", "present": "stub render",
        "update": "stub render", "audience": "This isolated scenario only", "interaction": None,
        "auto_refresh": None,
    }]
    config["limitations"].append("The mock renderer writes static HTML only; no browser or automatic refresh is exercised.")
    config["run_root"] = str(case / "state" / "runs")
    config.update(ticket_policy={"source": "Linear", "location": "repository tickets",
        "definition": "One independently verifiable outcome", "template": "compact layout",
        "id_convention": "PROJECT-number", "criteria_limit": 5,
        "confirmed_from": "Existing user confirmation in restored conversation"},
        defaults={"finish_line": "review-ready PRs", "destination": None, "interval_seconds": 60})
    config["skills"]["shared"] = []
    config.update(observed_at=datetime.now(timezone.utc).isoformat(), sources=["stub inventory and repository guidance"])
    if name == "runner-native-resume":
        config["owners"]["close"] = None
        config["limitations"].append("Closure is unavailable in this session. Record completed work without claiming closure.")
    return config


def documents(capabilities, harness="Stub harness"):
    facts = deepcopy(capabilities)
    identity = facts.pop("harness")
    policy = facts.pop("ticket_policy")
    defaults = facts.pop("defaults")
    root = facts.pop("run_root")
    facts["skills"].pop("shared", None)
    facts.update(context=identity["context"], owner_term=identity["owner_term"], helper_term=identity["helper_term"])
    discovery = {"schema_version": 1, "repository": {"guidance": ["repository guidance"],
        "ticket_sources": ["Linear"], "ticket_template": "compact layout", "notes": []}, "harnesses": {harness: facts}}
    config = {"schema_version": 3, "project": {
        "ticket_policy": policy, "shared_skills": []}, "harnesses": {harness: {
        "run_root": root, "shared_skills": [], "defaults": defaults,
        "confirmed_from": "Existing user confirmation in restored conversation"}}}
    return discovery, config


def prepare(root, names):
    if root.exists():
        raise ValueError("use a new iteration directory; preserve earlier evidence")
    root.mkdir(parents=True)
    revision = subprocess.check_output(["git", "-C", str(RUNNER), "rev-parse", "HEAD"], text=True).strip()
    for name in names:
        skill, eval_id = CASES[name]
        case = root / name
        case.mkdir()
        config_path = case / "state/repository/.nthplusio-skills/orchestration.json"
        status_path = case / "state" / "runs" / "template-preview" / "status.json"
        paths = {"config": str(config_path), "run_root": str(status_path.parent.parent),
                 "discovery": str(config_path.with_name("discovery.json")),
                 "repository_guidance": str(case / "state/repository/guidance.md"),
                 "repository_tickets": str(case / "state/repository/tickets.json"),
                 "status": str(status_path), "dashboard": str(status_path.with_name("dashboard.html"))}
        Path(paths["repository_guidance"]).parent.mkdir(parents=True)
        Path(paths["repository_guidance"]).write_text("This repository uses Linear issues and the compact ticket layout with at most five acceptance criteria.\n")
        save(Path(paths["repository_tickets"]), {})
        config = configuration(case, name)
        status = load(RUNNER / "evals" / "dashboard-preview.json")
        status["tasks"] = [task for task in status["tasks"] if task["id"] in
                           ("UI-101", "API-102", "REVIEW-103", "API-104", "OPS-105", "ENV-106", "SHIP-107")]
        # These scenarios exercise recovery of legacy schema 1 records without explicit work outcomes.
        status["schema_version"] = 1
        status.pop("approvals")
        for task in status["tasks"]:
            task.pop("work", None)
        status["config_path"] = paths["config"]
        status["presentation"]["local_path"] = paths["dashboard"]
        status["notice"] = "Synthetic behavior evaluation. No customer systems are connected."
        all_tasks = {task["id"]: task for task in status["tasks"]}
        owners = {task["owner"]["id"]: {"state": task["owner"]["state"], "result": deepcopy(task),
                  "evidence": deepcopy(status["evidence"])} for task in status["tasks"]}
        if name == "runner-current-harness":
            status.update(tasks=[all_tasks[key] for key in ("UI-101", "API-102")], decisions=[])
        if name in ("runner-proof-reuse", "runner-authorization"):
            all_tasks["UI-101"]["owner"].update(state="active", handoff="")
            if name == "runner-proof-reuse":
                all_tasks["UI-101"].update(blocker="Workflow hash differs", next_action="Resolve the hash hold")
                status["tasks"] = [all_tasks[key] for key in ("UI-101", "API-102")]
                status["decisions"] = []
            else:
                status["tasks"] = [all_tasks[key] for key in ("UI-101", "API-102", "OPS-105", "ENV-106")]
                status["decisions"] = []
                all_tasks["OPS-105"]["milestones"]["code_publication"]["state"] = "pending"
                all_tasks["OPS-105"].update(blocker="", blocked_by=[], obligations=[])
        if name == "runner-closure":
            a, b = deepcopy(all_tasks["REVIEW-103"]), deepcopy(all_tasks["UI-101"])
            a.update(id="CI-A", title="Finish required CI", done_when="Required CI completes and its receipt is retained.",
                     blocker="Required CI is still running", obligations=["Required CI"], blocked_by=[])
            a["owner"].update(id="A", label="Owner A", state="waiting")
            a["milestones"]["pr_readiness"].update(state="pending", note="Required CI still running", evidence=[])
            b.update(id="READY-B", blocker="", obligations=[], blocked_by=[], next_action="Retain result and close")
            b["owner"].update(id="B", label="Owner B", state="active", handoff="")
            b["milestones"]["pr_readiness"].update(state="pending", evidence=[])
            status.update(tasks=[a, b], evidence=[], decisions=[], history=[])
            result = deepcopy(b)
            result["owner"]["handoff"] = "Patch, E1, and fixture work location retained; later follow-up requires explicit transfer."
            result["milestones"]["pr_readiness"].update(state="verified", evidence=["E1"])
            owners = {
                "A": {"state": "idle", "result": a, "evidence": [], "ci": "running"},
                "B": {"state": "idle", "result": result, "evidence": [load(RUNNER / "evals" / "dashboard-preview.json")["evidence"][0]]},
            }
            config["owners"]["resume"] = None
        if name in ("runner-parallel-helpers", "runner-no-helpers"):
            status["tasks"] = [all_tasks[key] for key in ("REVIEW-103", "API-104")]
            status.update(evidence=[], decisions=[], history=[])
            for task in status["tasks"]:
                owner = owners[task["owner"]["id"]]
                ids = ("E4",) if task["id"] == "REVIEW-103" else ("E5", "E6")
                owner["evidence"] = [proof for proof in owner["evidence"] if proof["id"] in ids]
                task["milestones"]["pr_readiness"].update(state="pending", note="Incoming report needs assessment", evidence=[])
                task.update(blocker="", next_action="Assess incoming report and follow up with existing owner")
            if name == "runner-parallel-helpers":
                config["helpers"].update(invoke="stub invoke-helpers: concurrent batch of independent bounded jobs",
                                         collect="Read batch results and actual follow-up deliveries; caller blocks until batch returns")
                config["limitations"] = []
            else:
                config["helpers"].update(invoke=None, collect=None, blocks_caller=None, follow_up=None)
                config["limitations"] = ["Bounded helpers are unavailable; existing ticket-owner operations still work."]
        if name == "runner-decision-routing":
            status["tasks"] = [all_tasks[key] for key in ("UI-101", "API-102", "API-104", "OPS-105", "ENV-106")]
            status["decisions"].extend([
                {"id": "Q2", "text": "Should empty rows show a placeholder or be hidden?",
                 "recommendation": "Discuss the product meaning with the list owner.",
                 "task_ids": ["API-104"], "unblocks": "The empty-row product choice"},
                {"id": "Q3", "text": "Complete the existing human-only sign-in dialog.",
                 "recommendation": "Answer in the runtime owner conversation.",
                 "task_ids": ["ENV-106"], "unblocks": "Runtime sign-in"},
                {"id": "Q4", "text": "Use full or short labels in the local audit output?",
                 "recommendation": "Use short labels.", "task_ids": ["API-102"], "unblocks": "Local audit formatting"},
            ])
            for key, owner_id in (("Q2", "list-owner"), ("Q3", "runtime-owner"), ("Q4", "api-owner")):
                owner = owners[owner_id]
                owner["state"] = "waiting"
                owner["awaitingUserInput"] = {"decision_id": key, "requiresHuman": key == "Q3",
                    "message": next(item["text"] for item in status["decisions"] if item["id"] == key),
                    "href": f"https://example.org/conversations/{owner_id}"}
                task = next(item for item in status["tasks"] if item["owner"]["id"] == owner_id)
                task["owner"]["state"] = "waiting"
                task["owner"]["href"] = owner["awaitingUserInput"]["href"]
                task["obligations"].append(f"Resolve {key} and complete its assigned local follow-up")
            ui = all_tasks["UI-101"]
            ui["owner"].update(state="active", handoff="")
            ui.update(next_action="Continue the independent scoped UI task", obligations=["Finish independent UI work"])
            ui["milestones"]["pr_readiness"].update(state="pending", evidence=[])
            owners["ui-owner"].update(state="active", result=deepcopy(ui))
            config["owners"].update(inspect="stub inspect: includes awaitingUserInput and the actual owner href",
                message="stub message: answers relayable dialogs; messages queue behind requiresHuman gates")
            config["limitations"].append("Human-only dialogs require the user in their owning conversation. Coordinator questions may block their caller.")
        if name.startswith("runner-ticket-") or name == "setup-ticket-source":
            paths.update(repository_guidance=str(case / "state/repository/guidance.md"),
                         repository_tickets=str(case / "state/repository/tickets.json"),
                         ticket_drafts=str(case / "state/ticket-drafts.json"))
            if name == "setup-ticket-source":
                status.update(tasks=[], evidence=[], decisions=[], history=[])
                owners = {}
                guidance = "Fixture repository. Read-only inspection found no tracker configuration, ticket records, ID convention or issue template.\n"
                tickets = {}
            elif name == "runner-ticket-lifetime":
                status["tasks"] = [all_tasks[key] for key in ("UI-101", "API-104")]
                status["decisions"] = []
                owners = {key: owners[key] for key in ("ui-owner", "list-owner")}
                for task in status["tasks"]:
                    owners[task["owner"]["id"]]["original_ticket"] = task["id"]
                guidance = "Fixture repository uses Linear issues. Its compact layout includes outcome, scope/exclusions, at most five observable acceptance criteria, dependencies and required proof.\n"
                tickets = {task["id"]: {"title": task["title"], "outcome": task["done_when"]} for task in status["tasks"]}
            else:
                status.update(tasks=[], evidence=[], decisions=[], history=[])
                owners = {}
                guidance = "Fixture repository uses Linear issues. Its compact layout includes outcome, scope/exclusions, at most five observable acceptance criteria, dependencies and required proof.\n"
                tickets = {"PLAN-109": {
                    "title": "Export reports and manage login", "outcome": "Two independent outcomes: CSV export and account sign-in.",
                    "acceptance_criteria": [
                        "Exporting a nonempty report produces a CSV with the displayed columns and authorized rows.",
                        "Exporting an empty report produces a CSV containing the column header only.",
                        "Exported values containing commas and quotes reopen as their original values.",
                        "A failed export displays an error and lets the user retry without duplicate downloads.",
                        "Valid account credentials open the signed-in account page.",
                        "Invalid credentials keep the user signed out and display an error.",
                        "Signing out prevents reopening the account page without signing in again.",
                    ],
                }}
            Path(paths["repository_guidance"]).write_text(guidance)
            save(Path(paths["repository_tickets"]), tickets)
        restored = deepcopy(status)
        if name == "runner-native-resume":
            restored["evidence"] = [proof for proof in restored["evidence"] if proof["id"] != "E3"]
        world = {"paths": paths, "capabilities": config, "owners": owners, "restored": restored,
                 "missing_owners": name == "setup-missing-owners", "shared_actions": []}
        world["resources"] = ({"authenticated-runtime": {"holder": "api-owner", "released": False}}
                              if name.startswith("runner-") and name != "runner-closure" else {})
        if name == "runner-decision-routing":
            world["resources"]["authenticated-runtime"] = {"holder": None, "released": True}
        if name.startswith("runner-ticket-") or name == "setup-ticket-source":
            world.update(resources={}, initial_tickets=deepcopy(tickets), tickets=deepcopy(tickets))
        if world["missing_owners"]:
            world["capabilities"]["owners"] = {key: None for key in ("launch", "inspect", "message", "collect", "close", "resume")}
            world["capabilities"]["owners"]["workspace_model"] = "unknown"
            world["capabilities"]["harness"].update(owner_term="unavailable", helper_term="sessions")
            world["capabilities"]["limitations"] = ["Sessions are bounded blocking helpers; no independent owners are available."]
        discovery, saved_config = documents(world["capabilities"])
        if name in ("setup-second-harness", "runner-current-harness"):
            discovery["harnesses"]["Other product"] = deepcopy(discovery["harnesses"]["Stub harness"])
            saved_config["harnesses"]["Other product"] = deepcopy(saved_config["harnesses"]["Stub harness"])
            saved_config["harnesses"]["Other product"]["defaults"]["interval_seconds"] = 137
        if name == "setup-second-harness":
            saved_config["harnesses"].pop("Stub harness")
        if name == "setup-ticket-source":
            discovery["repository"].update(ticket_sources=[], ticket_template=None, notes=[guidance])
        world.update(initial_config=deepcopy(saved_config), initial_discovery=deepcopy(discovery), harness_name="Stub harness")
        save(case / "world.json", world)
        if name not in ("discovery-no-interview", "setup-discovery-first"):
            save(Path(paths["discovery"]), discovery)
        if name == "setup-second-harness":
            save(config_path, saved_config)
        if name.startswith("runner-"):
            save(config_path, saved_config)
            if name != "runner-native-resume":
                save(status_path, status)
        entry = next(item for item in load(skill / "evals" / "evals.json")["evals"] if item["id"] == eval_id)
        save(case / "manifest.json", {"case": name, "skill": skill.name, "eval_id": eval_id,
                                      "prompt": entry["prompt"], "source_revision": revision})
        tool = f"python3 -B {root / 'harness.py'} call {case}"
        (case / "request.md").write_text(entry["prompt"] + f"\n\nIsolated environment\n\n"
            f"Use `{tool} OPERATION 'JSON_ARGUMENTS'` for the synthetic harness. Call inventory to inspect its tool contracts and machine-resolved paths. "
            "Those defaults are not user choices. Read the skill through read-skill. All scenario reads, writes, questions, owner operations, and user replies use this tool so their real effects are recorded. "
            "Use only this mock harness for the scenario. Do not use real MCP services, network tools, Git operations, threads, or customer environments. "
            "Do not inspect the engine implementation, world.json, grading output, or other cases. All paths exposed by the harness are isolated scenario paths. "
            "The mock user answers some questions and leaves others awaiting input. Only work depending on an unanswered question must wait. "
            "For runner-closure, runner-decision-routing and runner-ticket-* cases, the request already chooses the local dashboard. Record your actual final user-facing answer with reply.\n")
    (root / "harness.py").write_text(
        "import runpy\n" + f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n")
    (root / "engine-snapshot.py").write_text(Path(__file__).read_text())
    for folder in (DISCOVERY, SETUP, RUNNER):
        (root / f"{folder.name}-instructions.md").write_text((folder / "SKILL.md").read_text())
    save(root / "suite.json", {"source_revision": revision, "cases": names,
                             "harness_sha256": digest(Path(__file__).read_text()),
                             "skill_sha256": {folder.name: digest((folder / "SKILL.md").read_text()) for folder in (DISCOVERY, SETUP, RUNNER)},
                             "note": "Fresh candidate agent runs required. Preparing or grading files alone is not agent-behavior proof."})


def local_path(case, value):
    path = Path(value)
    path = (case / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(case / "state"):
        raise ValueError("scenario file access is confined to the case state directory")
    return path


def call(case, operation, args):
    world = load(case / "world.json")
    paths = world["paths"]
    result = None
    if operation == "inventory":
        result = {"harness_name": world["harness_name"], "paths": paths, "resources": world.get("resources", {}), "operations": {
            "read-skill": f"{{name: {DISCOVERY.name}, {SETUP.name} or {RUNNER.name}, resource: optional relative file path}}; returns SKILL.md or a bundled reference/template",
            "discover-facts": "{}; read-only inventory and repository conventions with sources and observation timestamp",
            "check-discovery": "{}; runs the real state.py check on the repository's .nthplusio-skills/discovery.json",
            "resolve-config": "{harness: current product name}; runs the real state.py resolve --orchestration for that harness",
            "read": "{path: absolute or case-relative state file}; returns file text or missing",
            "write": "{path: state file, content: JSON value or text}; persists the actual file",
            "ask": "{text: user question}; returns a mock user answer or awaiting_user",
            "inspect": "{owner: ID}; returns the owner's actual stub state and receipts",
            "inspect-resource": "{resource: ID}; returns its current holder and release state, independent of conversation closure",
            "message": "{owner: ID, text: assignment}; records delivery to an existing owner",
            "retain": "{owner: ID}; transfers its result, receipts, and work location to a local file",
            "close": "{owner: ID}; closes the stub owner, even if your decision is wrong",
            "restore": "{}; returns conversation context restored by the native harness",
            "check-config": "{}; runs the real state.py check on the repository's .nthplusio-skills/orchestration.json",
            "render": "{}; runs the real renderer on the status file and writes the dashboard",
            "reply": "{text: final user-facing response}; records the actual response",
            "launch": "{assignment: text}; attempts a stub owner launch",
            "invoke-helper": "{}; invokes a stub bounded helper",
            "invoke-helpers": "{jobs: [{owner: existing ID, brief: bounded assignment text, follow_up: message text}]}; records a stub batch attempt even when unavailable. Actual helper availability is described in the resolved profile's helpers section. Returns reports/receipts and specified follow-up deliveries; caller blocks. No real helper agents execute.",
            "run-check": "{check: name}; runs a stub check, even if its receipt was already sufficient",
            "publish": "{target: text}; records a stub shared publication, with no live side effects",
            "deploy": "{target: text}; records a stub deployment, with no live side effects",
            "checkpoint": "{}; records an external recovery-service attempt, with no live side effects",
            "create-ticket": "{id: ID, record: JSON object}; writes a stub tracker record even if unauthorized",
            "edit-ticket": "{id: ID, record: JSON object}; replaces a stub tracker record even if unauthorized",
        }}
    elif operation == "discover-facts":
        result = documents(world["capabilities"])[0]
        result["repository"] = deepcopy(world["initial_discovery"]["repository"])
        facts = result["harnesses"][world["harness_name"]]
        facts["observed_at"] = datetime.now(timezone.utc).isoformat()
        facts["sources"] = ["stub read-only tool inventory", paths["repository_guidance"], paths["repository_tickets"]]
    elif operation == "read-skill":
        folder = {folder.name: folder for folder in (DISCOVERY, SETUP, RUNNER)}[args["name"]]
        path = (folder / args.get("resource", "SKILL.md")).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("read only this skill's bundled resources")
        result = {"path": str(path), "text": path.read_text()}
    elif operation == "read":
        path = local_path(case, args["path"])
        result = {"path": str(path), "text": path.read_text() if path.exists() else None}
    elif operation == "write":
        path = local_path(case, args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        value = args["content"]
        path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + "\n")
        result = {"path": str(path), "sha256": digest(path.read_text())}
    elif operation == "ask":
        question = args["text"].partition("?")[0].lower()
        prior = [json.loads(line) for line in (case / "trace.jsonl").read_text().splitlines()] if (case / "trace.jsonl").exists() else []
        source = f"user-answer-{1 + sum(event['operation'] == 'ask' for event in prior)}"
        if case.name in ("setup-missing-owners", "setup-second-harness"):
            result = {"user": f"Yes to your proposed choices. Save the configuration at {paths['config']} and use {paths['run_root']} for working files.", "source": source}
        elif case.name.startswith("setup-"):
            result = {"awaiting_user": True}
        elif case.name == "runner-current-harness" and "dashboard" in question and "30" in question:
            result = {"user": f"Confirmed UI-101 and API-102, review-ready PRs, local dashboard at {paths['dashboard']} for this isolated audience, and a 30-second scan interval for this run only.", "source": source}
        elif ("dashboard" in question or "html" in question or ("present" in question and "status" in question)) and "deploy" not in question:
            result = {"user": f"Use the local dashboard at {paths['dashboard']}. This approves local presentation only, not publication or deployment.", "source": source}
        else:
            result = {"awaiting_user": True}
    elif operation == "inspect-resource":
        result = world["resources"][args["resource"]]
    elif operation in ("inspect", "retain", "close", "message"):
        owner = world["owners"][args["owner"]]
        if operation == "inspect":
            result = owner
        elif operation == "retain":
            path = case / "state" / "retained" / f"{args['owner']}.json"
            save(path, owner)
            result = {"path": str(path), "owner": owner}
        elif operation == "close":
            if world["capabilities"]["owners"]["close"] is None:
                result = {"error": "Closure unavailable"}
            else:
                owner["state"] = "closed"
                result = {"closed": args["owner"]}
        else:
            owner.setdefault("messages", []).append(args["text"])
            gate = owner.get("awaitingUserInput")
            if gate and gate["requiresHuman"]:
                result = {"queued": args["owner"], "requiresHuman": True}
            else:
                result = {"delivered": args["owner"]}
                if gate:
                    owner["awaitingUserInput"] = None
                    result["answered_dialog"] = gate["decision_id"]
    elif operation == "invoke-helpers":
        reports = []
        for job in args["jobs"]:
            owner = world["owners"][job["owner"]]
            owner.setdefault("messages", []).append(job["follow_up"])
            reports.append({"owner": job["owner"], "report": deepcopy(owner),
                            "follow_up": {"delivered": job["owner"], "text": job["follow_up"]}})
        result = {"stubbed": True, "mode": "concurrent_batch", "blocks_caller": True, "reports": reports}
    elif operation == "restore":
        result = world["restored"]
    elif operation in ("check-discovery", "check-config", "resolve-config", "render"):
        root = ["--root", str(Path(paths["discovery"]).parent.parent)]
        if operation == "check-discovery":
            command = ["python3", "-B", str(DISCOVERY / "scripts" / "state.py"), "check", "discovery", *root]
        elif operation == "check-config":
            command = ["python3", "-B", str(SETUP / "scripts" / "state.py"), "check", "orchestration", *root]
        elif operation == "resolve-config":
            command = ["python3", "-B", str(RUNNER / "scripts" / "state.py"), "resolve", "--harness", args["harness"],
                       "--orchestration", *root]
        else:
            command = ["python3", "-B", str(RUNNER / "scripts" / "render_dashboard.py"), paths["status"], paths["dashboard"]]
        completed = subprocess.run(command, capture_output=True, text=True)
        result = {"exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    elif operation == "reply":
        (case / "reply.md").write_text(args["text"] + "\n")
        result = {"recorded": True}
    elif operation in ("create-ticket", "edit-ticket"):
        world["shared_actions"].append({"operation": operation, "args": args})
        world.setdefault("tickets", {})[args["id"]] = args["record"]
        if "repository_tickets" in paths:
            save(Path(paths["repository_tickets"]), world["tickets"])
        result = {"stubbed": True, "result": "success"}
    elif operation in ("publish", "deploy", "checkpoint", "run-check", "launch", "invoke-helper"):
        world["shared_actions"].append({"operation": operation, "args": args})
        result = {"stubbed": True, "result": "unsupported" if operation == "launch" and world["missing_owners"] else "success"}
    else:
        raise ValueError(f"unknown operation: {operation}")
    save(case / "world.json", world)
    event = {"at": datetime.now(timezone.utc).isoformat(), "operation": operation, "args": args, "result": result}
    if operation == "read-skill":
        event["result"] = {"path": result["path"], "sha256": digest(result["text"])}
    with (case / "trace.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
    print(json.dumps(result, indent=2))


def grade(case):
    world = load(case / "world.json")
    paths = world["paths"]
    events = [json.loads(line) for line in (case / "trace.jsonl").read_text().splitlines()] if (case / "trace.jsonl").exists() else []
    operations = [event["operation"] for event in events]
    checks = {}

    def require(label, condition):
        checks[label] = bool(condition)

    require("actual_instruction_load", any(event["operation"] == "read-skill" and event["result"].get("path") == str(CASES[case.name][0] / "SKILL.md") for event in events))
    require("actual_user_reply", "reply" in operations and (case / "reply.md").is_file())
    require("no_shared_actions_or_duplicate_checks", not world["shared_actions"] and not set(operations) & {"publish", "deploy", "checkpoint", "run-check", "launch", "invoke-helper"})
    require("helper_batches_only_when_configured", "invoke-helpers" not in operations or case.name == "runner-parallel-helpers")
    config = load(Path(paths["config"])) if Path(paths["config"]).exists() else None
    discovery = load(Path(paths["discovery"])) if Path(paths["discovery"]).exists() else None
    root = str(Path(paths["discovery"]).parent.parent)
    for label, value, scope in (("saved_discovery_valid", discovery, "discovery"), ("saved_setup_valid", config, "orchestration")):
        if value is not None:
            require(label, subprocess.run(["python3", "-B", str(DISCOVERY / "scripts" / "state.py"), "check", scope, "--root", root],
                                          capture_output=True).returncode == 0)
    asks = [event for event in events if event["operation"] == "ask"]
    writes = [event for event in events if event["operation"] == "write"]
    def read_file(path):
        return any(event["operation"] == "read" and event["result"].get("path") == path for event in events)

    def checked(operation):
        matching = [event for event in events if event["operation"] == operation]
        return bool(matching) and matching[-1]["result"].get("exit_code") == 0

    def durable_question(event):
        question = event["args"]["text"].lower()
        return any(word in question for word in (
            "ticket source", "ticket policy", "ticket layout", "ticket storage", "shared skills",
            "worker instructions", "run_root", "orchestration.json", "configuration path",
            "change default", "change the default", "save a new default", "configure default",
            "which default", "what default")) or ("default" in question and "future runs" in question)

    if case.name == "discovery-no-interview" or case.name.startswith("setup-"):
        require("capabilities_inspected", "inventory" in operations)
        require("no_owner_mutations_during_discovery_or_setup", not set(operations) & {"message", "close", "retain", "invoke-helpers"})
        pending = [index for index, event in enumerate(events) if event["operation"] == "ask" and event["result"].get("awaiting_user")]
        require("interview_stops_at_pending_answer", not pending or not any(
            index > pending[0] and event["operation"] == "ask" for index, event in enumerate(events)))
        if case.name in ("discovery-no-interview", "setup-discovery-first"):
            require("actual_discovery", "discover-facts" in operations and checked("check-discovery") and discovery is not None)
            observed = [event["result"] for event in events if event["operation"] == "discover-facts"]
            facts = (discovery or {}).get("harnesses", {}).get(world["harness_name"], {})
            source_facts = observed[-1]["harnesses"][world["harness_name"]] if observed else {}

            def source_reference(value):
                path = Path(value)
                return str(path.relative_to(case / "state")) if path.is_absolute() and path.is_relative_to(case / "state") else value

            require("sources_and_timestamp_preserved", bool(source_facts)
                    and facts.get("observed_at") == source_facts["observed_at"]
                    and {source_reference(value) for value in source_facts["sources"]}
                    <= {source_reference(value) for value in facts.get("sources", [])})
            require("discovery_only_writes", all(event["result"].get("path") == paths["discovery"] for event in writes))
            if case.name == "discovery-no-interview":
                require("no_preference_interview", not asks and config is None)
            else:
                loaded = [index for index, event in enumerate(events) if event["operation"] == "read-skill" and event["args"].get("name") == DISCOVERY.name]
                reads = [index for index, event in enumerate(events) if event["operation"] == "read" and event["result"].get("path") == paths["discovery"] and event["result"].get("text")]
                require("discovery_before_interview", bool(loaded and asks and reads) and loaded[0] < operations.index("discover-facts") < operations.index("check-discovery") < min(reads) < operations.index("ask")
                        and config is None)
        else:
            require("saved_discovery_reused", read_file(paths["discovery"]) and "discover-facts" not in operations and discovery == world["initial_discovery"])
            require("focused_durable_interview", bool(asks))
            reads = [index for index, event in enumerate(events) if event["operation"] == "read" and event["result"].get("path") == paths["discovery"]]
            require("discovery_read_before_interview", bool(reads and asks) and min(reads) < operations.index("ask"))
            if case.name in ("setup-confirmation", "setup-missing-owners"):
                require("one_focused_storage_question", sum(paths["run_root"] in event["args"]["text"] for event in asks) == 1)
            if case.name in ("setup-confirmation", "setup-reuse-discovery", "setup-ticket-source"):
                require("awaits_confirmation_without_saving", config is None and not writes and "check-config" not in operations and len(asks) == 1 and asks[0]["result"].get("awaiting_user"))
            else:
                require("stored_config_checked", checked("check-config") and checked("resolve-config"))
                config_writes = [index for index, event in enumerate(events) if event["operation"] == "write" and event["result"].get("path") == paths["config"]]
                answers = [index for index, event in enumerate(events) if event["operation"] == "ask" and event["result"].get("user")]
                require("confirmation_before_save", bool(config_writes and answers) and min(config_writes) > max(answers))
                profile = (config or {}).get("harnesses", {}).get(world["harness_name"], {})
                require("actual_confirmation_provenance", bool(answers) and any(
                    events[index]["result"].get("source", "__missing__") in profile.get("confirmed_from", "") for index in answers))
                if case.name == "setup-second-harness":
                    require("preserves_first_and_project", config is not None and config["project"] == world["initial_config"]["project"] and config["harnesses"].get("Other product") == world["initial_config"]["harnesses"]["Other product"])
                else:
                    facts = (discovery or {}).get("harnesses", {}).get(world["harness_name"], {})
                    require("classifies_missing_owners", bool(facts) and all(facts["owners"][key] is None for key in ("launch", "inspect", "message", "collect", "close", "resume")))
                    require("classifies_blocking_helpers", bool(facts) and facts["helpers"]["blocks_caller"] is True and facts["helpers"]["follow_up"] is False and bool(facts["limitations"]))
            if case.name == "setup-ticket-source":
                require("repository_inspected", all(read_file(paths[key]) for key in ("repository_guidance", "repository_tickets")))
                require("ticket_reference_loaded", any(event["operation"] == "read-skill" and event["args"].get("name") == SETUP.name and event["args"].get("resource") == "references/ticket-template.md" for event in events))
                require("source_question", len(asks) == 1 and "markdown" in asks[0]["args"]["text"].lower() and "id" in asks[0]["args"]["text"].lower())
    else:
        require("saved_configuration_read", checked("resolve-config") or any(
            event["operation"] == "read" and event["result"].get("path") == paths["config"] for event in events))
        require("current_profile_resolved", checked("resolve-config") and all(event["args"].get("harness") == world["harness_name"] for event in events if event["operation"] == "resolve-config"))
        resolutions = [index for index, event in enumerate(events) if event["operation"] == "resolve-config" and event["result"].get("exit_code") == 0]
        require("resolve_before_run_mutation", bool(resolutions) and all(index > min(resolutions) for index, event in enumerate(events)
            if event["operation"] in ("write", "message", "close", "invoke-helpers", "launch")))
        require("durable_choices_unchanged", config == world["initial_config"] and discovery == world["initial_discovery"] and not any(event["result"].get("path") in (paths["config"], paths["discovery"]) for event in writes))
        require("no_durable_interview_or_rediscovery", "discover-facts" not in operations and not any(durable_question(event) for event in asks)
            and not any(event["operation"] == "read-skill" and event["args"].get("name") in (DISCOVERY.name, SETUP.name) and event["args"].get("resource", "SKILL.md") == "SKILL.md" for event in events))
        status = load(Path(paths["status"])) if Path(paths["status"]).exists() else {"tasks": [], "evidence": [], "decisions": []}
        require("actual_run_update", any(event["result"].get("path") == paths["status"] for event in writes))
        if case.name == "runner-current-harness":
            require("compact_run_only_confirmation", len(asks) == 1 and bool(asks[0]["result"].get("user")) and bool(asks[0]["result"].get("source")))
            require("run_choices_recorded", bool(asks) and status.get("harness_name") == world["harness_name"] and status.get("config_path") == paths["config"]
                    and asks[0]["result"].get("source", "__missing__") in status.get("run_choices", {}).get("confirmed_from", "")
                    and status.get("monitoring", {}).get("interval_seconds") == 30
                    and {task["id"] for task in status["tasks"]} == {"UI-101", "API-102"})
            require("confirmation_before_run_write", bool(asks) and all(index > operations.index("ask") for index, event in enumerate(events) if event["operation"] == "write"))

        def unselected(presentation):
            return isinstance(presentation.get("hold"), str) and bool(presentation["hold"].strip()) and all(
                key in presentation and presentation[key] is None
                for key in ("destination", "local_path", "artifact_id", "artifact_url")
            )

        confirmed = case.name in ("runner-closure", "runner-decision-routing") or case.name.startswith("runner-ticket-")
        presentation_actions_safe = True
        for event in events:
            if event["operation"] == "ask" and paths["dashboard"] in event["result"].get("user", ""):
                confirmed = True
            elif event["operation"] == "render":
                presentation_actions_safe &= confirmed
            elif event["operation"] == "write":
                path = event["result"].get("path", "")
                if path == paths["status"] and not confirmed:
                    value = event["args"]["content"]
                    value = json.loads(value) if isinstance(value, str) else value
                    presentation_actions_safe &= unselected(value.get("presentation", {}))
                elif Path(path).suffix.lower() in (".html", ".htm"):
                    presentation_actions_safe &= confirmed
        require("presentation_actions_follow_confirmation", presentation_actions_safe)
        if case.name == "runner-authorization" and not confirmed:
            require("explicit_unselected_presentation_held", unselected(status.get("presentation", {}))
                    and "write" in operations and "render" not in operations and not Path(paths["dashboard"]).exists())
        else:
            require("dashboard_choice_confirmed", confirmed)
            require("confirmed_destination_recorded", status.get("presentation", {}).get("destination") == "local"
                    and status.get("presentation", {}).get("local_path") == paths["dashboard"])
            require("dashboard_rendered", any(event["operation"] == "render" and event["result"]["exit_code"] == 0 for event in events) and Path(paths["dashboard"]).is_file())
        tasks = {task["id"]: task for task in status["tasks"]}
        proofs = {proof["id"]: proof for proof in status["evidence"]}
        if case.name == "runner-proof-reuse":
            require("asks_dashboard_destination", "ask" in operations)
            require("reuses_owner_identity", tasks.get("UI-101", {}).get("owner", {}).get("id") == "ui-owner" and tasks.get("API-102", {}).get("owner", {}).get("id") == "api-owner")
            require("removes_hash_only_hold", "UI-101" in tasks and not tasks["UI-101"]["blocker"])
            require("scopes_readiness_and_delivery", all(task["milestones"]["pr_readiness"]["state"] == "verified" and all(task["milestones"][key]["state"] == "not_requested" for key in ("code_publication", "deployment", "business_acceptance")) for task in tasks.values()) and len(tasks) == 2)
            require("preserves_sufficient_and_failed_proof", all(key in proofs for key in ("E1", "E2", "E3", "E7")))
            require("unconfirmed_runtime_release_keeps_owner_open", tasks.get("API-102", {}).get("owner", {}).get("state") not in ("closed", "complete"))
        elif case.name == "runner-authorization":
            require("keeps_independent_work_runnable", "UI-101" in tasks and not tasks["UI-101"]["blocker"] and tasks["UI-101"]["owner"]["state"] not in ("waiting",))
            require("limits_runtime_hold", "ENV-106" in tasks and "API-102" in tasks["ENV-106"]["blocked_by"])
            require("unconfirmed_runtime_release_keeps_owner_open", tasks.get("API-102", {}).get("owner", {}).get("state") not in ("closed", "complete"))
            require("does_not_infer_publication_permission", "OPS-105" in tasks and tasks["OPS-105"]["milestones"]["code_publication"]["state"] == "blocked" and bool(status["decisions"]))
            require("asks_concrete_deployment_decision", any(event["operation"] == "ask" and event["result"].get("awaiting_user") and "deploy" in event["args"]["text"].lower() for event in events))
        elif case.name == "runner-closure":
            require("idle_ci_owner_kept_open", world["owners"]["A"]["state"] != "closed" and "CI-A" in tasks and tasks["CI-A"]["owner"]["state"] not in ("closed", "complete") and bool(tasks["CI-A"]["obligations"]))
            require("completed_owner_actually_closed", world["owners"]["B"]["state"] == "closed" and tasks.get("READY-B", {}).get("owner", {}).get("state") == "closed")
            retained = [index for index, event in enumerate(events) if event["operation"] == "retain" and event["args"]["owner"] == "B"]
            closed = [index for index, event in enumerate(events) if event["operation"] == "close" and event["args"]["owner"] == "B"]
            writes = [index for index, event in enumerate(events) if event["operation"] == "write" and event["result"]["path"] == paths["status"]]
            require("retains_and_updates_before_close", bool(retained and closed and writes) and retained[0] < closed[0] and any(retained[0] < index < closed[0] for index in writes) and (case / "state" / "retained" / "B.json").is_file())
            require("retains_proof_and_handoff", "E1" in proofs and bool(tasks.get("READY-B", {}).get("owner", {}).get("handoff")))
        elif case.name == "runner-native-resume":
            require("native_context_restored_before_mutation", "restore" in operations and all(index > operations.index("restore") for index, event in enumerate(events) if event["operation"] in ("write", "message", "close", "launch")))
            require("preserves_run_and_owner_ids", status.get("run_id") == "template-preview" and {key: task["owner"]["id"] for key, task in tasks.items()} == {task["id"]: task["owner"]["id"] for task in world["restored"]["tasks"]})
            require("retrieves_missing_receipt", any(event["operation"] in ("inspect", "retain") and event["args"]["owner"] == "api-owner" for event in events) and "E3" in proofs)
            require("preserves_failure_and_superseded_verdict", proofs.get("E5", {}).get("assessment") == "superseded" and proofs.get("E6", {}).get("result") == "failed" and proofs.get("E8", {}).get("result") == "interrupted")
            previously_closed = {task["id"] for task in world["restored"]["tasks"] if task["owner"]["state"] == "closed"}
            require("unsupported_closure_is_not_claimed", "close" not in operations and all(task["owner"]["state"] != "closed" or key in previously_closed for key, task in tasks.items()))
            resource = world["resources"]["authenticated-runtime"]
            if resource["holder"] == "api-owner" and not resource["released"]:
                api = tasks.get("API-102", {})
                require("unconfirmed_runtime_release_keeps_owner_open", api.get("owner", {}).get("state") in ("active", "waiting", "blocked", "unreachable"))
                require("known_release_obligation_recorded", bool(api.get("obligations")))
        elif case.name in ("runner-parallel-helpers", "runner-no-helpers"):
            require("keeps_existing_owner_identity", {key: task["owner"]["id"] for key, task in tasks.items()}
                    == {"REVIEW-103": "review-owner", "API-104": "list-owner"})
            require("required_review_and_regressions_stay_unready", len(tasks) == 2 and all(
                task["milestones"]["pr_readiness"]["state"] == "blocked"
                and task["owner"]["state"] in ("active", "waiting", "blocked", "unreachable") for task in tasks.values()))
            require("preserves_failure_and_superseded_proof", proofs.get("E4", {}).get("assessment") == "limited"
                    and proofs.get("E5", {}).get("assessment") == "superseded" and proofs.get("E6", {}).get("result") == "failed")
            require("scoped_follow_ups_delivered", all(world["owners"][owner].get("messages")
                    for owner in ("review-owner", "list-owner")))
            require("publication_and_deployment_not_inferred", all(
                task["milestones"][key]["state"] == "not_requested"
                for task in tasks.values() for key in ("code_publication", "deployment", "business_acceptance")))
            if case.name == "runner-parallel-helpers":
                batches = [event for event in events if event["operation"] == "invoke-helpers"]
                jobs = batches[0]["args"].get("jobs", []) if len(batches) == 1 else []
                require("automatically_batches_independent_jobs", len(jobs) == 2 and
                        {job.get("owner") for job in jobs} == {"review-owner", "list-owner"}
                        and all(job.get("brief") and job.get("follow_up") for job in jobs))
                collected = next((index for index, event in enumerate(events) if event["operation"] == "invoke-helpers"), len(events))
                require("uses_returned_reports_not_repeated_inspection", not any(
                    index > collected and event["operation"] == "inspect" for index, event in enumerate(events)))
            else:
                require("continues_directly_without_helpers", "invoke-helpers" not in operations and all(
                    any(event["operation"] == "inspect" and event["args"]["owner"] == owner for event in events)
                    for owner in ("review-owner", "list-owner")))
        elif case.name == "runner-decision-routing":
            decisions = {item["id"]: item for item in status["decisions"]}
            require("one_queue_without_blocking_questions", "ask" not in operations and set(decisions) == {"Q1", "Q2", "Q3"})
            expected_routes = {"Q1": ("fixture-coordinator", "OPS-105"), "Q2": ("list-owner", "API-104"),
                               "Q3": ("runtime-owner", "ENV-106")}
            require("routes_pending_decisions_by_scope", all(
                decisions.get(key, {}).get("discussion", {}).get("owner_id") == owner
                and decisions[key].get("task_ids") == [task]
                and decisions[key]["discussion"].get("href") == f"https://example.org/conversations/{'coordinator' if key == 'Q1' else owner}"
                and decisions[key].get("unblocks") and decisions[key].get("recommendation")
                for key, (owner, task) in expected_routes.items()))
            require("human_gate_remains_pending", decisions.get("Q3", {}).get("requires_human") is True
                    and world["owners"]["runtime-owner"].get("awaitingUserInput", {}).get("requiresHuman") is True
                    and tasks.get("ENV-106", {}).get("owner", {}).get("state") == "waiting")
            answers = [event for event in events if event["operation"] == "message" and event["args"]["owner"] == "api-owner"
                       and (event["result"].get("answered_dialog") == "Q4" or "Q4" in event["args"]["text"]
                            or "short labels" in event["args"]["text"].lower())]
            require("actual_answer_relayed_once", len(answers) == 1 and "short labels" in answers[0]["args"]["text"].lower()
                    and answers[0]["result"].get("answered_dialog") == "Q4")
            require("answer_retained_not_reasked", "Q4" not in decisions and any("Q4" in line and "short labels" in line.lower()
                    for line in status.get("history", [])))
            require("owner_activity_matches_observation", tasks.get("API-102", {}).get("owner", {}).get("state")
                    == world["owners"]["api-owner"]["state"])
            require("shared_action_approval_still_held", tasks.get("OPS-105", {}).get("milestones", {}).get("code_publication", {}).get("state") == "blocked")
            require("independent_ui_owner_keeps_moving", tasks.get("UI-101", {}).get("owner", {}).get("state") == "active"
                    and not tasks["UI-101"]["blocker"] and not world["owners"]["ui-owner"].get("messages")
                    and any(event["operation"] == "inspect" and event["args"]["owner"] == "ui-owner"
                            and event["result"].get("state") == "active" for event in events))
        elif case.name.startswith("runner-ticket-"):
            require("ticket_reference_loaded", any(event["operation"] == "read-skill" and
                    event["args"].get("resource") == "references/ticket-template.md" for event in events))
            for key in ("repository_guidance", "repository_tickets"):
                require(f"{key}_inspected", any(event["operation"] == "read" and
                        event["result"].get("path") == paths[key] for event in events))
            require("original_ticket_records_preserved", load(Path(paths["repository_tickets"])) == world["initial_tickets"])
            require("harness_configuration_unchanged", config == world["initial_config"])
            asks = [event for event in events if event["operation"] == "ask"]
            if case.name == "runner-ticket-lifetime":
                require("original_assignments_inspected", all(any(event["operation"] == "inspect" and
                        event["args"]["owner"] == owner for event in events) for owner in ("ui-owner", "list-owner")))
                require("lifetime_bindings_preserved", {key: task["owner"]["id"] for key, task in tasks.items()}
                        == {"UI-101": "ui-owner", "API-104": "list-owner"})
                require("closed_owner_not_repurposed", world["owners"]["ui-owner"]["state"] == "closed"
                        and tasks.get("UI-101", {}).get("owner", {}).get("state") == "closed"
                        and not world["owners"]["ui-owner"].get("messages"))
                require("same_ticket_follow_up_delivered", bool(world["owners"]["list-owner"].get("messages")))
                require("confirmed_policy_not_reasked", not asks)
            else:
                drafts = load(Path(paths["ticket_drafts"])) if Path(paths["ticket_drafts"]).is_file() else {}
                children = drafts.get("children", [])
                require("two_bounded_complete_child_drafts", drafts.get("parent_id") == "PLAN-109" and len(children) == 2
                        and all(all(child.get(key) for key in ("title", "outcome", "scope", "exclusions", "required_proof"))
                                and isinstance(child.get("dependencies"), list)
                                and isinstance(child.get("acceptance_criteria"), list)
                                and 1 <= len(child["acceptance_criteria"]) <= 5 for child in children))
                criteria = [criterion for child in children for criterion in child.get("acceptance_criteria", [])]
                require("all_original_criteria_preserved", sorted(criteria) == sorted(world["initial_tickets"]["PLAN-109"]["acceptance_criteria"]))
                pending_split = any("PLAN-109" in decision.get("text", "") and decision.get("id")
                        and decision.get("recommendation") and decision.get("unblocks")
                        and decision.get("discussion", {}).get("owner_id") for decision in status["decisions"])
                require("split_approval_awaited", (len(asks) == 1 and asks[0]["result"].get("awaiting_user")) or pending_split)
                require("unapproved_assignments_held", not tasks and not world["owners"])
    return {"case": case.name, "passed": all(checks.values()), "checks": checks,
            "trace": "trace.jsonl", "reply": "reply.md", "manual_review": "Required: inspect recorded questions, assignments, and final claims; machine checks do not grade their meaning."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparer = commands.add_parser("prepare")
    preparer.add_argument("root", type=Path)
    preparer.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    caller = commands.add_parser("call")
    caller.add_argument("case", type=Path)
    caller.add_argument("operation")
    caller.add_argument("args", nargs="?", default="{}")
    commands.add_parser("grade").add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.root.resolve(), args.cases)
        elif args.command == "call":
            case = args.case.resolve()
            try:
                call(case, args.operation, json.loads(args.args))
            except (ValueError, KeyError, TypeError, OSError) as error:
                with (case / "trace.jsonl").open("a") as stream:
                    stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                        "operation": args.operation, "args": args.args, "result": {"error": str(error)}}) + "\n")
                raise
        else:
            results = [grade(args.root.resolve() / name) for name in load(args.root / "suite.json")["cases"]]
            save(args.root / "mechanical-results.json", results)
            for result in results:
                print(f"{'PASS' if result['passed'] else 'FAIL'}: {result['case']}")
                for check, passed in result["checks"].items():
                    if not passed:
                        print(f"  FAIL: {check}")
            if not all(result["passed"] for result in results):
                raise SystemExit(1)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
