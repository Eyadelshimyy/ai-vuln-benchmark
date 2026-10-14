"""Run with python3 -B -m unittest discover -s <this directory> -v."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SKILL = Path(__file__).resolve().parent.parent
STATE = SKILL / "scripts" / "state.py"


def discovery_document():
    discovery = json.loads((SKILL / "assets" / "discovery.template.json").read_text())
    facts = discovery["harnesses"].pop("REPLACE_WITH_DISCOVERED_HARNESS_NAME")
    facts.update(context="Local terminal", observed_at="2026-10-09T12:00:00Z", sources=["Inspected tool inventory"],
                 owner_term="session", helper_term="subagent")
    facts["skills"]["load"] = "Load by skill name"
    other = deepcopy(facts)
    other.update(context="Orb", owner_term="thread")
    discovery["harnesses"] = {"Claude Code": facts, "Amp": other}
    return discovery


class StateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = self.root / "repository"
        self.discovery_path = self.repository / ".nthplusio-skills" / "discovery.json"
        self.discovery_path.parent.mkdir(parents=True)
        self.save(discovery_document())

    def save(self, discovery):
        self.discovery_path.write_text(json.dumps(discovery))

    def state(self, *arguments, cwd=None, root=True):
        command = [sys.executable, "-B", str(STATE), *arguments] + (["--root", str(self.repository)] if root else [])
        return subprocess.run(command, cwd=cwd or self.root, capture_output=True, text=True)

    def test_resolve_prints_the_entry_with_its_key_and_slug(self):
        for name in ("Claude Code", "claude-code", "CLAUDE CODE"):
            with self.subTest(name=name):
                result = self.state("resolve", "--harness", name)
                self.assertEqual(result.returncode, 0, result.stderr)
                entry = json.loads(result.stdout)
                self.assertEqual(entry["harness"], {"name": "Claude Code", "slug": "claude-code", "context": "Local terminal",
                                                    "owner_term": "session", "helper_term": "subagent"})
                self.assertEqual(entry["skills"]["load"], "Load by skill name")
                self.assertNotIn("run_root", entry)

    def test_unknown_harness_is_not_guessed(self):
        result = self.state("resolve", "--harness", "Claude")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("run harness-discovery in it", result.stderr)

    def test_keys_must_have_distinct_slugs(self):
        discovery = discovery_document()
        discovery["harnesses"]["claude code"] = deepcopy(discovery["harnesses"]["Claude Code"])
        self.save(discovery)
        result = self.state("check", "discovery")
        self.assertEqual(result.returncode, 1)
        self.assertIn("share the slug claude-code", result.stderr)
        discovery["harnesses"] = {"--": discovery["harnesses"]["Amp"]}
        self.save(discovery)
        self.assertIn("needs letters or digits", self.state("check", "discovery").stderr)

    def test_root_defaults_to_the_working_directorys_git_repository(self):
        subprocess.run(["git", "init", "-q", str(self.repository)], check=True)
        inside = self.repository / "nested" / "folder"
        inside.mkdir(parents=True)
        result = self.state("check", cwd=inside, root=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, f"PASS: discovery {self.discovery_path.resolve()}\n")
        outside = self.root / "not-a-repository"
        outside.mkdir()
        result = self.state("check", cwd=outside, root=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("pass --root", result.stderr)

    def test_optional_fields_are_checked_when_present(self):
        discovery = discovery_document()
        entry = discovery["harnesses"]["Amp"]
        entry["instructions"] = {"user": "~/.config/amp/AGENTS.md", "project": "AGENTS.md in each directory"}
        entry["ask_user"] = "A choice tool with two to five options, only when the user asked for questions"
        entry["models"] = {"select": "Start a thread with an agent mode", "list": "List the agent modes"}
        entry["skills"].update(locations=".agents/skills/ in the project", reload="The reload-skills tool")
        self.save(discovery)
        result = self.state("resolve", "--harness", "amp")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["models"]["list"], "List the agent modes")
        for section, field in (("instructions", "user"), ("models", "select"), ("skills", "reload"), ("helpers", "readonly")):
            with self.subTest(field=f"{section}.{field}"):
                broken = deepcopy(discovery)
                broken["harnesses"]["Amp"][section][field] = ""
                self.save(broken)
                result = self.state("check", "discovery")
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"Amp.{section}.{field} must be nonempty text", result.stderr)
        for field, value, message in (("ask_user", 3, "Amp.ask_user must be nonempty text"),
                                      ("models", "Task tool", "Amp.models must be an object")):
            with self.subTest(field=field):
                broken = deepcopy(discovery)
                broken["harnesses"]["Amp"][field] = value
                self.save(broken)
                self.assertIn(message, self.state("check", "discovery").stderr)

    def test_entries_saved_before_the_optional_fields_still_validate(self):
        discovery = discovery_document()
        for facts in discovery["harnesses"].values():
            for field in ("instructions", "ask_user", "models"):
                del facts[field]
            del facts["helpers"]["readonly"], facts["skills"]["locations"], facts["skills"]["reload"]
        self.save(discovery)
        result = self.state("check", "discovery")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_versions_this_tool_cannot_read_are_refused(self):
        for found, message in ((0, "is older than 1"), (2, "update these skills"), ("1", "update these skills")):
            with self.subTest(found=found):
                discovery = discovery_document()
                discovery["schema_version"] = found
                self.save(discovery)
                result = self.state("check", "discovery")
                self.assertEqual(result.returncode, 1)
                self.assertIn(message, result.stderr)

    def test_a_failure_names_exactly_one_skill(self):
        phrases = ("run harness-discovery", "run orchestration-setup", "run state.py migrate")

        def expect(arguments, phrase):
            result = self.state(*arguments)
            self.assertEqual(result.returncode, 1, result.stdout)
            present = [hint for hint in phrases if hint in result.stderr]
            self.assertEqual(present, [phrase], result.stderr)
            self.assertEqual(result.stderr.count(phrase), 1, result.stderr)

        discovery = discovery_document()
        del discovery["repository"]
        self.save(discovery)
        expect(("check", "discovery"), "run harness-discovery")

        self.discovery_path.write_text("{not JSON")
        expect(("resolve", "--harness", "Amp"), "run harness-discovery")

        discovery = discovery_document()
        discovery["schema_version"] = 0
        self.save(discovery)
        expect(("check", "discovery"), "run state.py migrate")

        discovery = discovery_document()
        discovery["schema_version"] = 2
        self.save(discovery)
        expect(("check", "discovery"), "run harness-discovery")

        discovery = discovery_document()
        discovery["harnesses"] = []
        self.save(discovery)
        expect(("check", "discovery"), "run harness-discovery")

        self.save(discovery_document())
        expect(("resolve", "--harness", "Claude"), "run harness-discovery")

        config_path = self.discovery_path.parent / "orchestration.json"
        config = {
            "schema_version": 3,
            "project": {"ticket_policy": {
                "source": "Linear", "location": "ENG", "definition": "One outcome",
                "template": "docs/tickets.md", "id_convention": "ENG-N", "criteria_limit": 3,
                "confirmed_from": "User confirmed",
            }, "shared_skills": []},
            "harnesses": {"Amp": {
                "run_root": "runs/amp", "shared_skills": [], "confirmed_from": "User confirmed",
                "defaults": {"finish_line": "review-ready PRs", "destination": None, "interval_seconds": 60},
            }},
        }
        config["harnesses"] = []
        config_path.write_text(json.dumps(config))
        expect(("resolve", "--harness", "Amp", "--orchestration"), "run orchestration-setup")

        config = {
            "schema_version": 3,
            "project": {"ticket_policy": {
                "source": "Linear", "location": "ENG", "definition": "One outcome",
                "template": "docs/tickets.md", "id_convention": "ENG-N", "criteria_limit": 0,
                "confirmed_from": "User confirmed",
            }, "shared_skills": []},
            "harnesses": {"Amp": {
                "run_root": "runs/amp", "shared_skills": [], "confirmed_from": "User confirmed",
                "defaults": {"finish_line": "review-ready PRs", "destination": None, "interval_seconds": 60},
            }},
        }
        config_path.write_text(json.dumps(config))
        expect(("check", "orchestration"), "run orchestration-setup")

        config["schema_version"] = 1
        config["project"]["ticket_policy"]["criteria_limit"] = 3
        config_path.write_text(json.dumps(config))
        expect(("check", "orchestration"), "run state.py migrate")

        config_path.write_text("{not JSON")
        expect(("resolve", "--harness", "Amp", "--orchestration"), "run orchestration-setup")

        config["harnesses"]["Amp"]["run_root"] = "C:/runs/amp"
        config["schema_version"] = 3
        config_path.write_text(json.dumps(config))
        repair = self.state("resolve", "--harness", "Amp", "--orchestration")
        self.assertEqual(repair.returncode, 1)
        self.assertIn("repair this harness profile with orchestration-setup", repair.stderr)
        self.assertNotIn("run orchestration-setup", repair.stderr)
        self.assertNotIn("run harness-discovery", repair.stderr)

        config["harnesses"]["Amp"]["run_root"] = "runs/amp"
        config_path.write_text(json.dumps(config))
        result = self.state("resolve", "--harness", "Amp", "--orchestration")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn("run_root", json.loads(result.stdout))


FIXTURES = Path(__file__).resolve().parent / "fixtures"
LEGACY = FIXTURES / "orchestration-v2"


class MigrationTests(unittest.TestCase):
    """Each fixture is a legacy folder exactly as an earlier version saved it."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = self.root / "repository"
        self.legacy = self.repository / ".orchestration"
        self.state_dir = self.repository / ".nthplusio-skills"
        shutil.copytree(LEGACY, self.legacy)
        self.run_file = self.legacy / "runs" / "amp" / "run-1" / "status.json"
        self.run_file.parent.mkdir(parents=True)
        self.run_file.write_text("{}")

    def state(self, *arguments):
        return subprocess.run([sys.executable, "-B", str(STATE), *arguments, "--root", str(self.repository)],
                              cwd=self.root, capture_output=True, text=True)

    def migrate(self, *arguments):
        result = self.state("migrate", *arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_every_fixture_migrates_to_the_current_schema(self):
        for fixture in sorted(path for path in FIXTURES.iterdir() if path.is_dir()):
            with self.subTest(fixture=fixture.name):
                repository = self.root / fixture.name
                shutil.copytree(fixture, repository / ("." + fixture.name.rsplit("-v", 1)[0]))
                migrated = subprocess.run([sys.executable, "-B", str(STATE), "migrate", "--root", str(repository)],
                                          capture_output=True, text=True)
                self.assertEqual(migrated.returncode, 0, migrated.stderr)
                checked = subprocess.run([sys.executable, "-B", str(STATE), "check", "--root", str(repository)],
                                         capture_output=True, text=True)
                self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_migrate_moves_upgrades_and_keeps_run_files_reachable(self):
        summary = self.migrate()
        self.assertTrue(summary["changed"])
        self.assertEqual(sorted(Path(target).name for _, target in summary["moved"]), ["discovery.json", "orchestration.json"])
        self.assertEqual(summary["upgraded"], [{"file": str(self.legacy / "config.json"), "from": 2, "to": 3}])
        self.assertEqual(summary["rebased_run_roots"], [{"harness": "Amp", "from": "runs/amp", "to": "../.orchestration/runs/amp"}])
        self.assertEqual((self.state_dir / "discovery.json").read_bytes(), (LEGACY / "discovery.json").read_bytes())
        config = json.loads((self.state_dir / "orchestration.json").read_text())
        self.assertEqual(config["schema_version"], 3)
        self.assertNotIn("discovery_file", config)
        self.assertFalse((self.legacy / "config.json").exists() or (self.legacy / "discovery.json").exists())
        self.assertTrue(self.run_file.exists())
        resolved = self.state("resolve", "--harness", "amp", "--orchestration")
        self.assertEqual(resolved.returncode, 0, resolved.stderr)
        self.assertEqual(json.loads(resolved.stdout)["run_root"], str((self.legacy / "runs" / "amp").resolve()))

    def test_dry_run_reports_the_changes_without_writing(self):
        summary = self.migrate("--dry-run")
        self.assertTrue(summary["dry_run"] and summary["changed"])
        self.assertEqual(len(summary["moved"]), 2)
        self.assertFalse(self.state_dir.exists())
        self.assertEqual((self.legacy / "config.json").read_bytes(), (LEGACY / "config.json").read_bytes())

    def test_a_second_run_changes_nothing(self):
        self.migrate()
        before = {path: path.read_bytes() for path in self.state_dir.iterdir()}
        self.assertFalse(self.migrate()["changed"])
        self.assertEqual({path: path.read_bytes() for path in self.state_dir.iterdir()}, before)

    def test_conflicting_current_state_is_refused_and_left_alone(self):
        self.state_dir.mkdir()
        current = json.loads((LEGACY / "discovery.json").read_text())
        current["harnesses"]["Amp"]["context"] = "Local terminal"
        (self.state_dir / "discovery.json").write_text(json.dumps(current))
        result = self.state("migrate")
        self.assertEqual(result.returncode, 1)
        self.assertIn("both exist and differ", result.stderr)
        self.assertTrue((self.legacy / "config.json").exists())
        self.assertFalse((self.state_dir / "orchestration.json").exists())

    def test_an_identical_legacy_copy_is_removed(self):
        self.state_dir.mkdir()
        shutil.copy(LEGACY / "discovery.json", self.state_dir / "discovery.json")
        summary = self.migrate()
        self.assertEqual(summary["removed_duplicates"], [str(self.legacy / "discovery.json")])
        self.assertFalse((self.legacy / "discovery.json").exists())
        self.assertTrue((self.state_dir / "orchestration.json").exists())

    def test_an_outside_configuration_is_copied_and_left_untouched(self):
        shutil.rmtree(self.legacy)
        outside = self.root / "project-orchestration"
        shutil.copytree(LEGACY, outside)
        summary = self.migrate("--from", str(outside / "config.json"))
        self.assertEqual(len(summary["copied"]), 2)
        self.assertEqual((outside / "config.json").read_bytes(), (LEGACY / "config.json").read_bytes())
        config = json.loads((self.state_dir / "orchestration.json").read_text())
        self.assertEqual(config["harnesses"]["Amp"]["run_root"], str((outside / "runs" / "amp").resolve()))

    def test_a_version_1_configuration_is_left_for_the_setup_interview(self):
        (self.legacy / "config.json").write_text(json.dumps({"schema_version": 1, "harness": "Amp"}))
        summary = self.migrate()
        self.assertIn("orchestration-setup interview", summary["interview"][0])
        self.assertEqual(json.loads((self.legacy / "config.json").read_text())["schema_version"], 1)
        self.assertTrue((self.state_dir / "discovery.json").exists())

    def test_check_and_resolve_name_the_migrate_command_for_legacy_state(self):
        for arguments in (("check",), ("resolve", "--harness", "Amp", "--orchestration")):
            with self.subTest(command=arguments[0]):
                result = self.state(*arguments)
                self.assertEqual(result.returncode, 1)
                self.assertIn("legacy state at", result.stderr)
                self.assertIn("run state.py migrate", result.stderr)

    def test_tracked_legacy_files_move_with_git(self):
        def git(*arguments):
            return subprocess.run(["git", "-C", str(self.repository), "-c", "user.name=Test",
                                   "-c", "user.email=test@example.com", *arguments],
                                  check=True, capture_output=True, text=True).stdout
        git("init", "-q")
        git("add", ".orchestration/config.json", ".orchestration/discovery.json")
        git("commit", "-q", "-m", "Save orchestration state")
        self.migrate()
        status = git("status", "--porcelain").splitlines()
        self.assertIn("R  .orchestration/discovery.json -> .nthplusio-skills/discovery.json", status)
        self.assertIn("RM .orchestration/config.json -> .nthplusio-skills/orchestration.json", status)


if __name__ == "__main__":
    unittest.main()
