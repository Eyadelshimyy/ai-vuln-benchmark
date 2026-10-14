"""The user starts at orchestration-run. Setup and discovery load when a caller needs them."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4]
SKILLS = ROOT / "skills" / "scoussens-skills"
README = ROOT / "README.md"
NAMES = ("orchestration-run", "orchestration-setup", "harness-discovery")


def description(name):
    text = (SKILLS / name / "SKILL.md").read_text()
    front = text.split("---", 2)[1]
    lines = front.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("description:"))
    body = []
    for line in lines[start + 1:]:
        if line.startswith("  "):
            body.append(line.strip())
        elif line.strip():
            break
    return " ".join(body)


def orchestration_section():
    readme = README.read_text()
    start = readme.index("## Orchestration\n")
    rest = readme[start + len("## Orchestration\n"):]
    end = rest.find("\n## ")
    return rest[:end]


def skill_for(prompt):
    """Explicit slash commands win. Any other orchestration request starts at the run."""
    descriptions = {name: description(name) for name in NAMES}
    for name in NAMES:
        if f"/{name}" in prompt:
            if f"/{name}" not in descriptions[name]:
                raise AssertionError(f"{name} no longer advertises /{name}")
            return name
    if "orchestrat" in prompt.lower():
        if "1. `/harness-discovery`" in orchestration_section():
            return "harness-discovery"
        return "orchestration-run"
    raise AssertionError(f"no skill claimed {prompt!r}")


class ChainContractTests(unittest.TestCase):
    def test_descriptions_name_who_calls_whom(self):
        descriptions = {name: description(name) for name in NAMES}
        for name, text in descriptions.items():
            self.assertLessEqual(len(text), 1024, name)
        run = descriptions["orchestration-run"]
        self.assertIn("orchestration-setup", run)
        self.assertIn("setting orchestration up", run)
        self.assertIn("do not run harness-discovery first", run)
        setup = descriptions["orchestration-setup"]
        self.assertIn("orchestration-run", setup)
        self.assertIn("/orchestration-setup", setup)
        self.assertIn("Do not start a run here", setup)
        discovery = descriptions["harness-discovery"]
        self.assertIn("orchestration-setup", discovery)
        self.assertIn("pstack-sync", discovery)
        self.assertIn("/harness-discovery", discovery)
        self.assertIn("Do not run it as the first step", discovery)

    def test_the_readme_starts_at_the_run(self):
        section = orchestration_section()
        self.assertIn("Start with `/orchestration-run`", section)
        self.assertNotIn("1. `/harness-discovery`", section)
        self.assertLess(section.index("`/orchestration-run`"), section.index("`/harness-discovery`"))

    def test_a_first_setup_starts_at_the_run(self):
        self.assertEqual(skill_for("set up orchestration for this repo"), "orchestration-run")

    def test_a_named_slash_stays_on_that_skill(self):
        self.assertEqual(skill_for("/harness-discovery"), "harness-discovery")
        self.assertEqual(skill_for("/orchestration-setup"), "orchestration-setup")

    def test_each_hand_off_is_stated_once_in_the_body(self):
        bodies = {name: (SKILLS / name / "SKILL.md").read_text().split("---", 2)[2] for name in NAMES}
        run = bodies["orchestration-run"]
        self.assertIn("When `resolve` exits 0", run)
        self.assertIn("do not read the", run)
        self.assertNotIn("Install `harness-discovery`, `orchestration-setup` and this skill together", run)
        for phrase in ("run orchestration-setup", "repair this harness profile with orchestration-setup", "run state.py migrate"):
            self.assertIn(phrase, run)
        self.assertIn("run harness-discovery", bodies["orchestration-setup"])
        self.assertEqual(sum(body.lower().count("load `orchestration-setup`") for body in bodies.values()), 1)
        self.assertEqual(sum(body.lower().count("load `harness-discovery`") for body in bodies.values()), 1)


if __name__ == "__main__":
    unittest.main()
