import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/test-formulae.sh"


class FormulaCITest(unittest.TestCase):
    def run_ci(self, testing="", added="", deleted="", os_name="macOS",
               arch="ARM64", fail_command=""):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brew = root / "brew"
            brew.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['BREW_CALLS'], 'a') as calls:\n"
                "    calls.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "sys.exit(1 if sys.argv[1] == "
                "os.environ['FAIL_COMMAND'] else 0)\n"
            )
            brew.chmod(0o755)
            calls = root / "calls.jsonl"
            env = {
                **os.environ,
                "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                "BREW_CALLS": str(calls),
                "FAIL_COMMAND": fail_command,
                "TESTING_FORMULAE": testing,
                "ADDED_FORMULAE": added,
                "DELETED_FORMULAE": deleted,
                "RUNNER_OS": os_name,
                "RUNNER_ARCH": arch,
            }
            result = subprocess.run(
                ["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
            )
            commands = (
                [json.loads(line) for line in calls.read_text().splitlines()]
                if calls.exists() else []
            )
            return result, commands

    def test_mixed_changes_preserve_stable_and_deleted_formulae(self):
        result, commands = self.run_ci(
            testing="omlx,zpiral,sl", added="omlx,sl", deleted="magus",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(commands, [
            ["test-bot", "--only-formulae", "--testing-formulae=zpiral,sl",
             "--added-formulae=sl", "--deleted-formulae=magus"],
            ["install", "--HEAD", "--build-from-source", "--with-audio",
             "scaryrawr/formulae/omlx"],
            ["test", "omlx"],
        ])

    def test_head_only_changes(self):
        for name in ("omlx", "scaryrawr/formulae/omlx"):
            with self.subTest(name=name):
                result, commands = self.run_ci(testing=name, added=name)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual([command[0] for command in commands],
                                 ["install", "test"])
                self.assertIn("--with-audio", commands[0])

    def test_unsupported_runners_still_test_stable_formulae(self):
        for os_name, arch in (("Linux", "X64"), ("macOS", "X64")):
            with self.subTest(os_name=os_name, arch=arch):
                result, commands = self.run_ci(
                    testing="zpiral,omlx", os_name=os_name, arch=arch,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(commands, [
                    ["test-bot", "--only-formulae",
                     "--testing-formulae=zpiral", "--added-formulae=",
                     "--deleted-formulae="],
                ])

    def test_deleted_formulae_are_not_installed(self):
        result, commands = self.run_ci(deleted="omlx,magus")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(commands, [
            ["test-bot", "--only-formulae", "--testing-formulae=",
             "--added-formulae=", "--deleted-formulae=omlx,magus"],
        ])

    def test_no_formula_changes(self):
        result, commands = self.run_ci()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(commands, [])

    def test_failures_propagate(self):
        for command, expected_calls in (("test-bot", 1), ("install", 2),
                                        ("test", 3)):
            with self.subTest(command=command):
                result, commands = self.run_ci(
                    testing="zpiral,omlx", fail_command=command,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(commands), expected_calls)


if __name__ == "__main__":
    unittest.main()
