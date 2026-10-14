"""The Cloud Agent install script's object fetch."""

import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "scripts" / "cloud-agent-install.sh"


def _object_fetch_command(script: str) -> str:
    logical: list[str] = []
    pending = ""
    for line in script.splitlines():
        piece = line.strip()
        if piece.endswith("\\"):
            pending += piece[:-1].strip() + " "
            continue
        pending += piece
        logical.append(pending.strip())
        pending = ""
    for line in logical:
        if "import_scene_assets.py" not in line or "--objects" not in line:
            continue
        command, _, _hook = line.partition("||")
        return command.strip()
    raise AssertionError("the install script does not fetch object meshes")


def test_the_install_script_fetches_objects_with_the_venv() -> None:
    """AC-ASSET-04: the object fetch uses the project virtualenv interpreter."""
    command = _object_fetch_command(INSTALL.read_text())
    interpreter = shlex.split(command)[0]
    assert interpreter == ".venv/bin/python"
    probe = subprocess.run(
        [interpreter, "-c", "import clave.assets.fetch"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr or probe.stdout
