"""Keep the Atheris extra within the wheel targets supported by the lock."""

from pathlib import Path
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement


def test_atheris_extra_platforms():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    requirements = [Requirement(value) for value in project["project"]["optional-dependencies"]["fuzz"]]
    atheris = next(requirement for requirement in requirements if requirement.name == "atheris")
    assert atheris.marker is not None

    environment = default_environment()
    environment.update(implementation_name="cpython", sys_platform="linux", platform_machine="x86_64")
    for version in ("3.12", "3.13", "3.14"):
        assert atheris.marker.evaluate({**environment, "python_version": version})

    for override in (
        {"python_version": "3.15"},
        {"implementation_name": "pypy"},
        {"sys_platform": "darwin"},
        {"sys_platform": "win32"},
        {"platform_machine": "aarch64"},
    ):
        assert not atheris.marker.evaluate({**environment, "python_version": "3.14", **override})
