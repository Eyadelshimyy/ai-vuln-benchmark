"""Repository-owned CI contracts for deterministic GPU smoke provisioning."""

from __future__ import annotations

from pathlib import Path


_CI_WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml"
_APT_SOURCES = "/tmp/fast-mlsirm-ubuntu.sources"  # noqa: S108 - asserts CI contract only


def _gpu_install_script() -> str:
    """Return the software-Vulkan provisioning shell from the CI workflow."""
    text = _CI_WORKFLOW.read_text(encoding="utf-8")
    start_marker = "      - name: Install software Vulkan adapter\n        run: |\n"
    end_marker = "      - name: Prove Vulkan compute adapter availability\n"
    assert start_marker in text, "gpu-smoke Vulkan provisioning step is missing"
    _, remainder = text.split(start_marker, 1)
    assert end_marker in remainder, "gpu-smoke Vulkan proof step is missing"
    script, _ = remainder.split(end_marker, 1)
    return script


def _logical_shell(script: str) -> str:
    """Join explicit shell continuations so command contracts stay readable."""
    return script.replace("\\\n", " ")


def _apt_commands() -> tuple[tuple[str, ...], ...]:
    """Return tokenized apt commands from only the GPU provisioning step."""
    script = _logical_shell(_gpu_install_script())
    return tuple(
        tuple(line.split())
        for line in script.splitlines()
        if "sudo apt-get" in line
    )


def test_gpu_smoke_uses_isolated_canonical_ubuntu_sources() -> None:
    """GPU provisioning must bypass runner mirrorlists that can pin dead mirrors."""
    script = _gpu_install_script()
    assert f"cat > {_APT_SOURCES} <<EOF" in script
    assert "URIs: https://archive.ubuntu.com/ubuntu" in script
    assert "URIs: https://security.ubuntu.com/ubuntu" in script
    assert (
        "Suites: ${VERSION_CODENAME} ${VERSION_CODENAME}-updates "
        "${VERSION_CODENAME}-backports"
    ) in script
    assert "Suites: ${VERSION_CODENAME}-security" in script


def test_gpu_smoke_apt_network_work_has_hard_deadlines() -> None:
    """Each apt operation must retain its exact bounded network contract."""
    source_options = (
        "-o",
        f"Dir::Etc::sourcelist={_APT_SOURCES}",
        "-o",
        "Dir::Etc::sourceparts=-",
    )
    update = (
        "timeout",
        "120s",
        "sudo",
        "apt-get",
        *source_options,
        "-o",
        "Acquire::Retries=2",
        "-o",
        "Acquire::http::Timeout=10",
        "-o",
        "Acquire::https::Timeout=10",
        "-o",
        "DPkg::Lock::Timeout=30",
        "update",
    )
    install = (
        "timeout",
        "180s",
        "sudo",
        "apt-get",
        *source_options,
        "-o",
        "Acquire::Retries=2",
        "-o",
        "Acquire::http::Timeout=10",
        "-o",
        "Acquire::https::Timeout=10",
        "-o",
        "DPkg::Lock::Timeout=30",
        "install",
        "--yes",
        "mesa-vulkan-drivers",
        "vulkan-tools",
    )

    assert _apt_commands() == (update, install)


def test_gpu_smoke_apt_lock_wait_is_bounded() -> None:
    """Both package-manager operations keep the exact 30-second lock bound."""
    for command in _apt_commands():
        assert "DPkg::Lock::Timeout=30" in command
