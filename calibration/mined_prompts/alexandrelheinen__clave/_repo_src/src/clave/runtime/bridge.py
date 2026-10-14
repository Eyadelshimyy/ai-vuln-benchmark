"""The process boundary: sockets, the child runtime, and one round trip.

Everything here exists because the safety layer runs in another process. That
is the point rather than an accident: the crate that can override a model links
no part of one, which it could not claim if it shared an address space with
PyTorch.

The cost is this file and the latency it adds, and the latency is measured
rather than assumed.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clave.errors import ClaveError
from clave.runtime.proposal import STOP, Outcome, Proposal

BINARY = "clave-sitl"
"""The runtime peer, built from `crates/clave-sitl`."""

BIND_TIMEOUT_SECONDS = 20.0
"""How long to wait for the child to bind its socket before giving up."""

REPLY_TIMEOUT_SECONDS = 30.0
"""How long one proposal may take before the runtime is declared stuck."""

RECEIVE_CAPACITY = 65_536
"""Largest datagram read, matching the runtime's own buffer."""


class BridgeError(ClaveError):
    """The runtime could not be started, reached, or understood."""


def locate_binary(root: Path) -> Path:
    """Find the built runtime.

    Args:
        root: Repository root.

    Returns:
        Path to the executable, preferring a release build.

    Raises:
        BridgeError: If neither profile has been built, saying how to build it.
    """
    for profile in ("release", "debug"):
        candidate = root / "target" / profile / BINARY
        if candidate.is_file():
            return candidate
    raise BridgeError(
        f"{BINARY} is not built. Run: cargo build --release -p clave-sitl"
    )


@dataclass(frozen=True)
class BridgePaths:
    """Where the four files the runtime needs live.

    Attributes:
        config: The envelope and routing policy, as JSON.
        proposals: Where the runtime binds and receives proposals.
        decisions: Where published decisions arrive, in the contract's CBOR.
        outcomes: Where one outcome per proposal arrives, as JSON.
    """

    config: Path
    proposals: Path
    decisions: Path
    outcomes: Path

    @classmethod
    def under(cls, directory: Path) -> BridgePaths:
        """Place all four under one directory."""
        return cls(
            config=directory / "runtime.json",
            proposals=directory / "proposals.sock",
            decisions=directory / "decisions.sock",
            outcomes=directory / "outcomes.sock",
        )


class Bridge:
    """One running runtime, and the sockets to talk to it.

    The producer binds its own two sockets before the child starts, because
    connecting a datagram socket to a path nothing is bound to fails at once
    rather than waiting.
    """

    def __init__(
        self,
        binary: Path,
        paths: BridgePaths,
        config: dict[str, Any],
        on_decision: Callable[[bytes], None] | None = None,
    ):
        """Prepare a bridge without starting anything.

        Args:
            binary: The built runtime.
            paths: Where its files go.
            config: The envelope and routing policy the runtime reads.
            on_decision: Receives the bytes of every decision the runtime
                published, in arrival order. This is the seam a ROS 2 publisher
                hangs on, and it is handed the published bytes rather than the
                proposal that produced them so the two cannot drift.
        """
        self._binary = binary
        self._paths = paths
        self._config = config
        self._on_decision = on_decision
        self._process: subprocess.Popen[bytes] | None = None
        self._decisions: socket.socket | None = None
        self._outcomes: socket.socket | None = None
        self._sender: socket.socket | None = None
        self.decisions_received = 0
        """How many published decisions arrived on the decision socket."""

    def __enter__(self) -> Bridge:
        """Write the configuration, bind, start the child, and wait for it.

        Returns:
            This bridge, running.

        Raises:
            BridgeError: If the child never binds its socket.
        """
        self._paths.config.parent.mkdir(parents=True, exist_ok=True)
        self._paths.config.write_text(json.dumps(self._config, indent=2) + "\n")
        for path in (
            self._paths.decisions,
            self._paths.outcomes,
            self._paths.proposals,
        ):
            path.unlink(missing_ok=True)

        self._decisions = _bind(self._paths.decisions)
        self._outcomes = _bind(self._paths.outcomes)
        self._process = subprocess.Popen(
            [
                str(self._binary),
                "--config",
                str(self._paths.config),
                "--listen",
                str(self._paths.proposals),
                "--decisions",
                str(self._paths.decisions),
                "--outcomes",
                str(self._paths.outcomes),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._await_bind()
        self._sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sender.connect(str(self._paths.proposals))
        return self

    def __exit__(self, *_: object) -> None:
        """Stop the child and close everything."""
        self.close()

    def submit(self, proposal: Proposal) -> Outcome:
        """Send one proposal and wait for its outcome.

        Args:
            proposal: What inference proposed.

        Returns:
            What the runtime did with it.

        Raises:
            BridgeError: If the runtime is not running or does not answer.
        """
        if self._sender is None or self._outcomes is None:
            raise BridgeError("the bridge is not running")
        self._sender.send(proposal.encode())
        try:
            payload = self._outcomes.recv(RECEIVE_CAPACITY)
        except TimeoutError as error:
            raise BridgeError(
                f"the runtime did not answer within {REPLY_TIMEOUT_SECONDS} seconds"
            ) from error
        outcome = Outcome.decode(payload)
        self._drain_decisions()
        return outcome

    def counters(self) -> dict[str, int]:
        """Stop the child and read the counters it printed.

        Returns:
            The runtime's own counts, which are the authority: the producer
            counts what it sent, and the runtime counts what it decided.

        Raises:
            BridgeError: If the runtime printed nothing readable.
        """
        stdout = self.close()
        try:
            read: dict[str, int] = json.loads(stdout or b"{}")
        except json.JSONDecodeError as error:
            raise BridgeError(f"the runtime printed no counters: {error}") from error
        return read

    def close(self) -> bytes:
        """Send the stop signal, wait for the child, and close the sockets.

        Returns:
            Whatever the child printed on standard output.
        """
        stdout = b""
        if self._sender is not None and self._process is not None:
            if self._process.poll() is None:
                self._sender.send(STOP)
            stdout, _ = self._process.communicate(timeout=REPLY_TIMEOUT_SECONDS)
            self._process = None
        self._drain_decisions()
        for handle in (self._sender, self._decisions, self._outcomes):
            if handle is not None:
                handle.close()
        self._sender = self._decisions = self._outcomes = None
        for path in (
            self._paths.decisions,
            self._paths.outcomes,
            self._paths.proposals,
        ):
            path.unlink(missing_ok=True)
        return stdout

    def _await_bind(self) -> None:
        """Wait for the child to create its socket, or say why it did not."""
        deadline = time.monotonic() + BIND_TIMEOUT_SECONDS
        while not self._paths.proposals.exists():
            if self._process is not None and self._process.poll() is not None:
                _, stderr = self._process.communicate()
                raise BridgeError(
                    f"the runtime exited before binding: {stderr.decode().strip()}"
                )
            if time.monotonic() > deadline:
                raise BridgeError(
                    f"the runtime did not bind within {BIND_TIMEOUT_SECONDS} seconds"
                )
            time.sleep(0.005)

    def _drain_decisions(self) -> None:
        """Count every published decision waiting on the decision socket."""
        if self._decisions is None:
            return
        self._decisions.setblocking(False)
        try:
            while True:
                try:
                    payload = self._decisions.recv(RECEIVE_CAPACITY)
                except (BlockingIOError, InterruptedError):
                    return
                self.decisions_received += 1
                if self._on_decision is not None:
                    self._on_decision(payload)
        finally:
            self._decisions.settimeout(REPLY_TIMEOUT_SECONDS)


def _bind(path: Path) -> socket.socket:
    """Bind a datagram socket at a path, with a read timeout."""
    handle = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    handle.bind(str(path))
    handle.settimeout(REPLY_TIMEOUT_SECONDS)
    return handle
