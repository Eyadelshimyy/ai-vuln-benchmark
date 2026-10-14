"""Typed errors for the adapter / plugin system."""

from __future__ import annotations


class AdapterError(Exception):
    """Base error for adapter registration and lookup failures."""


class AdapterNotFoundError(AdapterError):
    """Raised when an adapter id is not registered."""

    def __init__(self, kind: str, adapter_id: str, known: list[str]) -> None:
        self.kind = kind
        self.adapter_id = adapter_id
        self.known = known
        known_display = ", ".join(known) if known else "(none registered)"
        super().__init__(f"Unknown {kind} {adapter_id!r}. Known {kind}s: {known_display}.")


class DuplicateAdapterError(AdapterError):
    """Raised when registering an adapter id that already exists."""

    def __init__(self, kind: str, adapter_id: str, *, detail: str | None = None) -> None:
        self.kind = kind
        self.adapter_id = adapter_id
        self.detail = detail
        message = f"{kind} {adapter_id!r} is already registered."
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


class AdapterConflictError(AdapterError):
    """Raised when two adapters claim the same exclusive resource (e.g. file extension)."""


class UnsupportedCapabilityError(AdapterError):
    """Raised when an adapter is asked to perform work it does not implement."""

    def __init__(self, adapter_id: str, capability: str, *, detail: str | None = None) -> None:
        self.adapter_id = adapter_id
        self.capability = capability
        message = f"{adapter_id!r} does not support {capability}."
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


class AdapterNotImplementedError(AdapterError, NotImplementedError):
    """Raised for a reserved backend that is intentionally not implemented yet."""


class OutOfScopeError(AdapterError):
    """Raised when an operation targets a host that is not in authorized scope."""

    def __init__(self, target: str, *, detail: str | None = None) -> None:
        self.target = target
        message = f"Target {target!r} is not in scope."
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


class ActiveTestingNotPermittedError(AdapterError):
    """Raised when a target is in scope but active testing is not authorized."""

    def __init__(self, target: str, *, detail: str | None = None) -> None:
        self.target = target
        message = (
            f"Active testing is not permitted for {target!r}. "
            "Being in scope does not authorize active testing."
        )
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)
