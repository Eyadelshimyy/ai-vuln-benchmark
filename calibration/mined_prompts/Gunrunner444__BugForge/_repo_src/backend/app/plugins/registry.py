"""Generic adapter registry used by every plugin catalog.

Core orchestration looks up adapters by id (and optional aliases) instead of
branching on hard-coded language/provider names.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from app.plugins.errors import AdapterNotFoundError, DuplicateAdapterError


class AdapterRegistry[T]:
    """Register factories and create adapter instances by stable id.

    Factories may be zero-argument (stateless adapters) or accept keyword
    arguments such as ``settings=`` (AI providers). Callers pass through
    whatever the factory needs via :meth:`create`.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}
        self._aliases: dict[str, str] = {}
        self._descriptions: dict[str, str] = {}

    def register(
        self,
        adapter_id: str,
        factory: Callable[..., T],
        *,
        aliases: Iterable[str] = (),
        description: str = "",
        replace: bool = False,
    ) -> None:
        canonical = _normalize(adapter_id)
        if not canonical:
            raise ValueError(f"{self.kind} id must be a non-empty string")

        alias_keys = tuple(key for alias in aliases if (key := _normalize(alias)))
        self._validate_identity(canonical, alias_keys, replace=replace)

        snapshot_factories = dict(self._factories)
        snapshot_aliases = dict(self._aliases)
        snapshot_descriptions = dict(self._descriptions)
        try:
            self._commit_registration(
                canonical,
                factory,
                alias_keys,
                description=description,
                replace=replace,
            )
        except Exception:
            self._factories = snapshot_factories
            self._aliases = snapshot_aliases
            self._descriptions = snapshot_descriptions
            raise

    def _commit_registration(
        self,
        canonical: str,
        factory: Callable[..., T],
        alias_keys: tuple[str, ...],
        *,
        description: str,
        replace: bool,
    ) -> None:
        if replace:
            self._remove_aliases_for(canonical)
        self._factories[canonical] = factory
        self._descriptions[canonical] = description
        for alias_key in alias_keys:
            self._aliases[alias_key] = canonical
        self._aliases[canonical] = canonical

    def unregister(self, adapter_id: str) -> None:
        canonical = self._canonical_or_none(adapter_id)
        if canonical is None:
            return
        self._factories.pop(canonical, None)
        self._descriptions.pop(canonical, None)
        self._remove_aliases_for(canonical)

    def has(self, adapter_id: str) -> bool:
        return self._canonical_or_none(adapter_id) is not None

    def resolve_id(self, adapter_id: str) -> str:
        canonical = self._canonical_or_none(adapter_id)
        if canonical is None:
            raise AdapterNotFoundError(self.kind, adapter_id, self.available_ids())
        return canonical

    def available_ids(self) -> list[str]:
        return sorted(self._factories)

    def description(self, adapter_id: str) -> str:
        return self._descriptions.get(self.resolve_id(adapter_id), "")

    def create(self, adapter_id: str, *args: Any, **kwargs: Any) -> T:
        canonical = self.resolve_id(adapter_id)
        return self._factories[canonical](*args, **kwargs)

    def _validate_identity(
        self, canonical: str, alias_keys: tuple[str, ...], *, replace: bool
    ) -> None:
        existing_owner = self._owner_of(canonical)
        if existing_owner is not None and existing_owner != canonical:
            raise DuplicateAdapterError(
                self.kind,
                canonical,
                detail=f"{canonical!r} is already an alias of {existing_owner!r}",
            )
        if canonical in self._factories and not replace:
            raise DuplicateAdapterError(self.kind, canonical)

        for alias_key in alias_keys:
            if alias_key == canonical:
                continue
            if alias_key in self._factories:
                raise DuplicateAdapterError(
                    self.kind,
                    alias_key,
                    detail=f"alias {alias_key!r} conflicts with canonical id {alias_key!r}",
                )
            owner = self._aliases.get(alias_key)
            if owner is not None and owner != canonical:
                raise DuplicateAdapterError(
                    self.kind,
                    alias_key,
                    detail=f"alias {alias_key!r} is already registered for {owner!r}",
                )

    def _owner_of(self, adapter_id: str) -> str | None:
        key = _normalize(adapter_id)
        if key in self._factories:
            return key
        mapped = self._aliases.get(key)
        if mapped is not None and mapped in self._factories:
            return mapped
        return None

    def _canonical_or_none(self, adapter_id: str) -> str | None:
        return self._owner_of(adapter_id)

    def _remove_aliases_for(self, canonical: str) -> None:
        stale = [alias for alias, target in self._aliases.items() if target == canonical]
        for alias in stale:
            del self._aliases[alias]


def _normalize(adapter_id: str) -> str:
    return adapter_id.strip().lower()
