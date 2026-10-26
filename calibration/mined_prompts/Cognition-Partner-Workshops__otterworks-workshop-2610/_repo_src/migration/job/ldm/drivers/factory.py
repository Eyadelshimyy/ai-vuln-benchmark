"""Target driver construction from a picklable spec, so a Spark task can reopen the same target."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..config import LoadedManifest
from ..context import require_env
from ..errors import ConfigError
from .base import TargetDriver


@dataclass(frozen=True)
class TargetSpec:
    """Provider name plus the keyword arguments of its driver constructor. Contains credentials: never log it."""

    provider: str
    kwargs: dict[str, object] = field(default_factory=dict)

    def open(self) -> TargetDriver:
        if self.provider == "postgresql":
            from .postgresql import PostgresTarget

            return PostgresTarget(**self.kwargs)  # type: ignore[arg-type]
        if self.provider == "azuresql":
            from .azuresql import AzureSqlTarget

            return AzureSqlTarget(**self.kwargs)  # type: ignore[arg-type]
        raise ConfigError(f"target.provider {self.provider!r} has no driver in this build")


def target_spec(loaded: LoadedManifest, env: Mapping[str, str]) -> TargetSpec:
    tgt = loaded.manifest.target
    ce = tgt.connection_env
    host_kind = env.get("LDM_HOST", "local")
    if tgt.provider == "postgresql":
        assert ce.host and ce.port
        require_env(dict(env), [ce.host, ce.port, ce.database, ce.user, ce.password], "target postgresql")
        try:
            port = int(env[ce.port])
        except ValueError as e:
            raise ConfigError(f"{ce.port}={env[ce.port]!r} is not a port number") from e
        return TargetSpec(
            "postgresql",
            {
                "host": env[ce.host],
                "port": port,
                "database": env[ce.database],
                "user": env[ce.user],
                "password": env[ce.password],
                "sslmode": (env.get(ce.sslmode) if ce.sslmode else None) or "prefer",
                "ldm_host": host_kind,
            },
        )
    if tgt.provider == "azuresql":
        assert ce.server and ce.auth_mode and ce.managed_identity_client_id
        auth = env.get(ce.auth_mode) or "sql"
        require_env(dict(env), [ce.server, ce.database], "target azuresql")
        if auth == "sql":
            require_env(dict(env), [ce.user, ce.password], f"target azuresql ({ce.auth_mode}=sql)")
        elif auth == "managed-identity":
            require_env(
                dict(env), [ce.managed_identity_client_id], f"target azuresql ({ce.auth_mode}=managed-identity)"
            )
        else:
            raise ConfigError(f"{ce.auth_mode}={auth!r}: expected 'sql' or 'managed-identity'")
        return TargetSpec(
            "azuresql",
            {
                "server": env[ce.server],
                "database": env[ce.database],
                "auth": auth,
                "user": env.get(ce.user),
                "password": env.get(ce.password),
                "client_id": env.get(ce.managed_identity_client_id),
                "host": host_kind,
            },
        )
    raise ConfigError(f"target.provider {tgt.provider!r} has no driver in this build")
