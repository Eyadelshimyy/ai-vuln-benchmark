"""Multi-server SQL connection registry.

Each programme's ``EDMRef.serverName`` maps to a server entry here. Hundreds of
EDMs typically live on a handful of hosts (BERMUDA-SQL01, LONDON-SQL01, …);
each EDM is a separate *database* on those hosts.

Config sources (later wins nothing — file is primary, env default fills gaps):

* ``SQLSERVER_SERVERS_FILE`` → JSON map of serverName → connection settings
* ``SQLSERVER_DEFAULT_*`` env vars applied when a server entry omits credentials

Connection strings are built per (server, database) at query time. Connections
are retained per database until application shutdown.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerConnectionConfig:
    """One SQL Server host the workbench can open."""

    server_name: str
    host: str
    port: int = 1433
    user: str | None = None
    password: str | None = None
    driver: str = "ODBC Driver 18 for SQL Server"
    trust_server_certificate: bool = True
    encrypt: bool = True
    # Optional extra ODBC bits, e.g. "Authentication=ActiveDirectoryInteractive"
    extra_odbc: str | None = None

    def odbc_connect_string(self, database: str) -> str:
        """Build an ODBC connection string for ``database`` on this server."""
        parts = [
            f"DRIVER={{{self.driver}}}",
            f"SERVER={self.host},{self.port}",
            f"DATABASE={database}",
        ]
        if self.user:
            parts.append(f"UID={self.user}")
        if self.password:
            parts.append(f"PWD={self.password}")
        parts.append(f"Encrypt={'yes' if self.encrypt else 'no'}")
        parts.append(
            f"TrustServerCertificate={'yes' if self.trust_server_certificate else 'no'}"
        )
        if self.extra_odbc:
            parts.append(self.extra_odbc.rstrip(";"))
        return ";".join(parts) + ";"


@dataclass
class _PooledConnection:
    """Minimal per-server connection holder (one live conn, recreated on error)."""

    server_name: str
    database: str
    conn: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


class ConnectionRegistry:
    """Registry of named SQL hosts + lazy per-(server, db) connections."""

    def __init__(self) -> None:
        self._servers: dict[str, ServerConnectionConfig] = {}
        self._pools: dict[tuple[str, str], _PooledConnection] = {}
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}
        self._lock = threading.RLock()
        self._probe_cache: dict[str, tuple[bool, str, float]] = {}

    # ── registration ──────────────────────────────────────────────

    def register(self, cfg: ServerConnectionConfig) -> None:
        with self._lock:
            self._servers[cfg.server_name] = cfg
            # Drop pooled conns for this server so next use picks up new creds.
            stale = [k for k in self._pools if k[0] == cfg.server_name]
            for k in stale:
                self._close_pool_unlocked(k)

    def register_many(self, configs: list[ServerConnectionConfig]) -> None:
        for c in configs:
            self.register(c)

    def get(self, server_name: str) -> ServerConnectionConfig | None:
        return self._servers.get(server_name)

    def list_servers(self) -> list[ServerConnectionConfig]:
        return list(self._servers.values())

    def server_names(self) -> list[str]:
        return sorted(self._servers.keys())

    def __contains__(self, server_name: str) -> bool:
        return server_name in self._servers

    def __len__(self) -> int:
        return len(self._servers)

    # ── loading from disk / dict ───────────────────────────────────

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        *,
        default_user: str | None = None,
        default_password: str | None = None,
        default_driver: str = "ODBC Driver 18 for SQL Server",
    ) -> ConnectionRegistry:
        reg = cls()
        p = Path(path)
        if not p.exists():
            logger.warning("SQL servers file not found: %s", p)
            return reg
        raw = json.loads(p.read_text(encoding="utf-8"))
        reg.load_mapping(
            raw,
            default_user=default_user,
            default_password=default_password,
            default_driver=default_driver,
        )
        return reg

    def load_mapping(
        self,
        raw: dict[str, Any],
        *,
        default_user: str | None = None,
        default_password: str | None = None,
        default_driver: str = "ODBC Driver 18 for SQL Server",
    ) -> None:
        """Load ``{ serverName: { host, port, user, ... } }`` mapping."""
        for server_name, body in (raw or {}).items():
            if not isinstance(body, dict):
                continue
            host = body.get("host") or body.get("server") or body.get("hostname")
            if not host:
                logger.warning("Skipping server %s — no host", server_name)
                continue
            cfg = ServerConnectionConfig(
                server_name=str(server_name),
                host=str(host),
                port=int(body.get("port", 1433)),
                user=body.get("user") or body.get("uid") or default_user,
                password=body.get("password") or body.get("pwd") or default_password,
                driver=str(body.get("driver") or default_driver),
                trust_server_certificate=bool(
                    body.get("trustServerCertificate", body.get("trust_server_certificate", True))
                ),
                encrypt=bool(body.get("encrypt", True)),
                extra_odbc=body.get("extraOdbc") or body.get("extra_odbc"),
            )
            self.register(cfg)

    # ── connections ───────────────────────────────────────────────

    def connect(self, server_name: str, database: str) -> Any:
        """Return an open DB-API connection for ``server_name`` / ``database``.

        Requires ``pyodbc``. Reuses one connection per (server, database).
        """
        cfg = self.get(server_name)
        if cfg is None:
            raise KeyError(
                f"Server '{server_name}' is not in the connection registry. "
                f"Known: {self.server_names()}"
            )

        key = (server_name, database)
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        # Serialize only callers for this database. Cold connects to distinct
        # EDM databases remain concurrent.
        with key_lock:
            with self._lock:
                pooled = self._pools.get(key)
            if pooled is not None:
                with pooled.lock:
                    try:
                        cur = pooled.conn.cursor()
                        cur.execute("SELECT 1")
                        cur.close()
                        return pooled.conn
                    except Exception:
                        with self._lock:
                            self._close_pool_unlocked(key)

            try:
                import pyodbc  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover — optional dep
                raise RuntimeError(
                    "pyodbc is required for SQL Server access. "
                    "Install with: pip install 'peril-vista-backend[sql]'"
                ) from exc

            conn = pyodbc.connect(cfg.odbc_connect_string(database), timeout=15)
            conn.timeout = 60
            pooled = _PooledConnection(
                server_name=server_name, database=database, conn=conn
            )
            with self._lock:
                self._pools[key] = pooled
            return conn

    @contextmanager
    def connection(self, server_name: str, database: str):
        """Yield a pooled connection with exclusive DB-API access."""
        conn = self.connect(server_name, database)
        key = (server_name, database)
        with self._lock:
            pooled = self._pools[key]
        with pooled.lock:
            yield conn

    def close_all(self) -> None:
        with self._lock:
            for key in list(self._pools.keys()):
                self._close_pool_unlocked(key)
            self._key_locks.clear()

    def _close_pool_unlocked(self, key: tuple[str, str]) -> None:
        pooled = self._pools.pop(key, None)
        if pooled is None:
            return
        try:
            pooled.conn.close()
        except Exception:  # pragma: no cover
            pass

    def probe(self, server_name: str, *, database: str = "master") -> tuple[bool, str]:
        """Try opening a connection; return (ok, message)."""
        if server_name not in self._servers:
            return False, f"Server '{server_name}' not registered"
        try:
            with self.connection(server_name, database) as conn:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
            return True, "ok"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"


def load_registry_from_settings(
    *,
    servers_file: str | None,
    default_user: str | None = None,
    default_password: str | None = None,
    default_driver: str = "ODBC Driver 18 for SQL Server",
    inline_json: str | None = None,
) -> ConnectionRegistry:
    """Build a registry from settings knobs."""
    reg = ConnectionRegistry()
    if inline_json:
        try:
            reg.load_mapping(
                json.loads(inline_json),
                default_user=default_user,
                default_password=default_password,
                default_driver=default_driver,
            )
        except json.JSONDecodeError:
            logger.exception("SQLSERVER_SERVERS_JSON is not valid JSON")
    if servers_file:
        p = Path(servers_file)
        if not p.is_absolute():
            # Resolve relative to backend/ (same convention as mock_data_dir).
            backend_dir = Path(__file__).resolve().parents[2]
            p = (backend_dir / p).resolve()
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            reg.load_mapping(
                raw,
                default_user=default_user,
                default_password=default_password,
                default_driver=default_driver,
            )
        else:
            logger.warning("SQLSERVER_SERVERS_FILE not found: %s", p)
    return reg
