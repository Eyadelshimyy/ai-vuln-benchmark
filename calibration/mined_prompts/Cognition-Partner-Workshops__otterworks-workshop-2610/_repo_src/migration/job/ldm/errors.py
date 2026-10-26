"""Exit-code bearing exceptions (CONTRACTS.md §9.3)."""

from __future__ import annotations

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_RECONCILE = 2
EXIT_PURGE_GUARD = 3
EXIT_CONFIG = 4


class LdmError(Exception):
    exit_code = EXIT_UNEXPECTED

    def __init__(self, message: str, tables: dict[str, dict[str, int]] | None = None):
        super().__init__(message)
        self.tables = tables or {}


class ConfigError(LdmError):
    exit_code = EXIT_CONFIG


class ForeignRunError(ConfigError):
    """The run_id belongs to a run this invocation must not touch (different manifest); its row is left as is."""


class ReconcileError(LdmError):
    exit_code = EXIT_RECONCILE


class PurgeGuardError(LdmError):
    exit_code = EXIT_PURGE_GUARD


class SourceError(LdmError):
    """A Db2 / Oracle (or other source) error. The message always starts with 'SQLCODE=<n> SQLSTATE=<s>: '.

    `transient` is the driver's own verdict on whether a rolled-back batch may be retried; None leaves the decision
    to the SQLSTATE table in stages.purge (drivers whose client library does not report a SQLSTATE set it).
    """

    def __init__(self, sqlcode: int | None, sqlstate: str | None, text: str, transient: bool | None = None):
        self.sqlcode = sqlcode
        self.sqlstate = sqlstate
        self.text = text
        self.transient = transient
        super().__init__(f"SQLCODE={sqlcode if sqlcode is not None else '?'} SQLSTATE={sqlstate or '?????'}: {text}")


class TargetError(LdmError):
    """An Azure SQL error with ODBC SQLSTATE and SQL Server native error number."""

    def __init__(self, sqlstate: str | None, native_error: int | None, text: str):
        self.sqlstate = sqlstate
        self.native_error = native_error
        self.text = text
        super().__init__(f"SQLSTATE={sqlstate or '?????'} native={native_error}: {text}")
