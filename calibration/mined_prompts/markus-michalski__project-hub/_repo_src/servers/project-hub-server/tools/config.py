"""Configuration loader for project-hub."""
from __future__ import annotations

import sys
from pathlib import Path

_config_cache: dict | None = None


def get_config() -> dict:
    """Load config from ~/.project-hub/config.yaml, falling back to defaults."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    defaults = {
        "docs_root": str(Path.home() / ".project-hub" / "projects"),
        "db_path": str(Path.home() / ".project-hub" / "project-hub.db"),
        "user": {
            "name": "",
            "email": "",
            "organization": "",
        },
        "default_language": "en",
        "communication": {
            "email_signature": "",
            "default_tone": "professional",
        },
    }

    config_path = Path.home() / ".project-hub" / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                user_config = yaml.safe_load(f) or {}
            _deep_merge(defaults, user_config)
        except Exception as e:
            print(f"[project-hub] WARNING: Failed to load config from {config_path}: {e}", file=sys.stderr)

    # Expand ~ in paths
    for key in ("docs_root", "db_path"):
        if key in defaults:
            defaults[key] = str(Path(str(defaults[key])).expanduser())

    _config_cache = defaults
    return _config_cache


def get_docs_root() -> Path:
    """Return the configured docs root directory."""
    return Path(get_config()["docs_root"])


def get_db_path_from_config() -> Path:
    """Return the configured database path."""
    return Path(get_config()["db_path"])


def get_knowledge_root() -> Path:
    """Return the local knowledge base root directory."""
    cfg = get_config()
    raw = cfg.get("knowledge_root", str(Path.home() / ".project-hub" / "knowledge"))
    return Path(raw).expanduser()


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
