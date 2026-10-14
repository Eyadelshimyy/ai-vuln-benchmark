"""Central configuration loader for ComfyUI-LK-Model_Auto-Matching.

This module is the single source of truth for all configuration. It merges,
in increasing priority, the following sources into one logical config:

    _DEFAULT_CONFIG  <  core/data/matcher_config.json (strategy)
                    <  core/data/models_data.json   (tokenizer dictionary)
                    <  config.json (root, user secrets)
                    <  environment variables (LK_* whitelist)

Key spaces of each file are intentionally disjoint (db/matching <-> searcher/
features <-> tokenizer <-> secrets) so a deep merge never overwrites another
section. Loading is *soft-fail*: any parse/file error returns the default and
logs a warning. Values out of range are reverted to the default with a warning.
The node must never crash at load time.
"""
import copy
import json
import os
import logging

logger = logging.getLogger("LK.Config")

# ---------------------------------------------------------------------------
# Default configuration (covers every configurable key).
# Values == current effective values; DO NOT change without updating the
# matching code paths that depend on them.
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = {
    "db": {
        "enabled": True,
        "exact_score": 1.0,
        "basename_score": 0.99,
        "semantic_min_score": 0.35,
        "fuzzy_score_cutoff": 80,
        "civitai_score": 0.99,
    },
    "matching": {
        "use_db_fallback": True,
        "core_coverage_min": 0.6,
        "core_min_tokens": 2,
        "use_exact_match": True,
        "use_fuzzy_match": True,
        "use_variant_match": True,
        "use_legacy_match": True,
        "fuzzy_score_cutoff": 60.0,
        "variant_score_cutoff": 0.9,
        "legacy_score_cutoff": 75,
    },
    "searcher": {
        "network": {"timeout": 20, "impersonate": "chrome124"},
        "cache_ttl": 300,
        "early_stop_score": 0.7,
        "high_confidence_score": 0.9,
        "db_fallback_score": 0.85,
        "search_term_limit": 5,
        "source_preference": {
            "HuggingFace": 1.0,
            "ModelScope": 1.0,
            "CNB": 0.95,
            "Civitai": 0.9,
            "Liblib": 0.9,
        },
        "api": {
            "civitai": {
                "url": "https://civitai.com/api/v1/model-versions/by-hash",
                "limit": 20,
            },
            "huggingface": {
                "limit": 8,
                "hash_scores": {"exact": 0.98, "dir": 0.95, "file": 0.92},
            },
            "liblib": {"limit": 8},
        },
        "providers": {
            "modelscope": {"timeout": 25},
            "huggingface": {"scan_dirs": []},
        },
    },
    "features": {
        "chinese_tokenization": True,
        "version_aware": False,
    },
    # 镜像端点：国内优先。空字符串 = 使用官方域名（行为不变）。
    # CN 用户可将 hf_endpoint 设为 https://hf-mirror.com，
    # modelscope_endpoint 设为 https://modelscope.ai（国际站）。
    "mirrors": {
        "hf_endpoint": "",
        "modelscope_endpoint": "",
    },
    "tokenizer": {},   # populated from core/data/models_data.json at load time
    "secrets": {
        "civitai_api_key": "",
        "huggingface_token": "",
    },
}

# Environment-variable -> config-path whitelist (prevents arbitrary injection).
_ENV_MAP = {
    "LK_CIVITAI_API_KEY": ("secrets", "civitai_api_key"),
    "LK_HUGGINGFACE_TOKEN": ("secrets", "huggingface_token"),
    "LK_SEARCHER_TIMEOUT": ("searcher", "network", "timeout"),
    "LK_SEARCHER_EARLY_STOP": ("searcher", "early_stop_score"),
    "LK_SEARCHER_CACHE_TTL": ("searcher", "cache_ttl"),
    "LK_SEARCH_TERM_LIMIT": ("searcher", "search_term_limit"),
    "LK_HF_ENDPOINT": ("mirrors", "hf_endpoint"),
    "LK_MODELSCOPE_ENDPOINT": ("mirrors", "modelscope_endpoint"),
}

# File locations.
_ROOT = os.path.dirname(os.path.dirname(__file__))
_DATA = os.path.join(os.path.dirname(__file__), "data")
_CONFIG_PATH_USER = os.path.join(_ROOT, "envs", "config.json")           # secrets (gitignored)
_CONFIG_PATH_STRATEGY = os.path.join(_DATA, "matcher_config.json")  # strategy
_CONFIG_PATH_TOKENIZER = os.path.join(_DATA, "models_data.json")    # tokenizer dict

# Module-level singleton cache for the merged configuration.
_CONFIG_CACHE = None


# ---------------------------------------------------------------------------
# Merging / loading helpers
# ---------------------------------------------------------------------------
def _deep_merge(base, override):
    """Recursively deep-merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_json(path):
    """Load a JSON file. Soft-fail: FileNotFoundError -> {}, other -> warn + {}."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # malformed JSON etc. -> keep running with defaults
        logger.warning("[Config] 解析失败 %s: %s，使用默认值", path, exc)
        return {}


def _default_at(path):
    """Return the default value located at ``path`` (tuple of keys)."""
    node = _DEFAULT_CONFIG
    for part in path:
        node = node.get(part, {})
    return node


def _coerce(value, default):
    """Coerce an env-string ``value`` to the type of ``default``."""
    if isinstance(default, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "on", "y", "t")
    if isinstance(default, int):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    return value


def _apply_env(cfg):
    """Overlay whitelisted environment variables onto the merged config."""
    for env_key, path in _ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        default = _default_at(path)
        coerced = _coerce(raw, default)
        node = cfg
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = coerced


# ---------------------------------------------------------------------------
# Validation (soft-fail: revert to default + collect warnings)
# ---------------------------------------------------------------------------
def _revert_and_warn(node, key, default_section, errors):
    node[key] = copy.deepcopy(default_section.get(key))
    errors.append("{0} 值无效({1!r})，已回退默认 {2!r}".format(
        key, node[key], default_section.get(key)))


def _check_score(node, key, default_section, errors):
    val = node.get(key)
    if not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
        _revert_and_warn(node, key, default_section, errors)


def _check_number(node, key, default_section, errors):
    val = node.get(key)
    if not isinstance(val, (int, float)):
        _revert_and_warn(node, key, default_section, errors)


def _check_bool(node, key, default_section, errors):
    val = node.get(key)
    if not isinstance(val, bool):
        _revert_and_warn(node, key, default_section, errors)


def validate_config(cfg):
    """Soft validation. Returns (ok, [warnings]).

    Out-of-range scores, non-numeric cutoffs and non-bool flags are reverted
    to the default value; the warning is collected but the function never
    raises (node load safety is paramount).
    """
    errors = []
    try:
        db = cfg.setdefault("db", {})
        ddb = _DEFAULT_CONFIG["db"]
        for k in ("exact_score", "basename_score", "semantic_min_score", "civitai_score"):
            _check_score(db, k, ddb, errors)
        _check_number(db, "fuzzy_score_cutoff", ddb, errors)
        _check_bool(db, "enabled", ddb, errors)

        m = cfg.setdefault("matching", {})
        dm = _DEFAULT_CONFIG["matching"]
        _check_number(m, "fuzzy_score_cutoff", dm, errors)
        _check_number(m, "variant_score_cutoff", dm, errors)
        _check_number(m, "legacy_score_cutoff", dm, errors)
        for k in ("use_db_fallback", "use_exact_match", "use_fuzzy_match",
                  "use_variant_match", "use_legacy_match"):
            _check_bool(m, k, dm, errors)

        s = cfg.setdefault("searcher", {})
        ds = _DEFAULT_CONFIG["searcher"]
        _check_number(s, "cache_ttl", ds, errors)
        _check_number(s, "search_term_limit", ds, errors)
        for k in ("early_stop_score", "high_confidence_score", "db_fallback_score"):
            _check_score(s, k, ds, errors)
        net = s.setdefault("network", {})
        _check_number(net, "timeout", ds["network"], errors)

        f = cfg.setdefault("features", {})
        df = _DEFAULT_CONFIG["features"]
        _check_bool(f, "chinese_tokenization", df, errors)
        _check_bool(f, "version_aware", df, errors)
    except Exception as exc:  # defensive: validation must never crash loading
        errors.append("validate_config 异常: {0}".format(exc))
    ok = len(errors) == 0
    return ok, errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_all_config():
    """Merge all sources into the complete logical config (with tokenizer)."""
    cfg = copy.deepcopy(_DEFAULT_CONFIG)

    # 1. Strategy file (db / matching / searcher / features).
    cfg = _deep_merge(cfg, _load_json(_CONFIG_PATH_STRATEGY))

    # 2. Tokenizer dictionary (models_data.json) -> tokenizer section.
    tokenizer = _load_json(_CONFIG_PATH_TOKENIZER)
    if tokenizer:
        cfg["tokenizer"] = _deep_merge(cfg.get("tokenizer", {}), tokenizer)

    # 3. User secrets (root config.json).
    user = _load_json(_CONFIG_PATH_USER)
    if user:
        cfg["secrets"] = _deep_merge(cfg.get("secrets", {}), user)

    # 4. Environment overrides.
    _apply_env(cfg)

    # 5. Soft validation (reverts bad values, collects warnings).
    ok, errs = validate_config(cfg)
    for e in errs:
        logger.warning("[Config] %s", e)

    return cfg


def get_all_config():
    """Return the merged config, populating the module-level cache on first call."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_all_config()
    return _CONFIG_CACHE


def get_matcher_config():
    """Return {"db": {...}, "matching": {...}} for matcher.py / database.py.

    Derived from the global merged config so callers automatically pick up new
    defaults (e.g. db.civitai_score). Return shape is unchanged for callers.
    """
    cfg = get_all_config()
    return {"db": cfg.get("db", {}), "matching": cfg.get("matching", {})}


def get_searcher_config():
    """Return the ``searcher`` section of the merged config."""
    return get_all_config().get("searcher", {})


def get_tokenizer_config():
    """Return the ``tokenizer`` section (loaded from models_data.json)."""
    return get_all_config().get("tokenizer", {})


def get_features():
    """Return the ``features`` section (Phase 2 extension points)."""
    return get_all_config().get("features", {})


def get_secret(key):
    """Return a secret value from the secrets section (env-overridable)."""
    return get_user_config().get(key, "")


def get_user_config():
    """Return the secrets section (used by /get-config)."""
    return dict(get_all_config().get("secrets", {}))


def save_user_config(data):
    """Write ``data`` into the root config.json secrets section only.

    Merges into any existing file content and refreshes the cache.
    """
    data = data or {}
    existing = {}
    if os.path.exists(_CONFIG_PATH_USER):
        try:
            with open(_CONFIG_PATH_USER, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except Exception:
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update(data)
    # Keep only secret keys to avoid drifting the config.json shape.
    allowed = set(_DEFAULT_CONFIG["secrets"].keys())
    if allowed:
        existing = {k: existing[k] for k in allowed if k in existing}
    try:
        os.makedirs(os.path.dirname(_CONFIG_PATH_USER), exist_ok=True)
        with open(_CONFIG_PATH_USER, "w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=4)
    except Exception as exc:
        logger.warning("[Config] 写入 config.json 失败: %s", exc)
    # Refresh cached secrets so subsequent reads reflect the write.
    if _CONFIG_CACHE is not None:
        _CONFIG_CACHE["secrets"] = copy.deepcopy(existing)


def override_config(partial):
    """Test hook: deep-merge ``partial`` into the cached config and return it."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_all_config()
    _CONFIG_CACHE = _deep_merge(_CONFIG_CACHE, partial)
    return _CONFIG_CACHE


def reset_config():
    """Test hook: clear the cached config so the next read reloads from disk."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


# ---------------------------------------------------------------------------
# Backward-compatible aliases (kept for existing callers / regression tests)
# ---------------------------------------------------------------------------
def load_matcher_config(path=None):
    """Deprecated: use get_matcher_config(). Returns {db, matching}."""
    return get_matcher_config()


def override_matcher_config(config):
    """Deprecated alias: deep-merge ``config`` into the cached config."""
    override_config(config or {})
    return get_matcher_config()


def reset_matcher_config():
    """Deprecated alias: clear cache and return the refreshed matcher config."""
    reset_config()
    return get_matcher_config()
