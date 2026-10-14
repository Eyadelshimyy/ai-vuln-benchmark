from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_UNSAFE_DEFAULT_KEY = "change_me_in_production_please"

# Valid AI provider identifiers. Lookups go through the plugin catalog;
# this set is the configuration allow-list (unknown values fail fast).
_VALID_AI_PROVIDERS = {
    "mock",
    "openai",
    "anthropic",
    "ollama",
    "openai_compatible",
    "local",
    "mlx",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "BugForge"
    version: str = "1.8.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://bugforge:bugforge_dev@localhost:5432/bugforge"

    # Security — must be overridden in production
    secret_key: str = _UNSAFE_DEFAULT_KEY

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    # Analysis limits
    max_file_size_bytes: int = 5 * 1024 * 1024  # 5 MB
    max_repo_files: int = 10_000
    analysis_timeout_seconds: int = 300
    # Cross-file taint. Hitting a limit stops propagation and records a diagnostic.
    taint_max_files: int = 400
    taint_max_import_depth: int = 3
    taint_max_cross_file_rounds: int = 4
    taint_max_cross_file_edges: int = 2_000
    taint_cross_file_budget_ms: int = 1_500
    # Field paths such as obj.payload and obj["key"]. Zero disables field flow.
    taint_max_field_depth: int = 4
    taint_max_field_bindings: int = 2_000
    taint_max_alias_edges: int = 32

    # AI provider: mock | openai | anthropic | ollama | openai_compatible | local | mlx
    ai_provider: str = "mock"
    ai_model: str = "gpt-4o-mini"
    # API key comes from environment only — never commit a real key
    ai_api_key: str = ""
    # Base URL override: for Ollama use http://localhost:11434/v1
    ai_base_url: str = ""
    ai_max_context_chars: int = 32_000
    ai_max_output_tokens: int = 2_000
    ai_temperature: float = 0.1
    ai_timeout_seconds: int = 60
    ai_max_retries: int = 2
    ai_max_hypotheses: int = 3
    # Local MLX (Qwen etc.). Model name is configurable — not hard-coded in the engine.
    mlx_base_url: str = "http://127.0.0.1:8080/v1"
    mlx_model: str = "Qwen3.6-35B-A3B-8bit"
    ai_thinking_enabled: bool = False
    ai_native_json_mode: bool = False
    ai_max_context_tokens: int = 8192
    ai_security_enabled: bool = True

    # Language analyzers that should run during static analysis.
    # Empty = every registered adapter that implements STATIC_ANALYSIS
    # (Python plus JavaScript/TypeScript quality rules).
    # Comma-separated language ids, e.g. "python".
    language_analyzers: str = ""

    # GitHub Integration — credentials come from environment only, never committed
    github_token: str = ""
    github_app_id: str = ""
    github_app_private_key: str = ""

    # HackerOne Hacker API — never expose these to the frontend
    hackerone_api_username: str = ""
    hackerone_api_token: str = ""
    hackerone_api_base_url: str = "https://api.hackerone.com/v1"

    # Local operator authorization for HackerOne approval / live-testing gates.
    # Never persist these values in the database.
    bugforge_operator_token: str = ""
    bugforge_operator_identity: str = ""

    # HackerOne program-data freshness before live submission (hours).
    hackerone_program_max_age_hours: int = 24
    hackerone_weakness_max_age_hours: int = 168
    hackerone_attachment_max_bytes: int = 10 * 1024 * 1024
    bugforge_evidence_dir: str = ""

    # Guided security research agent budgets. The AI cannot raise these.
    security_agent_max_iterations: int = 20
    security_agent_max_tool_calls: int = 40
    security_agent_max_requests: int = 50
    security_agent_max_browser_actions: int = 20
    security_agent_max_fuzz_requests: int = 20
    security_agent_max_tokens: int = 200_000
    security_agent_max_scan_seconds: float = 300.0
    security_agent_max_identical_tool_calls: int = 2
    security_agent_identical_call_window_seconds: float = 120.0
    security_agent_source_excerpt_bytes: int = 32_768
    security_agent_source_max_lines: int = 200
    security_agent_approval_ttl_hours: int = 24
    security_agent_lab_roots: list[str] = []
    security_agent_exploratory_enabled: bool = False
    security_agent_max_exploratory_tests: int = 4
    security_agent_max_exploratory_iterations: int = 3
    security_agent_max_exploratory_seconds: float = 60.0
    security_agent_max_generated_test_bytes: int = 16_384
    security_agent_max_exploratory_artifact_bytes: int = 65_536
    security_agent_exploratory_require_docker: bool = True
    # Operator attests that the image already contains pytest. BugForge does not pip install.
    security_agent_exploratory_pytest_ready: bool = False
    security_agent_exploratory_python_image: str = ""
    # Local image that already contains forge. Empty means replay stays unavailable.
    # BugForge does not pull an image and does not fall back to host forge.
    security_agent_replay_image: str = ""
    # Local image that already contains ityfuzz. Empty means ItyFuzz stays unavailable.
    # BugForge does not pull an image and does not fall back to a host binary.
    security_agent_ityfuzz_image: str = ""
    # Local image for pinned runtime validation. Empty means runtime stays unavailable.
    # BugForge does not pull an image and does not fall back to a host binary.
    security_agent_runtime_image: str = ""
    # Fork execution stays off unless an operator sets both of these. Request text cannot.
    security_agent_fork_enabled: bool = False
    security_agent_fork_source: str = ""
    # Developer-only. The default research path does not execute host solc.
    solidity_host_compiler: bool = False
    # Bounty campaigns (Phase 51/52). Empty means the application database_url.
    # Campaign state is persisted there through the Phase 49 SqlStore; there is no
    # in-memory production fallback.
    bounty_campaign_database_url: str = ""
    # Bounded local stateful execution of VFCS harnesses with an installed fuzzer
    # (forge/echidna/medusa). Offline, no RPC, ffi disabled, never a fork or live chain.
    bounty_stateful_execution: bool = True
    bounty_stateful_timeout_seconds: int = 120
    # Bounded project-analysis profile. Defaults match the original scan limits.
    solidity_project_max_files: int = 40
    solidity_project_max_bytes: int = 200_000
    solidity_project_include_dependencies: bool = False
    solidity_compiler_emit_ir: bool = True
    solidity_yul_max_blocks: int = 32
    solidity_yul_max_nodes: int = 200
    solidity_cross_max_contracts: int = 48
    solidity_cross_max_edges: int = 240
    solidity_compiler_timeout_seconds: int = 20
    solidity_compiler_memory_mb: int = 512
    solidity_compiler_max_output: int = 1_000_000

    # -------------------------------------------------------------------
    # Autonomous Repository Discovery (v1.1.0)
    # -------------------------------------------------------------------
    # master switch: disabled | manual | scheduled
    discovery_mode: str = "disabled"
    # hours between scheduled discovery sweeps (0 = run once then stop)
    discovery_interval_hours: int = 24
    # minimum GitHub stars for a candidate to be considered
    discovery_min_stars: int = 1_000
    # maximum stars (0 = no upper bound)
    discovery_max_stars: int = 0
    # maximum repository disk size in KB (0 = no limit)
    discovery_max_size_kb: int = 50_000
    # maximum number of open files in the repository (0 = no limit)
    discovery_max_files: int = 5_000
    # comma-separated list of allowed primary languages (empty = all)
    discovery_languages: str = "Python,JavaScript,TypeScript,Go,Rust,Java"
    # require an OSI-recognised open-source license
    discovery_require_license: bool = True
    # skip forked repositories
    discovery_skip_forks: bool = True
    # skip archived repositories
    discovery_skip_archived: bool = True
    # require a commit within this many days (0 = no recency filter)
    discovery_max_staleness_days: int = 365
    # daily cap on new repository candidates (0 = unlimited)
    discovery_daily_repo_limit: int = 100
    # maximum simultaneous autonomous analyses
    discovery_max_concurrent: int = 2
    # daily cap on AI calls from the autonomous scanner (0 = unlimited)
    discovery_daily_ai_limit: int = 200
    # comma-separated topics to exclude
    discovery_excluded_topics: str = "exploit,hack,malware,pentest,ctf"
    # comma-separated owner logins to exclude
    discovery_excluded_owners: str = ""
    # permanent workspace root for cloned repositories (empty = auto-select)
    autonomous_workspace_dir: str = ""

    # -------------------------------------------------------------------
    # Eligibility / Safety Policy (v1.1.0)
    # -------------------------------------------------------------------
    # hard block repositories larger than this many KB
    safety_max_repo_size_kb: int = 100_000
    # hard block repositories with more than this many files
    safety_max_file_count: int = 20_000
    # hard block repositories with more than this many shell/CI scripts
    safety_max_script_count: int = 50
    # allow Docker execution of repository tests
    safety_allow_docker_exec: bool = True
    # allow network access inside sandbox containers
    safety_allow_sandbox_network: bool = False
    # allow dependency installation inside the sandbox
    safety_allow_dep_install: bool = False

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v = v.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info: object) -> str:
        env = getattr(getattr(info, "data", {}), "get", lambda k, d=None: d)(
            "environment", "development"
        )
        if v == _UNSAFE_DEFAULT_KEY and env == "production":
            raise ValueError(
                "SECRET_KEY must be set to a secure random value in production. "
                'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
            )
        return v

    @field_validator("discovery_mode")
    @classmethod
    def validate_discovery_mode(cls, v: str) -> str:
        valid = {"disabled", "manual", "scheduled"}
        if v not in valid:
            raise ValueError(f"discovery_mode must be one of {valid}")
        return v

    @field_validator(
        "taint_max_files",
        "taint_max_import_depth",
        "taint_max_cross_file_rounds",
        "taint_max_cross_file_edges",
        "taint_cross_file_budget_ms",
        "taint_max_field_depth",
        "taint_max_field_bindings",
        "taint_max_alias_edges",
    )
    @classmethod
    def validate_taint_limit(cls, value: int) -> int:
        if value < 0:
            raise ValueError("cross-file taint limits must be >= 0; 0 disables that propagation")
        return value

    @field_validator("ai_provider")
    @classmethod
    def validate_ai_provider(cls, v: str) -> str:
        if v not in _VALID_AI_PROVIDERS:
            raise ValueError(f"ai_provider must be one of {_VALID_AI_PROVIDERS}")
        return v

    def get_enabled_language_analyzers(self) -> list[str]:
        """Return configured language analyzer ids (empty = all capable adapters)."""
        if not self.language_analyzers.strip():
            return []
        return [lang.strip().lower() for lang in self.language_analyzers.split(",") if lang.strip()]

    def get_discovery_languages(self) -> list[str]:
        """Return the allowed language list (empty = all languages permitted)."""
        if not self.discovery_languages.strip():
            return []
        return [lang.strip() for lang in self.discovery_languages.split(",") if lang.strip()]

    def get_excluded_topics(self) -> set[str]:
        if not self.discovery_excluded_topics.strip():
            return set()
        return {t.strip().lower() for t in self.discovery_excluded_topics.split(",") if t.strip()}

    def get_excluded_owners(self) -> set[str]:
        if not self.discovery_excluded_owners.strip():
            return set()
        return {o.strip().lower() for o in self.discovery_excluded_owners.split(",") if o.strip()}

    def is_local_ai(self) -> bool:
        """Return True when the AI provider runs locally (no cloud cost)."""
        return self.ai_provider in {"mock", "ollama", "openai_compatible", "local", "mlx"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
