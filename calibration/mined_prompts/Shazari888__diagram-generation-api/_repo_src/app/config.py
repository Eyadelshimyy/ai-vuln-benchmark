from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Diagram Generation API"
    api_key: str

    database_url: str

    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    kroki_base_url: str = "https://kroki.io"
    mermaid_ink_base_url: str = "https://mermaid.ink"
    base_app_id: str | None = None

    x402_enabled: bool = False
    x402_facilitator_url: str = "https://x402.org/facilitator"
    x402_network: str = "eip155:84532"
    x402_pay_to: str | None = None
    x402_price_svg: str = "$0.05"
    x402_price_png: str = "$0.07"
    x402_builder_code: str | None = None
    cdp_api_key_id: str | None = None
    cdp_api_key_secret: str | None = None

    # Security
    enforce_https: bool = False
    rate_limit_free: str = "5/day"      # Free diagram cap per IP
    rate_limit_paid: str = "20/minute"
    rate_limit_global: str = "60/minute"
    use_llm_fallback: bool = False
    admin_ip_allowlist: str = ""        # Comma-separated IPs allowed to hit admin routes; empty = no restriction
    cors_allowed_origins: str = ""      # Comma-separated allowed origins; empty = block all cross-origin

    # Monitoring
    logtail_token: str | None = None  # Better Stack source token

    # Deprecated legacy config retained for older local experiments.
    # The active app is x402-only and does not require Stripe settings.


settings = Settings()
