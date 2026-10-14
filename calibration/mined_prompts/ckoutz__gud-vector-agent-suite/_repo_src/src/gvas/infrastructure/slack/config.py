from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SlackSettings(BaseSettings):
    """Slack adapter settings; values come from the environment only."""

    model_config = SettingsConfigDict(env_prefix="GVAS_SLACK_", env_file=".env", extra="ignore")

    signing_secret: str = ""
    bot_token: str = ""
    events_path: str = "/slack/events"
    request_max_age_seconds: int = 300
    installations: str = ""
    api_base_url: str = "https://slack.com/api"
    api_timeout_seconds: float = Field(default=30.0, gt=0)
    attachment_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1)

    @property
    def is_configured(self) -> bool:
        return bool(self.signing_secret and self.bot_token and self.installations)
