from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TelnyxSettings(BaseSettings):
    """Telnyx messaging channel settings, all read from ``GVAS_TELNYX_*``.

    The channel is optional as a set: ``public_key``, ``api_key`` and
    ``installations`` must all be present for the channel to be on, and the
    production composition rejects a deployment that sets only some of them.
    """

    model_config = SettingsConfigDict(env_prefix="GVAS_TELNYX_", env_file=".env", extra="ignore")

    public_key: str = ""
    api_key: str = ""
    installations: str = ""
    messaging_profile_id: str = ""
    webhook_path: str = "/telnyx/messaging"
    request_max_age_seconds: int = Field(default=300, gt=0)
    api_base_url: str = "https://api.telnyx.com/v2"
    api_timeout_seconds: float = Field(default=30.0, gt=0)

    @property
    def required_settings(self) -> dict[str, bool]:
        return {
            "GVAS_TELNYX_PUBLIC_KEY": bool(self.public_key),
            "GVAS_TELNYX_API_KEY": bool(self.api_key),
            "GVAS_TELNYX_INSTALLATIONS": bool(self.installations),
        }

    @property
    def is_configured(self) -> bool:
        return all(self.required_settings.values())

    @property
    def is_partially_configured(self) -> bool:
        return any(self.required_settings.values()) and not self.is_configured
