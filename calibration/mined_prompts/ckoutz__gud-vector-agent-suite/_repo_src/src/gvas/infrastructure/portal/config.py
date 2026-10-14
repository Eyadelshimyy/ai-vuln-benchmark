from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PortalSettings(BaseSettings):
    """Customer portal handoff settings, all read from ``GVAS_PORTAL_*``.

    Optional as a set: ``base_url`` and ``api_token`` must both be present for
    approved quotes to be handed to the portal, and the production composition
    rejects a deployment that sets only one of them. With neither set, quotes
    are emailed as before.
    """

    model_config = SettingsConfigDict(env_prefix="GVAS_PORTAL_", env_file=".env", extra="ignore")

    base_url: str = ""
    api_token: str = ""
    timeout_seconds: float = Field(default=30.0, gt=0)

    @property
    def required_settings(self) -> dict[str, bool]:
        return {
            "GVAS_PORTAL_BASE_URL": bool(self.base_url),
            "GVAS_PORTAL_API_TOKEN": bool(self.api_token),
        }

    @property
    def is_configured(self) -> bool:
        return all(self.required_settings.values())

    @property
    def is_partially_configured(self) -> bool:
        return any(self.required_settings.values()) and not self.is_configured
