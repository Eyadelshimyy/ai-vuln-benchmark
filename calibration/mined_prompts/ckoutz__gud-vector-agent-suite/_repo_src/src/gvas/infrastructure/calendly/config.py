from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from gvas.domain.identifiers import BusinessId

CALENDLY_USER_URI_PREFIX = "https://api.calendly.com/users/"


class CalendlySettings(BaseSettings):
    """Calendly appointment lookup settings, all read from ``GVAS_CALENDLY_*``.

    Optional as a set: ``token`` and ``installations`` must both be present for
    the lookup to be on, and the production composition rejects a deployment
    that sets only one of them.
    """

    model_config = SettingsConfigDict(env_prefix="GVAS_CALENDLY_", env_file=".env", extra="ignore")

    token: str = ""
    installations: str = ""
    api_base_url: str = "https://api.calendly.com"
    api_timeout_seconds: float = Field(default=30.0, gt=0)
    page_size: int = Field(default=100, ge=1, le=100)
    # Signing key of the webhook subscription created out-of-band
    # (POST /webhook_subscriptions); empty disables the webhook route.
    webhook_signing_key: str = ""

    @property
    def required_settings(self) -> dict[str, bool]:
        return {
            "GVAS_CALENDLY_TOKEN": bool(self.token),
            "GVAS_CALENDLY_INSTALLATIONS": bool(self.installations),
        }

    @property
    def is_configured(self) -> bool:
        return all(self.required_settings.values())

    @property
    def is_partially_configured(self) -> bool:
        return any(self.required_settings.values()) and not self.is_configured


class CalendlyInstallationError(ValueError):
    pass


class CalendlyInstallation(BaseModel):
    """One business bound to the Calendly user whose scheduled events it reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    business_id: BusinessId
    user_uri: str = Field(min_length=1)


def parse_calendly_installations(value: str) -> tuple[CalendlyInstallation, ...]:
    """Parse ``<business-uuid>=<calendly user uri>,...`` settings values."""

    installations: list[CalendlyInstallation] = []
    seen: set[BusinessId] = set()
    for entry in value.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        business, separator, user_uri = candidate.partition("=")
        if not separator or not business.strip() or not user_uri.strip():
            raise CalendlyInstallationError(f"invalid installation entry {candidate!r}")
        if not user_uri.strip().startswith(CALENDLY_USER_URI_PREFIX):
            raise CalendlyInstallationError(
                f"installation entry {candidate!r} must name a {CALENDLY_USER_URI_PREFIX}... uri"
            )
        try:
            business_id = BusinessId(UUID(business.strip()))
        except ValueError as error:
            raise CalendlyInstallationError(
                f"installation entry {candidate!r} has an invalid business id"
            ) from error
        if business_id in seen:
            raise CalendlyInstallationError(f"business {business_id} is listed twice")
        seen.add(business_id)
        try:
            installations.append(
                CalendlyInstallation(business_id=business_id, user_uri=user_uri.strip())
            )
        except ValidationError as error:
            raise CalendlyInstallationError(str(error)) from error
    if not installations:
        raise CalendlyInstallationError("no calendly installations configured")
    return tuple(installations)
