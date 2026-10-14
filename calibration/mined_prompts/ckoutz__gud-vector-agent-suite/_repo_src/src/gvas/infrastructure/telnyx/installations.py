import re
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gvas.domain.identifiers import BusinessId

TELNYX_SOURCE_NAMESPACE = "telnyx"
E164_PATTERN = re.compile(r"^\+[1-9]\d{1,14}$")


class TelnyxInstallationError(ValueError):
    pass


def _require_e164(value: str, *, role: str) -> str:
    if not E164_PATTERN.fullmatch(value):
        raise TelnyxInstallationError(f"{role} {value!r} is not an E.164 phone number")
    return value


class TelnyxInstallation(BaseModel):
    """A Telnyx business number bound to one business and its authorized owner numbers.

    Any handset can text the business number; only the configured
    ``owner_numbers`` may drive owner workflows, everyone else is ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    business_id: BusinessId
    telnyx_number: str = Field(min_length=1)
    owner_numbers: frozenset[str] = Field(min_length=1)

    @field_validator("telnyx_number")
    @classmethod
    def telnyx_number_is_e164(cls, value: str) -> str:
        return _require_e164(value, role="telnyx number")

    @field_validator("owner_numbers")
    @classmethod
    def owner_numbers_are_e164(cls, value: frozenset[str]) -> frozenset[str]:
        for number in value:
            _require_e164(number, role="owner number")
        return value

    @property
    def external_endpoint_id(self) -> str:
        return self.telnyx_number

    def is_authorized_owner(self, phone_number: str) -> bool:
        return phone_number in self.owner_numbers


class TelnyxInstallationDirectory(Protocol):
    async def find(self, telnyx_number: str) -> TelnyxInstallation | None: ...


def parse_telnyx_installations(value: str) -> tuple[TelnyxInstallation, ...]:
    """Parse ``<owner E.164>=<business-uuid>:<telnyx E.164>,...`` settings values.

    Owners of the same business number may be listed with ``|`` on the left-hand
    side. The deployment decides how many numbers and owners it accepts.
    """

    installations: list[TelnyxInstallation] = []
    for entry in value.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        owners, separator, remainder = candidate.partition("=")
        if not separator or not owners.strip():
            raise TelnyxInstallationError(f"invalid installation entry {candidate!r}")
        business, number_separator, telnyx_number = remainder.partition(":")
        if not number_separator or not telnyx_number.strip():
            raise TelnyxInstallationError(
                f"installation entry {candidate!r} names no telnyx number"
            )
        try:
            business_uuid = UUID(business.strip())
        except ValueError as error:
            raise TelnyxInstallationError(
                f"invalid business identifier in entry {candidate!r}"
            ) from error
        owner_numbers = frozenset(owner.strip() for owner in owners.split("|") if owner.strip())
        if not owner_numbers:
            raise TelnyxInstallationError(
                f"installation entry {candidate!r} lists no authorized owner numbers"
            )
        try:
            installation = TelnyxInstallation(
                business_id=BusinessId(business_uuid),
                telnyx_number=telnyx_number.strip(),
                owner_numbers=owner_numbers,
            )
        except ValidationError as error:
            raise TelnyxInstallationError(
                f"installation entry {candidate!r} has a phone number that is not E.164"
            ) from error
        installations.append(installation)
    return tuple(installations)


class StaticTelnyxInstallationDirectory:
    """Configuration-driven installation lookup keyed by the receiving Telnyx number."""

    def __init__(self, installations: tuple[TelnyxInstallation, ...]) -> None:
        self._by_number = {
            installation.telnyx_number: installation for installation in installations
        }

    @classmethod
    def from_setting(cls, value: str) -> "StaticTelnyxInstallationDirectory":
        return cls(parse_telnyx_installations(value))

    async def find(self, telnyx_number: str) -> TelnyxInstallation | None:
        return self._by_number.get(telnyx_number)
