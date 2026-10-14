from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from gvas.domain.identifiers import BusinessId

SLACK_SOURCE_NAMESPACE = "slack"


class SlackInstallationError(ValueError):
    pass


class SlackInstallation(BaseModel):
    """A Slack bot installation bound to exactly one business.

    Workspace membership does not grant owner authority: only the explicitly
    configured ``owner_user_ids`` may drive owner workflows for the business.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    business_id: BusinessId
    team_id: str = Field(min_length=1)
    owner_user_ids: frozenset[str] = Field(min_length=1)
    api_app_id: str | None = None

    @property
    def external_endpoint_id(self) -> str:
        if self.api_app_id is None:
            return self.team_id
        return f"{self.team_id}/{self.api_app_id}"

    def is_authorized_owner(self, user_id: str) -> bool:
        return user_id in self.owner_user_ids


class SlackInstallationDirectory(Protocol):
    async def find(self, team_id: str, api_app_id: str | None) -> SlackInstallation | None: ...


def parse_slack_installations(value: str) -> tuple[SlackInstallation, ...]:
    """Parse ``T0001=<business-uuid>:U0001|U0002,...`` settings values.

    Every entry must list at least one authorized owner user id, so a
    misconfigured workspace cannot silently authorize its whole membership.
    Multiple workspaces and multiple owners parse here; the deployment decides
    how many of each it accepts.
    """

    installations: list[SlackInstallation] = []
    for entry in value.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        team_id, separator, remainder = candidate.partition("=")
        if not separator or not team_id.strip():
            raise SlackInstallationError(f"invalid installation entry {candidate!r}")
        business, owner_separator, owners = remainder.partition(":")
        if not owner_separator:
            raise SlackInstallationError(
                f"installation entry {candidate!r} lists no authorized owner users"
            )
        try:
            business_uuid = UUID(business.strip())
        except ValueError as error:
            raise SlackInstallationError(
                f"invalid business identifier in entry {candidate!r}"
            ) from error
        owner_user_ids = frozenset(owner.strip() for owner in owners.split("|") if owner.strip())
        if not owner_user_ids:
            raise SlackInstallationError(
                f"installation entry {candidate!r} lists no authorized owner users"
            )
        installations.append(
            SlackInstallation(
                business_id=BusinessId(business_uuid),
                team_id=team_id.strip(),
                owner_user_ids=owner_user_ids,
            )
        )
    return tuple(installations)


class StaticSlackInstallationDirectory:
    """Configuration-driven installation lookup keyed by Slack workspace."""

    def __init__(self, installations: tuple[SlackInstallation, ...]) -> None:
        self._by_team = {installation.team_id: installation for installation in installations}

    @classmethod
    def from_setting(cls, value: str) -> "StaticSlackInstallationDirectory":
        return cls(parse_slack_installations(value))

    async def find(self, team_id: str, api_app_id: str | None) -> SlackInstallation | None:
        installation = self._by_team.get(team_id)
        if installation is None:
            return None
        if installation.api_app_id is not None and installation.api_app_id != api_app_id:
            return None
        return installation
