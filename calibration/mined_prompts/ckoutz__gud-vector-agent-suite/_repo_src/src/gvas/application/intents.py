from gvas.domain.intents import IntentResolution, IntentUnresolvedError
from gvas.domain.messages import NormalizedOwnerMessage


class UnconfiguredIntentResolver:
    async def resolve(self, message: NormalizedOwnerMessage) -> IntentResolution:
        raise IntentUnresolvedError("no intent resolver is configured")
