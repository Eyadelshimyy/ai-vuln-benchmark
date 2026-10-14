import json
from pathlib import Path

from gvas.application.templates import IndustryTemplateDefinition
from gvas.domain.templates import IndustryKey

SEED_DIRECTORY = Path(__file__).resolve().parent.parent / "industries"


class UnknownIndustrySeedError(LookupError):
    """The requested industry has no checked-in seed file."""


def load_industry_definitions(
    directory: Path | None = None,
) -> dict[IndustryKey, IndustryTemplateDefinition]:
    """Reads the checked-in seed files so industries stay reviewable in git."""
    seed_directory = directory or SEED_DIRECTORY
    definitions: dict[IndustryKey, IndustryTemplateDefinition] = {}
    for path in sorted(seed_directory.glob("*.json")):
        definition = IndustryTemplateDefinition.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if definition.industry_key in definitions:
            raise ValueError(f"duplicate industry seed {definition.industry_key}")
        definitions[definition.industry_key] = definition
    return definitions


def load_industry_definition(
    industry_key: IndustryKey, directory: Path | None = None
) -> IndustryTemplateDefinition:
    definitions = load_industry_definitions(directory)
    definition = definitions.get(industry_key)
    if definition is None:
        raise UnknownIndustrySeedError(f"no seed file defines industry {industry_key}")
    return definition
