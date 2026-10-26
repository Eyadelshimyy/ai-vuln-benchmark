"""Skill groupings and publications."""

from __future__ import annotations

import unicodedata

from pydantic import BaseModel, Field, HttpUrl


class SkillGroups(BaseModel):
    """Named skill categories; extra keys allowed via model_config."""

    model_config = {"extra": "allow"}

    programming: list[str] = Field(default_factory=list)
    machine_learning: list[str] = Field(default_factory=list)
    cloud: list[str] = Field(default_factory=list)
    statistics: list[str] = Field(default_factory=list)
    engineering: list[str] = Field(default_factory=list)

    def all_skills(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for values in self._group_values():
            for skill in values:
                key = skill.casefold()
                if key not in seen:
                    seen.add(key)
                    ordered.append(skill)
        return ordered

    def as_dict(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for name, values in self._groups():
            if values:
                result[name] = list(values)
        return result

    def count(self) -> int:
        return len(self.all_skills())

    def _groups(self) -> list[tuple[str, list[str]]]:
        data = self.model_dump()
        return [(str(k), list(v)) for k, v in data.items() if isinstance(v, list)]

    def _group_values(self) -> list[list[str]]:
        return [values for _, values in self._groups()]


def _name_tokens(name: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", name.casefold())
    ascii_name = "".join(c for c in folded if not unicodedata.combining(c))
    return {token for token in ascii_name.replace(",", " ").split() if len(token) > 2}


class Publication(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    journal: str | None = None
    year: int | None = None
    status: str | None = None
    doi: str | None = None
    url: HttpUrl | None = None
    authors: list[str] = Field(default_factory=list)

    def coauthors(self, self_name: str) -> list[str]:
        """Authors excluding the candidate, by overlap with their own name.

        Two shared name parts identify the candidate ('Jofré Flor, L.' vs the full
        name); one shared part is a namesake ('Leonardo Pérez') and stays.
        """
        me_tokens = _name_tokens(self_name)
        return [
            author
            for author in self.authors
            if len(me_tokens & _name_tokens(author)) < 2
        ]
