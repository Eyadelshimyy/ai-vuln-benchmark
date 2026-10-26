"""Domain models package."""

from jobbot.models.candidate import Candidate, PersonalInfo
from jobbot.models.education import Education
from jobbot.models.experience import Achievement, Experience
from jobbot.models.skill import Publication, SkillGroups
from jobbot.models.targets import ProfileTarget, TargetConstraints

__all__ = [
    "Achievement",
    "Candidate",
    "Education",
    "Experience",
    "PersonalInfo",
    "ProfileTarget",
    "Publication",
    "SkillGroups",
    "TargetConstraints",
]
