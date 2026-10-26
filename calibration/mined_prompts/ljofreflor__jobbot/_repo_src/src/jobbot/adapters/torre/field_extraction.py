"""Extract structured fields from Torre API responses for portal learning.

Torre provides structured data that can enrich the profile schema:
- Skills with experience requirements
- Remote work preferences  
- Compensation visibility
"""

from __future__ import annotations

from typing import Any

from jobbot.portals.form_learn import FieldKind, FormField


def extract_torre_fields(opportunity: dict[str, Any]) -> list[FormField]:
    """Extract pseudo-form-fields from Torre opportunity data.
    
    Torre API returns structured fields that don't come from HTML forms
    but represent requirements/preferences that should feed learning.
    
    Args:
        opportunity: Torre API opportunity dict
        
    Returns:
        List of FormField representing Torre's structured requirements
    """
    fields: list[FormField] = []
    
    # Remote work preference
    remote = opportunity.get("remote")
    if isinstance(remote, bool):
        fields.append(
            FormField(
                name="remote_work",
                label="Remote work",
                kind=FieldKind.CHECKBOX,
                required=False,
                is_screening=True,
            )
        )
    
    # Skills with experience requirements
    skills = opportunity.get("skills")
    if isinstance(skills, list):
        for skill in skills:
            if not isinstance(skill, dict):
                continue
            
            name = skill.get("name")
            experience = skill.get("experience")
            
            if name and experience:
                # e.g., "Python experience level"
                fields.append(
                    FormField(
                        name=f"skill_{_slugify(name)}_experience",
                        label=f"{name} experience level",
                        kind=FieldKind.SELECT,
                        required=False,
                        options=[
                            "potential-to-develop",
                            "novice",
                            "intermediate",
                            "advanced",
                            "expert",
                        ],
                        is_screening=True,
                    )
                )
    
    # Compensation visibility
    compensation = opportunity.get("compensation")
    if isinstance(compensation, dict):
        visible = compensation.get("visible")
        if isinstance(visible, bool):
            fields.append(
                FormField(
                    name="compensation_visible",
                    label="Show compensation/salary",
                    kind=FieldKind.CHECKBOX,
                    required=False,
                    is_screening=True,
                )
            )
    
    # Remote modality (office, hybrid, fully remote)
    remote_modality = opportunity.get("remote_modality")
    if remote_modality:
        fields.append(
            FormField(
                name="remote_modality",
                label="Remote work modality",
                kind=FieldKind.SELECT,
                required=False,
                options=["office", "hybrid", "fully-remote"],
                is_screening=True,
            )
        )
    
    # Opportunity type (job, freelance, internship, etc)
    opportunity_type = opportunity.get("type")
    if opportunity_type:
        fields.append(
            FormField(
                name="opportunity_type",
                label="Opportunity type",
                kind=FieldKind.SELECT,
                required=False,
                options=["job", "freelance", "internship", "volunteering"],
                is_screening=True,
            )
        )
    
    return fields


def _slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    return text.lower().replace(" ", "_").replace("+", "plus").replace("#", "sharp")
