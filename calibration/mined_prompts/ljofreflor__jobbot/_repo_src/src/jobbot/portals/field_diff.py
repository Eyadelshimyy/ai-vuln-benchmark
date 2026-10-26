"""Diff form fields vs profile.yaml schema to discover new fields."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from jobbot.portals.form_learn import FieldKind, FormField, FormKnowledge


@dataclass(frozen=True)
class NewFieldCandidate:
    """A field that appears in forms but not in profile.yaml schema."""

    field_hash: str
    labels_seen: list[str]
    kind: FieldKind
    portals: list[str]
    required_count: int
    total_count: int
    options: list[str]
    example_accepts: list[str]

    @property
    def frequency(self) -> int:
        """How many portals ask for this field."""
        return len(self.portals)

    @property
    def required_ratio(self) -> float:
        """Fraction of portals that mark this field as required."""
        return self.required_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def is_common(self) -> bool:
        """Field appears in multiple portals (not just one idiosyncrasy)."""
        return self.frequency >= 2


def normalize_field_label(label: str) -> str:
    """Normalize label for semantic comparison.
    
    'email' ≈ 'correo electrónico' ≈ 'e-mail' → 'email'
    """
    import unicodedata
    
    # Lowercase, strip punctuation, collapse spaces
    normalized = label.lower().strip()
    normalized = normalized.replace("-", " ").replace("_", " ")
    
    # Remove accents
    normalized = "".join(
        c for c in unicodedata.normalize("NFD", normalized)
        if unicodedata.category(c) != "Mn"
    )
    
    normalized = " ".join(normalized.split())
    
    # Common synonyms (without accents now)
    synonyms = {
        "e mail": "email",
        "correo electronico": "email",
        "correo": "email",
        "telefono": "phone",
        "celular": "phone",
        "movil": "phone",
        "curriculum": "cv",
        "resume": "cv",
        "hoja de vida": "cv",
        "carta de presentacion": "cover letter",
        "nombre completo": "full name",
        "apellido": "last name",
        "direccion": "address",
        "pais": "country",
        "ciudad": "city",
        "visa de trabajo": "work authorization",
        "autorizacion de trabajo": "work authorization",
        "permiso de trabajo": "work authorization",
    }
    
    for old, new in synonyms.items():
        if old in normalized:
            normalized = normalized.replace(old, new)
    
    return normalized


def field_semantic_hash(label: str, kind: FieldKind) -> str:
    """Stable hash based on semantic content, not exact wording.
    
    Two labels with the same meaning → same hash.
    """
    normalized = normalize_field_label(label)
    semantic_key = f"{normalized}|{kind.value}"
    return hashlib.sha256(semantic_key.encode()).hexdigest()[:16]


def extract_profile_schema_fields() -> set[str]:
    """Extract field semantic hashes from profile.yaml schema.
    
    These are the fields we already know about.
    """
    # Known profile.yaml sections and their common field labels
    known_fields = [
        # personal
        ("name", FieldKind.TEXT),
        ("full name", FieldKind.TEXT),
        ("first name", FieldKind.TEXT),
        ("last name", FieldKind.TEXT),
        ("email", FieldKind.EMAIL),
        ("phone", FieldKind.PHONE),
        ("city", FieldKind.TEXT),
        ("country", FieldKind.TEXT),
        ("address", FieldKind.TEXT),
        ("linkedin", FieldKind.URL),
        ("github", FieldKind.URL),
        ("website", FieldKind.URL),
        ("portfolio", FieldKind.URL),
        
        # application basics
        ("cv", FieldKind.FILE),
        ("resume", FieldKind.FILE),
        ("curriculum", FieldKind.FILE),
        ("cover letter", FieldKind.FILE),
        ("carta", FieldKind.FILE),
        
        # experience (implied from profile structure)
        ("company", FieldKind.TEXT),
        ("title", FieldKind.TEXT),
        ("position", FieldKind.TEXT),
        ("start date", FieldKind.DATE),
        ("end date", FieldKind.DATE),
        ("description", FieldKind.LONG_TEXT),
        
        # education
        ("degree", FieldKind.TEXT),
        ("institution", FieldKind.TEXT),
        ("university", FieldKind.TEXT),
        
        # skills (list)
        ("skills", FieldKind.TEXT),
        ("technologies", FieldKind.TEXT),
    ]
    
    return {field_semantic_hash(label, kind) for label, kind in known_fields}


def diff_form_fields(
    forms: list[FormKnowledge],
    *,
    min_frequency: int = 1,
) -> list[NewFieldCandidate]:
    """Find fields that appear in forms but not in profile.yaml schema.
    
    Args:
        forms: All observed form knowledge
        min_frequency: Only return fields seen in >= N portals
    
    Returns:
        List of new field candidates, sorted by frequency (descending)
    """
    known_hashes = extract_profile_schema_fields()
    
    # Group form fields by semantic hash
    field_groups: dict[str, list[tuple[FormField, str]]] = {}  # hash → [(field, portal)]
    
    for form in forms:
        if not form.readable or not form.fields:
            continue
        
        portal = form.url
        for field in form.fields:
            # Skip identity/basic fields that every form asks
            if field.is_screening is False:
                continue
            
            field_hash = field_semantic_hash(field.label, field.kind)
            
            # Skip if we already have this in profile.yaml
            if field_hash in known_hashes:
                continue
            
            if field_hash not in field_groups:
                field_groups[field_hash] = []
            field_groups[field_hash].append((field, portal))
    
    # Build candidates from groups
    candidates: list[NewFieldCandidate] = []
    
    for field_hash, occurrences in field_groups.items():
        portals = list({portal for _, portal in occurrences})
        
        # Skip if below frequency threshold
        if len(portals) < min_frequency:
            continue
        
        # Collect all labels seen for this field
        labels_seen = list({field.label for field, _ in occurrences})
        
        # Use the most common kind (usually they agree)
        kinds = [field.kind for field, _ in occurrences]
        kind = max(set(kinds), key=kinds.count)
        
        # Count how many mark it as required
        required_count = sum(1 for field, _ in occurrences if field.required)
        
        # Collect options from select/radio fields
        all_options: list[str] = []
        for field, _ in occurrences:
            all_options.extend(field.options)
        unique_options = list(dict.fromkeys(all_options))  # preserve order, dedup
        
        # Collect file accepts
        accepts: list[str] = []
        for field, _ in occurrences:
            accepts.extend(field.accepts)
        unique_accepts = list(dict.fromkeys(accepts))
        
        candidates.append(
            NewFieldCandidate(
                field_hash=field_hash,
                labels_seen=labels_seen,
                kind=kind,
                portals=portals,
                required_count=required_count,
                total_count=len(occurrences),
                options=unique_options[:10],  # Limit to first 10 options
                example_accepts=unique_accepts,
            )
        )
    
    # Sort by frequency (descending), then by required ratio
    candidates.sort(key=lambda c: (c.frequency, c.required_ratio), reverse=True)
    
    return candidates


def has_semantic_match(label: str, kind: FieldKind, known_hashes: set[str]) -> bool:
    """Check if a field semantically matches any known field."""
    field_hash = field_semantic_hash(label, kind)
    return field_hash in known_hashes
