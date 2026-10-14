"""Project type management — built-in and user-defined types."""
from __future__ import annotations

import datetime
import re
import shutil
from pathlib import Path
from typing import Optional

import yaml

from .config import get_config

# Built-in types ship alongside the plugin
_BUILTIN_TYPES_ROOT = Path(__file__).parent.parent.parent.parent / "project-types"


def _get_user_types_root() -> Path:
    """Return ~/.project-hub/project-types/ (user-managed custom types)."""
    cfg = get_config()
    raw = cfg.get("project_types_root", str(Path.home() / ".project-hub" / "project-types"))
    return Path(raw).expanduser()


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug.strip("-")


def _read_type_meta(type_dir: Path) -> dict:
    """Read README.md frontmatter from a type directory."""
    readme = type_dir / "README.md"
    if not readme.exists():
        return {"name": type_dir.name, "label": type_dir.name, "description": ""}

    content = readme.read_text(encoding="utf-8")
    meta: dict = {"name": type_dir.name, "label": type_dir.name, "description": ""}

    if content.startswith("---"):
        lines = content.split("\n")
        end = next((i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
        if end:
            for line in lines[1:end]:
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip()

    return meta


def list_project_types() -> list[dict]:
    """List all available project types: built-in + custom (user-defined).

    Custom types override built-ins when names collide.
    Each entry includes a 'source' field: 'built-in' or 'custom'.
    """
    result: dict[str, dict] = {}

    if _BUILTIN_TYPES_ROOT.exists():
        for d in sorted(_BUILTIN_TYPES_ROOT.iterdir()):
            if d.is_dir():
                meta = _read_type_meta(d)
                result[d.name] = {**meta, "source": "built-in", "path": str(d)}

    user_root = _get_user_types_root()
    if user_root.exists():
        for d in sorted(user_root.iterdir()):
            if d.is_dir():
                meta = _read_type_meta(d)
                result[d.name] = {**meta, "source": "custom", "path": str(d)}

    return sorted(result.values(), key=lambda x: x["name"])


def get_project_type(name: str) -> Optional[dict]:
    """Get project type details. Checks user directory first, then built-ins."""
    user_dir = _get_user_types_root() / name
    if user_dir.is_dir():
        meta = _read_type_meta(user_dir)
        readme = user_dir / "README.md"
        return {
            **meta,
            "source": "custom",
            "path": str(user_dir),
            "readme": readme.read_text(encoding="utf-8") if readme.exists() else "",
        }

    builtin_dir = _BUILTIN_TYPES_ROOT / name
    if builtin_dir.is_dir():
        meta = _read_type_meta(builtin_dir)
        readme = builtin_dir / "README.md"
        return {
            **meta,
            "source": "built-in",
            "path": str(builtin_dir),
            "readme": readme.read_text(encoding="utf-8") if readme.exists() else "",
        }

    return None


def create_project_type(
    name: str,
    description: str = "",
    phases: Optional[list[str]] = None,
    contacts: Optional[list[str]] = None,
) -> dict:
    """Create a custom project type scaffold in ~/.project-hub/project-types/{name}/."""
    slug = _slugify(name)
    user_root = _get_user_types_root()
    type_dir = user_root / slug

    if type_dir.exists():
        return {"error": f"Project type '{slug}' already exists", "path": str(type_dir)}

    is_override = (_BUILTIN_TYPES_ROOT / slug).is_dir()

    type_dir.mkdir(parents=True, exist_ok=True)
    (type_dir / "knowledge").mkdir(exist_ok=True)

    phases_section = ""
    if phases:
        phases_section = "\n## Standard Phases\n\n" + "\n".join(
            f"{i + 1}. **{p}**" for i, p in enumerate(phases)
        )

    contacts_section = ""
    if contacts:
        contacts_section = "\n## Standard Contacts\n\n" + "\n".join(
            f"- {c}" for c in contacts
        )

    readme_content = f"""---
name: {slug}
label: {name}
description: {description}
---

# {name}
{phases_section}
{contacts_section}
""".strip() + "\n"

    (type_dir / "README.md").write_text(readme_content, encoding="utf-8")

    return {
        "name": slug,
        "label": name,
        "description": description,
        "source": "custom",
        "path": str(type_dir),
        "overrides_builtin": is_override,
        "created": True,
    }


def delete_project_type(name: str) -> dict:
    """Delete a custom project type. Built-in types cannot be deleted."""
    if (_BUILTIN_TYPES_ROOT / name).is_dir():
        return {"error": f"Cannot delete built-in type '{name}' — only custom types can be removed"}

    user_dir = _get_user_types_root() / name
    if not user_dir.exists():
        return {"error": f"Custom type '{name}' not found"}

    shutil.rmtree(user_dir)
    return {"deleted": True, "name": name}


# ---------------------------------------------------------------------------
# Template support
# ---------------------------------------------------------------------------

def _coerce_str(value: object) -> str:
    """Coerce any scalar YAML-parsed value to a plain string."""
    if value is None:
        return ""
    if isinstance(value, datetime.date):
        return value.isoformat()
    return str(value)


def get_project_template(project_type: str) -> dict:
    """Return the template markdown for a project type.

    Checks user-defined types first, then built-ins.
    Falls back to the generic template when no specific template exists.
    """
    for root in [_get_user_types_root(), _BUILTIN_TYPES_ROOT]:
        template_file = root / project_type / "template.md"
        if template_file.exists():
            return {
                "project_type": project_type,
                "template_content": template_file.read_text(encoding="utf-8"),
            }

    generic_template = _BUILTIN_TYPES_ROOT / "generic" / "template.md"
    if generic_template.exists():
        content = generic_template.read_text(encoding="utf-8")
        content = re.sub(r"(?m)^project_type:\s*generic\s*$", f"project_type: {project_type}", content, count=1)
        return {
            "project_type": "generic",
            "template_content": content,
            "fallback": True,
        }

    return {"error": f"No template found for type '{project_type}'"}


def parse_project_template(content: str) -> dict:
    """Parse a filled-out project template and return its field values.

    Expects YAML frontmatter delimited by '---' markers.
    Date values are coerced to ISO-format strings.
    """
    content = content.strip()
    if not content.startswith("---"):
        return {"error": "No YAML frontmatter found — content must start with '---'"}

    lines = content.split("\n")
    end = next((i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---"), None)
    if end is None:
        return {"error": "Unclosed YAML frontmatter — missing closing '---'"}

    yaml_block = "\n".join(lines[1:end])
    try:
        data = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as exc:
        return {"error": f"YAML parse error: {exc}"}

    for key, val in data.items():
        if isinstance(val, (list, dict)):
            return {"error": f"Field '{key}' contains a {type(val).__name__} value. All fields must be plain text."}

    return {k: _coerce_str(v) for k, v in data.items()}


def create_project_from_template(template_content: str = "", file_path: str = "") -> dict:
    """Create a project from a filled-out template.

    Accepts either the template as a string (template_content) or a path to
    a saved template file (file_path). Both must not be provided at once.
    """
    from tools.projects import create_project  # late import — avoids circular dependency

    if file_path:
        if not file_path.endswith(".md"):
            return {"error": "file_path must point to a .md file"}
        path = Path(file_path).expanduser()
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        template_content = path.read_text(encoding="utf-8")

    if not template_content:
        return {"error": "Either template_content or file_path is required"}

    fields = parse_project_template(template_content)
    if "error" in fields:
        return fields

    name = fields.get("name", "").strip()
    if not name:
        return {"error": "Field 'name' is required and cannot be empty"}

    project_type = fields.get("project_type", "generic").strip() or "generic"

    return create_project(
        name=name,
        project_type=project_type,
        description=fields.get("description", ""),
        market=fields.get("market", ""),
        products=fields.get("products", ""),
        phase=fields.get("phase", ""),
        go_live=fields.get("go_live", ""),
        budget=fields.get("budget", ""),
        notes=fields.get("notes", ""),
    )
