from __future__ import annotations

import re
import subprocess
from pathlib import Path


PLANTUML_TIMEOUT_SECONDS = 60
PLANTUML_OPTIONS = ("-tsvg", "-pipe", "-failfast2", "-nometadata")
FETCH_SCRIPT_HINT = "run scripts/fetch_plantuml.sh"
ERROR_SVG_PATTERN = re.compile(r"syntax error", re.IGNORECASE)
XML_PROLOG_PATTERN = re.compile(r"\A<\?xml[^?]*\?>\s*")


class DiagramError(RuntimeError):
    pass


def render_svg(source: str, jar: Path) -> str:
    if not jar.is_file():
        raise DiagramError(f"PlantUML jar not found at {jar}; {FETCH_SCRIPT_HINT}")

    try:
        result = subprocess.run(
            ["java", "-jar", str(jar), *PLANTUML_OPTIONS],
            input=source,
            text=True,
            capture_output=True,
            check=False,
            timeout=PLANTUML_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise DiagramError("Java runtime not found on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise DiagramError(
            f"PlantUML timed out after {PLANTUML_TIMEOUT_SECONDS} seconds"
        ) from error

    svg = result.stdout.lstrip("\ufeff \t\r\n")
    diagnostic = result.stderr.strip()
    if (
        result.returncode != 0
        or ERROR_SVG_PATTERN.search(svg)
        or not svg.startswith("<svg")
    ):
        if not diagnostic:
            error_line = next(
                (line.strip() for line in svg.splitlines() if "Syntax Error" in line),
                "",
            )
            diagnostic = (
                error_line or f"PlantUML exited with status {result.returncode}"
            )
        raise DiagramError(diagnostic)

    return XML_PROLOG_PATTERN.sub("", svg, count=1)
