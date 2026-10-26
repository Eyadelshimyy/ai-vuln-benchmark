"""Inspect page accessibility tree for selector discovery."""

from __future__ import annotations

from playwright.sync_api import Page
from rich.console import Console

_ROLES: tuple[tuple[str, str], ...] = (
    ("heading", "headings"),
    ("button", "buttons"),
    ("link", "links"),
    ("textbox", "inputs"),
)


def inspect_page(page: Page, console: Console | None = None) -> dict[str, list[str]]:
    out = console or Console()
    info: dict[str, list[str]] = {
        "headings": [],
        "buttons": [],
        "links": [],
        "inputs": [],
    }
    for role, key in _ROLES:
        locs = page.get_by_role(role)  # type: ignore[arg-type]
        count = min(locs.count(), 40)
        for i in range(count):
            try:
                text = locs.nth(i).inner_text(timeout=1_000).strip()
            except Exception:  # noqa: BLE001
                text = ""
            if text:
                info[key].append(text[:120])
    out.print(f"URL: {page.url}")
    for key, values in info.items():
        out.print(f"\n[bold]{key}[/bold] ({len(values)})")
        for value in values[:20]:
            out.print(f"  - {value}")
    return info
