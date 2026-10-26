"""Selector resilience helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from playwright.sync_api import Locator, Page

T = TypeVar("T")

LocatorFactory = Callable[[Page], Locator]


class SelectorResolutionError(Exception):
    def __init__(self, message: str, *, url: str, debug_path: str | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.debug_path = debug_path


def click_first_available(
    page: Page,
    factories: Sequence[LocatorFactory],
    *,
    description: str,
) -> None:
    errors: list[str] = []
    for factory in factories:
        locator = factory(page)
        try:
            if locator.count() == 0:
                errors.append("count=0")
                continue
            locator.first.click(timeout=5_000)
            return
        except Exception as exc:  # noqa: BLE001 — try next selector
            errors.append(str(exc))
    msg = (
        f"UI appears to have changed.\nExpected:\n{description}\nURL:\n{page.url}\n"
        f"Attempts:\n" + "\n".join(errors)
    )
    raise SelectorResolutionError(msg, url=page.url)


def first_text(page: Page, factories: Sequence[LocatorFactory]) -> str | None:
    for factory in factories:
        locator = factory(page)
        try:
            if locator.count() == 0:
                continue
            text = locator.first.inner_text(timeout=3_000).strip()
            if text:
                return text
        except Exception:  # noqa: BLE001
            continue
    return None
