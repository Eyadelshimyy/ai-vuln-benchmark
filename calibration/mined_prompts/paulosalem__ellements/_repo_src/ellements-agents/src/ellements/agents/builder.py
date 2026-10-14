"""Fluent builder for assembling agents on top of any :class:`AgentBackend`.

The builder unifies three concerns:

1. **Instructions** — direct text, a file, or a rendered template.
2. **Persona + Guideline** — composed through
   :class:`ellements.core.PromptContext` + :class:`PersonaLibrary` /
   :class:`GuidelineLibrary`.
3. **Tools** — accumulated in a :class:`~ellements.core.ToolRegistry`
   and forwarded to the backend, which adapts them to its native
   tool format.

There are **no** alias methods, **no** ``run_sync`` flavours, **no**
``.mustache.md`` legacy branch, and **no** lazy circular-import hacks.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ellements.core import (
    GuidelineLibrary,
    PersonaLibrary,
    ToolRegistry,
)
from ellements.core.prompting import PromptContext
from ellements.core.templating import render_prompt_template

from .backend import AgentBackend


class AgentBuilder:
    """Compose an agent backend with tools, instructions, and prompt context.

    Example:
        >>> from ellements.agents import AgentBuilder, OpenAIAgentsBackend
        >>> agent = (
        ...     AgentBuilder("YouTubeExpert", backend=OpenAIAgentsBackend())
        ...     .with_instructions("Find the best YouTube videos")
        ...     .with_model("gpt-4o-mini")
        ...     .with_tools(youtube_search_tools())
        ...     .build()
        ... )
    """

    def __init__(self, name: str, *, backend: AgentBackend) -> None:
        self.name = name
        self.backend = backend
        self.model: str | None = None
        self.tools = ToolRegistry()
        self.instructions: str | None = None
        self.instruction_template: str | None = None
        self.template_renderer: Callable[..., str] | None = None
        self.prompt_context = PromptContext()

    # ── Identity / model ───────────────────────────────────────────

    def with_model(self, model: str) -> AgentBuilder:
        self.model = model
        return self

    # ── Tools ──────────────────────────────────────────────────────

    def with_tools(self, tools: ToolRegistry | Mapping[str, Any]) -> AgentBuilder:
        """Bulk-add tools from a :class:`ToolRegistry` or ``{name: callable}`` mapping."""
        for name, tool in tools.items():
            self.tools.register(tool, name=name)
        return self

    def with_tool(self, name: str, tool: Any) -> AgentBuilder:
        """Add a single tool under *name*."""
        self.tools.register(tool, name=name)
        return self

    # ── Instructions ───────────────────────────────────────────────

    def with_instructions(self, instructions: str) -> AgentBuilder:
        """Set inline instructions text."""
        self.instructions = instructions
        return self

    def with_instructions_from_file(
        self, path: Path | str, *, fallback: str | None = None
    ) -> AgentBuilder:
        """Load instructions from *path*.

        Raises:
            FileNotFoundError: When the file is missing and no
                ``fallback`` is provided.
        """
        try:
            self.instructions = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            if fallback is None:
                raise
            self.instructions = fallback
        return self

    def with_instructions_template(
        self,
        template_name: str,
        *,
        renderer: Callable[..., str] | None = None,
    ) -> AgentBuilder:
        """Render instructions from a Mustache template.

        For split templates the builder loads
        ``{template_name}.system.mustache.md`` for instructions and
        ``{template_name}.user.mustache.md`` (when present) for the
        guideline message returned by
        :meth:`get_guideline_user_message`.

        Args:
            template_name: Template basename. May be qualified with a
                package-relative path (e.g. ``"writing/poem"``).
            renderer: Override the default ``render_prompt_template``
                helper (which loads from this package's ``prompts/``).
        """
        self.instruction_template = template_name
        self.template_renderer = renderer or _default_template_renderer
        return self

    # ── Persona ────────────────────────────────────────────────────

    def with_persona(self, persona_folder: Path | str) -> AgentBuilder:
        """Attach a :class:`PersonaLibrary` loaded from *persona_folder*."""
        self.prompt_context.attach_persona_library(
            PersonaLibrary(persona_folder), fallback_to_first=True
        )
        return self

    def with_persona_id(self, persona_id: str) -> AgentBuilder:
        """Select an active persona by its registered ID."""
        if self.prompt_context.persona_library is None:
            raise ValueError(
                "Persona library not attached. Call .with_persona(folder) first."
            )
        self.prompt_context.set_persona_by_id(persona_id)
        return self

    def with_persona_data(self, persona_data: dict[str, Any]) -> AgentBuilder:
        """Activate a persona from inline data (no library required)."""
        self.prompt_context.set_persona_from_data(persona_data)
        return self

    def with_persona_from_path(self, path: Path | str) -> AgentBuilder:
        """Activate a persona from an arbitrary file path."""
        self.prompt_context.set_persona_from_path(path)
        return self

    # ── Guideline ──────────────────────────────────────────────────

    def with_guideline(self, guideline_folder: Path | str) -> AgentBuilder:
        """Attach a :class:`GuidelineLibrary` loaded from *guideline_folder*."""
        self.prompt_context.attach_guideline_library(
            GuidelineLibrary(guideline_folder)
        )
        return self

    def with_guideline_id(self, guideline_id: str) -> AgentBuilder:
        """Select an active guideline by its registered ID."""
        if self.prompt_context.guideline_library is None:
            raise ValueError(
                "Guideline library not attached. Call .with_guideline(folder) first."
            )
        self.prompt_context.set_guideline_by_id(guideline_id)
        return self

    def with_guideline_text(self, text: str) -> AgentBuilder:
        """Activate a guideline from inline text."""
        self.prompt_context.set_guideline_from_text(text)
        return self

    def with_guideline_from_path(self, path: Path | str) -> AgentBuilder:
        """Activate a guideline from an arbitrary file path."""
        self.prompt_context.set_guideline_from_path(path)
        return self

    def clear_guideline(self) -> None:
        """Deactivate the currently selected guideline."""
        self.prompt_context.clear_guideline()

    # ── Convenience read-out ───────────────────────────────────────

    @property
    def current_persona_id(self) -> str | None:
        return self.prompt_context.current_persona_id

    @property
    def current_guideline_id(self) -> str | None:
        return self.prompt_context.current_guideline_id

    def get_guideline_user_message(self) -> str | None:
        """Return the rendered guideline user-message, if available.

        Only meaningful when both a guideline is active and the
        instructions are template-driven with a sibling
        ``{template}.user.mustache.md`` file.
        """
        guideline_text = self.prompt_context.get_guideline_text()
        if not guideline_text or self.instruction_template is None:
            return None
        if self.template_renderer is None:
            return None
        user_template = f"{self.instruction_template}.user.mustache.md"
        try:
            rendered = self.template_renderer(
                user_template, guideline=guideline_text
            )
        except FileNotFoundError:
            return None
        return rendered.strip() or None

    # ── Build ──────────────────────────────────────────────────────

    def build(self) -> Any:
        """Materialize the agent through the configured backend."""
        if self.model is None:
            raise ValueError(
                "Model must be set before building. Call .with_model(...)"
            )

        instructions = self._resolve_instructions()
        return self.backend.create_agent(
            name=self.name,
            instructions=instructions,
            tools=self.tools,
            model=self.model,
        )

    async def run(self, task: str, *, max_turns: int = 10, **kwargs: Any) -> Any:
        """Build and run the agent on *task* using the backend."""
        agent = self.build()
        return await self.backend.run(
            agent, task, max_turns=max_turns, **kwargs
        )

    # ── Internals ──────────────────────────────────────────────────

    def _resolve_instructions(self) -> str:
        if self.instruction_template is not None:
            if self.template_renderer is None:
                raise ValueError("Template renderer is not set")
            persona_text = self.prompt_context.get_persona_text()
            if persona_text is None:
                raise ValueError(
                    "Template-mode instructions require an active persona. "
                    "Call .with_persona(...) and select a persona first."
                )
            system_template = f"{self.instruction_template}.system.mustache.md"
            return self.template_renderer(
                system_template, persona=persona_text, guideline=""
            )
        if self.instructions is not None:
            return self.instructions
        raise ValueError(
            "Instructions are not set. Use .with_instructions(...) "
            "or .with_instructions_template(...)."
        )


# ── Module-level helpers ───────────────────────────────────────────


def _default_template_renderer(template_name: str, /, **context: Any) -> str:
    """Default template renderer rooted at this package's ``prompts/``."""
    return render_prompt_template(
        template_name,
        package="ellements.agents",
        resource_root="prompts",
        **context,
    )


def check_openai_api_key() -> bool:
    """Return whether ``OPENAI_API_KEY`` is set; print a hint when not."""
    if os.getenv("OPENAI_API_KEY"):
        return True
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("  export OPENAI_API_KEY='your-api-key'")
    return False


def load_prompt_file(
    path: Path | str, *, fallback: str | None = None
) -> str:
    """Load a prompt file with an optional fallback string."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        if fallback is None:
            raise
        return fallback


__all__ = [
    "AgentBuilder",
    "check_openai_api_key",
    "load_prompt_file",
]
