"""Get on Board profile HITL client — permanent profile maintainer."""

from __future__ import annotations

import logging
from pathlib import Path

from jobbot.adapters.getonboard.draft import (
    fields_from_seed_text,
    load_permanent_profile,
    permanent_profile_md_path,
    save_permanent_profile,
)
from jobbot.adapters.getonboard.package import (
    PROFILE_EDIT_URL,
    RESUMES_HINT_URL,
    build_getonboard_sync_package,
    render_getonboard_sync_markdown,
)
from jobbot.browser import background
from jobbot.config import JobbotConfig, load_config
from jobbot.nlp.refine import RefineResult, refine_permanent_profile
from jobbot.profile.loader import load_profile

logger = logging.getLogger("jobbot.getonboard")


class GetOnBoardProfileClient:
    """Maintain permanent GoB profile texts + open editors (HITL)."""

    def __init__(self, config: JobbotConfig | None = None) -> None:
        self.config = config or load_config()

    @classmethod
    def from_config(cls, config: JobbotConfig) -> GetOnBoardProfileClient:
        return cls(config)

    def prepare_package(
        self,
        *,
        cold: bool = False,
        use_llm: bool = False,
        seed_experiencia: str | None = None,
        seed_formacion: str | None = None,
    ) -> tuple[Path, RefineResult]:
        """Refine permanent profile (cumulative by default) + sync sheet."""
        candidate = load_profile(self.config.profile_path)
        previous = None if cold else load_permanent_profile(self.config.output_dir)
        if seed_experiencia:
            previous = fields_from_seed_text(
                experiencia=seed_experiencia,
                formacion=seed_formacion or (previous.formacion_academica if previous else ""),
                candidate=candidate,
            )
        result = refine_permanent_profile(
            candidate,
            previous,
            use_llm=use_llm and not cold,
        )
        md_path = save_permanent_profile(result.fields, self.config.output_dir)
        logger.info(
            "Wrote permanent GoB profile (%s, kept=%s added=%s dropped=%s): %s",
            result.mode,
            result.kept_paragraphs,
            result.added_paragraphs,
            result.dropped_paragraphs,
            md_path,
        )

        package = build_getonboard_sync_package(candidate)
        out_dir = self.config.output_dir / "getonboard"
        out_dir.mkdir(parents=True, exist_ok=True)
        sync_path = out_dir / "sync_package.md"
        sync_path.write_text(render_getonboard_sync_markdown(package), encoding="utf-8")
        logger.info("Wrote Get on Board sync package: %s", sync_path)
        return md_path, result

    def show_permanent_paths(self) -> tuple[Path, Path | None]:
        md = permanent_profile_md_path(self.config.output_dir)
        loaded = load_permanent_profile(self.config.output_dir)
        return md, md if loaded is not None and md.is_file() else None

    def open_profile_edit(self) -> str:
        background.open_url(PROFILE_EDIT_URL)
        return PROFILE_EDIT_URL

    def open_resumes(self) -> str:
        background.open_url(RESUMES_HINT_URL)
        return RESUMES_HINT_URL
