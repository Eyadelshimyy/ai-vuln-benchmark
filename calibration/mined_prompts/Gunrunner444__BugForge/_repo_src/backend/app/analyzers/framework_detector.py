from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.analyzers.framework_registry import FrameworkSpec, get_framework_registry


@dataclass
class FrameworkInfo:
    name: str
    language: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


class FrameworkDetector:
    """Detect application frameworks from files and manifests."""

    def detect(self, repo_root: Path, all_file_paths: list[Path]) -> list[FrameworkInfo]:
        relative_paths = {
            p.relative_to(repo_root) for p in all_file_paths if p.is_relative_to(repo_root)
        }
        file_names = {p.name for p in relative_paths}
        relative_strs = {str(p) for p in relative_paths}
        dir_names = {str(part) for p in relative_paths for part in p.parts[:-1]}

        results: list[FrameworkInfo] = []
        for spec in get_framework_registry().all_specs():
            info = self._check_framework(spec, repo_root, file_names, dir_names, relative_strs)
            if info:
                results.append(info)
        return results

    def _check_framework(
        self,
        spec: FrameworkSpec,
        repo_root: Path,
        file_names: set[str],
        dir_names: set[str],
        relative_strs: set[str],
    ) -> FrameworkInfo | None:
        evidence: list[str] = []
        score = 0.0

        for fname in spec.indicator_files:
            if fname in file_names or fname in relative_strs:
                evidence.append(fname)
                score += 0.4

        for dname in spec.indicator_dirs:
            if dname in dir_names or any(rel.startswith(f"{dname}/") for rel in relative_strs):
                evidence.append(f"{dname}/")
                score += 0.2

        config_files = spec.config_files or (
            "pyproject.toml",
            "requirements.txt",
            "requirements-dev.txt",
            "setup.cfg",
        )
        if spec.config_patterns:
            for config_file in config_files:
                config_path = repo_root / config_file
                if not config_path.exists():
                    continue
                try:
                    content = config_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for pattern in spec.config_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        evidence.append(config_file)
                        score += 0.4
                        break

        for manifest, patterns in spec.manifests.items():
            path = repo_root / manifest
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for pattern in patterns:
                if re.search(pattern, content):
                    evidence.append(manifest)
                    score += 0.5
                    break

        if score > 0:
            return FrameworkInfo(
                name=spec.name,
                language=spec.language,
                confidence=min(score, 1.0),
                evidence=evidence,
            )
        return None
