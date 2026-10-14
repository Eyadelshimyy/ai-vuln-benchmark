"""Framework detection registry.

Specs live here rather than as language-specific branches in the security
engine. New frameworks are added by registering a :class:`FrameworkSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FrameworkSpec:
    name: str
    language: str
    indicator_files: tuple[str, ...] = ()
    indicator_dirs: tuple[str, ...] = ()
    config_patterns: tuple[str, ...] = ()
    config_files: tuple[str, ...] = ()
    manifests: dict[str, tuple[str, ...]] = field(default_factory=dict)
    route_receivers: tuple[str, ...] = ()
    constructors: tuple[str, ...] = ()
    constructor_modules: tuple[str, ...] = ()


_PYTHON_CONFIG = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.cfg",
    "Pipfile",
)


DEFAULT_FRAMEWORKS: tuple[FrameworkSpec, ...] = (
    FrameworkSpec(
        name="django",
        language="python",
        indicator_files=("manage.py",),
        config_patterns=(r"django",),
        config_files=_PYTHON_CONFIG,
    ),
    FrameworkSpec(
        name="flask",
        language="python",
        config_patterns=(r"flask",),
        config_files=_PYTHON_CONFIG,
        constructors=("Flask",),
        constructor_modules=("flask",),
    ),
    FrameworkSpec(
        name="fastapi",
        language="python",
        config_patterns=(r"fastapi",),
        config_files=_PYTHON_CONFIG,
        constructors=("FastAPI",),
        constructor_modules=("fastapi",),
    ),
    FrameworkSpec(
        name="pytest",
        language="python",
        indicator_files=("pytest.ini", "conftest.py"),
        indicator_dirs=("tests", "test"),
        config_patterns=(r"\[tool\.pytest", r"\[pytest\]", r"pytest"),
        config_files=_PYTHON_CONFIG,
    ),
    FrameworkSpec(
        name="sqlalchemy",
        language="python",
        config_patterns=(r"sqlalchemy",),
        config_files=_PYTHON_CONFIG,
    ),
    FrameworkSpec(
        name="pydantic",
        language="python",
        config_patterns=(r"pydantic",),
        config_files=_PYTHON_CONFIG,
    ),
    FrameworkSpec(
        name="next.js",
        language="javascript/typescript",
        indicator_files=("next.config.js", "next.config.ts", "next.config.mjs"),
        manifests={"package.json": (r'"next"',)},
    ),
    FrameworkSpec(
        name="react",
        language="javascript/typescript",
        manifests={"package.json": (r'"react"',)},
    ),
    FrameworkSpec(
        name="express",
        language="javascript",
        manifests={"package.json": (r'"express"',)},
        constructors=("express",),
        constructor_modules=("express",),
    ),
    FrameworkSpec(
        name="nestjs",
        language="javascript/typescript",
        manifests={"package.json": (r'"@nestjs/core"', r'"@nestjs/common"')},
    ),
    FrameworkSpec(
        name="rails",
        language="ruby",
        indicator_files=("config/application.rb", "Rakefile"),
        indicator_dirs=("app/controllers", "config/environments"),
        manifests={"Gemfile": (r"\brails\b", r"gem ['\"]rails['\"]")},
    ),
    FrameworkSpec(
        name="sinatra",
        language="ruby",
        manifests={"Gemfile": (r"\bsinatra\b", r"gem ['\"]sinatra['\"]")},
    ),
    FrameworkSpec(
        name="gin",
        language="go",
        manifests={"go.mod": (r"github.com/gin-gonic/gin",)},
    ),
    FrameworkSpec(
        name="actix",
        language="rust",
        manifests={"Cargo.toml": (r"actix-web",)},
    ),
    FrameworkSpec(
        name="spring",
        language="java",
        indicator_files=("pom.xml", "build.gradle", "build.gradle.kts"),
        config_patterns=(r"springframework", r"spring-boot"),
        config_files=("pom.xml", "build.gradle", "build.gradle.kts"),
    ),
    FrameworkSpec(
        name="laravel",
        language="php",
        indicator_files=("artisan",),
        manifests={"composer.json": (r'"laravel/framework"',)},
    ),
    FrameworkSpec(
        name="ktor",
        language="kotlin",
        manifests={"build.gradle.kts": (r"io.ktor",), "build.gradle": (r"io.ktor",)},
    ),
    FrameworkSpec(
        name="vapor",
        language="swift",
        manifests={"Package.swift": (r"vapor",)},
    ),
    FrameworkSpec(
        name="echo",
        language="go",
        manifests={"go.mod": (r"github.com/labstack/echo",)},
    ),
    FrameworkSpec(
        name="fiber",
        language="go",
        manifests={"go.mod": (r"github.com/gofiber/fiber",)},
    ),
    FrameworkSpec(
        name="axum",
        language="rust",
        manifests={"Cargo.toml": (r"\baxum\b",)},
    ),
    FrameworkSpec(
        name="rocket",
        language="rust",
        manifests={"Cargo.toml": (r"\brocket\b",)},
    ),
    FrameworkSpec(
        name="symfony",
        language="php",
        manifests={"composer.json": (r'"symfony/framework-bundle"', r'"symfony/http-foundation"')},
        indicator_files=("symfony.lock",),
    ),
    FrameworkSpec(
        name="foundry",
        language="solidity",
        indicator_files=("foundry.toml",),
    ),
    FrameworkSpec(
        name="hardhat",
        language="solidity",
        indicator_files=("hardhat.config.js", "hardhat.config.ts"),
    ),
    FrameworkSpec(
        name="servlet",
        language="java",
        config_patterns=(r"javax\.servlet", r"jakarta\.servlet", r"HttpServlet"),
        config_files=("pom.xml", "build.gradle", "build.gradle.kts"),
        indicator_files=("web.xml",),
    ),
)


class FrameworkRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, FrameworkSpec] = {}

    def register(self, spec: FrameworkSpec, *, replace: bool = False) -> None:
        key = spec.name.strip().lower()
        if not key:
            raise ValueError("framework name must be non-empty")
        if key in self._specs and not replace:
            raise ValueError(f"framework {key!r} is already registered")
        self._specs[key] = spec

    def get(self, name: str) -> FrameworkSpec | None:
        return self._specs.get(name.strip().lower())

    def all_specs(self) -> list[FrameworkSpec]:
        return list(self._specs.values())


def default_framework_registry() -> FrameworkRegistry:
    registry = FrameworkRegistry()
    for spec in DEFAULT_FRAMEWORKS:
        registry.register(spec)
    return registry


_REGISTRY: FrameworkRegistry | None = None


def get_framework_registry() -> FrameworkRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = default_framework_registry()
    return _REGISTRY
