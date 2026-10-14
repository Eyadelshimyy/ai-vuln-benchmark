"""Built-in language adapters.

Programming languages with a Tree-sitter (or CPython AST) backend implement
parse, entity/import extraction, scope-aware data flow, and security analysis.
Markup/query formats get specialized analysis. Remaining auxiliaries stay
detection-only.
"""

from __future__ import annotations

from app.adapters.languages.base import LanguageAdapter
from app.adapters.languages.profile import ProfileLanguageAdapter
from app.analysis.quality import quality_rules_for
from app.domain.language import LanguageCapability, ParserTier


class DetectionLanguageAdapter(LanguageAdapter):
    """Shared implementation for languages that can be detected but not analyzed."""

    def __init__(
        self,
        language_id: str,
        display_name: str,
        extensions: frozenset[str],
        *,
        is_source: bool,
    ) -> None:
        self._language_id = language_id
        self._display_name = display_name
        self._extensions = extensions
        caps = {LanguageCapability.DETECTION}
        if is_source:
            caps.add(LanguageCapability.SOURCE)
        self._capabilities = frozenset(caps)

    @property
    def language_id(self) -> str:
        return self._language_id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def file_extensions(self) -> frozenset[str]:
        return self._extensions

    @property
    def capabilities(self) -> frozenset[LanguageCapability]:
        return self._capabilities

    def parser_tier(self) -> ParserTier:
        return ParserTier.DETECTION_ONLY


class JavaScriptAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__(
            "javascript",
            "JavaScript",
            frozenset({".js", ".mjs", ".cjs", ".jsx"}),
            rules=quality_rules_for("javascript"),
        )


class TypeScriptAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__(
            "typescript",
            "TypeScript",
            frozenset({".ts", ".tsx"}),
            rules=quality_rules_for("typescript"),
        )


class RubyAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("ruby", "Ruby", frozenset({".rb"}), rules=quality_rules_for("ruby"))


class CAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("c", "C", frozenset({".c", ".h"}), rules=quality_rules_for("c"))


class CppAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__(
            "cpp",
            "C++",
            frozenset({".cpp", ".cc", ".cxx", ".hpp"}),
            rules=quality_rules_for("cpp"),
        )


class GoAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("go", "Go", frozenset({".go"}), rules=quality_rules_for("go"))


class RustAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("rust", "Rust", frozenset({".rs"}), rules=quality_rules_for("rust"))


class JavaAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("java", "Java", frozenset({".java"}), rules=quality_rules_for("java"))


class PHPAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("php", "PHP", frozenset({".php"}), rules=quality_rules_for("php"))


class KotlinAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__(
            "kotlin", "Kotlin", frozenset({".kt", ".kts"}), rules=quality_rules_for("kotlin")
        )


class SwiftAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("swift", "Swift", frozenset({".swift"}), rules=quality_rules_for("swift"))


class CSharpAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("csharp", "C#", frozenset({".cs"}), rules=quality_rules_for("csharp"))


class SolidityAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__(
            "solidity",
            "Solidity",
            frozenset({".sol"}),
            rules=quality_rules_for("solidity"),
        )

    def analysis_diagnostics(self) -> dict[str, object]:
        from app.adapters.languages.capabilities import capability_matrix, promotion_stage

        report = super().analysis_diagnostics()
        report["capability_matrix"] = capability_matrix(self.language_id)
        report["promotion_stage"] = promotion_stage(self.language_id)
        return report


class ShellAdapter(ProfileLanguageAdapter):
    def __init__(self) -> None:
        super().__init__(
            "shell", "Shell", frozenset({".sh", ".bash", ".zsh"}), rules=quality_rules_for("shell")
        )


class SpecializedLanguageAdapter(ProfileLanguageAdapter):
    """Syntax-aware specialized analysis (not a full application-language taint suite)."""

    def __init__(self, language_id: str, display_name: str, extensions: frozenset[str]) -> None:
        super().__init__(language_id, display_name, extensions, specialized=True)


class HtmlAdapter(SpecializedLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("html", "HTML", frozenset({".html", ".htm"}))


class CssAdapter(SpecializedLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("css", "CSS", frozenset({".css"}))


class ScssAdapter(SpecializedLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("scss", "SCSS", frozenset({".scss", ".sass"}))


class SqlAdapter(SpecializedLanguageAdapter):
    def __init__(self) -> None:
        super().__init__("sql", "SQL", frozenset({".sql"}))


# Config/docs remain detection-only.
AUXILIARY_LANGUAGES: tuple[DetectionLanguageAdapter, ...] = (
    DetectionLanguageAdapter("yaml", "YAML", frozenset({".yaml", ".yml"}), is_source=False),
    DetectionLanguageAdapter("json", "JSON", frozenset({".json"}), is_source=False),
    DetectionLanguageAdapter("toml", "TOML", frozenset({".toml"}), is_source=False),
    DetectionLanguageAdapter("markdown", "Markdown", frozenset({".md"}), is_source=False),
    DetectionLanguageAdapter(
        "restructuredtext",
        "reStructuredText",
        frozenset({".rst"}),
        is_source=False,
    ),
    DetectionLanguageAdapter("r", "R", frozenset({".r"}), is_source=True),
    DetectionLanguageAdapter("scala", "Scala", frozenset({".scala"}), is_source=True),
    DetectionLanguageAdapter("dart", "Dart", frozenset({".dart"}), is_source=True),
    DetectionLanguageAdapter("lua", "Lua", frozenset({".lua"}), is_source=True),
    DetectionLanguageAdapter("elixir", "Elixir", frozenset({".ex", ".exs"}), is_source=True),
    DetectionLanguageAdapter("vyper", "Vyper", frozenset({".vy"}), is_source=True),
)

PROGRAMMING_LANGUAGE_ADAPTERS: tuple[type[LanguageAdapter], ...] = (
    JavaScriptAdapter,
    TypeScriptAdapter,
    RubyAdapter,
    CAdapter,
    CppAdapter,
    GoAdapter,
    RustAdapter,
    JavaAdapter,
    PHPAdapter,
    KotlinAdapter,
    SwiftAdapter,
    CSharpAdapter,
    ShellAdapter,
    SolidityAdapter,
    HtmlAdapter,
    CssAdapter,
    ScssAdapter,
    SqlAdapter,
)
