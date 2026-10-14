import ast
from collections import OrderedDict
from pathlib import Path


def extract_test_names(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(node.name)
    return names


_FEATURE_AREA_MAP = {
    "test_analyze_endpoint": "Sentiment Analysis",
    "test_auth_endpoints": "Authentication",
    "test_news_comparison_endpoint": "News Comparison",
    "test_workflow": "Analysis Pipeline",
    "test_app_flow": "App Experience",
    "test_login_page": "Login",
}


def feature_area_from_filename(stem: str) -> str:
    stem = Path(stem).stem
    if stem in _FEATURE_AREA_MAP:
        return _FEATURE_AREA_MAP[stem]
    return stem.removeprefix("test_").replace("_", " ").title()


def humanize_name(test_name: str) -> str:
    s = test_name.removeprefix("test_").replace("_", " ")
    return s[0].upper() + s[1:] if s else s


def render_features_markdown(features: OrderedDict[str, list[str]]) -> str:
    lines = [
        "# QSent Features",
        "",
        "> Auto-generated from the integration and e2e test suite. Each section reflects a product feature area; each item is a verified behavior.",
    ]
    for area, behaviors in features.items():
        lines.append("")
        lines.append(f"## {area}")
        lines.append("")
        for behavior in behaviors:
            lines.append(f"- {behavior}")
    lines.append("")
    return "\n".join(lines)


def collect_features(tests_dir: Path) -> OrderedDict[str, list[str]]:
    result: OrderedDict[str, list[str]] = OrderedDict()
    candidate_dirs = [tests_dir / "e2e", tests_dir / "integration"]
    files = sorted(
        f
        for subdir in candidate_dirs
        if subdir.exists()
        for f in subdir.glob("test_*.py")
    )
    for file_path in files:
        names = extract_test_names(file_path)
        if not names:
            continue
        area = feature_area_from_filename(file_path.stem)
        result.setdefault(area, []).extend(humanize_name(n) for n in names)
    return result


def check_docs(docs_path: Path, generated: str) -> bool:
    if not docs_path.exists():
        return False
    return docs_path.read_text().replace("\r\n", "\n") == generated


if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).parent.parent
    features = collect_features(repo_root / "tests")
    generated = render_features_markdown(features)
    docs_path = repo_root / "docs" / "FEATURES.md"

    if "--check" in sys.argv:
        if check_docs(docs_path, generated):
            print("docs/FEATURES.md is up to date.")
            sys.exit(0)
        else:
            print("docs/FEATURES.md is missing or out of date. Run: python scripts/generate_features.py")
            sys.exit(1)
    else:
        docs_path.write_text(generated)
        print("docs/FEATURES.md written.")
        sys.exit(0)
