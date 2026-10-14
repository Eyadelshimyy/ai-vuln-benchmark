"""Smoke: MCP server defines all expected tool handlers."""
import ast
from pathlib import Path

SERVER_PY = Path(__file__).parent.parent.parent / "servers" / "project-hub-server" / "server.py"

EXPECTED_TOOLS = {
    # Session
    "tool_get_session",
    "tool_set_session",
    "tool_clear_session",
    # Projects
    "tool_list_projects",
    "tool_get_project",
    "tool_get_project_by_id",
    "tool_create_project",
    "tool_update_project",
    "tool_delete_project",
    "tool_list_docs",
    "tool_link_project",
    "tool_unlink_project",
    # Contacts
    "tool_list_contacts",
    "tool_add_contact",
    "tool_update_contact",
    "tool_delete_contact",
    # Notes
    "tool_list_notes",
    "tool_get_note",
    "tool_add_note",
    "tool_update_note",
    "tool_delete_note",
    # Attachments
    "tool_attach_file",
    "tool_attach_files",
    "tool_list_attachments",
    "tool_remove_attachment",
    # Search
    "tool_search_notes",
    "tool_search_contacts",
    # Knowledge
    "tool_list_knowledge",
    "tool_get_knowledge",
    "tool_get_all_knowledge",
    "tool_save_knowledge",
    "tool_delete_knowledge",
    # Reports
    "tool_generate_report",
    # Templates
    "tool_get_project_template",
    "tool_create_project_from_template",
}


def _get_defined_functions() -> set[str]:
    tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def test_server_file_exists():
    assert SERVER_PY.exists(), f"server.py not found: {SERVER_PY}"


def test_server_is_valid_python():
    ast.parse(SERVER_PY.read_text(encoding="utf-8"))


def test_all_tools_defined():
    """All expected @mcp.tool() handlers are defined in server.py."""
    defined = _get_defined_functions()
    missing = EXPECTED_TOOLS - defined
    assert not missing, f"Missing tool functions: {sorted(missing)}"


def test_server_module_actually_imports_and_registers_tools():
    """Regression guard (project-hub#123): the tests above only `ast.parse` server.py, which
    would accept an import of a removed class (e.g. `mcp.server.fastmcp.FastMCP`, gone in mcp
    2.0.0) without ever executing it. This test performs the real import, independent of the
    response-wrapping contract tests in test_project_types.py that happen to import `server`
    only as a side effect of testing something else.
    """
    from server import mcp

    tool_names = {tool.name for tool in mcp._tool_manager.list_tools()}
    missing = EXPECTED_TOOLS - tool_names
    assert not missing, f"Registered tools missing at runtime: {sorted(missing)}"


def test_claude_md_has_skill_routing_heading():
    """CLAUDE.md must exist and contain the ## Skill Routing heading."""
    claude_md = Path(__file__).parent.parent.parent / "CLAUDE.md"
    assert claude_md.exists(), "CLAUDE.md not found at repo root"
    assert "## Skill Routing" in claude_md.read_text(encoding="utf-8"), (
        "CLAUDE.md is missing '## Skill Routing' section"
    )
