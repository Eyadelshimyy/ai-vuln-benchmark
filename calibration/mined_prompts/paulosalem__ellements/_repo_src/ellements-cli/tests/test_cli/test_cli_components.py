"""Unit tests for ellements.cli building blocks and CliPrinter."""

from io import StringIO

from ellements.cli import (
    CliPrinter,
    app_header,
    chat_input_prompt,
    chat_message,
    error,
    issue_line,
    issues_block,
    key_value_table,
    result_markdown,
    section_rule,
    stats_footer,
    step_log,
    success,
    tool_use_badge,
    warning,
    welcome_banner,
)
from rich.console import Console


def _capture(fn, *args, **kwargs) -> str:
    """Run *fn* with a captured Console and return the plain-text output."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    fn(*args, console=console, **kwargs)
    return buf.getvalue()


# ── Building blocks ───────────────────────────────────────────────────


class TestAppHeader:
    def test_renders_title(self):
        out = _capture(app_header, "My App")
        assert "My App" in out

    def test_renders_subtitle_items(self):
        out = _capture(app_header, "App", {"Model": "gpt-4o", "File": "x.md"})
        assert "Model" in out
        assert "gpt-4o" in out
        assert "File" in out

    def test_renders_icon(self):
        out = _capture(app_header, "App", icon="🚀")
        assert "🚀" in out


class TestKeyValueTable:
    def test_renders_keys_and_values(self):
        out = _capture(key_value_table, {"name": "Alice", "age": "30"})
        assert "name" in out
        assert "Alice" in out

    def test_boolean_formatting(self):
        out = _capture(key_value_table, {"flag": True, "other": False})
        assert "true" in out
        assert "false" in out

    def test_empty_dict_no_output(self):
        out = _capture(key_value_table, {})
        assert out.strip() == ""

    def test_custom_title(self):
        out = _capture(key_value_table, {"k": "v"}, title="Config")
        assert "Config" in out


class TestSectionRule:
    def test_renders_title(self):
        out = _capture(section_rule, "Results")
        assert "Results" in out


class TestStatsFooter:
    def test_renders_stats(self):
        out = _capture(stats_footer, {"Calls": 5, "Time": "2.1s"})
        assert "Calls" in out
        assert "5" in out
        assert "2.1s" in out


class TestIssueLine:
    def test_warning(self):
        out = _capture(issue_line, "warning", "missing var")
        assert "WARNING" in out
        assert "missing var" in out

    def test_error(self):
        out = _capture(issue_line, "error", "bad file")
        assert "ERROR" in out

    def test_suggestion(self):
        out = _capture(issue_line, "suggestion", "try X")
        assert "SUGGESTION" in out


class TestIssuesBlock:
    def test_renders_multiple_issues(self):
        issues = [
            {"type": "warning", "message": "msg1"},
            {"type": "error", "message": "msg2"},
        ]
        out = _capture(issues_block, issues)
        assert "msg1" in out
        assert "msg2" in out

    def test_empty_issues_no_output(self):
        out = _capture(issues_block, [])
        assert out.strip() == ""


class TestStatusHelpers:
    def test_success(self):
        out = _capture(success, "all good")
        assert "✓" in out
        assert "all good" in out

    def test_error(self):
        out = _capture(error, "failed")
        assert "✗" in out
        assert "failed" in out

    def test_warning(self):
        out = _capture(warning, "careful")
        assert "⚠" in out
        assert "careful" in out


class TestStepLog:
    def test_mid_position(self):
        out = _capture(step_log, "doing stuff")
        assert "├─" in out
        assert "doing stuff" in out

    def test_last_position(self):
        out = _capture(step_log, "final", position="last")
        assert "└─" in out

    def test_cont_position(self):
        out = _capture(step_log, "more", position="cont")
        assert "│" in out


class TestResultMarkdown:
    def test_renders_markdown_content(self):
        buf_stderr = StringIO()
        buf_stdout = StringIO()
        c_err = Console(file=buf_stderr, force_terminal=True, width=120)
        c_out = Console(file=buf_stdout, force_terminal=True, width=120)
        result_markdown("# Hello World", console=c_err, output_console=c_out)
        assert "Hello World" in buf_stdout.getvalue()
        # Section rule goes to stderr
        assert "Result" in buf_stderr.getvalue()


# ── CliPrinter ────────────────────────────────────────────────────────


def _make_printer(verbose: bool = False) -> tuple[CliPrinter, StringIO, StringIO]:
    """Create a CliPrinter with captured consoles."""
    buf_err = StringIO()
    buf_out = StringIO()
    printer = CliPrinter("TestApp", icon="🧪", verbose=verbose)
    printer.console = Console(file=buf_err, force_terminal=True, width=120)
    printer.output_console = Console(file=buf_out, force_terminal=True, width=120)
    return printer, buf_err, buf_out


class TestCliPrinter:
    def test_header(self):
        printer, buf, _ = _make_printer()
        printer.header({"Model": "gpt-4o"})
        out = buf.getvalue()
        assert "TestApp" in out
        assert "🧪" in out
        assert "Model" in out

    def test_variables(self):
        printer, buf, _ = _make_printer()
        printer.variables({"topic": "cats"})
        assert "topic" in buf.getvalue()
        assert "cats" in buf.getvalue()

    def test_status(self):
        printer, buf, _ = _make_printer()
        printer.status("Working…")
        assert "Working" in buf.getvalue()

    def test_event_silent_without_renderer(self):
        printer, buf, _ = _make_printer(verbose=True)
        printer.event("composing", {"model": "gpt-4o"})
        assert buf.getvalue().strip() == ""

    def test_event_silent_when_not_verbose_even_with_renderer(self):
        calls: list[tuple[str, dict]] = []

        class _Renderer:
            def render(self, event_type, data, console):
                calls.append((event_type, dict(data)))

        printer, buf, _ = _make_printer(verbose=False)
        printer.event_renderer = _Renderer()
        printer.event("composing", {"model": "gpt-4o"})
        assert calls == []
        assert buf.getvalue().strip() == ""

    def test_event_dispatches_to_plugin_renderer_when_verbose(self):
        calls: list[tuple[str, dict]] = []

        class _Renderer:
            def render(self, event_type, data, console):
                console.print(f"EV[{event_type}]={data['name']}")
                calls.append((event_type, dict(data)))

        printer, buf, _ = _make_printer(verbose=True)
        printer.event_renderer = _Renderer()
        printer.event("tool_call", {"name": "read_file"})
        assert calls == [("tool_call", {"name": "read_file"})]
        assert "read_file" in buf.getvalue()

    def test_result_markdown(self):
        printer, buf_err, buf_out = _make_printer()
        printer.result_markdown("# Hello")
        assert "Hello" in buf_out.getvalue()

    def test_result_json(self):
        printer, _, buf_out = _make_printer()
        printer.result_json({"key": "value"})
        out = buf_out.getvalue()
        assert "key" in out
        assert "value" in out

    def test_issues(self):
        printer, buf, _ = _make_printer()
        printer.issues([{"type": "warning", "message": "watch out"}])
        assert "watch out" in buf.getvalue()

    def test_stats(self):
        printer, buf, _ = _make_printer()
        printer.stats({"Calls": 5}, elapsed=1.234)
        out = buf.getvalue()
        assert "5" in out
        assert "1.2s" in out

    def test_file_written(self):
        printer, buf, _ = _make_printer()
        printer.file_written("/tmp/out.md")
        out = buf.getvalue()
        assert "out.md" in out
        assert "✓" in out

    def test_done(self):
        printer, buf, _ = _make_printer()
        printer.done()
        assert "Done" in buf.getvalue()

    def test_success_error_warning(self):
        printer, buf, _ = _make_printer()
        printer.success("ok")
        printer.error("bad")
        printer.warning("hmm")
        out = buf.getvalue()
        assert "ok" in out
        assert "bad" in out
        assert "hmm" in out


# ── Chat components ───────────────────────────────────────────────────


class TestChatMessage:
    def test_user_message(self):
        out = _capture(chat_message, "user", "Hello there")
        assert "You" in out
        assert "Hello there" in out

    def test_assistant_message_to_stdout(self):
        buf_err = StringIO()
        buf_out = StringIO()
        c_err = Console(file=buf_err, force_terminal=True, width=120)
        c_out = Console(file=buf_out, force_terminal=True, width=120)
        chat_message("assistant", "# Response", console=c_err, output_console=c_out)
        # Panel with content goes to stdout
        assert "Response" in buf_out.getvalue()
        # Label "Assistant" appears in stdout panel title
        assert "Assistant" in buf_out.getvalue()

    def test_system_message(self):
        out = _capture(chat_message, "system", "System notice")
        assert "System" in out
        assert "System notice" in out

    def test_custom_label(self):
        out = _capture(chat_message, "user", "Hi", label="Agent")
        assert "Agent" in out


class TestChatInputPrompt:
    def test_default(self):
        prompt = chat_input_prompt()
        assert "You" in prompt
        assert "›" in prompt

    def test_custom_label(self):
        prompt = chat_input_prompt("Human")
        assert "Human" in prompt


class TestToolUseBadge:
    def test_renders_tool_name(self):
        out = _capture(tool_use_badge, "stock_lookup")
        assert "stock_lookup" in out
        assert "🔧" in out

    def test_renders_custom_tool(self):
        out = _capture(tool_use_badge, "web_search")
        assert "web_search" in out


class TestWelcomeBanner:
    def test_basic(self):
        out = _capture(welcome_banner, "My App")
        assert "My App" in out

    def test_with_subtitle(self):
        out = _capture(welcome_banner, "My App", {"Model": "gpt-4o"})
        assert "Model" in out
        assert "gpt-4o" in out

    def test_with_commands(self):
        out = _capture(
            welcome_banner, "My App",
            commands={"quit": "Exit", "/help": "Show help"},
        )
        assert "quit" in out
        assert "Exit" in out
        assert "Commands" in out

    def test_with_icon(self):
        out = _capture(welcome_banner, "My App", icon="🤖")
        assert "🤖" in out


# ── CliPrinter chat methods ──────────────────────────────────────────


class TestCliPrinterChat:
    def test_welcome(self):
        printer, buf, _ = _make_printer()
        printer.welcome(
            {"Model": "sonnet"},
            commands={"quit": "Exit"},
        )
        out = buf.getvalue()
        assert "TestApp" in out
        assert "Model" in out
        assert "quit" in out

    def test_user_message(self):
        printer, buf, _ = _make_printer()
        printer.user_message("Hello")
        assert "Hello" in buf.getvalue()
        assert "You" in buf.getvalue()

    def test_assistant_message(self):
        printer, _, buf_out = _make_printer()
        printer.assistant_message("# Reply")
        out = buf_out.getvalue()
        assert "Reply" in out
        assert "Assistant" in out

    def test_tool_badge(self):
        printer, buf, _ = _make_printer()
        printer.tool_badge("web_search")
        out = buf.getvalue()
        assert "web_search" in out
        assert "🔧" in out

    def test_input_prompt(self):
        printer, _, _ = _make_printer()
        prompt = printer.input_prompt()
        assert "You" in prompt
        assert "›" in prompt

    def test_thinking(self):
        printer, buf, _ = _make_printer()
        # Just verify it's a context manager that doesn't crash
        with printer.thinking("Processing…"):
            pass

    def test_goodbye(self):
        printer, buf, _ = _make_printer()
        printer.goodbye()
        assert "Goodbye" in buf.getvalue()

    def test_goodbye_custom(self):
        printer, buf, _ = _make_printer()
        printer.goodbye("See you later!")
        assert "See you later" in buf.getvalue()
