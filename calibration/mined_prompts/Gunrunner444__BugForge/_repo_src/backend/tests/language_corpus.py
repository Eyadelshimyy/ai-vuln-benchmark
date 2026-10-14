"""Regression corpus for the profile-driven parser and taint engine.

Each case is a minimal reproducer for a parser/security behavior that must
not regress. Cases document both expected recognitions and forbidden
false positives.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusCase:
    name: str
    language: str
    filename: str
    source: str
    expect_calls: tuple[str, ...] = ()
    unexpected_calls: tuple[str, ...] = ()
    expect_bindings: tuple[str, ...] = ()
    expect_entities: tuple[str, ...] = ()
    expect_imports: tuple[str, ...] = ()
    notes: str = ""


PARSER_CORPUS: tuple[CorpusCase, ...] = (
    CorpusCase(
        name="string_literal_eval_is_not_a_call",
        language="javascript",
        filename="str.js",
        source='const msg = "eval(userInput)";\n',
        unexpected_calls=("eval",),
        notes="Text inside quotes must never become a CallSite.",
    ),
    CorpusCase(
        name="comment_eval_is_not_a_call",
        language="javascript",
        filename="cmt.js",
        source="// eval(userInput)\nconst x = 1;\n",
        unexpected_calls=("eval",),
    ),
    CorpusCase(
        name="template_interpolation_is_code",
        language="javascript",
        filename="tpl.js",
        source="const s = `hi ${eval(user)}`;\n",
        expect_calls=("eval",),
        notes="`${}` interpolations are executable and must be scanned.",
    ),
    CorpusCase(
        name="escaped_quotes_in_call_args",
        language="javascript",
        filename="esc.js",
        source=r"""run("say \"hello(\")")""" + "\n",
        expect_calls=("run",),
        unexpected_calls=("hello",),
    ),
    CorpusCase(
        name="nested_parens_and_strings",
        language="javascript",
        filename="nest.js",
        source='outer(inner("a(b)"), extra());\n',
        expect_calls=("outer", "inner", "extra"),
    ),
    CorpusCase(
        name="optional_chaining_call",
        language="javascript",
        filename="opt.js",
        source="obj?.method(user);\n",
        expect_calls=("method",),
    ),
    CorpusCase(
        name="multiline_import",
        language="javascript",
        filename="imp.js",
        source="import {\n  foo,\n  bar as baz\n} from 'mod';\n",
        expect_imports=("mod",),
    ),
    CorpusCase(
        name="c_brace_on_next_line",
        language="c",
        filename="fn.c",
        source='#include <stdio.h>\nvoid run(void)\n{\n    puts("x");\n}\n',
        expect_entities=("run",),
        expect_imports=("stdio.h",),
        expect_calls=("puts",),
    ),
    CorpusCase(
        name="member_assignment_is_sink_shaped",
        language="javascript",
        filename="dom.js",
        source="el.innerHTML = user;\n",
        expect_calls=("innerHTML",),
    ),
    CorpusCase(
        name="constructor_new_function",
        language="javascript",
        filename="nf.js",
        source="const f = new Function(userInput);\n",
        expect_calls=("Function",),
    ),
    CorpusCase(
        name="if_is_not_a_call",
        language="javascript",
        filename="if.js",
        source="if (ready) { run(x); }\n",
        expect_calls=("run",),
        unexpected_calls=("if",),
    ),
    CorpusCase(
        name="destructure_from_req",
        language="javascript",
        filename="d.js",
        source="const { query } = req;\n",
        expect_bindings=("query",),
    ),
    CorpusCase(
        name="php_hash_comment_not_call",
        language="php",
        filename="c.php",
        source="<?php\n# eval($x);\n$y = 1;\n",
        unexpected_calls=("eval",),
    ),
    CorpusCase(
        name="java_runtime_exec_chain",
        language="java",
        filename="R.java",
        source="class R { void t(String cmd) { Runtime.getRuntime().exec(cmd); } }\n",
        expect_calls=("exec",),
    ),
    CorpusCase(
        name="go_import_block",
        language="go",
        filename="m.go",
        source='package m\nimport (\n    "fmt"\n    "os/exec"\n)\n',
        expect_imports=("fmt", "os/exec"),
    ),
)
