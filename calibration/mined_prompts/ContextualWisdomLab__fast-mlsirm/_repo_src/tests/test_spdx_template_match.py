"""Synthetic tests for tools/spdx_template_match.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "spdx_template_match.py"
_spec = importlib.util.spec_from_file_location("spdx_template_match", SCRIPT)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

# Same shape as the SPDX MIT template: an unbounded copyright var before the grant.
TPL = ('<<beginOptional>>MIT License\n\n<<endOptional>> <<var;name="copyright";original="Copyright (c) <year> <holders>";'
       'match=".{0,5000}">>\nPermission is granted, subject to keeping this notice.\n'
       'IN NO EVENT SHALL <<var;name="holder";original="THE AUTHORS";match=".+">> BE LIABLE.')
BODY = "Permission is granted, subject to keeping this notice.\nIN NO EVENT SHALL THE AUTHORS BE LIABLE."


def test_copyright_line_and_optional_title_match():
    spans = M.match("MIT License\n\nCopyright (c) 2020 Example\n" + BODY, TPL)
    assert spans == [{"span": "optional", "template_original": "", "text": "MIT License"},
                     {"span": "var:copyright", "template_original": "Copyright (c) <year> <holders>",
                      "text": "Copyright (c) 2020 Example"}]


def test_copyright_var_cannot_absorb_another_license():
    """Defect: the loose match lets '.{0,5000}' swallow a whole preceding license; strict rejects it."""
    text = ("Redistribution and use in source and binary forms are permitted provided that the\n"
            "following conditions are met: do not use the name of the author.\n"
            "Copyright (c) 2020 Example\n" + BODY)
    assert M.match(text, TPL, strict=False) is not None   # pre-fix behaviour
    assert M.match(text, TPL) is None


def test_unbounded_holder_var_is_capped():
    long_holder = "THE AUTHORS " + "AND ALSO YOU MUST NOT SELL THIS SOFTWARE " * 10
    text = "Copyright (c) 2020 Example\n" + BODY.replace("THE AUTHORS", long_holder)
    assert M.match(text, TPL, strict=False) is not None   # pre-fix behaviour
    assert M.match(text, TPL) is None
    assert M.match(text.replace(long_holder, "THE COPYRIGHT OWNER"), TPL) is not None


def test_changed_or_extra_condition_never_matches():
    assert M.match("Copyright (c) 2020 Example\n" + BODY.replace("keeping", "removing"), TPL) is None
    assert M.match("Copyright (c) 2020 Example\n" + BODY + "\nNo commercial use.", TPL) is None


def test_very_long_text_is_rejected_quickly():
    assert M.match("Copyright (c) 2020 Example\n" + "word " * 50000 + BODY, TPL) is None


def test_comment_indicators_and_leading_title_are_ignored():
    """SPDX Matching Guidelines: ignore code comment markers and a license title at the start."""
    commented = "/*\n * Example Project License\n * Copyright (c) 2020 Example\n *\n" + \
        "".join(f" * {line}\n" for line in BODY.splitlines()) + " */\n"
    assert M.match(commented, TPL) is not None
    assert M.match(commented.replace("keeping", "removing"), TPL) is None


def test_tex_style_quotes_are_equivalent():
    tpl = '<<var;name="copyright";original="x";match=".{0,50}">>\nPROVIDED "AS IS" ONLY.'
    assert M.match("Copyright 2020 X\nPROVIDED ``AS IS'' ONLY.", tpl) is not None
    assert M.match("Copyright 2020 X\nPROVIDED ``AS WAS'' ONLY.", tpl) is None
