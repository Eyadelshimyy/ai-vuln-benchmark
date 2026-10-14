"""Language grammar maps for Tree-sitter → SyntaxGraph extraction.

Maps are data, not core ``if language ==`` branches in the security engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Grammar:
    ts_name: str
    function_types: frozenset[str]
    class_types: frozenset[str] = field(default_factory=frozenset)
    call_types: frozenset[str] = field(default_factory=frozenset)
    assignment_types: frozenset[str] = field(default_factory=frozenset)
    import_types: frozenset[str] = field(default_factory=frozenset)
    comment_types: frozenset[str] = field(default_factory=frozenset)
    string_types: frozenset[str] = field(default_factory=frozenset)
    interpolation_types: frozenset[str] = field(default_factory=frozenset)
    identifier_types: frozenset[str] = field(default_factory=frozenset)
    parameter_container_types: frozenset[str] = field(default_factory=frozenset)
    return_types: frozenset[str] = field(default_factory=frozenset)
    branch_types: frozenset[str] = field(default_factory=frozenset)
    loop_types: frozenset[str] = field(default_factory=frozenset)
    exception_types: frozenset[str] = field(default_factory=frozenset)
    member_types: frozenset[str] = field(default_factory=frozenset)
    constructor_types: frozenset[str] = field(default_factory=frozenset)
    decorator_types: frozenset[str] = field(default_factory=frozenset)
    field_types: frozenset[str] = field(default_factory=frozenset)
    name_fields: tuple[str, ...] = ("name",)
    import_kind: str = "import"
    dollar_idents: bool = False
    block_types: frozenset[str] = field(default_factory=frozenset)
    # function: assignments bind to nearest function. block: current block.
    # mixed: var→function, let/const→block (JavaScript/TypeScript).
    lexical_model: str = "block"
    function_scoped_keywords: frozenset[str] = field(default_factory=frozenset)
    block_scoped_keywords: frozenset[str] = field(default_factory=frozenset)
    quality_unwrap_names: frozenset[str] = field(default_factory=frozenset)
    quality_deprecated_calls: frozenset[str] = field(default_factory=frozenset)
    quality_panic_names: frozenset[str] = field(default_factory=frozenset)
    quality_for_loop_types: frozenset[str] = field(default_factory=frozenset)
    quality_posix_test_types: frozenset[str] = field(default_factory=frozenset)
    quality_force_unwrap_child_types: frozenset[str] = field(default_factory=frozenset)
    quality_unquoted_expansion: bool = False
    quality_blank_ident: bool = False


_COMMON_COMMENT = frozenset({"comment", "line_comment", "block_comment"})
_COMMON_ID = frozenset(
    {
        "identifier",
        "property_identifier",
        "field_identifier",
        "type_identifier",
        "simple_identifier",
        "name",
        "word",
        "variable_name",
        "shorthand_property_identifier",
        "shorthand_property_identifier_pattern",
    }
)

GRAMMARS: dict[str, Grammar] = {
    "python": Grammar(
        ts_name="python",
        function_types=frozenset({"function_definition"}),
        class_types=frozenset({"class_definition"}),
        call_types=frozenset({"call"}),
        assignment_types=frozenset({"assignment", "augmented_assignment", "named_expression"}),
        import_types=frozenset({"import_statement", "import_from_statement"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string", "concatenated_string"}),
        interpolation_types=frozenset({"interpolation"}),
        identifier_types=frozenset({"identifier"}),
        parameter_container_types=frozenset({"parameters"}),
        return_types=frozenset({"return_statement"}),
        branch_types=frozenset({"if_statement", "match_statement"}),
        loop_types=frozenset({"for_statement", "while_statement"}),
        exception_types=frozenset({"try_statement", "except_clause"}),
        member_types=frozenset({"attribute"}),
        decorator_types=frozenset({"decorator"}),
        import_kind="import",
        block_types=frozenset(
            {
                "block",
                "if_statement",
                "for_statement",
                "while_statement",
                "match_statement",
                "except_clause",
                "lambda",
                "list_comprehension",
                "set_comprehension",
                "dictionary_comprehension",
                "generator_expression",
            }
        ),
        lexical_model="function",
    ),
    "javascript": Grammar(
        ts_name="javascript",
        function_types=frozenset(
            {
                "function_declaration",
                "generator_function_declaration",
                "function_expression",
                "arrow_function",
                "method_definition",
                "generator_function",
            }
        ),
        class_types=frozenset({"class_declaration", "class"}),
        call_types=frozenset({"call_expression"}),
        assignment_types=frozenset(
            {
                "assignment_expression",
                "variable_declarator",
                "public_field_definition",
            }
        ),
        import_types=frozenset({"import_statement", "import_clause", "export_statement"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string", "template_string"}),
        interpolation_types=frozenset({"template_substitution"}),
        identifier_types=_COMMON_ID,
        parameter_container_types=frozenset({"formal_parameters"}),
        return_types=frozenset({"return_statement"}),
        branch_types=frozenset({"if_statement", "switch_statement", "ternary_expression"}),
        loop_types=frozenset(
            {"for_statement", "for_in_statement", "while_statement", "do_statement"}
        ),
        exception_types=frozenset({"try_statement", "catch_clause"}),
        member_types=frozenset({"member_expression", "subscript_expression"}),
        constructor_types=frozenset({"new_expression"}),
        decorator_types=frozenset({"decorator"}),
        import_kind="import",
        block_types=frozenset(
            {
                "statement_block",
                "if_statement",
                "switch_case",
                "switch_default",
                "for_statement",
                "for_in_statement",
                "while_statement",
                "do_statement",
                "try_statement",
                "catch_clause",
                "finally_clause",
            }
        ),
        lexical_model="mixed",
        function_scoped_keywords=frozenset({"var"}),
        block_scoped_keywords=frozenset({"let", "const"}),
    ),
    "typescript": Grammar(
        ts_name="typescript",
        function_types=frozenset(
            {
                "function_declaration",
                "generator_function_declaration",
                "function_expression",
                "arrow_function",
                "method_definition",
                "function_signature",
            }
        ),
        class_types=frozenset(
            {
                "class_declaration",
                "class",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
                "internal_module",
                "module",
            }
        ),
        call_types=frozenset({"call_expression"}),
        assignment_types=frozenset({"assignment_expression", "variable_declarator"}),
        import_types=frozenset({"import_statement", "export_statement"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string", "template_string"}),
        interpolation_types=frozenset({"template_substitution"}),
        identifier_types=_COMMON_ID,
        parameter_container_types=frozenset({"formal_parameters"}),
        return_types=frozenset({"return_statement"}),
        branch_types=frozenset({"if_statement", "switch_statement"}),
        loop_types=frozenset({"for_statement", "for_in_statement", "while_statement"}),
        exception_types=frozenset({"try_statement", "catch_clause"}),
        member_types=frozenset({"member_expression", "subscript_expression"}),
        constructor_types=frozenset({"new_expression"}),
        decorator_types=frozenset({"decorator"}),
        import_kind="import",
        block_types=frozenset(
            {
                "statement_block",
                "if_statement",
                "switch_case",
                "switch_default",
                "for_statement",
                "for_in_statement",
                "while_statement",
                "catch_clause",
            }
        ),
        lexical_model="mixed",
        function_scoped_keywords=frozenset({"var"}),
        block_scoped_keywords=frozenset({"let", "const"}),
    ),
    "tsx": Grammar(
        ts_name="tsx",
        function_types=frozenset(
            {
                "function_declaration",
                "function_expression",
                "arrow_function",
                "method_definition",
            }
        ),
        class_types=frozenset({"class_declaration", "interface_declaration"}),
        call_types=frozenset({"call_expression"}),
        assignment_types=frozenset({"assignment_expression", "variable_declarator"}),
        import_types=frozenset({"import_statement", "export_statement"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string", "template_string"}),
        interpolation_types=frozenset({"template_substitution", "jsx_expression"}),
        identifier_types=_COMMON_ID,
        parameter_container_types=frozenset({"formal_parameters"}),
        return_types=frozenset({"return_statement"}),
        exception_types=frozenset({"try_statement", "catch_clause"}),
        member_types=frozenset({"member_expression"}),
        constructor_types=frozenset({"new_expression"}),
        import_kind="import",
        block_types=frozenset({"statement_block", "if_statement", "catch_clause"}),
        lexical_model="mixed",
        function_scoped_keywords=frozenset({"var"}),
        block_scoped_keywords=frozenset({"let", "const"}),
    ),
    "ruby": Grammar(
        ts_name="ruby",
        function_types=frozenset({"method", "singleton_method"}),
        class_types=frozenset({"class", "module"}),
        call_types=frozenset({"call"}),
        assignment_types=frozenset({"assignment", "operator_assignment"}),
        import_types=frozenset(),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string", "heredoc_body", "heredoc_beginning"}),
        interpolation_types=frozenset({"interpolation"}),
        identifier_types=frozenset({"identifier", "constant"}),
        parameter_container_types=frozenset({"method_parameters", "parameters"}),
        return_types=frozenset({"return"}),
        branch_types=frozenset({"if", "unless", "case"}),
        loop_types=frozenset({"while", "until", "for"}),
        exception_types=frozenset({"begin", "rescue"}),
        member_types=frozenset({"call"}),
        import_kind="require",
        block_types=frozenset(
            {"then", "else", "if", "unless", "case", "when", "do_block", "block", "rescue"}
        ),
        lexical_model="function",
        quality_for_loop_types=frozenset({"for"}),
    ),
    "c": Grammar(
        ts_name="c",
        function_types=frozenset({"function_definition"}),
        class_types=frozenset({"struct_specifier", "enum_specifier", "union_specifier"}),
        call_types=frozenset({"call_expression"}),
        assignment_types=frozenset({"assignment_expression", "init_declarator"}),
        import_types=frozenset({"preproc_include"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string_literal", "char_literal", "raw_string_literal"}),
        identifier_types=frozenset({"identifier", "field_identifier"}),
        parameter_container_types=frozenset({"parameter_list"}),
        return_types=frozenset({"return_statement"}),
        branch_types=frozenset({"if_statement", "switch_statement"}),
        loop_types=frozenset({"for_statement", "while_statement", "do_statement"}),
        member_types=frozenset({"field_expression"}),
        import_kind="include",
        block_types=frozenset(
            {
                "compound_statement",
                "if_statement",
                "switch_statement",
                "for_statement",
                "while_statement",
                "do_statement",
            }
        ),
        lexical_model="block",
        quality_deprecated_calls=frozenset({"gets", "gets_s", "strcpy", "sprintf"}),
    ),
    "cpp": Grammar(
        ts_name="cpp",
        function_types=frozenset({"function_definition"}),
        class_types=frozenset(
            {
                "class_specifier",
                "struct_specifier",
                "enum_specifier",
                "namespace_definition",
            }
        ),
        call_types=frozenset({"call_expression"}),
        assignment_types=frozenset({"assignment_expression", "init_declarator"}),
        import_types=frozenset({"preproc_include", "using_declaration"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string_literal", "char_literal", "raw_string_literal"}),
        identifier_types=frozenset(
            {"identifier", "field_identifier", "type_identifier", "namespace_identifier"}
        ),
        parameter_container_types=frozenset({"parameter_list"}),
        return_types=frozenset({"return_statement"}),
        branch_types=frozenset({"if_statement", "switch_statement"}),
        loop_types=frozenset(
            {"for_statement", "while_statement", "do_statement", "for_range_loop"}
        ),
        exception_types=frozenset({"try_statement", "catch_clause"}),
        member_types=frozenset({"field_expression", "qualified_identifier"}),
        import_kind="include",
        block_types=frozenset(
            {
                "compound_statement",
                "if_statement",
                "switch_statement",
                "for_statement",
                "while_statement",
                "do_statement",
                "for_range_loop",
                "try_statement",
                "catch_clause",
            }
        ),
        lexical_model="block",
        quality_deprecated_calls=frozenset({"gets", "strcpy", "sprintf"}),
    ),
    "go": Grammar(
        ts_name="go",
        function_types=frozenset({"function_declaration", "method_declaration"}),
        class_types=frozenset({"type_declaration", "type_spec"}),
        call_types=frozenset({"call_expression"}),
        assignment_types=frozenset({"short_var_declaration", "assignment_statement", "var_spec"}),
        import_types=frozenset({"import_declaration", "import_spec"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"interpreted_string_literal", "raw_string_literal"}),
        identifier_types=frozenset({"identifier", "field_identifier", "package_identifier"}),
        parameter_container_types=frozenset({"parameter_list"}),
        return_types=frozenset({"return_statement"}),
        branch_types=frozenset({"if_statement", "expression_switch_statement"}),
        loop_types=frozenset({"for_statement"}),
        member_types=frozenset({"selector_expression"}),
        import_kind="import",
        block_types=frozenset(
            {
                "block",
                "if_statement",
                "expression_switch_statement",
                "for_statement",
            }
        ),
        lexical_model="block",
        quality_panic_names=frozenset({"panic"}),
        quality_blank_ident=True,
    ),
    "rust": Grammar(
        ts_name="rust",
        function_types=frozenset({"function_item"}),
        class_types=frozenset({"struct_item", "enum_item", "impl_item", "trait_item", "mod_item"}),
        call_types=frozenset({"call_expression", "macro_invocation"}),
        assignment_types=frozenset({"let_declaration", "assignment_expression"}),
        import_types=frozenset({"use_declaration"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string_literal", "raw_string_literal", "char_literal"}),
        identifier_types=frozenset({"identifier", "field_identifier", "type_identifier"}),
        parameter_container_types=frozenset({"parameters"}),
        return_types=frozenset({"return_expression"}),
        branch_types=frozenset({"if_expression", "match_expression"}),
        loop_types=frozenset({"loop_expression", "while_expression", "for_expression"}),
        member_types=frozenset({"field_expression", "scoped_identifier"}),
        import_kind="use",
        block_types=frozenset(
            {
                "block",
                "if_expression",
                "match_expression",
                "loop_expression",
                "while_expression",
                "for_expression",
            }
        ),
        lexical_model="block",
        quality_unwrap_names=frozenset({"unwrap", "expect", "unwrap_err", "unwrap_or"}),
        quality_panic_names=frozenset({"panic", "todo", "unimplemented"}),
    ),
    "java": Grammar(
        ts_name="java",
        function_types=frozenset({"method_declaration", "constructor_declaration"}),
        class_types=frozenset(
            {
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            }
        ),
        call_types=frozenset({"method_invocation"}),
        assignment_types=frozenset({"assignment_expression", "variable_declarator"}),
        import_types=frozenset({"import_declaration"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string_literal", "text_block"}),
        identifier_types=frozenset({"identifier", "type_identifier"}),
        parameter_container_types=frozenset({"formal_parameters"}),
        return_types=frozenset({"return_statement"}),
        branch_types=frozenset({"if_statement", "switch_expression"}),
        loop_types=frozenset({"for_statement", "enhanced_for_statement", "while_statement"}),
        exception_types=frozenset({"try_statement", "catch_clause"}),
        member_types=frozenset({"field_access", "scoped_identifier"}),
        constructor_types=frozenset({"object_creation_expression"}),
        decorator_types=frozenset({"marker_annotation", "annotation"}),
        field_types=frozenset({"field_declaration"}),
        import_kind="import",
        block_types=frozenset(
            {
                "block",
                "if_statement",
                "switch_expression",
                "for_statement",
                "enhanced_for_statement",
                "while_statement",
                "try_statement",
                "catch_clause",
            }
        ),
        lexical_model="block",
        quality_deprecated_calls=frozenset({"printStackTrace"}),
    ),
    "php": Grammar(
        ts_name="php",
        function_types=frozenset({"function_definition", "method_declaration"}),
        class_types=frozenset({"class_declaration", "interface_declaration", "trait_declaration"}),
        call_types=frozenset(
            {
                "function_call_expression",
                "member_call_expression",
                "scoped_call_expression",
                "echo_statement",
                "print_intrinsic",
                "shell_command_expression",
                "include_expression",
                "include_once_expression",
                "require_expression",
                "require_once_expression",
            }
        ),
        assignment_types=frozenset({"assignment_expression"}),
        import_types=frozenset(
            {"namespace_use_declaration", "include_expression", "require_expression"}
        ),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset(
            {"string", "encapsed_string", "heredoc", "nowdoc", "shell_command_expression"}
        ),
        interpolation_types=frozenset({"encapsed_string", "shell_command_expression"}),
        identifier_types=frozenset({"name", "variable_name"}),
        parameter_container_types=frozenset({"formal_parameters"}),
        return_types=frozenset({"return_statement"}),
        branch_types=frozenset({"if_statement", "switch_statement"}),
        loop_types=frozenset({"for_statement", "foreach_statement", "while_statement"}),
        exception_types=frozenset({"try_statement", "catch_clause"}),
        member_types=frozenset({"member_access_expression", "scoped_property_access_expression"}),
        dollar_idents=True,
        import_kind="use",
        block_types=frozenset(
            {
                "compound_statement",
                "if_statement",
                "switch_statement",
                "for_statement",
                "foreach_statement",
                "while_statement",
                "try_statement",
                "catch_clause",
            }
        ),
        lexical_model="function",
        quality_deprecated_calls=frozenset({"mysql_query", "split"}),
    ),
    "kotlin": Grammar(
        ts_name="kotlin",
        function_types=frozenset({"function_declaration"}),
        class_types=frozenset({"class_declaration", "object_declaration", "companion_object"}),
        call_types=frozenset({"call_expression"}),
        assignment_types=frozenset({"property_declaration", "assignment"}),
        import_types=frozenset({"import_header"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset(
            {"string_literal", "line_string_literal", "multi_line_string_literal"}
        ),
        interpolation_types=frozenset({"interpolated_expression", "interpolated_identifier"}),
        identifier_types=frozenset({"simple_identifier", "type_identifier"}),
        parameter_container_types=frozenset({"function_value_parameters"}),
        return_types=frozenset({"return_expression"}),
        branch_types=frozenset({"if_expression", "when_expression"}),
        loop_types=frozenset({"for_statement", "while_statement"}),
        exception_types=frozenset({"try_expression", "catch_block"}),
        member_types=frozenset({"navigation_expression", "navigation_suffix"}),
        decorator_types=frozenset({"annotation"}),
        import_kind="import",
        block_types=frozenset(
            {
                "control_structure_body",
                "if_expression",
                "when_expression",
                "for_statement",
                "while_statement",
                "try_expression",
                "catch_block",
            }
        ),
        lexical_model="block",
        quality_unwrap_names=frozenset({"!!"}),
        quality_force_unwrap_child_types=frozenset({"!!"}),
    ),
    "swift": Grammar(
        ts_name="swift",
        function_types=frozenset({"function_declaration", "init_declaration"}),
        class_types=frozenset(
            {
                "class_declaration",
                "struct_declaration",
                "enum_declaration",
                "protocol_declaration",
                "actor_declaration",
            }
        ),
        call_types=frozenset({"call_expression"}),
        assignment_types=frozenset({"property_declaration", "assignment"}),
        import_types=frozenset({"import_declaration"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset(
            {"line_string_literal", "multi_line_string_literal", "raw_string_literal"}
        ),
        interpolation_types=frozenset({"interpolated_expression"}),
        identifier_types=frozenset({"simple_identifier", "type_identifier"}),
        parameter_container_types=frozenset({"parameter"}),
        return_types=frozenset({"return_statement"}),
        branch_types=frozenset({"if_statement", "guard_statement", "switch_statement"}),
        loop_types=frozenset({"for_statement", "while_statement", "repeat_while_statement"}),
        exception_types=frozenset({"do_statement", "catch_clause"}),
        member_types=frozenset({"navigation_expression", "navigation_suffix"}),
        import_kind="import",
        block_types=frozenset(
            {
                "statements",
                "if_statement",
                "guard_statement",
                "switch_statement",
                "for_statement",
                "while_statement",
                "repeat_while_statement",
                "do_statement",
                "catch_clause",
            }
        ),
        lexical_model="block",
        quality_unwrap_names=frozenset({"unsafelyUnwrapped"}),
        quality_force_unwrap_child_types=frozenset({"bang"}),
    ),
    "csharp": Grammar(
        ts_name="csharp",
        function_types=frozenset(
            {"method_declaration", "constructor_declaration", "local_function_statement"}
        ),
        class_types=frozenset(
            {
                "class_declaration",
                "interface_declaration",
                "struct_declaration",
                "enum_declaration",
                "record_declaration",
                "namespace_declaration",
            }
        ),
        call_types=frozenset({"invocation_expression"}),
        assignment_types=frozenset(
            {
                "assignment_expression",
                "variable_declarator",
                "local_declaration_statement",
            }
        ),
        import_types=frozenset({"using_directive"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset(
            {
                "string_literal",
                "verbatim_string_literal",
                "interpolated_string_expression",
                "raw_string_literal",
            }
        ),
        interpolation_types=frozenset({"interpolation"}),
        identifier_types=frozenset({"identifier"}),
        parameter_container_types=frozenset({"parameter_list"}),
        return_types=frozenset({"return_statement"}),
        branch_types=frozenset({"if_statement", "switch_statement"}),
        loop_types=frozenset({"for_statement", "foreach_statement", "while_statement"}),
        exception_types=frozenset({"try_statement", "catch_clause"}),
        member_types=frozenset({"member_access_expression", "conditional_access_expression"}),
        constructor_types=frozenset({"object_creation_expression"}),
        decorator_types=frozenset({"attribute"}),
        import_kind="using",
        block_types=frozenset(
            {
                "block",
                "if_statement",
                "switch_statement",
                "for_statement",
                "foreach_statement",
                "while_statement",
                "try_statement",
                "catch_clause",
            }
        ),
        lexical_model="block",
    ),
    "shell": Grammar(
        ts_name="bash",
        function_types=frozenset({"function_definition"}),
        call_types=frozenset({"command", "redirected_statement"}),
        assignment_types=frozenset({"variable_assignment"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string", "raw_string", "ansi_c_string", "heredoc_body"}),
        interpolation_types=frozenset(
            {"expansion", "simple_expansion", "command_substitution", "process_substitution"}
        ),
        identifier_types=frozenset({"word", "variable_name", "command_name"}),
        return_types=frozenset({"return"}),
        branch_types=frozenset({"if_statement", "case_statement"}),
        loop_types=frozenset({"for_statement", "while_statement", "c_style_for_statement"}),
        import_kind="source",
        block_types=frozenset(
            {
                "compound_statement",
                "if_statement",
                "case_statement",
                "for_statement",
                "while_statement",
                "function_definition",
            }
        ),
        lexical_model="function",
        quality_unquoted_expansion=True,
        quality_posix_test_types=frozenset({"test_command"}),
    ),
    "html": Grammar(
        ts_name="html",
        function_types=frozenset({"script_element"}),
        call_types=frozenset({"script_element"}),
        assignment_types=frozenset({"attribute"}),
        comment_types=frozenset({"comment"}),
        string_types=frozenset({"quoted_attribute_value", "attribute_value"}),
        identifier_types=frozenset({"tag_name", "attribute_name"}),
        import_kind="markup",
    ),
    "css": Grammar(
        ts_name="css",
        function_types=frozenset({"rule_set"}),
        call_types=frozenset({"call_expression"}),
        assignment_types=frozenset({"declaration"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string_value"}),
        identifier_types=frozenset({"property_name", "tag_name", "class_name", "function_name"}),
        import_kind="import",
    ),
    "scss": Grammar(
        ts_name="scss",
        function_types=frozenset({"rule_set", "mixin_statement", "function_statement"}),
        call_types=frozenset({"call_expression", "include_statement"}),
        assignment_types=frozenset({"declaration", "variable_declaration"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string_value"}),
        identifier_types=frozenset({"property_name", "variable_name", "function_name"}),
        import_kind="import",
    ),
    "sql": Grammar(
        ts_name="sql",
        function_types=frozenset({"statement", "create_function"}),
        call_types=frozenset({"invocation", "function_call"}),
        comment_types=_COMMON_COMMENT,
        string_types=frozenset({"string", "literal"}),
        identifier_types=frozenset({"identifier", "object_reference"}),
        import_kind="sql",
    ),
}

# BugForge language id → grammar key (tsx files use the tsx grammar)
GRAMMAR_FOR_LANGUAGE: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "ruby": "ruby",
    "c": "c",
    "cpp": "cpp",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "php": "php",
    "kotlin": "kotlin",
    "swift": "swift",
    "csharp": "csharp",
    "shell": "shell",
    "html": "html",
    "css": "css",
    "scss": "scss",
    "sql": "sql",
}


def grammar_for(language_id: str, *, filename: str = "") -> Grammar | None:
    if filename.endswith(".tsx"):
        return GRAMMARS.get("tsx")
    key = GRAMMAR_FOR_LANGUAGE.get(language_id)
    if key is None:
        return None
    return GRAMMARS.get(key)
