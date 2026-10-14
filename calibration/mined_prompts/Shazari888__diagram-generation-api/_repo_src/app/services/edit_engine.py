import re
from typing import Any, Optional


class EditEngineException(Exception):
    """Exception raised when an edit operation fails."""

    pass


class DiagramEditEngine:
    """
    Deterministic diagram source editor.
    Applies operations in order to transform diagram source code.
    """

    @staticmethod
    def apply_operation(source: str, operation: dict[str, Any]) -> str:
        """
        Apply a single edit operation to diagram source.

        Args:
            source: The diagram source code
            operation: Dictionary with 'type' and operation-specific fields

        Returns:
            Modified diagram source

        Raises:
            EditEngineException: If the operation is invalid or fails
        """
        op_type = operation.get("type")

        if op_type == "replace_text":
            return DiagramEditEngine._replace_text(
                source,
                operation.get("from"),
                operation.get("to"),
            )
        elif op_type == "rename_node":
            return DiagramEditEngine._rename_node(
                source,
                operation.get("from"),
                operation.get("to"),
            )
        elif op_type == "add_line":
            return DiagramEditEngine._add_line(source, operation.get("line"))
        elif op_type == "remove_line_contains":
            return DiagramEditEngine._remove_line_contains(
                source,
                operation.get("contains"),
            )
        elif op_type == "prepend_text":
            return DiagramEditEngine._prepend_text(source, operation.get("text"))
        elif op_type == "append_text":
            return DiagramEditEngine._append_text(source, operation.get("text"))
        elif op_type == "set_node_shape":
            return DiagramEditEngine._set_node_shape(
                source,
                operation.get("node_id"),
                operation.get("shape"),
                operation.get("label"),
            )
        elif op_type == "set_node_color":
            return DiagramEditEngine._set_node_color(
                source,
                operation.get("node_id"),
                operation.get("fill"),
                operation.get("stroke"),
                operation.get("text_color"),
                operation.get("stroke_width_px"),
            )
        elif op_type == "set_node_font_size":
            return DiagramEditEngine._set_node_font_size(
                source,
                operation.get("node_id"),
                operation.get("font_size_px"),
            )
        elif op_type == "set_node_size":
            return DiagramEditEngine._set_node_size(
                source,
                operation.get("node_id"),
                operation.get("font_size_px"),
                operation.get("padding_px"),
            )
        elif op_type == "set_link_color":
            return DiagramEditEngine._set_link_color(
                source,
                operation.get("stroke"),
                operation.get("text_color"),
                operation.get("stroke_width_px"),
            )
        elif op_type == "set_theme":
            return DiagramEditEngine._set_theme(source, operation.get("theme"))
        elif op_type == "set_global_font_size":
            return DiagramEditEngine._set_global_font_size(
                source,
                operation.get("font_size_px"),
            )
        else:
            raise EditEngineException(f"Unknown operation type: {op_type}")

    @staticmethod
    def apply_operations(
        source: str, operations: list[dict[str, Any]]
    ) -> tuple[str, int]:
        """
        Apply multiple operations in sequence.

        Args:
            source: The diagram source code
            operations: List of operation dictionaries

        Returns:
            Tuple of (modified_source, count_of_operations_applied)

        Raises:
            EditEngineException: On first operation failure
        """
        modified = source
        for i, op in enumerate(operations):
            try:
                modified = DiagramEditEngine.apply_operation(modified, op)
            except EditEngineException as exc:
                raise EditEngineException(
                    f"Operation {i} ({op.get('type')}): {str(exc)}"
                ) from exc
        return modified, len(operations)

    # ========================================================================
    # Individual Operation Implementations
    # ========================================================================

    @staticmethod
    def _replace_text(source: str, from_text: str, to: str) -> str:
        """Replace all occurrences of from_text with to."""
        if not from_text:
            raise EditEngineException("from_text cannot be empty")
        return source.replace(from_text, to)

    @staticmethod
    def _rename_node(source: str, from_name: str, to_name: str) -> str:
        """
        Rename a node by replacing patterns like:
        - A[...] → B[...]  (node definition)
        - A -->|label| B   (node reference)
        """
        if not from_name or not to_name:
            raise EditEngineException("from and to node names cannot be empty")

        result = source
        escaped_from = re.escape(from_name)

        # Replace node definition: A[label] -> to[label] or to{label} etc.
        pattern = r"\b" + escaped_from + r"([\[\{\(])"
        result = re.sub(pattern, to_name + r"\1", result)

        # Replace node references in arrows: --> from_name becomes --> to_name
        pattern = r"(-+[>|]*\s*)" + escaped_from + r"(\s*[->]|\s|$|\]|\}|\))"
        result = re.sub(pattern, r"\1" + to_name + r"\2", result)

        # Replace starting node references: from_name --> becomes to_name -->
        pattern = r"^" + escaped_from + r"(\s*-->|\s*-\.->|\s*===)"
        result = re.sub(pattern, to_name + r"\1", result, flags=re.MULTILINE)

        return result


    @staticmethod
    def _add_line(source: str, line: str) -> str:
        """Append a line to the diagram source."""
        if not line:
            raise EditEngineException("line cannot be empty")
        return source.rstrip() + "\n" + line

    @staticmethod
    def _remove_line_contains(source: str, contains: str) -> str:
        """Remove all lines that contain the given substring."""
        if not contains:
            raise EditEngineException("contains cannot be empty")
        lines = source.split("\n")
        filtered = [line for line in lines if contains not in line]
        return "\n".join(filtered)

    @staticmethod
    def _prepend_text(source: str, text: str) -> str:
        """Prepend text to the diagram source."""
        if not text:
            raise EditEngineException("text cannot be empty")
        return text + "\n" + source.lstrip()

    @staticmethod
    def _append_text(source: str, text: str) -> str:
        """Append text to the diagram source."""
        if not text:
            raise EditEngineException("text cannot be empty")
        return source.rstrip() + "\n" + text

    @staticmethod
    def _set_node_shape(
        source: str,
        node_id: str,
        shape: str,
        label: Optional[str] = None,
    ) -> str:
        """
        Set node shape for Mermaid diagrams.
        Example: A --> B becomes A[text] or A{text} depending on shape.
        """
        if not node_id or not shape:
            raise EditEngineException("node_id and shape are required")

        # Map shape to Mermaid syntax
        shape_map = {
            "rectangle": "[{}]",
            "round": "({})",
            "stadium": "([{}])",
            "subroutine": "[[{}]]",
            "cylindrical": "[({})]",
            "circle": "(({}))",
            "asymmetric": ">{}]",
            "rhombus": "{{{}}",
            "hexagon": "{{{}}}",
            "parallelogram": "[\\{}\\]",
            "parallelogram_alt": "[/{}\\]",
            "trapezoid": "[\\{}/]",
            "trapezoid_alt": "[\\{}\\]",
        }

        if shape not in shape_map:
            raise EditEngineException(f"Unknown shape: {shape}")

        bracket_template = shape_map[shape]

        # Find existing node definition with id
        escaped_node_id = re.escape(node_id)
        pattern = r"\b" + escaped_node_id + r"\s*[\[\(\{][^\]\)\}]*[\]\)\}]"
        match = re.search(pattern, source)

        if not match:
            raise EditEngineException(f"Node '{node_id}' not found in diagram source")

        # Extract label from existing definition or use provided label
        existing = match.group(0)
        if label:
            new_text = label
        else:
            # Try to extract label from existing node
            text_match = re.search(r"[\[\(\{]([^\]\)\}]*)[\]\)\}]", existing)
            new_text = text_match.group(1) if text_match else node_id

        new_node = node_id + bracket_template.format(new_text)
        return source[: match.start()] + new_node + source[match.end() :]

    @staticmethod
    def _set_node_color(
        source: str,
        node_id: str,
        fill: Optional[str] = None,
        stroke: Optional[str] = None,
        text_color: Optional[str] = None,
        stroke_width_px: Optional[int] = None,
    ) -> str:
        """
        Set node color in Mermaid diagrams using style directives.
        Appends or updates style definition.
        """
        if not node_id:
            raise EditEngineException("node_id is required")

        if not any([fill, stroke, text_color, stroke_width_px]):
            raise EditEngineException(
                "At least one of fill, stroke, text_color, stroke_width_px is required"
            )

        # Build style string
        style_parts = []
        if fill:
            style_parts.append(f"fill:{fill}")
        if stroke:
            style_parts.append(f"stroke:{stroke}")
        if text_color:
            style_parts.append(f"color:{text_color}")
        if stroke_width_px:
            style_parts.append(f"stroke-width:{stroke_width_px}px")

        style_str = ",".join(style_parts)

        # Check if style already exists for this node
        style_pattern = rf"style {re.escape(node_id)} [^;\n]*"
        if re.search(style_pattern, source):
            # Update existing style
            return re.sub(
                style_pattern,
                f"style {node_id} {style_str}",
                source,
            )
        else:
            # Append new style
            return source.rstrip() + f"\nstyle {node_id} {style_str}"

    @staticmethod
    def _set_node_font_size(source: str, node_id: str, font_size_px: int) -> str:
        """Set font size for a specific node."""
        if not node_id or not font_size_px:
            raise EditEngineException("node_id and font_size_px are required")

        # For Mermaid, use style directive
        style_str = f"font-size:{font_size_px}px"
        style_pattern = rf"style {re.escape(node_id)} [^;\n]*"
        if re.search(style_pattern, source):
            return re.sub(
                style_pattern,
                f"style {node_id} {style_str}",
                source,
            )
        else:
            return source.rstrip() + f"\nstyle {node_id} {style_str}"

    @staticmethod
    def _set_node_size(
        source: str,
        node_id: str,
        font_size_px: Optional[int] = None,
        padding_px: Optional[int] = None,
    ) -> str:
        """Set node size (font size and/or padding)."""
        if not node_id:
            raise EditEngineException("node_id is required")

        if not any([font_size_px, padding_px]):
            raise EditEngineException(
                "At least one of font_size_px or padding_px is required"
            )

        style_parts = []
        if font_size_px:
            style_parts.append(f"font-size:{font_size_px}px")
        if padding_px:
            style_parts.append(f"padding:{padding_px}px")

        style_str = ",".join(style_parts)
        style_pattern = rf"style {re.escape(node_id)} [^;\n]*"
        if re.search(style_pattern, source):
            return re.sub(
                style_pattern,
                f"style {node_id} {style_str}",
                source,
            )
        else:
            return source.rstrip() + f"\nstyle {node_id} {style_str}"

    @staticmethod
    def _set_link_color(
        source: str,
        stroke: Optional[str] = None,
        text_color: Optional[str] = None,
        stroke_width_px: Optional[int] = None,
    ) -> str:
        """Set link (edge) colors globally in Mermaid."""
        if not any([stroke, text_color, stroke_width_px]):
            raise EditEngineException(
                "At least one of stroke, text_color, stroke_width_px is required"
            )

        style_parts = []
        if stroke:
            style_parts.append(f"stroke:{stroke}")
        if text_color:
            style_parts.append(f"color:{text_color}")
        if stroke_width_px:
            style_parts.append(f"stroke-width:{stroke_width_px}px")

        style_str = ",".join(style_parts)

        # Check if linkStyle already exists
        if "linkStyle" in source:
            # Update existing — find all linkStyle lines and append/update
            linkstyle_pattern = r"linkStyle [0-9, ]+"
            if re.search(linkstyle_pattern, source):
                # Just append; let renderer interpret
                return source.rstrip() + f"\nlinkStyle default {style_str}"
        else:
            return source.rstrip() + f"\nlinkStyle default {style_str}"

    @staticmethod
    def _set_theme(source: str, theme: str) -> str:
        """Set diagram theme in Mermaid."""
        if not theme:
            raise EditEngineException("theme is required")

        # Check if theme is already set
        if "%%{init:" in source:
            # Update existing init block
            pattern = r"%%{init:.*?}%%"
            if re.search(pattern, source, re.DOTALL):
                return re.sub(
                    pattern,
                    f"{{% init: {{ 'theme': '{theme}' }} %}}",
                    source,
                    flags=re.DOTALL,
                )
        # Prepend new init block
        return f"{{% init: {{ 'theme': '{theme}' }} %}}\n{source}"

    @staticmethod
    def _set_global_font_size(source: str, font_size_px: int) -> str:
        """Set global font size in diagram config."""
        if not font_size_px:
            raise EditEngineException("font_size_px is required")

        # For Mermaid, use config block
        config_str = f"{{% init: {{ 'fontSize': {font_size_px} }} %}}"

        if "%%{init:" in source:
            pattern = r"%%{init:.*?}%%"
            return re.sub(
                pattern,
                config_str,
                source,
                flags=re.DOTALL,
            )
        else:
            return config_str + "\n" + source
