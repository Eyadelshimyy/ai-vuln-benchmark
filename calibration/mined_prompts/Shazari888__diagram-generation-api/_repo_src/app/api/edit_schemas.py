from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Operation Types (Discriminated Union)
# ============================================================================


class ReplaceTextOperation(BaseModel):
    type: Literal["replace_text"]
    from_text: str = Field(..., alias="from", min_length=1, max_length=1000)
    to: str = Field(..., min_length=0, max_length=1000)

    class Config:
        populate_by_name = True


class RenameNodeOperation(BaseModel):
    type: Literal["rename_node"]
    from_text: str = Field(..., alias="from", min_length=1, max_length=500)
    to: str = Field(..., min_length=1, max_length=500)

    class Config:
        populate_by_name = True


class AddLineOperation(BaseModel):
    type: Literal["add_line"]
    line: str = Field(..., min_length=1, max_length=2000)


class RemoveLineContainsOperation(BaseModel):
    type: Literal["remove_line_contains"]
    contains: str = Field(..., min_length=1, max_length=500)


class PrependTextOperation(BaseModel):
    type: Literal["prepend_text"]
    text: str = Field(..., min_length=1, max_length=2000)


class AppendTextOperation(BaseModel):
    type: Literal["append_text"]
    text: str = Field(..., min_length=1, max_length=2000)


class SetNodeShapeOperation(BaseModel):
    type: Literal["set_node_shape"]
    node_id: str = Field(..., min_length=1, max_length=500)
    shape: Literal[
        "rectangle",
        "round",
        "stadium",
        "subroutine",
        "cylindrical",
        "circle",
        "asymmetric",
        "rhombus",
        "hexagon",
        "parallelogram",
        "parallelogram_alt",
        "trapezoid",
        "trapezoid_alt",
    ]
    label: Optional[str] = Field(None, max_length=500)


class SetNodeColorOperation(BaseModel):
    type: Literal["set_node_color"]
    node_id: str = Field(..., min_length=1, max_length=500)
    fill: Optional[str] = Field(None, max_length=20)
    stroke: Optional[str] = Field(None, max_length=20)
    text_color: Optional[str] = Field(None, max_length=20)
    stroke_width_px: Optional[int] = Field(None, gt=0, le=100)

    @model_validator(mode="after")
    def validate_at_least_one_style(self):
        if not any([self.fill, self.stroke, self.text_color, self.stroke_width_px]):
            raise ValueError("At least one style field (fill, stroke, text_color, stroke_width_px) is required")
        return self


class SetNodeFontSizeOperation(BaseModel):
    type: Literal["set_node_font_size"]
    node_id: str = Field(..., min_length=1, max_length=500)
    font_size_px: int = Field(..., gt=0, le=200)


class SetNodeSizeOperation(BaseModel):
    type: Literal["set_node_size"]
    node_id: str = Field(..., min_length=1, max_length=500)
    font_size_px: Optional[int] = Field(None, gt=0, le=200)
    padding_px: Optional[int] = Field(None, gt=0, le=100)

    @model_validator(mode="after")
    def validate_at_least_one_size_field(self):
        if self.font_size_px is None and self.padding_px is None:
            raise ValueError("At least one of font_size_px or padding_px is required")
        return self


class SetLinkColorOperation(BaseModel):
    type: Literal["set_link_color"]
    stroke: Optional[str] = Field(None, max_length=20)
    text_color: Optional[str] = Field(None, max_length=20)
    stroke_width_px: Optional[int] = Field(None, gt=0, le=100)

    @model_validator(mode="after")
    def validate_at_least_one_style(self):
        if not any([self.stroke, self.text_color, self.stroke_width_px]):
            raise ValueError("At least one style field (stroke, text_color, stroke_width_px) is required")
        return self


class SetThemeOperation(BaseModel):
    type: Literal["set_theme"]
    theme: Literal["default", "neutral", "dark", "forest", "base"]


class SetGlobalFontSizeOperation(BaseModel):
    type: Literal["set_global_font_size"]
    font_size_px: int = Field(..., gt=0, le=200)


# Union of all operation types (discriminated by "type" field)
EditOperation = Union[
    ReplaceTextOperation,
    RenameNodeOperation,
    AddLineOperation,
    RemoveLineContainsOperation,
    PrependTextOperation,
    AppendTextOperation,
    SetNodeShapeOperation,
    SetNodeColorOperation,
    SetNodeFontSizeOperation,
    SetNodeSizeOperation,
    SetLinkColorOperation,
    SetThemeOperation,
    SetGlobalFontSizeOperation,
]


# ============================================================================
# Edit Request / Response
# ============================================================================


class EditDiagramRequest(BaseModel):
    diagram_source: str = Field(..., min_length=1, max_length=50000)
    diagram_type: str = Field(default="mermaid", pattern="^(mermaid|d2|plantuml|graphviz)$")
    format: str = Field(default="svg", pattern="^(svg|png)$")
    operations: list[EditOperation] = Field(default_factory=list, max_length=100)
    render_after_edit: bool = Field(default=True)


class EditDiagramResponse(BaseModel):
    ok: bool = True
    endpoint: str
    diagram_type: str
    format: str
    edited_diagram_source: str
    rendered: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
