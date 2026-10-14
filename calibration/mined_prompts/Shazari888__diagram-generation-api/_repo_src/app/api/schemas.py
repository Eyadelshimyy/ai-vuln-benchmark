from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class GenerateDiagramRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    diagram_type: str = Field(default="mermaid", pattern="^(mermaid|d2|plantuml|graphviz)$")
    format: str = Field(default="svg", pattern="^(svg|png)$")


class DiagramResponse(BaseModel):
    id: UUID
    prompt: str
    source: str
    diagram_type: str
    created_at: datetime

    model_config = {"from_attributes": True}


class GenerateDiagramResponse(BaseModel):
    diagram: DiagramResponse
    rendered: str
    format: str


class HealthResponse(BaseModel):
    status: str = "ok"
