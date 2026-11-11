"""Pydantic models for structured field extraction."""
from typing import Optional
from pydantic import BaseModel, Field


class FieldExtraction(BaseModel):
    """A single extracted field with its provenance."""

    value: Optional[str] = Field(
        default=None,
        description="The extracted value, as a string. Null if not found.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="The unit of measurement. Null if unitless.",
    )
    citation_text: Optional[str] = Field(
        default=None,
        description="Verbatim quote from the datasheet containing this value.",
    )
    confidence: str = Field(
        default="medium",
        description="One of: 'high', 'medium', 'low'.",
    )
    notes: Optional[str] = Field(default=None)


class FieldSpec(BaseModel):
    name: str
    display_name: str
    query: str
    search_terms: list[str]

    class Config:
        frozen = True