"""Request / response models (validated by Pydantic)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Priority = Literal["Low", "Medium", "High", "Urgent"]
Status = Literal["Open", "In Progress", "Resolved", "Closed"]
FieldType = Literal["text", "dropdown", "number", "date", "checkbox"]


def _strip(v: Any) -> Any:
    return v.strip() if isinstance(v, str) else v


class TicketCreate(BaseModel):
    customer_name: str = Field(..., min_length=1, max_length=200)
    subject: str = Field(..., min_length=1, max_length=300)
    description: str = Field("", max_length=5000)
    category: Optional[str] = Field(None, max_length=100, description="Leave empty to let the ML model predict it")
    priority: Priority = "Medium"
    status: Status = "Open"
    custom_fields: dict[str, Any] = Field(default_factory=dict)

    _strip_fields = field_validator("customer_name", "subject", "description", "category", mode="before")(_strip)

    @field_validator("customer_name", "subject")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("description", mode="before")
    @classmethod
    def none_to_empty(cls, v: Any) -> Any:
        return "" if v is None else v


class TicketUpdate(BaseModel):
    customer_name: Optional[str] = Field(None, min_length=1, max_length=200)
    subject: Optional[str] = Field(None, min_length=1, max_length=300)
    description: Optional[str] = Field(None, max_length=5000)
    category: Optional[str] = Field(None, max_length=100)
    priority: Optional[Priority] = None
    status: Optional[Status] = None
    custom_fields: Optional[dict[str, Any]] = None

    _strip_fields = field_validator("customer_name", "subject", "description", "category", mode="before")(_strip)


class ResponseUpdate(BaseModel):
    final_response: str = Field(..., min_length=1, max_length=10000)
    approve: bool = False

    @field_validator("final_response", mode="before")
    @classmethod
    def strip_text(cls, v: Any) -> Any:
        v = _strip(v)
        if not v:
            raise ValueError("response must not be empty")
        return v


class FieldDefinitionCreate(BaseModel):
    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]{0,49}$", description="snake_case key, e.g. warranty_id")
    label: str = Field(..., min_length=1, max_length=100)
    field_type: FieldType
    required: bool = False
    options: Optional[list[str]] = None
    help_text: Optional[str] = Field(None, max_length=300)
    sort_order: int = 100

    @field_validator("options")
    @classmethod
    def clean_options(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return v
        cleaned = [o.strip() for o in v if o and o.strip()]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("options must be unique")
        return cleaned


class FieldDefinitionUpdate(BaseModel):
    label: Optional[str] = Field(None, min_length=1, max_length=100)
    required: Optional[bool] = None
    options: Optional[list[str]] = None
    help_text: Optional[str] = Field(None, max_length=300)
    sort_order: Optional[int] = None
    is_active: Optional[bool] = None


class PredictRequest(BaseModel):
    subject: str = ""
    description: str = ""
