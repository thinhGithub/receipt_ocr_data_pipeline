"""Structured receipt data validation components."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def validate_structured_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Return non-destructive validation diagnostics for extracted fields."""
    errors: dict[str, list[str]] = {}
    timestamp = fields.get("timestamp")
    if timestamp:
        try:
            datetime.fromisoformat(str(timestamp))
        except ValueError:
            errors.setdefault("timestamp", []).append("invalid_iso_timestamp")
    total_cost = fields.get("total_cost")
    if total_cost is not None and (not isinstance(total_cost, int) or total_cost < 0):
        errors.setdefault("total_cost", []).append("invalid_non_negative_integer")
    for field in ("seller", "address"):
        value = fields.get(field)
        if value is not None and not str(value).strip():
            errors.setdefault(field, []).append("empty_string")
    return {"is_valid": not errors, "errors": errors}
