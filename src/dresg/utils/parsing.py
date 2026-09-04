from __future__ import annotations


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer")
    if len(set(items)) != len(items):
        raise ValueError(f"Duplicate ids are not allowed: {items}")
    return items
