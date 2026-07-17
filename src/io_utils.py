"""Minimal CSV helpers.

The demo deliberately avoids pandas so that it runs cleanly on a bare
numpy + scipy + matplotlib install. These helpers read and write the small
CSV tables used across the pipeline.
"""
from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dictionaries to a CSV file with a header row."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: str) -> list[dict]:
    """Read a CSV file into a list of dictionaries (all values as strings)."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_float(value: Any) -> float:
    """Parse a CSV string into a float, mapping blanks to NaN."""
    if value is None or value == "":
        return float("nan")
    return float(value)


def write_json(path: str, obj: Any) -> None:
    """Write an object to pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Recursively round floats and normalise numpy scalars for JSON output."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj
