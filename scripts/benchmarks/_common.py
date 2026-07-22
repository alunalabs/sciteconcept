"""Shared validation and file helpers for public benchmark scripts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, received {value!r}")
    return number


def unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=np.float64), copy=False)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, received {array.shape}")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1.0e-12)


def load_embeddings(path: str | Path, *, expected_rows: int | None = None) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid embedding matrix {path}: {values.shape}")
    if expected_rows is not None and values.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding row mismatch for {path}: {values.shape[0]} != {expected_rows}"
        )
    return np.asarray(values, dtype=np.float32)


def parse_assignment(value: str, *, separator: str = "=") -> tuple[str, Path]:
    if separator not in value:
        raise ValueError(f"Expected NAME{separator}PATH, received {value!r}")
    name, path = value.split(separator, 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"Expected NAME{separator}PATH, received {value!r}")
    return name.strip(), Path(path).expanduser()


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Refusing to write an empty benchmark table")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(materialized[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(materialized)
