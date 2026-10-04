from __future__ import annotations

FIELD_WIDTH_YARDS = 53.3
FIELD_LENGTH_YARDS = 120.0


def fmt(value: float) -> str:
    return f"{value:.3f}"


def parse_float(value: str) -> float:
    if value in {"", "NA"}:
        return 0.0
    return float(value)


def normalize_xy(x: float, y: float, play_direction: str) -> tuple[float, float]:
    if play_direction == "left":
        return FIELD_LENGTH_YARDS - x, FIELD_WIDTH_YARDS - y
    return x, y
