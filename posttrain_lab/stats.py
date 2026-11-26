"""Small numeric helpers with no dependencies."""

from __future__ import annotations

from typing import Iterable, Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0 to 100) with linear interpolation."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def describe(values: Iterable[float], digits: int = 2) -> dict:
    """Count, mean and a few percentiles of a list of numbers."""
    data = [float(v) for v in values]
    if not data:
        return {"n": 0}
    return {
        "n": len(data),
        "mean": round(sum(data) / len(data), digits),
        "min": round(min(data), digits),
        "p50": round(percentile(data, 50), digits),
        "p90": round(percentile(data, 90), digits),
        "p95": round(percentile(data, 95), digits),
        "max": round(max(data), digits),
    }


def fit_line(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least-squares intercept and slope of y against x."""
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return (sum(ys) / n if n else float("nan")), 0.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
    return mean_y - slope * mean_x, slope
