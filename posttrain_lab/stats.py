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


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def macro_mean(scores: Sequence[float], groups: Sequence[str]) -> float:
    """The mean of the per-group means, so every group counts the same."""
    by_group: dict[str, list[float]] = {}
    for score, group in zip(scores, groups):
        by_group.setdefault(group, []).append(score)
    return _mean([_mean(values) for values in by_group.values()])


def bootstrap(
    scores: Sequence[float],
    *,
    clusters: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    macro: bool = False,
    resamples: int = 2000,
    seed: int = 0,
    keep_draws: bool = False,
) -> dict:
    """A point estimate with a 95% percentile bootstrap interval.

    ``clusters`` names the unit that is resampled (both orders of a pairwise item
    share one cluster, so they are drawn together). With ``macro`` the statistic is
    the mean of the group means, and resampling is done inside each group.
    """
    import random

    n = len(scores)
    if n == 0:
        return {"n": 0, "estimate": None, "low": None, "high": None}
    clusters = list(clusters) if clusters is not None else [str(i) for i in range(n)]
    groups = list(groups) if groups is not None else [""] * n
    members: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        members.setdefault(cluster, []).append(index)
    cluster_group = {cluster: groups[indexes[0]] for cluster, indexes in members.items()}
    strata: dict[str, list[str]] = {}
    for cluster, group in cluster_group.items():
        strata.setdefault(group if macro else "", []).append(cluster)

    def statistic(indexes: list[int]) -> float:
        chosen = [scores[i] for i in indexes]
        return macro_mean(chosen, [groups[i] for i in indexes]) if macro else _mean(chosen)

    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        indexes: list[int] = []
        for names in strata.values():
            for _ in names:
                indexes.extend(members[names[rng.randrange(len(names))]])
        draws.append(statistic(indexes))
    result = {
        "n": n,
        "estimate": round(statistic(list(range(n))), 4),
        "low": round(percentile(draws, 2.5), 4),
        "high": round(percentile(draws, 97.5), 4),
    }
    if keep_draws:
        result["draws"] = draws
    return result


def paired_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    clusters: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    macro: bool = False,
    resamples: int = 2000,
    seed: int = 0,
    keep_draws: bool = False,
) -> dict:
    """``second`` minus ``first`` on the same items, with a 95% bootstrap interval
    and an exact McNemar test on the items where the two disagree."""
    from math import comb

    differences = [b - a for a, b in zip(first, second)]
    result = bootstrap(
        differences, clusters=clusters, groups=groups, macro=macro, resamples=resamples, seed=seed, keep_draws=keep_draws
    )
    gained = sum(1 for a, b in zip(first, second) if b > a)
    lost = sum(1 for a, b in zip(first, second) if a > b)
    discordant = gained + lost
    p_value = 1.0
    if discordant:
        tail = sum(comb(discordant, k) for k in range(0, min(gained, lost) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    result.update(gained=gained, lost=lost, mcnemar_p=round(p_value, 6))
    return result
