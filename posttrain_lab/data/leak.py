"""Keep benchmark items, and anything close to one, out of training data.

Three checks, each against the question text of every benchmark item:

* exact: the same string;
* normalized: the same after lower-casing and dropping punctuation and extra spaces;
* similar: the Jaccard similarity of character n-gram sets is at or above a threshold,
  computed after every run of digits is replaced by a single 0, so that a problem
  with the same wording and different numbers still counts as close.

The similarity search is exact. It uses the usual prefix filter: when shingles are
put in one fixed order, two sets can only reach the threshold if their rarest few
shingles overlap, so only those need an index.
"""

from __future__ import annotations

import math
import re
from collections import Counter

NGRAM = 5
THRESHOLD = 0.5


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def mask_numbers(text: str) -> str:
    return re.sub(r"\d+", "0", text)


def shingles(text: str, n: int = NGRAM) -> frozenset[str]:
    if len(text) <= n:
        return frozenset([text])
    return frozenset(text[i : i + n] for i in range(len(text) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _prefix(ordered: list[str], threshold: float) -> list[str]:
    return ordered[: len(ordered) - math.ceil(threshold * len(ordered)) + 1]


def leak_check(candidates: list[dict], references: list[tuple[str, str]], *, n: int = NGRAM, threshold: float = THRESHOLD) -> tuple[list[dict], list[dict], dict]:
    """Split candidates into kept and dropped. ``references`` are (label, text) pairs.

    Each dropped candidate is returned as ``{"id", "reason", "reference", "similarity"}``.
    """
    exact = {text.strip(): label for label, text in references}
    normalized = {normalize(text): label for label, text in references}
    candidate_sets = {c["id"]: shingles(mask_numbers(normalize(c["question"])), n) for c in candidates}
    largest = max((len(s) for s in candidate_sets.values()), default=0)
    smallest = min((len(s) for s in candidate_sets.values()), default=0)

    # A reference much longer or shorter than every candidate cannot reach the threshold.
    reference_sets: list[tuple[str, frozenset]] = []
    out_of_range = 0
    for label, text in references:
        masked = mask_numbers(normalize(text))
        if len(masked) - n + 1 > largest / threshold + 1 or len(masked) + n < smallest * threshold:
            out_of_range += 1
            continue
        reference_sets.append((label, shingles(masked, n)))

    frequency: Counter = Counter()
    for _, shingle_set in reference_sets:
        frequency.update(shingle_set)

    def order(shingle_set: frozenset) -> list[str]:
        return sorted(shingle_set, key=lambda s: (frequency.get(s, 0), s))  # rarest first

    index: dict[str, list[int]] = {}
    for position, (_, shingle_set) in enumerate(reference_sets):
        for shingle in _prefix(order(shingle_set), threshold):
            index.setdefault(shingle, []).append(position)

    kept, dropped, compared = [], [], 0
    for candidate in candidates:
        question = candidate["question"]
        if question.strip() in exact:
            dropped.append({"id": candidate["id"], "reason": "exact", "reference": exact[question.strip()], "similarity": 1.0})
            continue
        if normalize(question) in normalized:
            dropped.append({"id": candidate["id"], "reason": "normalized", "reference": normalized[normalize(question)], "similarity": 1.0})
            continue
        mine = candidate_sets[candidate["id"]]
        best, best_label = 0.0, None
        for position in {p for shingle in _prefix(order(mine), threshold) for p in index.get(shingle, ())}:
            label, theirs = reference_sets[position]
            if not (threshold * len(mine) <= len(theirs) <= len(mine) / threshold):
                continue
            compared += 1
            similarity = jaccard(mine, theirs)
            if similarity > best:
                best, best_label = similarity, label
        if best >= threshold:
            dropped.append({"id": candidate["id"], "reason": "similar", "reference": best_label, "similarity": round(best, 4)})
        else:
            kept.append(candidate)
    stats = {
        "candidates": len(candidates),
        "references": len(references),
        "references_too_long_or_short_to_match": out_of_range,
        "pairs_compared_in_full": compared,
        "ngram": n,
        "threshold": threshold,
        "dropped": dict(Counter(d["reason"] for d in dropped)),
        "kept": len(kept),
    }
    return kept, dropped, stats


def closest(candidate: str, references: list[tuple[str, str]], n: int = NGRAM) -> tuple[float, str | None]:
    """The most similar reference by brute force. Slow; for tests and spot checks."""
    mine = shingles(mask_numbers(normalize(candidate)), n)
    best, best_label = 0.0, None
    for label, text in references:
        similarity = jaccard(mine, shingles(mask_numbers(normalize(text)), n))
        if similarity > best:
            best, best_label = similarity, label
    return best, best_label
