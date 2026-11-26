"""Small generated prompts for the smoke test. Every expected answer comes from code."""

from __future__ import annotations

import random
import re

SYSTEM_CONCISE = "Answer concisely."

_FIRST_NAMES = ["Lena", "Marcus", "Priya", "Tomas", "Aiko", "Daniel", "Sofia", "Omar", "Greta", "Ines"]
_LAST_NAMES = ["Ortiz", "Lindqvist", "Banerjee", "Okafor", "Tanaka", "Reyes", "Novak", "Haddad", "Keller", "Moreau"]
_CITIES = ["Lisbon", "Tucson", "Osaka", "Leeds", "Tampere", "Durban", "Quito", "Gdansk", "Hobart", "Calgary"]
_ITEMS = ["standing desk", "bike rack", "espresso grinder", "tent", "monitor arm", "bookshelf", "space heater"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]


def _user(text: str, system: str | None = None) -> list[dict]:
    messages = [{"role": "system", "content": system}] if system else []
    return [*messages, {"role": "user", "content": text}]


def throughput_prompt(rng: random.Random) -> list[dict]:
    """A short question whose answer needs a line or two of work."""
    a, b = rng.randint(12, 49), rng.randint(12, 49)
    return _user(f"What is {a} * {b}?", SYSTEM_CONCISE)


def arithmetic_item(rng: random.Random) -> dict:
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    op = rng.choice(["+", "-", "*"])
    if op == "*":
        a, b = rng.randint(11, 99), rng.randint(3, 19)
    answer = {"+": a + b, "-": a - b, "*": a * b}[op]
    return {
        "kind": "arithmetic",
        "messages": _user(f"What is {a} {op} {b}? Reply with only the number."),
        "answer": str(answer),
    }


def judgment_item(rng: random.Random, correct: bool | None = None) -> dict:
    """A pointwise judging prompt: a problem, a response, and whether it is right."""
    if correct is None:
        correct = rng.random() < 0.5
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    op = rng.choice(["+", "-", "*"])
    truth = {"+": a + b, "-": a - b, "*": a * b}[op]
    given = truth
    if not correct:
        given = truth + rng.choice([-20, -10, -2, -1, 1, 2, 10, 20])
    text = (
        f"A student was asked: What is {a} {op} {b}?\n"
        f"The student answered: {given}\n"
        "Is the student's answer correct? Answer with one word, yes or no."
    )
    return {"kind": "judgment", "messages": _user(text), "answer": "yes" if correct else "no"}


def extraction_item(rng: random.Random) -> dict:
    name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
    city = rng.choice(_CITIES)
    item = rng.choice(_ITEMS)
    invoice = f"INV-{rng.randint(10000, 99999)}"
    order = f"PO-{rng.randint(1000, 9999)}"
    total = f"{rng.randint(40, 900)}.{rng.randint(0, 99):02d}"
    date = f"{rng.randint(1, 28)} {rng.choice(_MONTHS)} 2026"
    text = (
        f"On {date}, {name} of {city} ordered one {item} under purchase order {order}. "
        f"The invoice {invoice} lists a total of ${total}, due within 30 days."
    )
    field, answer = rng.choice(
        [("invoice number", invoice), ("purchase order number", order), ("customer's city", city), ("customer's full name", name)]
    )
    return {
        "kind": "extraction",
        "messages": _user(f"Text: {text}\n\nExtract the {field} from the text. Reply with only that value."),
        "answer": answer,
    }


def behaviour_items(seed: int, per_kind: int) -> list[dict]:
    """A fixed mix of arithmetic, yes/no judgments and short extractions."""
    rng = random.Random(seed)
    items = []
    for index in range(per_kind):
        items.append(arithmetic_item(rng))
        items.append(judgment_item(rng, correct=(index % 2 == 0)))
        items.append(extraction_item(rng))
    for index, item in enumerate(items):
        item["id"] = f"{item['kind']}-{index:03d}"
    return items


def judgment_items(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    items = [judgment_item(rng, correct=(index % 2 == 0)) for index in range(n)]
    for index, item in enumerate(items):
        item["id"] = f"judgment-{index:03d}"
    return items


def long_judgment_item(rng: random.Random, sentences: int) -> dict:
    """A claim to check against a long generated source text.

    Used to measure how fast the service reads long prompts. The claim is true or
    false by construction.
    """
    facts = []
    for _ in range(sentences):
        name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        facts.append((name, rng.choice(_CITIES), rng.choice(_ITEMS), rng.randint(2, 97), rng.choice(_MONTHS)))
    lines = [
        f"Record {i + 1}: {name} in {city} ordered {count} units of the {item} in {month}."
        for i, (name, city, item, count, month) in enumerate(facts)
    ]
    index = rng.randrange(len(facts))
    name, city, item, count, month = facts[index]
    supported = rng.random() < 0.5
    stated = count if supported else count + rng.choice([-1, 1, 5, 10])
    claim = f"Record {index + 1} says {name} ordered {stated} units."
    text = (
        "Source:\n" + "\n".join(lines) + f"\n\nClaim: {claim}\n"
        "Is the claim supported by the source? Answer with one word, yes or no."
    )
    return {"kind": "claim", "messages": _user(text), "answer": "yes" if supported else "no"}


def filler_message(sentences: int) -> list[dict]:
    """A long, dull user message of a controllable length, for the context-limit probe."""
    line = "The quick brown fox jumps over the lazy dog near the old stone bridge."
    return _user(" ".join([line] * sentences) + "\n\nReply with the single word done.")


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize(text: str) -> str:
    return text.strip().strip("\"'`*").rstrip(".").strip().lower()


def check_answer(item: dict, text: str) -> dict:
    """Whether a reply is in the requested format, and whether it is right.

    ``parseable`` is strict: the reply is the bare value and nothing else. ``correct``
    is lenient: the right value can be pulled out of whatever was written.
    """
    expected = item["answer"]
    cleaned = _normalize(text)
    if item["kind"] == "arithmetic":
        strict = bool(re.fullmatch(r"-?\d[\d,]*", cleaned))
        numbers = [n.replace(",", "") for n in _NUMBER.findall(text)]
        value = cleaned.replace(",", "") if strict else (numbers[-1] if numbers else None)
        return {"parseable": strict, "correct": value == expected}
    if item["kind"] in ("judgment", "claim"):
        strict = cleaned in ("yes", "no")
        words = re.findall(r"\b(yes|no)\b", text.lower())
        value = cleaned if strict else (words[0] if words else None)
        return {"parseable": strict, "correct": value == expected}
    # Extraction has no fixed format beyond "only that value": one short line.
    strict = len(text.strip().splitlines()) == 1 and len(cleaned.split()) <= 5
    return {"parseable": strict, "correct": expected.lower() in text.lower()}
