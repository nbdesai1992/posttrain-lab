"""Procedural reasoning problems whose answers are fixed by a program.

Every problem is built from a seed, a family and a level (1 is easiest, 5 hardest).
The answer is computed by the generator before any model sees the problem. Nothing
here is taken from, or modelled on, an item of any benchmark: the templates and the
wording were written for this repository.

Bump ``GENERATOR_VERSION`` whenever a change alters the problems a seed produces.
"""

from __future__ import annotations

import datetime
import math
import random
from typing import Callable

GENERATOR_VERSION = 1
LEVELS = (1, 2, 3, 4, 5)
NUMBER_SUFFIX = "Solve the problem. Give only the final answer as a number, with no units and no explanation."
TEXT_SUFFIX = "Solve the problem. Give only the final answer, with no explanation."

NAMES = ["Asha", "Bruno", "Carmen", "Dmitri", "Elif", "Farid", "Greta", "Hugo", "Ines", "Jonas", "Kavya", "Leona", "Mateo", "Noor", "Otis", "Petra"]
PLACES = ["warehouse", "library", "bakery", "nursery", "depot", "workshop", "aquarium", "orchard", "print shop", "bike store"]
THINGS = ["crates", "books", "loaves", "seedlings", "pallets", "lamps", "tanks", "baskets", "posters", "helmets"]


# ------------------------------------------------------------- arithmetic


def _expression(rng: random.Random, leaves: int, low: int, high: int, ops: str) -> tuple[str, int]:
    if leaves == 1:
        value = rng.randint(low, high)
        return str(value), value
    split = rng.randint(1, leaves - 1)
    left_text, left = _expression(rng, split, low, high, ops)
    right_text, right = _expression(rng, leaves - split, low, high, ops)
    op = rng.choice(ops)
    value = {"+": left + right, "-": left - right, "*": left * right}[op]
    if split > 1:
        left_text = f"({left_text})"
    if leaves - split > 1:
        right_text = f"({right_text})"
    return f"{left_text} {op} {right_text}", value


def arithmetic(rng: random.Random, level: int) -> dict:
    leaves, low, high, ops = {1: (2, 11, 99, "+-"), 2: (3, 11, 99, "+-*"), 3: (5, 11, 199, "+-*"), 4: (7, 101, 999, "+-*"), 5: (10, 1001, 9999, "+-*")}[level]
    while True:
        text, value = _expression(rng, leaves, low, high, ops)
        if abs(value) < 10**15:
            return {"question": f"What is the value of {text}?", "answer": value, "params": {"terms": leaves, "low": low, "high": high}}


# ------------------------------------------------------------ word problems


def stock_story(rng: random.Random, level: int) -> dict:
    """A count that changes through a series of events."""
    steps = {1: 1, 2: 2, 3: 4, 4: 6, 5: 9}[level]
    place, thing, name = rng.choice(PLACES), rng.choice(THINGS), rng.choice(NAMES)
    scale = {1: 40, 2: 90, 3: 300, 4: 900, 5: 4000}[level]
    count = rng.randint(scale // 2, scale) * 4
    lines = [f"{name} runs a {place} that starts the week with {count} {thing}."]
    for _ in range(steps):
        kind = rng.choice(["receive", "send", "batches", "share", "double", "damaged"])
        if kind == "receive" or count < 10:
            days, per_day = rng.randint(2, 6), rng.randint(3, scale // 4 + 3)
            count += days * per_day
            lines.append(f"A supplier then delivers {per_day} {thing} a day for {days} days.")
        elif kind == "send" and count > 20:
            orders = rng.randint(2, 5)
            each = rng.randint(1, max(1, count // (orders * 3)))
            count -= orders * each
            lines.append(f"After that, {orders} customers each collect {each} {thing}.")
        elif kind == "batches":
            batches, size = rng.randint(2, 7), rng.randint(4, scale // 5 + 4)
            count += batches * size
            lines.append(f"The team makes {batches} more batches of {size} {thing} each.")
        elif kind == "share" and count % 4 == 0 and count > 40:
            fraction = rng.choice([2, 4])
            count -= count // fraction
            lines.append(f"One {'half' if fraction == 2 else 'quarter'} of the {thing} on hand at that point is moved to another site.")
        elif kind == "double" and count < scale * 6:
            count *= 2
            lines.append(f"A matching shipment arrives that doubles the number of {thing} on hand.")
        else:
            lost = rng.randint(1, max(1, min(count // 5, scale // 6 + 1)))
            count -= lost
            lines.append(f"During a check, {lost} {thing} are found damaged and thrown away.")
    lines.append(f"How many {thing} does the {place} have now?")
    return {"question": " ".join(lines), "answer": count, "params": {"events": steps, "scale": scale}}


def rates(rng: random.Random, level: int) -> dict:
    """Totals built from several rates, each running for its own length of time."""
    legs = {1: 1, 2: 2, 3: 3, 4: 5, 5: 7}[level]
    big = level >= 4
    kind = rng.choice(["travel", "machines", "tank"])
    if kind == "travel":
        name, total, parts = rng.choice(NAMES), 0, []
        for _ in range(legs):
            speed, hours = rng.randint(30, 140 if big else 90), rng.randint(2, 9)
            total += speed * hours
            parts.append(f"{hours} hours at {speed} km per hour")
        question = f"{name} drives for " + ", then ".join(parts) + ". How many kilometres does the whole trip cover?"
    elif kind == "machines":
        total, parts = 0, []
        for index in range(legs):
            rate, hours, machines = rng.randint(12, 400 if big else 80), rng.randint(3, 12), rng.randint(1, 6 if big else 3)
            total += rate * hours * machines
            parts.append(f"line {index + 1} has {machines} machine{'s' if machines > 1 else ''} that each make {rate} parts an hour and runs for {hours} hours")
        question = "In a factory, " + "; ".join(parts) + ". How many parts are made in total?"
    else:
        begin = rng.randint(200, 900) * (10 if big else 1)
        volume, parts = begin, []
        for index in range(legs):
            rate, minutes = rng.randint(4, 60), rng.randint(5, 40)
            fills = index % 2 == 0 or volume < rate * minutes
            volume += rate * minutes if fills else -rate * minutes
            parts.append(f"a {'pump adds' if fills else 'drain removes'} {rate} litres a minute for {minutes} minutes")
        question = f"A tank holds {begin} litres. Then " + ", then ".join(parts) + ". How many litres are in the tank at the end?"
        total = volume
    return {"question": question, "answer": total, "params": {"legs": legs, "kind": kind}}


# ---------------------------------------------------------------- sequences


def sequence(rng: random.Random, level: int) -> dict:
    if level == 1:
        start, step, n = rng.randint(2, 40), rng.randint(2, 15), rng.randint(6, 15)
        answer = start + (n - 1) * step
        rule = f"starts at {start} and each term is {step} more than the one before"
    elif level == 2:
        start, ratio, n = rng.randint(2, 30), rng.randint(2, 5), rng.randint(4, 9)
        answer = start * ratio ** (n - 1)
        rule = f"starts at {start} and each term is {ratio} times the one before"
    elif level == 3:
        start, mult, add, n = rng.randint(1, 20), rng.randint(2, 4), rng.randint(-9, 12), rng.randint(5, 8)
        value = start
        for _ in range(n - 1):
            value = mult * value + add
        answer = value
        rule = f"starts at {start} and each term is {mult} times the one before, {'plus' if add >= 0 else 'minus'} {abs(add)}"
    else:
        first, second = rng.randint(1, 40), rng.randint(1, 40)
        p, q = (1, 1) if level == 4 else (rng.randint(2, 3), rng.randint(1, 3))
        n = rng.randint(10, 14) if level == 4 else rng.randint(12, 16)
        a, b = first, second
        for _ in range(n - 2):
            a, b = b, p * b + q * a
        answer = b
        tail = "the sum of the two terms before it" if (p, q) == (1, 1) else f"{p} times the term before it plus {q} times the term before that"
        rule = f"starts with {first} and {second}, and from the third term on each term is {tail}"
    return {"question": f"A sequence {rule}. What is term number {n} of the sequence?", "answer": answer, "params": {"n": n}}


# ------------------------------------------------------------ number theory


def number_theory(rng: random.Random, level: int) -> dict:
    if level == 1:
        a, b = rng.randint(50, 999), rng.randint(3, 19)
        return {"question": f"What is the remainder when {a} is divided by {b}?", "answer": a % b, "params": {"kind": "remainder"}}
    if level == 2:
        g = rng.randint(2, 40)
        a, b = g * rng.randint(2, 30), g * rng.randint(2, 30)
        return {"question": f"What is the greatest common divisor of {a} and {b}?", "answer": math.gcd(a, b), "params": {"kind": "gcd"}}
    if level == 3:
        values = [rng.randint(4, 60) for _ in range(3)]
        return {"question": f"What is the least common multiple of {values[0]}, {values[1]} and {values[2]}?", "answer": math.lcm(*values), "params": {"kind": "lcm"}}
    if level == 4:
        base, exponent, modulus = rng.randint(2, 19), rng.randint(9, 30), rng.randint(11, 97)
        return {"question": f"What is the remainder when {base} to the power {exponent} is divided by {modulus}?", "answer": pow(base, exponent, modulus), "params": {"kind": "power_mod"}}
    base, exponent, modulus = rng.randint(102, 989), rng.randint(120, 900), rng.randint(1009, 9973)
    return {"question": f"What is the remainder when {base} to the power {exponent} is divided by {modulus}?", "answer": pow(base, exponent, modulus), "params": {"kind": "power_mod_large"}}


# ------------------------------------------------------------------- tables


def table(rng: random.Random, level: int) -> dict:
    rows = {1: 3, 2: 5, 3: 6, 4: 10, 5: 16}[level]
    high = {1: 20, 2: 40, 3: 90, 4: 300, 5: 900}[level]
    teams = ["north", "south", "east", "west"]
    people = rng.sample(NAMES, min(rows, len(NAMES)))
    records = [(people[i % len(people)] + ("" if i < len(people) else " II"), rng.choice(teams), rng.randint(2, high), rng.randint(2, 60 if level > 2 else 12)) for i in range(rows)]
    lines = [f"{name}: team {team}, {units} units sold at {price} dollars each" for name, team, units, price in records]
    team = rng.choice(sorted({r[1] for r in records}))
    if level == 1:
        name, _, units, _ = rng.choice(records)
        question, answer = f"How many units did {name} sell?", units
    elif level == 2:
        question, answer = f"How many units did team {team} sell in total?", sum(r[2] for r in records if r[1] == team)
    elif level == 3:
        question, answer = f"What is the total revenue in dollars of team {team}?", sum(r[2] * r[3] for r in records if r[1] == team)
    else:
        for _ in range(10):
            cut = rng.randint(high // 4, high // 2)
            if any(r[1] != team and r[2] > cut for r in records):
                break
        question = f"What is the total revenue in dollars of the people who are not in team {team} and who sold more than {cut} units?"
        answer = sum(r[2] * r[3] for r in records if r[1] != team and r[2] > cut)
    return {"question": "Sales records:\n" + "\n".join(lines) + f"\n\n{question}", "answer": answer, "params": {"rows": rows}}


# --------------------------------------------------------------- code trace


def code_trace(rng: random.Random, level: int) -> dict:
    a, b, m = rng.randint(2, 9), rng.randint(3, 12), rng.randint(3, 11)
    if level == 1:
        code = f"x = {rng.randint(3, 30)}\ny = x * {a} + {b}\nz = y - x\nprint(z)"
    elif level == 2:
        code = f"total = {rng.randint(0, 20)}\nfor i in range(1, {rng.randint(6, 16)}):\n    if i % {rng.randint(2, 5)} == 0:\n        total += i * {a} + {b}\nprint(total)"
    elif level == 3:
        code = (
            f"total = {b}\nfor i in range({rng.randint(8, 14)}):\n    if total % {m} < {max(1, m // 2)}:\n        total += i * {a}\n"
            f"    else:\n        total -= {b}\nprint(total)"
        )
    elif level == 4:
        code = (
            f"n = {rng.randint(40, 400)}\nsteps = 0\nwhile n != 1:\n    if n % 2 == 0:\n        n = n // 2\n    else:\n        n = 3 * n + 1\n"
            f"    steps += 1\n    if steps == {rng.randint(18, 30)}:\n        break\nprint(n + steps)"
        )
    else:
        code = (
            f"state = {rng.randint(3, 50)}\ntotal = 0\nfor i in range({rng.randint(5, 8)}):\n    for j in range({rng.randint(5, 8)}):\n"
            f"        state = (state * {a} + i * j + {b}) % {rng.randint(97, 499)}\n        if state % {m} == 0:\n            total += state\n"
            f"        else:\n            total += j\nprint(total + state)"
        )
    printed: list[int] = []
    exec(code, {"__builtins__": {"range": range, "print": printed.append}})  # the code is our own template
    return {"question": f"What does this Python program print?\n\n{code}", "answer": printed[0], "params": {"lines": code.count(chr(10)) + 1}}


# ------------------------------------------------------------ dates and time


def calendar(rng: random.Random, level: int) -> dict:
    if level <= 2:
        hour, minute = rng.randint(6, 20), rng.choice([0, 5, 10, 20, 25, 40, 45, 50])
        spans = [rng.randint(15, 170) for _ in range(1 if level == 1 else 3)]
        end = hour * 60 + minute + sum(spans)
        parts = " and then ".join(f"{s} minutes" for s in spans)
        question = f"A session starts at {hour}:{minute:02d} and runs for {parts} with no breaks. How many minutes after midnight does it end?"
        return {"question": question, "answer": end, "params": {"kind": "clock"}}
    start = datetime.date(rng.randint(2021, 2027), rng.randint(1, 12), rng.randint(1, 28))
    if level == 3:
        end = start + datetime.timedelta(days=rng.randint(40, 400))
        question = f"How many days are there from {start.day} {start:%B %Y} to {end.day} {end:%B %Y}, counting the last day but not the first?"
        return {"question": question, "answer": (end - start).days, "params": {"kind": "days_between"}}
    stops = 3 if level == 4 else 6
    day, total, parts = start, 0, []
    for _ in range(stops):
        gap = rng.randint(20, 300)
        nxt = day + datetime.timedelta(days=gap)
        parts.append(f"{nxt.day} {nxt:%B %Y}")
        total += gap if len(parts) % 2 == 1 else 0
        day = nxt
    question = (
        f"A project has checkpoints on {start.day} {start:%B %Y}, " + ", ".join(parts) + ". "
        "Counting the days from each checkpoint to the next, add up the gaps that come first, third, fifth and so on in that order. What is the total number of days?"
    )
    return {"question": question, "answer": total, "params": {"kind": "alternating_gaps", "stops": stops}}


# ----------------------------------------------------------------- counting


def counting(rng: random.Random, level: int) -> dict:
    if level == 1:
        a, n = rng.randint(3, 12), rng.randint(60, 400)
        return {"question": f"How many whole numbers from 1 to {n} are multiples of {a}?", "answer": n // a, "params": {"kind": "multiples"}}
    if level == 2:
        a, b, n = rng.randint(3, 9), rng.randint(4, 11), rng.randint(100, 900)
        answer = sum(1 for k in range(1, n + 1) if k % a == 0 or k % b == 0)
        return {"question": f"How many whole numbers from 1 to {n} are multiples of {a} or of {b} (or both)?", "answer": answer, "params": {"kind": "either"}}
    if level == 3:
        a, b = rng.randint(3, 8), rng.randint(4, 12)
        low = rng.randint(50, 500)
        high = low + rng.randint(200, 2000)
        answer = sum(1 for k in range(low, high + 1) if k % a == 0 and k % b != 0)
        return {"question": f"How many whole numbers from {low} to {high} inclusive are multiples of {a} but not multiples of {b}?", "answer": answer, "params": {"kind": "but_not"}}
    if level == 4:
        a, b, c, n = rng.randint(3, 6), rng.randint(5, 9), rng.randint(7, 13), rng.randint(500, 5000)
        answer = sum(1 for k in range(1, n + 1) if k % a == 0 or k % b == 0 or k % c == 0)
        return {"question": f"How many whole numbers from 1 to {n} are multiples of at least one of {a}, {b} and {c}?", "answer": answer, "params": {"kind": "three_sets"}}
    n, target = rng.randint(2000, 60000), rng.randint(9, 22)
    answer = sum(1 for k in range(1, n + 1) if sum(map(int, str(k))) == target)
    return {"question": f"How many whole numbers from 1 to {n} have digits that add up to exactly {target}?", "answer": answer, "params": {"kind": "digit_sum"}}


# ----------------------------------------------------------------- equations


def _signed(text: str) -> str:
    return text.replace("+ -", "- ")


def equations(rng: random.Random, level: int) -> dict:
    x, y, z = rng.randint(-12, 25), rng.randint(-12, 25), rng.randint(-9, 15)
    if level == 1:
        a, b = rng.randint(2, 12), rng.randint(-30, 60)
        return {"question": _signed(f"Solve for x: {a}x + {b} = {a * x + b}."), "answer": x, "params": {"unknowns": 1}}
    if level == 2:
        a, c = rng.randint(3, 15), rng.randint(-40, 40)
        b = rng.choice([v for v in range(2, 10) if v != a])
        return {"question": _signed(f"Solve for x: {a}x + {c} = {b}x + {(a - b) * x + c}."), "answer": x, "params": {"unknowns": 1}}
    size = 2 if level == 3 else 3
    while True:
        m = [[rng.randint(1, 9) * rng.choice([1, 1, -1]) for _ in range(size)] for _ in range(size)]
        if size == 2:
            det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        else:
            det = (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )
        if det != 0:
            break
    if size == 2:
        lines = [f"{r[0]}x + {r[1]}y = {r[0] * x + r[1] * y}" for r in m]
        return {"question": _signed("Solve the system " + " and ".join(lines) + ". What is x + y?"), "answer": x + y, "params": {"unknowns": 2}}
    scale = 1 if level == 4 else 7
    x, y, z = x * scale, y * scale, z * scale
    lines = [f"{r[0]}x + {r[1]}y + {r[2]}z = {r[0] * x + r[1] * y + r[2] * z}" for r in m]
    ask, answer = ("x + y + z", x + y + z) if level == 4 else ("2x - 3y + 5z", 2 * x - 3 * y + 5 * z)
    return {"question": _signed("Solve the system " + ", ".join(lines) + f". What is {ask}?"), "answer": answer, "params": {"unknowns": 3}}


# ------------------------------------------------------------------ ordering


def ordering(rng: random.Random, level: int) -> dict:
    """Who stands where in a line, from clues about neighbours. The answer is a name."""
    people = {1: 3, 2: 4, 3: 5, 4: 7, 5: 9}[level]
    line = rng.sample(NAMES, people)
    clues = [f"{line[i + 1]} is directly behind {line[i]}" if rng.random() < 0.5 else f"{line[i]} is directly in front of {line[i + 1]}" for i in range(people - 1)]
    rng.shuffle(clues)
    position = rng.randint(1, people)
    ordinal = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"][position - 1]
    question = f"{people} people stand in a single line, one behind the other. " + ". ".join(clues) + f". Who is {ordinal} in the line, counting from the front?"
    return {"question": question, "answer": line[position - 1], "params": {"people": people}, "answer_type": "text"}


FAMILIES: dict[str, Callable[[random.Random, int], dict]] = {
    "arithmetic": arithmetic,
    "stock_story": stock_story,
    "rates": rates,
    "sequence": sequence,
    "number_theory": number_theory,
    "table": table,
    "code_trace": code_trace,
    "calendar": calendar,
    "counting": counting,
    "equations": equations,
    "ordering": ordering,
}


def make_item(family: str, level: int, index: int, seed: int) -> dict:
    """The problem a (seed, family, level, index) names. The same inputs always give the same item."""
    rng = random.Random(f"{GENERATOR_VERSION}:{seed}:{family}:{level}:{index}")
    made = FAMILIES[family](rng, level)
    answer_type = made.get("answer_type", "number")
    return {
        "id": f"{family}-L{level}-{index:05d}",
        "family": family,
        "level": level,
        "params": made["params"],
        "question": made["question"],
        "answer": str(made["answer"]),
        "answer_type": answer_type,
        "prompt": made["question"] + "\n\n" + (NUMBER_SUFFIX if answer_type == "number" else TEXT_SUFFIX),
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
    }


def grade_answer(item: dict, text: str) -> dict:
    """Whether a reply gives the item's answer. Numbers are compared as numbers."""
    from ..benchmarks.simple import parse_number

    if item["answer_type"] == "number":
        value, strict = parse_number(text)
        return {"score": float(value is not None and abs(value - float(item["answer"])) < 1e-6), "strict": strict}
    lines = [line for line in text.strip().splitlines() if line.strip()]
    said = (lines[0] if lines else "").strip().strip("*`'\".,!: ").lower()
    return {"score": float(said == item["answer"].lower()), "strict": len(lines) == 1}
