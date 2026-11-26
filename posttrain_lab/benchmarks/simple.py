"""Benchmarks graded by a label or an answer key that ships with the data."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .base import Benchmark, Item, fixed_sample, parse_choice, read_jsonl, read_parquet, strip_code_fence, user

LETTERS = "ABCDEFGHIJ"
YES_NO = ("yes", "no")


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _choice_grade(item: Item, reply: dict, choices: tuple[str, ...]) -> dict:
    choice, strict = parse_choice(reply.get("text", ""), choices)
    return {"score": float(choice is not None and choice.lower() == str(item.answer).lower()), "choice": choice, "strict": strict}


def pairwise_metrics(items: list[Item], grades: list[dict], first: str) -> dict:
    """Numbers that only make sense when every pair is shown in both orders."""
    by_pair: dict[str, list[tuple[Item, dict]]] = {}
    for item, grade in zip(items, grades):
        by_pair.setdefault(item.cluster, []).append((item, grade))
    complete = [pair for pair in by_pair.values() if len(pair) == 2]
    consistent = sum(1 for (i1, g1), (i2, g2) in complete if g1["choice"] and g2["choice"] and (g1["score"] == g2["score"]))
    both_right = sum(1 for (_, g1), (_, g2) in complete if g1["score"] == 1 and g2["score"] == 1)
    answered = [(i, g) for i, g in zip(items, grades) if g["choice"]]
    correct_first = [(i, g) for i, g in zip(items, grades) if i.answer == first]
    correct_second = [(i, g) for i, g in zip(items, grades) if i.answer != first]
    return {
        "pairs": len(complete),
        "swap_consistency": _rate(consistent, len(complete)),
        "both_orders_correct": _rate(both_right, len(complete)),
        "picked_first_position": _rate(sum(1 for _, g in answered if g["choice"] == first), len(answered)),
        "accuracy_when_correct_is_first": _rate(sum(int(g["score"]) for _, g in correct_first), len(correct_first)),
        "accuracy_when_correct_is_second": _rate(sum(int(g["score"]) for _, g in correct_second), len(correct_second)),
        "unparsed": sum(1 for g in grades if not g["choice"]),
    }


class JudgeBench(Benchmark):
    name = "judgebench"
    title = "JudgeBench, pairwise"
    job = "Judging answers"
    role = "target"
    graded_by = "labels that ship with the benchmark"
    source = "judgebench"
    choices = ("A", "B")
    chance = 0.5
    max_tokens_fast = 16

    PROMPT = (
        "You are judging two responses to the same question. Exactly one of them is correct.\n\n"
        "Question:\n{question}\n\nResponse A:\n{first}\n\nResponse B:\n{second}\n\n"
        "Which response is correct? Answer with one letter, A or B."
    )

    def rows(self, root: Path) -> list[dict]:
        rows = []
        for split in ("gpt", "claude"):
            for row in read_jsonl(root / self.source / f"data/{split}-00000-of-00001.jsonl"):
                rows.append({**row, "split": split})
        return rows

    @staticmethod
    def family(source: str) -> str:
        return "mmlu-pro" if source.startswith("mmlu-pro") else source

    def load(self, root: Path, seed: int) -> list[Item]:
        items = []
        for row in self.rows(root):
            correct = "A" if row["label"] == "A>B" else "B"
            for order in ("ab", "ba"):
                first, second = (row["response_A"], row["response_B"]) if order == "ab" else (row["response_B"], row["response_A"])
                answer = correct if order == "ab" else ("B" if correct == "A" else "A")
                items.append(
                    Item(
                        id=f"{row['pair_id']}:{order}",
                        messages=user(self.PROMPT.format(question=row["question"], first=first, second=second)),
                        answer=answer,
                        group=self.family(row["source"]),
                        cluster=row["pair_id"],
                    )
                )
        return items

    def grade(self, item: Item, reply: dict) -> dict:
        return _choice_grade(item, reply, self.choices)

    def extra_metrics(self, items: list[Item], grades: list[dict]) -> dict:
        return pairwise_metrics(items, grades, first="A")


class JudgeBenchPointwise(JudgeBench):
    name = "judgebench_pointwise"
    title = "JudgeBench, one response at a time"
    choices = ("yes", "no", "Yes", "No")

    PROMPT = (
        "You are checking a response to a question.\n\nQuestion:\n{question}\n\nResponse:\n{response}\n\n"
        "Is the response correct? Answer with one word, yes or no."
    )

    def load(self, root: Path, seed: int) -> list[Item]:
        items = []
        for row in self.rows(root):
            for side in ("A", "B"):
                is_correct = (row["label"] == "A>B") == (side == "A")
                items.append(
                    Item(
                        id=f"{row['pair_id']}:{side}",
                        messages=user(self.PROMPT.format(question=row["question"], response=row[f"response_{side}"])),
                        answer="yes" if is_correct else "no",
                        group=self.family(row["source"]),
                        cluster=row["pair_id"],
                    )
                )
        return items

    def grade(self, item: Item, reply: dict) -> dict:
        return _choice_grade(item, reply, YES_NO)

    def extra_metrics(self, items: list[Item], grades: list[dict]) -> dict:
        wrong = [g for i, g in zip(items, grades) if i.answer == "no"]
        right = [g for i, g in zip(items, grades) if i.answer == "yes"]
        false_accept = _rate(sum(1 for g in wrong if g["choice"] == "yes"), len(wrong))
        false_reject = _rate(sum(1 for g in right if g["choice"] == "no"), len(right))
        return {
            "false_accept_rate": false_accept,  # said yes to an incorrect response
            "false_reject_rate": false_reject,  # said no to a correct response
            "said_yes": _rate(sum(1 for g in grades if g["choice"] == "yes"), len(grades)),
            "unparsed": sum(1 for g in grades if not g["choice"]),
        }


class LLMBar(Benchmark):
    name = "llmbar"
    title = "LLMBar"
    job = "Judging answers"
    role = "target"
    graded_by = "labels that ship with the benchmark"
    source = "llmbar"
    choices = ("1", "2")
    chance = 0.5
    max_tokens_fast = 16
    SUBSETS = ("Natural", "Adversarial_Neighbor", "Adversarial_GPTInst", "Adversarial_GPTOut", "Adversarial_Manual")

    PROMPT = (
        "You are judging which of two outputs better follows an instruction.\n\n"
        "Instruction:\n{instruction}\n\nOutput 1:\n{first}\n\nOutput 2:\n{second}\n\n"
        "Which output better follows the instruction? Answer with one digit, 1 or 2."
    )

    def load(self, root: Path, seed: int) -> list[Item]:
        items = []
        for subset in self.SUBSETS:
            rows = json.loads((root / self.source / f"{subset}.json").read_text(encoding="utf-8"))
            for index, row in enumerate(rows):
                for order in ("ab", "ba"):
                    first, second = (row["output_1"], row["output_2"]) if order == "ab" else (row["output_2"], row["output_1"])
                    answer = str(row["label"]) if order == "ab" else str(3 - row["label"])
                    items.append(
                        Item(
                            id=f"{subset}-{index}:{order}",
                            messages=user(self.PROMPT.format(instruction=row["input"], first=first, second=second)),
                            answer=answer,
                            group=subset,
                            cluster=f"{subset}-{index}",
                        )
                    )
        return items

    def grade(self, item: Item, reply: dict) -> dict:
        return _choice_grade(item, reply, self.choices)

    def extra_metrics(self, items: list[Item], grades: list[dict]) -> dict:
        return pairwise_metrics(items, grades, first="1")


class RewardBench2(Benchmark):
    name = "rewardbench2"
    title = "RewardBench 2, best of four"
    job = "Judging answers"
    role = "target"
    graded_by = "labels that ship with the benchmark"
    source = "rewardbench2"
    choices = ("A", "B", "C", "D")
    chance = 0.25
    macro = True
    max_tokens_fast = 16

    PROMPT = (
        "A user sent the prompt below, and four responses were written. Pick the best response.\n\n"
        "Prompt:\n{prompt}\n\n{responses}\n\nWhich response is best? Answer with one letter: A, B, C or D."
    )

    def load(self, root: Path, seed: int) -> list[Item]:
        items = []
        for row in read_parquet(root / self.source / "data/test-00000-of-00001.parquet"):
            # The Ties subset has several right answers and its own rating-based
            # metric, so it does not fit a pick-one task and is left out.
            if row["subset"] == "Ties" or len(row["chosen"]) != 1 or len(row["rejected"]) != 3:
                continue
            candidates = [(row["chosen"][0], True)] + [(text, False) for text in row["rejected"]]
            random.Random(f"{seed}:{row['id']}").shuffle(candidates)
            responses = "\n\n".join(f"Response {LETTERS[i]}:\n{text}" for i, (text, _) in enumerate(candidates))
            answer = LETTERS[[is_best for _, is_best in candidates].index(True)]
            items.append(
                Item(
                    id=f"{row['subset']}:{row['id']}",  # the dataset's own ids repeat across subsets
                    messages=user(self.PROMPT.format(prompt=row["prompt"], responses=responses)),
                    answer=answer,
                    group=row["subset"],
                )
            )
        return items

    def grade(self, item: Item, reply: dict) -> dict:
        return _choice_grade(item, reply, self.choices)

    def extra_metrics(self, items: list[Item], grades: list[dict]) -> dict:
        by_position = {}
        for letter in self.choices:
            subset = [g for i, g in zip(items, grades) if i.answer == letter]
            by_position[letter] = _rate(sum(int(g["score"]) for g in subset), len(subset))
        return {"accuracy_by_position_of_best": by_position, "unparsed": sum(1 for g in grades if not g["choice"])}


class MMLUPro(Benchmark):
    name = "mmlu_pro"
    title = "MMLU-Pro (sample of 500)"
    job = "Instructions and knowledge"
    role = "watch"
    graded_by = "answer key"
    source = "mmlu_pro"
    choices = tuple(LETTERS)
    chance = 0.1
    max_tokens_fast = 16
    SAMPLE = 500

    def load(self, root: Path, seed: int) -> list[Item]:
        rows = read_parquet(root / self.source / "data/test-00000-of-00001.parquet")
        items = []
        for row in fixed_sample(rows, self.SAMPLE, seed):
            options = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(row["options"]))
            prompt = f"{row['question']}\n\n{options}\n\nAnswer with the letter of the correct option and nothing else."
            items.append(
                Item(id=str(row["question_id"]), messages=user(prompt), answer=row["answer"], group=row["category"],
                     extra={"options": len(row["options"])})
            )
        return items

    def grade(self, item: Item, reply: dict) -> dict:
        return _choice_grade(item, reply, tuple(LETTERS[: item.extra["options"]]))


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _bare_number(text: str) -> float | None:
    cleaned = text.strip().strip("*`$ \n").rstrip(".%").replace(",", "").replace("$", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_number(text: str) -> tuple[float | None, bool]:
    """The number a reply gives. Strict when the reply is a bare number.

    A reply that opens with a bare number on its own line and then keeps talking is
    read as that number. Otherwise the last number in the reply is taken."""
    value = _bare_number(text)
    if value is not None:
        return value, True
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and _bare_number(lines[0]) is not None:
        return _bare_number(lines[0]), False
    found = _NUMBER.findall(text)
    if not found:
        return None, False
    try:
        return float(found[-1].replace(",", "")), False
    except ValueError:
        return None, False


class GSM8K(Benchmark):
    name = "gsm8k"
    title = "GSM8K"
    job = "Reasoning without the wait"
    role = "candidate"
    graded_by = "answer key"
    source = "gsm8k"
    max_tokens_fast = 256

    PROMPT = "{question}\n\nSolve the problem. Give only the final answer as a number, with no units and no explanation."

    def load(self, root: Path, seed: int) -> list[Item]:
        rows = read_parquet(root / self.source / "main/test-00000-of-00001.parquet")
        return [
            Item(
                id=f"gsm8k-{index}",
                messages=user(self.PROMPT.format(question=row["question"])),
                answer=row["answer"].split("####")[-1].strip().replace(",", ""),
            )
            for index, row in enumerate(rows)
        ]

    def grade(self, item: Item, reply: dict) -> dict:
        value, strict = parse_number(reply.get("text", ""))
        correct = value is not None and abs(value - float(item.answer)) < 1e-6
        return {"score": float(correct), "value": value, "strict": strict}


def _has_outside_reference(node) -> bool:
    if isinstance(node, dict):
        reference = node.get("$ref")
        if isinstance(reference, str) and not reference.startswith("#"):
            return True
        return any(_has_outside_reference(value) for value in node.values())
    if isinstance(node, list):
        return any(_has_outside_reference(value) for value in node)
    return False


class JSONSchemaBench(Benchmark):
    name = "jsonschemabench"
    title = "JSONSchemaBench (sample of 1,000)"
    job = "Filling a schema"
    role = "candidate"
    graded_by = "a JSON Schema validator"
    source = "jsonschemabench"
    max_tokens_fast = 2048
    SAMPLE = 1000

    PROMPT = (
        "Write one JSON value that is valid against the JSON Schema below. "
        "Reply with the JSON only, with no explanation and no code fence.\n\n{schema}"
    )

    def gradable(self, root: Path) -> tuple[list[dict], int]:
        """Schemas that can be checked offline, and how many were set aside."""
        from jsonschema.validators import validator_for

        kept, dropped = [], 0
        for row in read_parquet(root / self.source / "data/test-00000-of-00001.parquet"):
            try:
                schema = json.loads(row["json_schema"])
                validator_for(schema).check_schema(schema)
                if _has_outside_reference(schema):
                    raise ValueError("refers to a document outside itself")
            except Exception:
                dropped += 1
                continue
            kept.append(row)
        return kept, dropped

    def load(self, root: Path, seed: int) -> list[Item]:
        rows, _ = self.gradable(root)
        return [
            Item(id=row["unique_id"], messages=user(self.PROMPT.format(schema=row["json_schema"])), answer=row["json_schema"])
            for row in fixed_sample(rows, self.SAMPLE, seed)
        ]

    def grade(self, item: Item, reply: dict) -> dict:
        from jsonschema.validators import validator_for
        from referencing import Registry

        text = strip_code_fence(reply.get("text", ""), "json")
        try:
            value = json.loads(text)
        except (ValueError, RecursionError):
            return {"score": 0.0, "error": "not JSON"}
        schema = json.loads(item.answer)
        try:
            # An empty registry: a schema is never allowed to fetch anything.
            validator = validator_for(schema)(schema, registry=Registry())
            error = next(iter(validator.iter_errors(value)), None)
        except Exception as exc:  # a reference the schema cannot resolve, or a regex it cannot compile
            return {"score": 0.0, "error": f"could not validate: {type(exc).__name__}"}
        return {"score": float(error is None), "error": error.message[:200] if error else None}

    def extra_metrics(self, items: list[Item], grades: list[dict]) -> dict:
        return {
            "not_json": sum(1 for g in grades if g.get("error") == "not JSON"),
            "invalid_against_schema": sum(1 for g in grades if g["score"] == 0 and g.get("error") != "not JSON"),
        }
