"""Generators, the leak check and the set builder."""

from __future__ import annotations

import json
import random
import re

import pytest

from posttrain_lab.benchmarks.base import Benchmark, Item, user
from posttrain_lab.data import generators
from posttrain_lab.data.leak import closest, jaccard, leak_check, mask_numbers, normalize, shingles
from posttrain_lab.data.sets import build_sets, cell_of, length_matched, permuted_answers, take_tokens
from posttrain_lab.steps.base import StepContext, run_step
from posttrain_lab.steps.build_data import BuildData
from posttrain_lab.steps.filter import Filter
from posttrain_lab.store import JsonlStore
from posttrain_lab.testing import FakeBackend

from conftest import run

# ------------------------------------------------------------ generators


@pytest.mark.parametrize("family", list(generators.FAMILIES))
def test_each_family_is_repeatable_and_its_answers_grade(family):
    for level in generators.LEVELS:
        if level > 5 and family not in generators.DEEP_FAMILIES:
            with pytest.raises(ValueError):
                generators.make_item(family, level, 0, 11)
            continue
        first = [generators.make_item(family, level, i, 11) for i in range(30)]
        assert first == [generators.make_item(family, level, i, 11) for i in range(30)]
        assert [i["question"] for i in first] != [generators.make_item(family, level, i, 12)["question"] for i in range(30)]
        for item in first:
            assert item["id"] == f"{family}-L{level}-{first.index(item):05d}" and item["prompt"].startswith(item["question"])
            assert generators.grade_answer(item, item["answer"])["score"] == 1.0
            assert generators.grade_answer(item, "I do not know")["score"] == 0.0
            if item["answer_type"] == "number":
                assert str(int(item["answer"])) == item["answer"]
                assert generators.grade_answer(item, str(int(item["answer"]) + 1))["score"] == 0.0
                assert item["prompt"].endswith(generators.NUMBER_SUFFIX)


def test_problems_grow_with_the_level():
    for family in ("arithmetic", "stock_story", "rates", "table", "ordering"):
        lengths = [sum(len(generators.make_item(family, level, i, 3)["question"]) for i in range(40)) for level in (1, 2, 3, 4, 5)]
        assert lengths == sorted(lengths), family


def test_answers_are_right_when_checked_another_way():
    for index in range(40):
        item = generators.make_item("arithmetic", 3, index, 5)
        expression = item["question"].removeprefix("What is the value of ").removesuffix("?")
        assert re.fullmatch(r"[0-9+\-* ()]+", expression) and eval(expression) == int(item["answer"])  # digits and operators only
        trace = generators.make_item("code_trace", 2, index, 5)
        printed: list[int] = []
        exec(trace["question"].split("\n\n", 1)[1], {"__builtins__": {"range": range, "print": printed.append}})
        assert printed == [int(trace["answer"])]
        line = generators.make_item("ordering", 3, index, 5)
        behind = dict(re.findall(r"(\w+) is directly behind (\w+)", line["question"])) | {b: a for a, b in re.findall(r"(\w+) is directly in front of (\w+)", line["question"])}
        front = (set(behind.values()) - set(behind)).pop()
        order = [front]
        ahead_of = {v: k for k, v in behind.items()}
        while order[-1] in ahead_of:
            order.append(ahead_of[order[-1]])
        position = ["first", "second", "third", "fourth", "fifth"].index(re.search(r"Who is (\w+) in the line", line["question"]).group(1))
        assert order[position] == line["answer"]


def test_digit_sum_counts_match_brute_force():
    for n in (7, 99, 1000, 4821, 60000):
        brute: dict[int, int] = {}
        for k in range(0, n + 1):
            total = sum(map(int, str(k)))
            brute[total] = brute.get(total, 0) + 1
        assert {k: v for k, v in generators.digit_sum_counts(n).items() if v} == brute
    item = generators.make_item("counting", 6, 0, 1)
    n, target = map(int, re.findall(r"\d+", item["question"])[1:3])
    assert int(item["answer"]) == generators.digit_sum_counts(n).get(target, 0)


def test_grading_reads_the_answer_first_and_names_without_case():
    number = {"answer": "42", "answer_type": "number"}
    assert generators.grade_answer(number, "42\n\nCheck: 6 times 7.")["score"] == 1.0
    assert generators.grade_answer(number, "It is 41 or 42")["score"] == 1.0 and generators.grade_answer(number, "42 or 41")["score"] == 0.0
    name = {"answer": "Greta", "answer_type": "text"}
    assert generators.grade_answer(name, "greta.")["score"] == 1.0 and generators.grade_answer(name, "Hugo")["score"] == 0.0


# ------------------------------------------------------------ leak check

REFERENCES = [
    ("gsm:1", "Mara has 12 apples and buys 30 more. How many apples does she have?"),
    ("gsm:2", "A train leaves the station at noon travelling 80 km per hour for 3 hours. How far does it go?"),
    ("long:1", "word " * 5000),
]


def test_leak_check_finds_exact_normalized_and_similar_items():
    candidates = [
        {"id": "same", "question": "Mara has 12 apples and buys 30 more. How many apples does she have?"},
        {"id": "spacing", "question": "mara has 12 apples,  and buys 30 more -- how many apples does she have"},
        {"id": "numbers", "question": "Mara has 57 apples and buys 4 more. How many apples does she have?"},
        {"id": "fresh", "question": "What is the least common multiple of 14, 21 and 35?"},
    ]
    kept, dropped, stats = leak_check(candidates, REFERENCES)
    assert [c["id"] for c in kept] == ["fresh"]
    assert {d["id"]: d["reason"] for d in dropped} == {"same": "exact", "spacing": "normalized", "numbers": "similar"}
    assert all(d["reference"] == "gsm:1" for d in dropped)
    assert stats["dropped"] == {"exact": 1, "normalized": 1, "similar": 1} and stats["references_too_long_or_short_to_match"] == 1
    assert stats["kept"] == 1 and stats["threshold"] == 0.5 and stats["ngram"] == 5


def test_normalizing_and_masking():
    assert normalize("  What's 3 + 4?\n") == "what s 3 4"
    assert mask_numbers("buy 120 for 3") == "buy 0 for 0"
    assert jaccard(shingles("abcdefgh"), shingles("abcdefgh")) == 1.0 and jaccard(shingles("abcdefgh"), shingles("zzzzzzzz")) == 0.0


def test_the_indexed_search_agrees_with_brute_force():
    rng = random.Random(4)
    words = ["tank", "pump", "litres", "minutes", "adds", "removes", "each", "how", "many", "books", "crates", "then", "a", "the", "holds", "of"]
    def sentence(n):
        return " ".join(rng.choice(words) for _ in range(n))
    references = [(f"r{i}", sentence(rng.randint(8, 30))) for i in range(300)]
    candidates = [{"id": f"c{i}", "question": sentence(rng.randint(8, 30))} for i in range(200)]
    # Plant some near copies so there is something to find.
    for i in range(0, 200, 10):
        label, text = references[i]
        candidates[i]["question"] = text + " " + rng.choice(words)
    for threshold in (0.3, 0.5, 0.8):
        kept, dropped, _ = leak_check(candidates, references, threshold=threshold)
        expected = {c["id"] for c in candidates if closest(c["question"], references)[0] >= threshold or normalize(c["question"]) in {normalize(t) for _, t in references}}
        assert {d["id"] for d in dropped} == expected and len(kept) + len(dropped) == 200
        assert expected, threshold


# ---------------------------------------------------------------- sets


def test_cells():
    assert [cell_of(True, True), cell_of(False, True), cell_of(False, False), cell_of(True, False)] == ["easy", "middle", "hard", "rare"]


def labeled_items(n=3000, seed=1):
    rng = random.Random(seed)
    cells = ["middle"] * 5 + ["easy"] * 3 + ["hard"] * 3 + ["rare"]
    return [
        {"id": f"i{k:04d}", "cell": rng.choice(cells), "family": rng.choice(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
         "answer": str(rng.randint(0, 400)), "train_tokens": rng.randint(40, 160)}
        for k in range(n)
    ]


def test_take_tokens_is_nested_and_close_to_the_budget():
    tokens = {f"x{k}": 10 + k % 7 for k in range(200)}
    order = list(tokens)
    small, large = take_tokens(order, tokens, 500), take_tokens(order, tokens, 1500)
    assert large[: len(small)] == small
    assert abs(sum(tokens[i] for i in small) - 500) <= 8 and abs(sum(tokens[i] for i in large) - 1500) <= 8


def test_training_sets_are_nested_matched_and_disjoint_from_what_is_held_out():
    items = labeled_items()
    built = build_sets(items, seed=9, dev_per_cell=30, sizes=(50, 100, 400), main_size=100)
    sets, cell, tokens = built["sets"], {i["id"]: i["cell"] for i in items}, {i["id"]: i["train_tokens"] for i in items}
    assert built == build_sets(items, seed=9, dev_per_cell=30, sizes=(50, 100, 400), main_size=100)
    assert build_sets(items, seed=10, dev_per_cell=30, sizes=(50, 100, 400), main_size=100)["sets"]["middle-100"]["items"] != sets["middle-100"]["items"]
    for size, smaller in ((100, 50), (400, 100)):
        for kind in ("middle", "random"):
            assert sets[f"{kind}-{size}"]["items"][: sets[f"{kind}-{smaller}"]["n_items"]] == sets[f"{kind}-{smaller}"]["items"]
    for size in (50, 100, 400):
        assert sets[f"middle-{size}"]["n_items"] == size and {cell[i] for i in sets[f"middle-{size}"]["items"]} == {"middle"}
        assert abs(sets[f"random-{size}"]["train_tokens"] - sets[f"middle-{size}"]["train_tokens"]) <= 160
        assert len({cell[i] for i in sets[f"random-{size}"]["items"]}) > 1
    budget = sets["middle-100"]["train_tokens"]
    for name, only in (("easy-100", "easy"), ("hard-100", "hard")):
        assert {cell[i] for i in sets[name]["items"]} == {only} and abs(sets[name]["train_tokens"] - budget) <= 160
    held_out = {i for ids in built["dev"].values() for i in ids} | set(built["prompt_examples"])
    assert all(len(built["dev"][c]) == 30 for c in ("easy", "middle", "hard")) and len(built["prompt_examples"]) == 8
    assert {cell[i] for i in built["prompt_examples"]} == {"middle"} and len({next(x["family"] for x in items if x["id"] == i) for i in built["prompt_examples"]}) == 8
    for name, entry in sets.items():
        assert not held_out & set(entry["items"]), name
        assert len(set(entry["items"])) == entry["n_items"] and entry["train_tokens"] == sum(tokens[i] for i in entry["items"])


def test_shuffled_labels_and_length_matching():
    items = labeled_items()
    built = build_sets(items, seed=9, dev_per_cell=30, sizes=(50, 100, 400), main_size=100)
    answers, tokens = {i["id"]: i["answer"] for i in items}, {i["id"]: i["train_tokens"] for i in items}
    main, wrong = built["sets"]["middle-100"]["items"], built["sets"]["shuffled-labels-100"]
    assert wrong["items"] == main and sorted(wrong["answers"].values()) == sorted(answers[i] for i in main)
    assert wrong["unchanged_answers"] == sum(1 for i in main if wrong["answers"][i] == answers[i]) <= 2
    matched = built["sets"]["length-matched-random-100"]["items"]
    assert len(matched) == 100 and len(set(matched)) == 100
    assert sorted(tokens[i] for i in matched) == sorted(tokens[i] for i in main)  # the pool is large enough for exact matches
    assert permuted_answers(["a", "b", "c"], {"a": "1", "b": "2", "c": "3"}, 1) in ({"a": "2", "b": "3", "c": "1"}, {"a": "3", "b": "1", "c": "2"})
    assert length_matched(["t"], ["p", "q"], {"t": 50, "p": 10, "q": 49}, 1) == ["q"]


# ---------------------------------------------------------------- steps


class TinyBenchmark(Benchmark):
    name, source = "tiny", "tiny"

    def load(self, root, seed):
        return [Item(id="t1", messages=user("wrapped"), source_text="What is the value of 12 + 30?"), Item(id="t2", messages=user("How many loaves are left after the fair?"))]


def context(tmp_path, renderer, step, client, **config):
    lines: list[str] = []
    return StepContext(step=step, config=config, renderer=renderer, runs_dir=tmp_path / "runs" / step, results_dir=tmp_path / "results" / step,
                       store=JsonlStore(tmp_path / "runs" / step / "results.jsonl"), client=client, out=lines.append), lines


def test_build_data_generates_counts_tokens_and_drops_leaks(tmp_path, renderer, make_client):
    client, backend = make_client()
    config = {"seed": 2, "counts": {"arithmetic": [40, 0, 0, 0, 0], "ordering": [0, 10, 0, 0, 0]}, "_registry": {"tiny": TinyBenchmark()}}
    ctx, lines = context(tmp_path, renderer, "build-data", client, **config)
    assert run(run_step(BuildData(), ctx, dry_run=True)) == 0 and not (tmp_path / "runs").exists()
    assert run(run_step(BuildData(), ctx)) == 0 and backend.sample_calls == []
    kept = [json.loads(line) for line in (ctx.runs_dir / "candidates.jsonl").read_text().splitlines()]
    dropped = [json.loads(line) for line in (ctx.runs_dir / "dropped.jsonl").read_text().splitlines()]
    summary = json.loads((ctx.runs_dir / "summary.json").read_text())
    # Level 1 arithmetic is "What is the value of A + B?": the same wording as the planted benchmark question.
    assert len(kept) == 10 and {d["reason"] for d in dropped} <= {"similar", "exact", "normalized"} and len(dropped) == 40
    assert all(d["reference"] == "tiny:t1" for d in dropped)
    assert summary["generated"] == 50 and summary["kept"] == 10 and summary["by_family"]["ordering"]["L2"] == 10
    assert summary["leak_check"]["references"] == 2
    item = kept[0]
    assert item["prompt_tokens"] == len(renderer.build_prompt([{"role": "user", "content": item["prompt"]}], 0.0)) + 2
    assert item["train_tokens"] > item["prompt_tokens"] - 2 and item["generator_version"] == generators.GENERATOR_VERSION


def test_filter_labels_cells_and_writes_the_dataset(tmp_path, renderer, make_client):
    build_dir = tmp_path / "runs" / "build-data"
    build_dir.mkdir(parents=True)
    items = []
    for index in range(120):
        item = generators.make_item("arithmetic", 1 + index % 3, index, 2)
        item["prompt_tokens"], item["train_tokens"] = 30 + index % 5, 34 + index % 5
        items.append(item)
    (build_dir / "candidates.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    (build_dir / "summary.json").write_text(json.dumps({"generated": 120, "kept": 120, "repeats_skipped": 0, "by_family": {"arithmetic": {"L1": 40, "L2": 40, "L3": 40}},
                                                        "leak_check": {"references": 5, "ngram": 5, "threshold": 0.5, "dropped": {}}}))
    by_prompt = {" ".join(item["prompt"].split()): item for item in items}

    def solve(forced):
        """Right on level 1 either way, right on level 2 only when it reasons, wrong on level 3."""

        def answer(text):
            item = next(v for k, v in by_prompt.items() if text.endswith(k))
            return item["answer"] if item["level"] == 1 or (item["level"] == 2 and not forced) else "0"

        return answer

    backend = FakeBackend(renderer, answer=solve(False), forced_answer=solve(True))
    client, _ = make_client(backend, max_concurrency=4, call_log=tmp_path / "runs" / "filter" / "calls.jsonl")
    ctx, lines = context(tmp_path, renderer, "filter", client, seed=5, settings=["fast", "reasoning"], sizes=[5, 10, 20], main_size=10, dev_per_cell=3, model_id="m:free", client_={"max_concurrency": 4})
    assert run(run_step(Filter(), ctx)) == 0 and len(backend.sample_calls) == 240
    out = tmp_path / "results" / "data"
    summary = json.loads((out / "summary.json").read_text())
    assert {c: summary["cells"]["overall"][c]["count"] for c in ("easy", "middle", "hard", "rare")} == {"easy": 40, "middle": 40, "hard": 40, "rare": 0}
    assert summary["cells"]["by_level"]["L2"]["middle"]["share"] == 1.0 and summary["accuracy"] == {"fast": round(40 / 120, 4), "reasoning": round(80 / 120, 4)}
    assert summary["sets"]["middle-20"]["n_items"] == 20 and summary["sets"]["middle-20"]["cells"]["middle"] == 20
    assert summary["dev"] == {"easy": 3, "middle": 3, "hard": 3, "rare": 0} and len(summary["prompt_examples"]) == 8
    assert summary["available_after_holding_out"]["middle"] == 40 - 3 - 8
    sets = json.loads((out / "sets.json").read_text())
    assert set(sets["sets"]) == {"middle-5", "random-5", "middle-10", "random-10", "middle-20", "random-20", "easy-10", "hard-10", "shuffled-labels-10", "length-matched-random-10"}
    import gzip, hashlib
    rows = [json.loads(line) for line in gzip.open(out / "items.jsonl.gz", "rt")]
    assert "canary" in rows[0] and len(rows) == 121 and rows[1]["cell"] in ("easy", "middle", "hard")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["files"]["items.jsonl.gz"]["sha256"] == hashlib.sha256((out / "items.jsonl.gz").read_bytes()).hexdigest()
    report = (out / "REPORT.md").read_text()
    assert "## The four cells" in report and "## Training sets" in report and "middle band: 40" in report
    figure = json.loads((out / "figure-data.json").read_text())
    assert [c["cell"] for c in figure["four_cells"]] == ["easy", "middle", "hard", "rare"] and figure["by_family"][0]["family"] == "arithmetic"
