"""Loaders, graders and statistics, on small made-up fixtures.

Nothing here is a benchmark item. Tests that need a downloaded checker are skipped
when the survey's source directory is not present.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from posttrain_lab.benchmarks import BENCHMARKS, NOT_RUN, bfcl, sandbox, sources
from posttrain_lab.benchmarks.base import Item, fixed_sample, parse_choice, strip_code_fence
from posttrain_lab.benchmarks.code import MBPP
from posttrain_lab.benchmarks.ifeval import IFEval
from posttrain_lab.benchmarks.simple import (
    GSM8K, JSONSchemaBench, JudgeBench, JudgeBenchPointwise, LLMBar, MMLUPro, RewardBench2, parse_number,
)
from posttrain_lab.stats import bootstrap, macro_mean, paired_difference
from posttrain_lab.steps.survey import classify, reference_subsample

SOURCES = Path(__file__).resolve().parent.parent / "runs" / "survey" / "sources"
needs_sources = pytest.mark.skipif(not (SOURCES / "bfcl").exists(), reason="benchmark sources are not downloaded")


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(pa.Table.from_pylist(rows), path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text, expected, strict",
    [
        ("A", "A", True),
        (" b.\n", "B", True),
        ("**A**", "A", True),
        ("(B)", "B", True),
        ("The answer is B.", "B", False),
        ("Response A is correct because B is wrong.", "A", False),
        ("B, since the proof is complete", "B", False),
        ("Neither is good", None, False),
        ("", None, False),
    ],
)
def test_parse_choice_letters(text, expected, strict):
    assert parse_choice(text, ("A", "B")) == (expected, strict)


def test_parse_choice_accepts_a_repeated_choice_but_not_a_mixed_one():
    assert parse_choice("AA", ("A", "B")) == ("A", False)
    assert parse_choice("yesyes", ("yes", "no")) == ("yes", False)
    assert parse_choice("I I", tuple("ABCDEFGHIJ")) == ("I", False)
    assert parse_choice("yesno", ("yes", "no")) == (None, False)
    assert parse_choice("AB", ("A", "B")) == (None, False)


def test_parse_choice_words_and_digits():
    assert parse_choice("Yes.", ("yes", "no")) == ("yes", True)
    assert parse_choice("No, the sum is wrong.", ("yes", "no")) == ("no", False)
    assert parse_choice("Output 2", ("1", "2")) == ("2", False)
    assert parse_choice("12", ("1", "2")) == (None, False)


@pytest.mark.parametrize(
    "text, value, strict",
    [("18", 18.0, True), ("$1,250.50", 1250.5, True), ("-3", -3.0, True), ("The total is 42 apples.", 42.0, False),
     ("3 + 4 = 7", 7.0, False), ("no idea", None, False), ("15%", 15.0, True)],
)
def test_parse_number(text, value, strict):
    assert parse_number(text) == (value, strict)


def test_strip_code_fence():
    assert strip_code_fence("```python\nx = 1\n```\nDone.", "python") == "x = 1"
    assert strip_code_fence("x = 1") == "x = 1"
    assert strip_code_fence("```json\n{\"a\": 1}", "json") == '{"a": 1}'


def test_fixed_sample_is_repeatable_and_nested():
    items = list(range(100))
    small, large = fixed_sample(items, 10, 3), fixed_sample(items, 40, 3)
    assert small == fixed_sample(items, 10, 3) and len(small) == 10
    assert set(small) <= set(large) and small == sorted(small)
    assert fixed_sample(items, 10, 4) != small
    assert fixed_sample(items, 500, 3) == items


def test_reference_subsample_keeps_both_orders_of_a_pair():
    items = [Item(id=f"p{p}:{o}", messages=[], cluster=f"p{p}") for p in range(50) for o in ("ab", "ba")]
    chosen = reference_subsample(items, 20, seed=1)
    assert len(chosen) == 20
    assert all((f"{i.split(':')[0]}:ab" in chosen) == (f"{i.split(':')[0]}:ba" in chosen) for i in chosen)
    assert chosen == reference_subsample(items, 20, seed=1)


# -------------------------------------------------------------- judging


def judge_fixture(root: Path) -> None:
    rows = [
        {"pair_id": "p1", "original_id": 1, "source": "mmlu-pro-law", "question": "Is water wet?", "response_model": "m",
         "response_A": "Yes it is.", "response_B": "No it is dry.", "label": "A>B"},
        {"pair_id": "p2", "original_id": 2, "source": "livebench-math", "question": "2+2?", "response_model": "m",
         "response_A": "5", "response_B": "4", "label": "B>A"},
    ]
    write_jsonl(root / "judgebench/data/gpt-00000-of-00001.jsonl", rows[:1])
    write_jsonl(root / "judgebench/data/claude-00000-of-00001.jsonl", rows[1:])


def test_judgebench_shows_each_pair_in_both_orders(tmp_path):
    judge_fixture(tmp_path)
    bench = JudgeBench()
    items = bench.load(tmp_path, seed=0)
    assert [i.id for i in items] == ["p1:ab", "p1:ba", "p2:ab", "p2:ba"]
    assert [i.answer for i in items] == ["A", "B", "B", "A"]
    assert [i.group for i in items] == ["mmlu-pro", "mmlu-pro", "livebench-math", "livebench-math"]
    assert items[0].cluster == items[1].cluster == "p1"
    first, swapped = items[0].messages[0]["content"], items[1].messages[0]["content"]
    assert first.index("Yes it is.") < first.index("No it is dry.") and swapped.index("No it is dry.") < swapped.index("Yes it is.")
    # A judge that always says A: right once per pair, never consistent.
    grades = [bench.grade(item, {"text": "A"}) for item in items]
    assert [g["score"] for g in grades] == [1.0, 0.0, 0.0, 1.0]
    metrics = bench.extra_metrics(items, grades)
    assert metrics["swap_consistency"] == 0.0 and metrics["both_orders_correct"] == 0.0
    assert metrics["picked_first_position"] == 1.0 and metrics["unparsed"] == 0
    assert metrics["accuracy_when_correct_is_first"] == 1.0 and metrics["accuracy_when_correct_is_second"] == 0.0
    # A judge that is always right is consistent.
    right = [bench.grade(item, {"text": item.answer}) for item in items]
    assert bench.extra_metrics(items, right)["swap_consistency"] == 1.0


def test_judgebench_pointwise_separates_false_accepts_from_false_rejects(tmp_path):
    judge_fixture(tmp_path)
    bench = JudgeBenchPointwise()
    items = bench.load(tmp_path, seed=0)
    assert [(i.id, i.answer) for i in items] == [("p1:A", "yes"), ("p1:B", "no"), ("p2:A", "no"), ("p2:B", "yes")]
    assert "Response A" not in items[0].messages[0]["content"]
    always_yes = [bench.grade(item, {"text": "Yes"}) for item in items]
    metrics = bench.extra_metrics(items, always_yes)
    assert metrics["false_accept_rate"] == 1.0 and metrics["false_reject_rate"] == 0.0 and metrics["said_yes"] == 1.0
    mixed = [bench.grade(item, {"text": text}) for item, text in zip(items, ["no", "no", "yes", "maybe"])]
    metrics = bench.extra_metrics(items, mixed)
    assert metrics["false_accept_rate"] == 0.5 and metrics["false_reject_rate"] == 0.5 and metrics["unparsed"] == 1


def test_llmbar_flips_the_label_with_the_order(tmp_path):
    for subset in LLMBar.SUBSETS:
        rows = [{"input": f"Say hi ({subset})", "output_1": "hi", "output_2": "bye", "label": 1}]
        (tmp_path / "llmbar").mkdir(exist_ok=True)
        (tmp_path / "llmbar" / f"{subset}.json").write_text(json.dumps(rows))
    bench = LLMBar()
    items = bench.load(tmp_path, seed=0)
    assert len(items) == 10 and [i.answer for i in items[:2]] == ["1", "2"]
    assert bench.grade(items[1], {"text": "2"})["score"] == 1.0 and bench.grade(items[1], {"text": "Output 1"})["score"] == 0.0


def test_rewardbench2_shuffles_once_and_skips_ties(tmp_path):
    rows = [
        {"id": str(n), "prompt": f"question {n}", "chosen": ["best"], "rejected": ["bad one", "bad two", "bad three"], "subset": "Math"}
        for n in range(8)
    ] + [{"id": "t", "prompt": "tie", "chosen": ["a", "b"], "rejected": ["c", "d", "e"], "subset": "Ties"}]
    write_parquet(tmp_path / "rewardbench2/data/test-00000-of-00001.parquet", rows)
    bench = RewardBench2()
    items = bench.load(tmp_path, seed=5)
    assert len(items) == 8 and all(i.group == "Math" for i in items)
    assert items[0].id == "Math:0" and len({i.id for i in items}) == 8
    assert [i.answer for i in items] == [i.answer for i in bench.load(tmp_path, seed=5)]
    assert len({i.answer for i in items}) > 1  # the best response is not always in the same place
    for item in items:
        shown = re.findall(r"Response ([A-D]):\n(.+)", item.messages[0]["content"])
        assert dict((text, letter) for letter, text in shown)["best"] == item.answer
        assert bench.grade(item, {"text": item.answer})["score"] == 1.0


def test_mmlu_pro_letters_follow_the_number_of_options(tmp_path):
    rows = [{"question_id": n, "question": f"q{n}", "options": ["w", "x", "y", "z"][: 3 + n % 2], "answer": "C",
             "answer_index": 2, "cot_content": "", "category": "law", "src": "s"} for n in range(6)]
    write_parquet(tmp_path / "mmlu_pro/data/test-00000-of-00001.parquet", rows)
    bench = MMLUPro()
    items = bench.load(tmp_path, seed=0)
    assert len(items) == 6 and "C. y" in items[0].messages[0]["content"]
    three_options = next(i for i in items if i.extra["options"] == 3)
    assert bench.grade(three_options, {"text": "C"})["score"] == 1.0
    assert bench.grade(three_options, {"text": "D"}) == {"score": 0.0, "choice": None, "strict": False}


def test_gsm8k_reads_the_answer_key(tmp_path):
    write_parquet(tmp_path / "gsm8k/main/test-00000-of-00001.parquet", [{"question": "Two plus two?", "answer": "2 + 2 = <<2+2=4>>4\n#### 1,004"}])
    bench = GSM8K()
    (item,) = bench.load(tmp_path, seed=0)
    assert item.answer == "1004"
    assert bench.grade(item, {"text": "1,004"}) == {"score": 1.0, "value": 1004.0, "strict": True}
    assert bench.grade(item, {"text": "She has 1004 in total."})["strict"] is False
    assert bench.grade(item, {"text": "1005"})["score"] == 0.0 and bench.grade(item, {"text": ""})["score"] == 0.0


def test_jsonschemabench_validates_offline(tmp_path):
    good = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}, "required": ["n"]}
    rows = [
        {"json_schema": json.dumps(good), "unique_id": "good"},
        {"json_schema": json.dumps({"$ref": "https://example.com/other.json"}), "unique_id": "outside"},
        {"json_schema": json.dumps({"type": "nonsense"}), "unique_id": "invalid"},
        {"json_schema": "{not json", "unique_id": "broken"},
    ]
    write_parquet(tmp_path / "jsonschemabench/data/test-00000-of-00001.parquet", rows)
    bench = JSONSchemaBench()
    kept, dropped = bench.gradable(tmp_path)
    assert [row["unique_id"] for row in kept] == ["good"] and dropped == 3
    (item,) = bench.load(tmp_path, seed=0)
    assert bench.grade(item, {"text": '{"n": 3}'})["score"] == 1.0
    assert bench.grade(item, {"text": '```json\n{"n": 3}\n```'})["score"] == 1.0
    assert bench.grade(item, {"text": '{"n": 0}'})["score"] == 0.0
    assert bench.grade(item, {"text": "n is three"}) == {"score": 0.0, "error": "not JSON"}
    grades = [bench.grade(item, {"text": t}) for t in ('{"n": 3}', '{"n": 0}', "oops")]
    assert bench.extra_metrics([item] * 3, grades) == {"not_json": 1, "invalid_against_schema": 1}


# ------------------------------------------------------------------ code


def mbpp_fixture(root: Path) -> None:
    write_parquet(
        root / "mbpp/sanitized/test-00000-of-00001.parquet",
        [{"source_file": "f", "task_id": 7, "prompt": "Write a function that doubles a number.", "code": "def double(x): return 2 * x",
          "test_imports": ["import math"], "test_list": ["assert double(2) == 4", "assert double(0) == 0"]}],
    )


def test_mbpp_builds_the_program_from_the_reply_and_the_tests(tmp_path):
    mbpp_fixture(tmp_path)
    bench = MBPP()
    (item,) = bench.load(tmp_path, seed=0)
    assert "assert double(2) == 4" in item.messages[0]["content"]
    program = bench.program(item, {"text": "Here you go:\n```python\ndef double(x):\n    return x + x\n```"})
    assert program.splitlines()[0] == "import math" and "Here you go" not in program
    assert program.rstrip().endswith("assert double(0) == 0")
    assert bench.grade(item, {"text": "  "}) == {"score": 0.0, "status": "empty"}


@pytest.mark.skipif(not sandbox.docker_available(), reason="Docker is not running")
def test_sandbox_runs_code_without_network_or_write_access(tmp_path):
    mbpp_fixture(tmp_path)
    bench = MBPP()
    (item,) = bench.load(tmp_path, seed=0)
    assert bench.grade(item, {"text": "```python\ndef double(x):\n    return 2 * x\n```"})["score"] == 1.0
    assert bench.grade(item, {"text": "```python\ndef double(x):\n    return x\n```"})["status"] == "failed"
    assert sandbox.run_python("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)").passed is False
    assert sandbox.run_python("open('/etc/x', 'w')").passed is False
    assert sandbox.run_python("while True: pass", timeout_s=2).status == "timeout"


def test_ifeval_loader(tmp_path):
    write_jsonl(tmp_path / "ifeval/ifeval_input_data.jsonl", [{"key": 9, "prompt": "Write two words.", "instruction_id_list": ["x:y"], "kwargs": [{}]}])
    (item,) = IFEval().load(tmp_path, seed=0)
    assert item.id == "9" and item.answer["instruction_id_list"] == ["x:y"] and item.messages[0]["content"] == "Write two words."


@needs_sources
def test_ifeval_public_checker_grades_a_made_up_prompt():
    bench = IFEval()
    bench.prepare(SOURCES)
    item = Item(id="1", messages=[{"role": "user", "content": "Write at least five words and use no commas."}],
                answer={"instruction_id_list": ["length_constraints:number_words", "punctuation:no_comma"],
                        "kwargs": [{"relation": "at least", "num_words": 5, "num_sentences": None}, {}]})
    assert bench.grade(item, {"text": "one two three four five six"})["score"] == 1.0
    partly = bench.grade(item, {"text": "one, two, three, four, five, six"})
    assert partly["score"] == 0.0 and partly["followed"] == 1 and partly["instructions"] == 2
    assert bench.extra_metrics([item], [partly])["instruction_level_strict"] == 0.5


# ------------------------------------------------------------------ BFCL


def test_bfcl_decodes_calls_without_evaluating_them():
    assert bfcl.decode_calls("[area(base=10, height=-2.5), log(msg='hi', tags=['a', 'b'], opts={'k': True})]") == [
        {"area": {"base": 10, "height": -2.5}}, {"log": {"msg": "hi", "tags": ["a", "b"], "opts": {"k": True}}}]
    assert bfcl.decode_calls("```\nmath.sqrt(x=4)\n```") == [{"math.sqrt": {"x": 4}}]
    assert bfcl.decode_calls("f(n=2 * 3 + 1, s='a' + 'b')") == [{"f": {"n": 7, "s": "ab"}}]
    assert bfcl.decode_calls("[]") == []
    for prose in ("I cannot do that.", "[1, 2]", "f(x=__import__('os').system('true') + 1)", "f(x=(lambda: 1))"):
        with pytest.raises(Exception):
            bfcl.decode_calls(prose)
    # A call inside an argument is kept as text and never run.
    assert bfcl.decode_calls("f(x=open('/etc/passwd'))") == [{"f": {"x": "open('/etc/passwd')"}}]


def test_bfcl_native_tool_calls_are_written_in_the_text_format():
    text = bfcl.native_calls_as_text([{"name": "cd", "args": {"folder": "docs"}}, {"name": "pwd", "args": {}}, {"name": "set", "args": {"on": True, "n": 2}}])
    assert text == "[cd(folder='docs'), pwd(), set(on=True, n=2)]"
    assert bfcl.decode_calls(text) == [{"cd": {"folder": "docs"}}, {"pwd": {}}, {"set": {"on": True, "n": 2}}]


def test_bfcl_writes_calls_back_out():
    decoded = bfcl.decode_calls("[cd(folder='docs'), mv(source='a.txt', destination='b')]")
    assert bfcl.call_strings(decoded) == ["cd(folder='docs')", "mv(source='a.txt', destination='b')"]
    assert bfcl.is_call_list(decoded) and not bfcl.is_call_list([{"a": 1}]) and not bfcl.is_call_list("x")


class Counter:
    """A made-up back end with one piece of state."""

    def __init__(self):
        self.total = 0

    def add(self, amount: int = 1):
        self.total += amount
        return {"total": self.total}

    def name(self):
        return "counter"

    def _secret(self):
        return "hidden"


def test_bfcl_executes_only_public_methods_with_literal_arguments():
    counter = Counter()
    results = bfcl.execute_calls(
        ["add(amount=2)", "add(3)", "name()", "_secret()", "print('x')", "add(amount=__import__('os').getpid())", "add(amount=add(1)['total'])"],
        {"Counter": counter},
    )
    assert results[:3] == ['{"total": 2}', '{"total": 5}', "counter"]
    assert all(r.startswith("Error during execution") for r in results[3:])
    assert counter.total == 5


def test_bfcl_multi_turn_conversation_and_grading(monkeypatch):
    monkeypatch.setitem(bfcl._modules, "prompts", type("P", (), {
        "PROMPT_STYLE_TEMPLATES": {"classic": {"persona": "p.", "task": "t", "tool_call_no_tag": "{output_format}{param_types}",
                                               "multiturn_behavior": "m", "available_tools": "{format}:{functions}"}},
        "PROMPT_TEMPLATE_MAPPING": {"plaintext": "{persona}{task}{tool_call_format}{multiturn_behavior}{available_tools}"},
        "OUTPUT_FORMAT_MAPPING": {"python": "[f()]"}, "PARAM_TYPE_MAPPING": {"python": ""},
        "DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING": "{functions}\nmore functions"}))
    monkeypatch.delitem(bfcl._modules, "ast_checker", raising=False)

    class Checks:
        @staticmethod
        def state_checker(model, truth):
            same = all(vars(model[k]) == vars(truth[k]) for k in truth)
            return {"valid": same, "error_type": "multi_turn:instance_state_mismatch"}

        @staticmethod
        def response_checker(model_results, truth_results, turn):
            return {"valid": all(r in model_results for r in truth_results), "error_type": "multi_turn:execution_response_mismatch"}

    monkeypatch.setitem(bfcl._modules, "multi_turn", Checks)
    monkeypatch.setattr(bfcl, "new_instances", lambda entry: {"Counter": Counter()})
    bench = bfcl.BFCLMultiTurn()
    entry = {"id": "multi_turn_base_0", "question": [[{"role": "user", "content": "add two"}], [], [{"role": "user", "content": "add one more"}]],
             "function": [{"name": "add"}], "held_back": {"1": [{"name": "name"}]}}
    item = Item(id=entry["id"], messages=[], answer=[["add(amount=2)"], [], ["add(amount=1)"]], group="multi_turn_base", extra=entry)

    def scripted(replies):
        seen, queue = [], iter(replies)

        async def ask(messages):
            seen.append(messages)
            return next(queue)

        return ask, seen

    ask, seen = scripted(["[add(amount=2)]", "Done.", "Nothing to add.", "[add(amount=1)]", "[]"])
    reply = asyncio.run(bench.converse(item, ask))
    assert reply == {"turns": [["[add(amount=2)]", "Done."], ["Nothing to add."], ["[add(amount=1)]", "[]"]], "forced_stop": False}
    assert seen[0][0]["role"] == "system" and seen[0][1] == {"role": "user", "content": "add two"}
    assert "'content': '{\"total\": 2}'" in seen[1][-1]["content"] and seen[1][-1]["role"] == "user"
    assert seen[2][-1]["content"].endswith("more functions")  # the held-back function arrives in turn 1
    assert bench.grade(item, reply) == {"score": 1.0, "error": None}

    ask, _ = scripted(["[add(amount=5)]", "ok", "ok", "[add(amount=1)]", "ok"])
    assert bench.grade(item, asyncio.run(bench.converse(item, ask)))["error"] == "instance_state_mismatch"
    ask, _ = scripted(["I will not.", "ok", "ok"])
    assert bench.grade(item, asyncio.run(bench.converse(item, ask)))["error"] == "empty_turn_model_response"
    ask, _ = scripted(["[add(amount=0)]"] * 100)
    stuck = asyncio.run(bench.converse(item, ask))
    assert stuck["forced_stop"] and len(stuck["turns"][0]) == bench.MAX_STEPS + 1
    assert bench.grade(item, stuck) == {"score": 0.0, "error": "force_terminated"}


@needs_sources
def test_bfcl_ast_checker_on_a_made_up_function():
    bench = bfcl.BFCL()
    bench.prepare(SOURCES)
    function = [{"name": "book_table", "description": "Book a table.", "parameters": {"type": "dict", "properties": {
        "people": {"type": "integer", "description": "How many."}, "city": {"type": "string", "description": "Where."}}, "required": ["people", "city"]}}]
    item = Item(id="simple_python_x", messages=[], answer=[{"book_table": {"people": [4], "city": ["Lisbon", "lisbon"]}}],
                group="simple_python", extra={"function": function})
    assert bench.grade(item, {"text": "[book_table(people=4, city='Lisbon')]"})["score"] == 1.0
    assert bench.grade(item, {"text": "[book_table(people=5, city='Lisbon')]"})["score"] == 0.0
    assert bench.grade(item, {"text": "[book_table(people=4)]"})["score"] == 0.0
    assert bench.grade(item, {"text": "Sure, booking now."})["error"].startswith("could not decode")
    irrelevant = Item(id="irrelevance_x", messages=[], answer=None, group="irrelevance", extra={"function": function})
    assert bench.grade(irrelevant, {"text": "None of the functions fit."})["score"] == 1.0
    assert bench.grade(irrelevant, {"text": "[book_table(people=4, city='Lisbon')]"})["score"] == 0.0
    assert "Note that the provided function is in Python 3 syntax." in bfcl.first_turn([{"role": "user", "content": "hi"}], bfcl.with_python_hint(function))[0]["content"]


# ---------------------------------------------------------------- sources


def test_every_source_is_pinned_to_a_revision_and_a_checksum():
    pins = sources.load_pins()
    for source in sources.all_sources():
        assert re.search(r"/[0-9a-f]{40}/", source.url), source.url
        assert re.fullmatch(r"[0-9a-f]{64}", pins.get(source.url, "")), source.url
        assert source.licence
    assert {b.source for b in BENCHMARKS.values()} <= set(sources.licences())
    assert set(NOT_RUN) == {"llm_aggrefact", "wildguardtest"}
    assert all("needs a token" in entry["reason"] for entry in NOT_RUN.values())


def test_a_changed_file_fails_its_checksum(tmp_path, monkeypatch):
    source = sources.Source("demo", "a.txt", "https://example.invalid/" + "0" * 40 + "/a.txt", "MIT")
    monkeypatch.setattr(sources, "all_sources", lambda: [source])
    monkeypatch.setattr(sources, "load_pins", lambda: {source.url: sources.hashlib.sha256(b"expected").hexdigest()})
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "a.txt").write_bytes(b"expected")
    assert sources.ensure(tmp_path, out=lambda line: None) == {"demo": tmp_path / "demo"}
    (tmp_path / "demo" / "a.txt").write_bytes(b"tampered")
    with pytest.raises(sources.ChecksumError):
        sources.ensure(tmp_path, out=lambda line: None)


# ------------------------------------------------------------------ stats


def test_bootstrap_interval_brackets_the_mean():
    scores = [1.0] * 70 + [0.0] * 30
    result = bootstrap(scores, seed=1)
    assert result["estimate"] == 0.7 and result["n"] == 100
    assert 0.58 < result["low"] < 0.7 < result["high"] < 0.8
    assert bootstrap(scores, seed=1) == result
    assert bootstrap([], seed=1)["estimate"] is None


def test_bootstrap_resamples_pairs_together():
    # Pairs are either both right or both wrong, so there are really 20 observations.
    scores = [1.0, 1.0] * 10 + [0.0, 0.0] * 10
    clusters = [f"p{i // 2}" for i in range(40)]
    independent = bootstrap(scores, seed=2)
    paired = bootstrap(scores, clusters=clusters, seed=2)
    assert paired["high"] - paired["low"] > independent["high"] - independent["low"]


def test_macro_mean_weights_groups_equally():
    scores, groups = [1.0] * 90 + [0.0] * 10, ["big"] * 90 + ["small"] * 10
    assert macro_mean(scores, groups) == 0.5
    result = bootstrap(scores, groups=groups, macro=True, seed=3)
    assert result["estimate"] == 0.5 and result["low"] == result["high"] == 0.5


def test_paired_difference_and_mcnemar():
    fast = [0.0] * 30 + [1.0] * 70
    slow = [1.0] * 25 + [0.0] * 5 + [1.0] * 68 + [0.0] * 2
    result = paired_difference(fast, slow, seed=4)
    assert result["estimate"] == 0.23 and result["gained"] == 25 and result["lost"] == 2
    assert result["low"] > 0.1 and result["mcnemar_p"] < 0.001
    same = paired_difference(fast, fast, seed=4)
    assert same["estimate"] == 0.0 and same["mcnemar_p"] == 1.0


def test_target_rule():
    target, watch = JudgeBench(), MMLUPro()
    assert classify(target, 10.0) == "training target" and classify(target, 9.9) == "in between"
    assert classify(target, 3.0) == "usable as is" and classify(target, -4.0) == "usable as is"
    assert classify(target, 3.1) == "in between" and classify(target, None) == "not measured"
    assert classify(watch, 25.0) == "watch only (training target)"
