"""BFCL V4: the single-turn and multi-turn categories that a program can grade.

What comes from the Berkeley Function Calling Leaderboard, downloaded at a pinned
revision and imported unchanged (Apache-2.0): the data, the default system prompt,
the AST checker for single-turn items, the simulated back ends for multi-turn items,
and the state and response checks.

What is written here instead of imported: the prompt assembly, the decoding of a
reply into calls, and the execution of calls against the back ends. BFCL's own
versions pull in every provider's SDK, need tree-sitter for Java and JavaScript, and
run model output through ``eval``. The ports below follow BFCL's logic and never
evaluate model output as code. Scores are from this harness, not BFCL's.

When the model is free to reason it sometimes ignores the text format BFCL asks for
and makes the call in its own tool-call format. Those calls are read as the reply,
in every setting alike (``native_calls_as_text``).

Left out: ``simple_java`` and ``simple_javascript`` (their decoders need
tree-sitter), and the agentic categories ``web_search`` and ``memory`` (an outside
search service, and memory back ends with an embedding model).
"""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

from .base import Benchmark, Item, fixed_sample, read_jsonl
from .sources import BFCL_MULTI_TURN

SINGLE_TURN = [
    "simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance",
]
PYTHON_HINT = " Note that the provided function is in Python 3 syntax."
BACKEND_FILES = {
    "GorillaFileSystem": "gorilla_file_system", "MathAPI": "math_api", "MessageAPI": "message_api",
    "TwitterAPI": "posting_api", "TicketAPI": "ticket_api", "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking", "VehicleControlAPI": "vehicle_control",
}
_modules: dict = {}


def setup(root: Path) -> dict:
    """Import the downloaded BFCL modules. Their heavy model table is replaced by a stub."""
    if "multi_turn" in _modules and "ast_checker" in _modules:
        return _modules
    base = str(root / "bfcl")
    if base not in sys.path:
        sys.path.insert(0, base)
    stub = types.ModuleType("bfcl_eval.constants.model_config")
    stub.MODEL_CONFIG_MAPPING = defaultdict(lambda: types.SimpleNamespace(underscore_to_dot=False))
    sys.modules.setdefault(stub.__name__, stub)
    _modules["prompts"] = importlib.import_module("bfcl_eval.constants.default_prompts")
    _modules["backends"] = importlib.import_module("bfcl_eval.constants.executable_backend_config")
    _modules["enums"] = importlib.import_module("bfcl_eval.constants.enums")
    _modules["ast_checker"] = importlib.import_module("bfcl_eval.eval_checker.ast_eval.ast_checker")
    _modules["multi_turn"] = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker")
    return _modules


# ------------------------------------------------------------------ prompts


def system_prompt(functions: list[dict]) -> str:
    """BFCL's default prompt for models that are told the functions in plain text."""
    prompts = _modules["prompts"]
    style = prompts.PROMPT_STYLE_TEMPLATES["classic"]
    return prompts.PROMPT_TEMPLATE_MAPPING["plaintext"].format(
        persona=style["persona"],
        task=style["task"],
        tool_call_format=style["tool_call_no_tag"].format(
            output_format=prompts.OUTPUT_FORMAT_MAPPING["python"], param_types=prompts.PARAM_TYPE_MAPPING["python"]
        ),
        multiturn_behavior=style["multiturn_behavior"],
        available_tools=style["available_tools"].format(format="json", functions=json.dumps(functions, indent=4)),
    )


def with_python_hint(functions: list[dict]) -> list[dict]:
    functions = copy.deepcopy(functions)
    for function in functions:
        function["description"] = function["description"] + PYTHON_HINT
    return functions


def first_turn(messages: list[dict], functions: list[dict]) -> list[dict]:
    """Put the system prompt first, in front of any system message the item brings."""
    messages = copy.deepcopy(messages)
    prompt = system_prompt(functions)
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = prompt + "\n\n" + messages[0]["content"]
        return messages
    return [{"role": "system", "content": prompt}, *messages]


# ----------------------------------------------------------------- decoding


def _arithmetic(node: ast.BinOp):
    operations = {
        ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    }
    left, right = _value(node.left), _value(node.right)
    numbers = isinstance(left, (int, float)) and isinstance(right, (int, float))
    if type(node.op) in operations and (numbers or (isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str))):
        return operations[type(node.op)](left, right)
    if isinstance(node.op, ast.Pow) and numbers and abs(right) <= 64 and abs(left) <= 1e6:
        return left**right
    raise ValueError("unsupported expression in an argument")


def _value(node: ast.AST):
    """One argument value. Follows BFCL's ``resolve_ast_by_type`` without ``eval``."""
    if isinstance(node, ast.Constant):
        return "..." if node.value is Ellipsis else node.value
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return -node.operand.value
        raise ValueError("unsupported unary expression")
    if isinstance(node, ast.List):
        return [_value(element) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_value(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return {_value(key): _value(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.BinOp):
        return _arithmetic(node)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return ast.unparse(node) if not node.keywords else _call(node)
    if isinstance(node, ast.Subscript):
        return ast.unparse(node.value) + "[" + ast.unparse(node.slice) + "]"
    raise ValueError(f"unsupported argument syntax: {type(node).__name__}")


def _call(node: ast.Call) -> dict:
    parts = []
    target = node.func
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    return {".".join(reversed(parts)): {keyword.arg: _value(keyword.value) for keyword in node.keywords}}


def decode_calls(text: str) -> list[dict]:
    """A reply such as ``[f(a=1), g(b='x')]`` as ``[{"f": {"a": 1}}, {"g": {"b": "x"}}]``.
    Raises if the reply is not a list of calls."""
    result = text.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    body = ast.parse(result.strip().strip("'"), mode="eval").body
    if isinstance(body, ast.Call):
        return [_call(body)]
    calls = []
    for element in body.elts:
        if not isinstance(element, ast.Call):
            raise ValueError("the reply contains something that is not a call")
        calls.append(_call(element))
    return calls


def is_call_list(decoded) -> bool:
    return isinstance(decoded, list) and all(
        isinstance(call, dict) and len(call) == 1 and isinstance(next(iter(call.values())), dict) for call in decoded
    )


def _nested(value) -> str:
    if isinstance(value, dict):
        if value and all(isinstance(inner, dict) for inner in value.values()):
            name = next(iter(value))
            return f"{name}({', '.join(f'{k}={_nested(v)}' for k, v in value[name].items())})"
        return "{" + ", ".join(f"'{k}': {_nested(v)}" for k, v in value.items()) + "}"
    return repr(value)


def call_strings(decoded: list[dict]) -> list[str]:
    """Decoded calls written back out as ``name(arg=value)`` strings."""
    return [
        f"{name}({', '.join(f'{k}={_nested(v)}' for k, v in arguments.items())})"
        for call in decoded
        for name, arguments in call.items()
    ]


def native_calls_as_text(tool_calls: list[dict]) -> str:
    """Calls the model made in its own tool-call format, written the way BFCL asks
    for them, so the rest of the harness treats both forms alike."""
    return "[" + ", ".join(call_strings([{call["name"]: call["args"]} for call in tool_calls])) + "]"


def decode_for_execution(text: str) -> list[str]:
    return call_strings(decode_calls(text))


# ---------------------------------------------------------------- execution


def new_instances(entry: dict) -> dict:
    """Fresh simulated back ends for one multi-turn item, in their starting state."""
    config = _modules["backends"]
    long_context = "long_context" in entry["id"] or "composite" in entry["id"]
    instances = {}
    for class_name in entry["involved_classes"]:
        module = importlib.import_module(config.CLASS_FILE_PATH_MAPPING[class_name])
        instance = getattr(module, class_name)()
        if class_name not in config.STATELESS_CLASSES:
            instance._load_scenario(copy.deepcopy(entry["initial_config"].get(class_name, {})), long_context=long_context)
        instances[class_name] = instance
    return instances


def _evaluate(node: ast.AST, methods: dict):
    if isinstance(node, ast.Call):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name not in methods:
            raise NameError(f"name '{name}' is not defined")
        arguments = [_evaluate(argument, methods) for argument in node.args]
        keywords = {keyword.arg: _evaluate(keyword.value, methods) for keyword in node.keywords}
        return methods[name](*arguments, **keywords)
    return ast.literal_eval(node)


def execute_calls(calls: list[str], instances: dict) -> list[str]:
    """Run call strings against the back ends and return what each call gave back.

    Only public methods of the given instances can be called, and arguments must be
    literals or further such calls. Nothing is passed to ``eval``.
    """
    methods = {
        name: method
        for instance in instances.values()
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod)
        if not name.startswith("_")
    }
    results = []
    for call in calls:
        try:
            value = _evaluate(ast.parse(call.strip(), mode="eval").body, methods)
            if isinstance(value, str):
                results.append(value)
            elif isinstance(value, dict):
                try:
                    results.append(json.dumps(value))
                except (TypeError, ValueError):
                    results.append(str(value))
            else:
                results.append(str(value))
        except Exception as exc:
            results.append(f"Error during execution: {exc}")
    return results


# --------------------------------------------------------------- benchmarks


def _category(entry_id: str) -> str:
    return entry_id.rsplit("_", 1)[0]


class BFCL(Benchmark):
    name = "bfcl"
    title = "BFCL V4, single turn"
    job = "Calling tools"
    role = "target"
    graded_by = "BFCL's AST checker"
    source = "bfcl"
    macro = True
    max_tokens_fast = 512
    reads_tool_calls = True

    def load(self, root: Path, seed: int) -> list[Item]:
        setup(root)
        data = root / self.source / "bfcl_eval" / "data"
        items = []
        for category in SINGLE_TURN:
            answers = {}
            answer_file = data / "possible_answer" / f"BFCL_v4_{category}.json"
            if answer_file.exists():
                answers = {row["id"]: row["ground_truth"] for row in read_jsonl(answer_file)}
            for entry in read_jsonl(data / f"BFCL_v4_{category}.json"):
                items.append(
                    Item(
                        id=entry["id"],
                        messages=first_turn(entry["question"][0], with_python_hint(entry["function"])),
                        answer=answers.get(entry["id"]),
                        group=category,
                        extra={"function": entry["function"]},
                    )
                )
        return items

    def prepare(self, root: Path) -> None:
        setup(root)

    def grade(self, item: Item, reply: dict) -> dict:
        try:
            decoded = decode_calls(reply.get("text", ""))
        except Exception as exc:
            decoded, problem = None, f"{type(exc).__name__}"
        calls = bool(decoded) and is_call_list(decoded) and not (len(decoded) == 1 and not decoded[0])
        if "irrelevance" in item.group:
            return {"score": float(not calls), "called": calls}
        if "relevance" in item.group:
            return {"score": float(calls), "called": calls}
        if decoded is None:
            return {"score": 0.0, "error": f"could not decode: {problem}"}
        if not is_call_list(decoded):
            return {"score": 0.0, "error": "not in function-call format"}
        checked = _modules["ast_checker"].ast_checker(
            item.extra["function"], decoded, item.answer, _modules["enums"].Language.PYTHON, item.group, "posttrain-lab"
        )
        return {"score": float(checked["valid"]), "error": None if checked["valid"] else checked.get("error_type")}

    def extra_metrics(self, items: list[Item], grades: list[dict]) -> dict:
        errors: dict[str, int] = {}
        for grade in grades:
            if grade["score"] == 0 and grade.get("error"):
                kind = grade["error"].split(":")[0]
                errors[kind] = errors.get(kind, 0) + 1
        return {"error_types": dict(sorted(errors.items(), key=lambda kv: -kv[1])[:6])}


class BFCLMultiTurn(Benchmark):
    name = "bfcl_multi_turn"
    title = "BFCL V4, multi turn (sample of 200)"
    job = "Calling tools"
    role = "target"
    graded_by = "BFCL's simulated back ends, with its state and response checks"
    source = "bfcl"
    macro = True
    multi_step = True
    reads_tool_calls = True
    max_tokens_fast = 512
    MAX_STEPS = 20  # per turn, as in BFCL
    # An item takes about a dozen calls, and each resends some 12,000 tokens of
    # function documentation. To stay a light user of the free tier the survey takes
    # a fixed sample of each category (out of 200). Samples with the same seed are
    # nested, so raising this number later keeps every item already run.
    PER_CATEGORY = 50
    reference_size = 60

    def load(self, root: Path, seed: int) -> list[Item]:
        setup(root)
        data = root / self.source / "bfcl_eval" / "data"
        config = _modules["backends"]
        docs = {
            class_name: read_jsonl(data / "multi_turn_func_doc" / config.MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name])
            for class_name in BACKEND_FILES
        }
        items = []
        for category in BFCL_MULTI_TURN:
            answers = {row["id"]: row["ground_truth"] for row in read_jsonl(data / "possible_answer" / f"BFCL_v4_{category}.json")}
            for entry in fixed_sample(read_jsonl(data / f"BFCL_v4_{category}.json"), self.PER_CATEGORY, seed):
                functions = [doc for class_name in entry["involved_classes"] for doc in docs[class_name]]
                held_back = {}
                for turn, names in entry.get("missed_function", {}).items():
                    held_back[turn] = [doc for doc in functions if doc["name"] in names]
                    functions = [doc for doc in functions if doc["name"] not in names]
                entry = {**entry, "function": with_python_hint(functions), "held_back": {t: with_python_hint(d) for t, d in held_back.items()}}
                items.append(Item(id=entry["id"], messages=[], answer=answers[entry["id"]], group=category, extra=entry))
        return items

    def prepare(self, root: Path) -> None:
        setup(root)

    # The conversation: one or more model steps per user turn, each followed by
    # running the calls it made and showing the model the results.

    def turn_message(self, item: Item, turn: int) -> list[dict]:
        held_back = item.extra["held_back"].get(str(turn))
        if held_back is not None:
            text = _modules["prompts"].DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(functions=held_back)
            return [{"role": "user", "content": text}]
        return copy.deepcopy(item.extra["question"][turn])

    async def converse(self, item: Item, ask) -> dict:
        """Run the item. ``ask(messages)`` returns the model's reply text."""
        instances = new_instances(item.extra)
        messages: list[dict] = []
        turns: list[list[str]] = []
        forced_stop = False
        for turn in range(len(item.extra["question"])):
            new = self.turn_message(item, turn)
            messages = first_turn(new, item.extra["function"]) if turn == 0 else [*messages, *new]
            replies: list[str] = []
            while True:
                reply = await ask(messages)
                replies.append(reply)
                messages = [*messages, {"role": "assistant", "content": reply}]
                try:
                    calls = decode_for_execution(reply)
                except Exception:
                    break
                if not calls or calls == [""]:
                    break
                results = execute_calls(calls, instances)
                shown = [{"role": "tool", "name": call, "content": result} for call, result in zip(calls, results)]
                messages = [*messages, {"role": "user", "content": repr(shown)}]
                if len(replies) > self.MAX_STEPS:
                    forced_stop = True
                    break
            turns.append(replies)
            if forced_stop:
                break
        return {"turns": turns, "forced_stop": forced_stop}

    def grade(self, item: Item, reply: dict) -> dict:
        checker = _modules["multi_turn"]
        turns, truth = reply.get("turns") or [], item.answer
        if reply.get("forced_stop") or len(turns) != len(truth):
            return {"score": 0.0, "error": "force_terminated"}
        model_instances, truth_instances = new_instances(item.extra), new_instances(item.extra)
        every_result: list[str] = []
        for index, expected in enumerate(truth):
            steps = []
            for text in turns[index]:
                try:
                    calls = decode_for_execution(text)
                except Exception:
                    continue
                if calls and calls != [""]:
                    steps.append(calls)
            for calls in steps:
                every_result.extend(execute_calls(calls, model_instances))
            expected_results = execute_calls(expected, truth_instances)
            if not expected:
                continue
            if not steps:
                return {"score": 0.0, "error": "empty_turn_model_response", "turn": index}
            for check in (
                checker.state_checker(model_instances, truth_instances),
                checker.response_checker(every_result, expected_results, index),
            ):
                if not check["valid"]:
                    return {"score": 0.0, "error": check["error_type"].split(":")[-1], "turn": index}
        return {"score": 1.0, "error": None}

    def extra_metrics(self, items: list[Item], grades: list[dict]) -> dict:
        errors: dict[str, int] = {}
        for grade in grades:
            if grade.get("error"):
                errors[grade["error"]] = errors.get(grade["error"], 0) + 1
        return {"error_types": dict(sorted(errors.items(), key=lambda kv: -kv[1]))}
