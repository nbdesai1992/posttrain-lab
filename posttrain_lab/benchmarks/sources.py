"""Where every benchmark file comes from, pinned to a revision and a checksum.

Nothing here needs a login. Files are downloaded once into the survey's own data
directory (``runs/survey/sources/`` by default) and checked against the SHA-256
values in ``pins.json``. Benchmark items must never be read by a step that builds
training data, other than to check for leaks.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PINS_PATH = Path(__file__).with_name("pins.json")
HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
GH = "https://raw.githubusercontent.com/{repo}/{rev}/{path}"

BFCL_REPO, BFCL_REV = "ShishirPatil/gorilla", "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_ROOT = "berkeley-function-call-leaderboard/bfcl_eval/"
IFEVAL_REPO, IFEVAL_REV = "google-research/google-research", "e6890f85757dd84e27ca6df2dd30651dafad28e0"

BFCL_SINGLE_TURN = [
    "simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple",
    "irrelevance", "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "live_irrelevance", "live_relevance",
]
BFCL_MULTI_TURN = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
BFCL_NO_ANSWER_FILE = {"irrelevance", "live_irrelevance", "live_relevance"}
BFCL_BACKENDS = [
    "gorilla_file_system", "math_api", "message_api", "posting_api", "ticket_api", "trading_bot",
    "travel_booking", "vehicle_control", "long_context",
]


@dataclass(frozen=True)
class Source:
    group: str  # the benchmark (or checker) the file belongs to
    path: str  # where it is stored under the data root, relative to the group
    url: str
    licence: str


def _hf(group: str, repo: str, rev: str, path: str, licence: str) -> Source:
    return Source(group, path, HF.format(repo=repo, rev=rev, path=path), licence)


def _gh(group: str, repo: str, rev: str, path: str, licence: str, store_as: str | None = None) -> Source:
    return Source(group, store_as or path, GH.format(repo=repo, rev=rev, path=path), licence)


def all_sources() -> list[Source]:
    sources = [
        _hf("judgebench", "ScalerLab/JudgeBench", "57dd5e0b9817d07f05ec8f45a91b2ce1e310e308", "data/gpt-00000-of-00001.jsonl", "MIT"),
        _hf("judgebench", "ScalerLab/JudgeBench", "57dd5e0b9817d07f05ec8f45a91b2ce1e310e308", "data/claude-00000-of-00001.jsonl", "MIT"),
        _hf("rewardbench2", "allenai/reward-bench-2", "7ff08853b0d5686e79b13fda8677024f566a104a", "data/test-00000-of-00001.parquet", "ODC-BY"),
        _hf("jsonschemabench", "epfl-dlab/JSONSchemaBench", "5bd0f4640badc6f3f02df796421d21cb0ca0b141", "data/test-00000-of-00001.parquet", "MIT"),
        _hf("gsm8k", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet", "MIT"),
        _hf("mbpp", "google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c", "sanitized/test-00000-of-00001.parquet", "CC-BY-4.0"),
        _hf("ifeval", "google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "ifeval_input_data.jsonl", "Apache-2.0"),
        _hf("mmlu_pro", "TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "data/test-00000-of-00001.parquet", "MIT"),
    ]
    for subset in ("Natural", "Adversarial/Neighbor", "Adversarial/GPTInst", "Adversarial/GPTOut", "Adversarial/Manual"):
        sources.append(
            _gh("llmbar", "princeton-nlp/LLMBar", "900616bff90b6c6c8e1681f7d079250637c55992",
                f"Dataset/LLMBar/{subset}/dataset.json", "MIT", store_as=f"{subset.replace('/', '_')}.json")
        )
    # The official IFEval checker, imported from the downloaded copy.
    for name in ("instructions.py", "instructions_registry.py", "instructions_util.py", "evaluation_lib.py"):
        sources.append(
            _gh("ifeval_checker", IFEVAL_REPO, IFEVAL_REV, f"instruction_following_eval/{name}", "Apache-2.0",
                store_as=f"instruction_following_eval/{name}")
        )
    # Sentence tokenizer data the IFEval checker loads through NLTK.
    sources.append(
        _gh("nltk_data", "nltk/nltk_data", "4f15a3d89eefe9748ec1c05be495d91289197155",
            "packages/tokenizers/punkt_tab.zip", "Apache-2.0", store_as="tokenizers/punkt_tab.zip")
    )
    # BFCL V4 data, and the parts of its evaluation code that grade without any service.
    def bfcl(path: str) -> Source:
        return _gh("bfcl", BFCL_REPO, BFCL_REV, BFCL_ROOT + path, "Apache-2.0", store_as="bfcl_eval/" + path)

    for category in BFCL_SINGLE_TURN + BFCL_MULTI_TURN:
        sources.append(bfcl(f"data/BFCL_v4_{category}.json"))
        if category not in BFCL_NO_ANSWER_FILE:
            sources.append(bfcl(f"data/possible_answer/BFCL_v4_{category}.json"))
    for doc in ("gorilla_file_system", "math_api", "message_api", "posting_api", "ticket_api", "trading_bot", "travel_booking", "vehicle_control"):
        sources.append(bfcl(f"data/multi_turn_func_doc/{doc}.json"))
    for module in (
        "constants/default_prompts.py", "constants/enums.py", "constants/type_mappings.py",
        "constants/executable_backend_config.py",
        "eval_checker/ast_eval/ast_checker.py",
        "eval_checker/ast_eval/type_convertor/java_type_converter.py",
        "eval_checker/ast_eval/type_convertor/js_type_converter.py",
        "eval_checker/multi_turn_eval/multi_turn_checker.py",
        "eval_checker/multi_turn_eval/multi_turn_utils.py",
        *(f"eval_checker/multi_turn_eval/func_source_code/{name}.py" for name in BFCL_BACKENDS),
    ):
        sources.append(bfcl(module))
    return sources


def load_pins() -> dict[str, str]:
    return json.loads(PINS_PATH.read_text()) if PINS_PATH.exists() else {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ChecksumError(RuntimeError):
    pass


def ensure(root: Path, groups: set[str] | None = None, *, update_pins: bool = False, out=print) -> dict[str, Path]:
    """Download whatever is missing and verify every file. Returns {group: directory}.

    ``update_pins`` records the checksums of freshly downloaded files. It is for the
    maintainer adding or bumping a source, never for a normal run.
    """
    pins = load_pins()
    directories: dict[str, Path] = {}
    for source in all_sources():
        if groups is not None and source.group not in groups:
            continue
        target = root / source.group / source.path
        directories[source.group] = root / source.group
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(source.url, headers={"User-Agent": "posttrain-lab"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            partial = target.with_suffix(target.suffix + ".part")
            partial.write_bytes(data)
            partial.rename(target)
            out(f"downloaded {source.group}/{source.path} ({len(data):,} bytes)")
        actual = sha256(target)
        if update_pins:
            pins[source.url] = actual
        elif pins.get(source.url) != actual:
            raise ChecksumError(
                f"{target} does not match its pinned checksum (expected {pins.get(source.url)}, got {actual}). "
                "Delete the file to download it again."
            )
    if update_pins:
        PINS_PATH.write_text(json.dumps(dict(sorted(pins.items())), indent=1) + "\n")
    return directories


def licences() -> dict[str, str]:
    return {source.group: source.licence for source in all_sources()}
