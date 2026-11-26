"""IFEval, graded by the benchmark's own public checker.

The checker is Google's ``instruction_following_eval`` code, downloaded at a pinned
revision (see ``sources.py``) and imported from that copy. Nothing in it is changed.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from .base import Benchmark, Item, read_jsonl, user


class IFEval(Benchmark):
    name = "ifeval"
    title = "IFEval"
    job = "Instructions and knowledge"
    role = "watch"
    graded_by = "the benchmark's public checker"
    source = "ifeval"
    max_tokens_fast = 2048
    _lib = None

    def load(self, root: Path, seed: int) -> list[Item]:
        return [
            Item(
                id=str(row["key"]),
                messages=user(row["prompt"]),
                answer={"instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"]},
            )
            for row in read_jsonl(root / self.source / "ifeval_input_data.jsonl")
        ]

    def prepare(self, root: Path) -> None:
        if IFEval._lib is not None:
            return
        import langdetect
        import nltk

        data = root / "nltk_data"
        if not (data / "tokenizers" / "punkt_tab" / "english").exists():
            with zipfile.ZipFile(data / "tokenizers" / "punkt_tab.zip") as archive:
                archive.extractall(data / "tokenizers")
        if str(data) not in nltk.data.path:
            nltk.data.path.insert(0, str(data))
        langdetect.DetectorFactory.seed = 0  # language detection is random unless seeded
        checker = str(root / "ifeval_checker")
        if checker not in sys.path:
            sys.path.insert(0, checker)
        from instruction_following_eval import evaluation_lib

        IFEval._lib = evaluation_lib

    def grade(self, item: Item, reply: dict) -> dict:
        lib = IFEval._lib
        prompt = item.messages[-1]["content"]
        # The data lists every possible argument and leaves the unused ones empty.
        kwargs = [{k: v for k, v in entry.items() if v is not None} for entry in item.answer["kwargs"]]
        example = lib.InputExample(
            key=int(item.id), instruction_id_list=item.answer["instruction_id_list"], prompt=prompt, kwargs=kwargs
        )
        responses = {prompt: reply.get("text", "")}
        strict = lib.test_instruction_following_strict(example, responses)
        loose = lib.test_instruction_following_loose(example, responses)
        return {
            "score": float(strict.follow_all_instructions),
            "instructions": len(strict.follow_instruction_list),
            "followed": sum(map(bool, strict.follow_instruction_list)),
            "loose": float(loose.follow_all_instructions),
            "loose_followed": sum(map(bool, loose.follow_instruction_list)),
        }

    def extra_metrics(self, items: list[Item], grades: list[dict]) -> dict:
        total = sum(g["instructions"] for g in grades) or 1
        return {
            "instruction_level_strict": round(sum(g["followed"] for g in grades) / total, 4),
            "prompt_level_loose": round(sum(g["loose"] for g in grades) / max(len(grades), 1), 4),
            "instruction_level_loose": round(sum(g["loose_followed"] for g in grades) / total, 4),
        }
