"""SearchQA task adapter: data loading, prompts, and EM scoring.

* **Data**: ``<split_dir>/{train,val,test}/items.json``
  (train 400 / val 200 / test 1400; fields ``id, question, context, answers``).
  The split file format originates from the baseline method's released data.
* **System prompt**: fixed template below; ``skill_section`` is empty
  (skills act on hidden states, not on the prompt text).
* **User turn**: ``## Context`` (truncated to 6000 chars at ``[DOC]``
  boundaries) + ``## Question``.
* **Prompting**: ``apply_chat_template(add_generation_prompt=True,
  enable_thinking=False)``.
* **Metric**: EM (SQuAD normalization + ``<answer>...</answer>`` extraction),
  computed after truncating the generation at ``</answer>``
  (:func:`cut_at_answer`).
"""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from pathlib import Path

from .probe_paths import data_root

__all__ = ["SearchQATask", "TASKS", "get_task", "cut_at_answer"]

MAX_CONTEXT_CHARS = 6000
SPLITS = ("train", "val", "test")

_SYSTEM_TEMPLATE = """You are an expert question answering agent.

{skill_section}## Task Format
You will receive a CONTEXT containing document passages and a QUESTION.
Read the context carefully and answer the question based on the information provided.

## Answer Format
Think step by step, then provide your final answer inside <answer>...</answer> tags.
Keep your answer concise — typically a few words or a short phrase.
Do not repeat the question. Do not include unnecessary explanation in the answer tags.

Example:
<answer>Abraham Lincoln</answer>
"""


def cut_at_answer(text: str) -> str:
    """Truncate the generation right after the first ``</answer>``."""
    cut = text.find("</answer>")
    return text[: cut + len("</answer>")] if cut != -1 else text


# --------------------------------------------------------------------------- #
# Scoring: self-contained local implementation
# --------------------------------------------------------------------------- #
def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()


def _extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def _fallback_em(prediction: str, gold_answers: list) -> float:
    """Exact match after SQuAD-style normalization."""
    norm_pred = _normalize_answer(prediction)
    for gold in gold_answers:
        if _normalize_answer(gold) == norm_pred:
            return 1.0
    return 0.0


def _fallback_f1(prediction: str, gold_answers: list) -> float:
    norm_pred = _normalize_answer(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return 1.0 if any(not _normalize_answer(g).split() for g in gold_answers) else 0.0
    best = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n = sum(common.values())
        if n == 0:
            continue
        p, r = n / len(pred_tokens), n / len(gold_tokens)
        best = max(best, 2 * p * r / (p + r))
    return best


class _Scorer:
    """Self-contained EM/F1 scorer for SearchQA."""

    def score(self, text: str, item: dict) -> dict:
        gold = item["answers"]
        text = cut_at_answer(text)
        ans = _extract_answer(text)
        return {"em": _fallback_em(ans, gold), "f1": _fallback_f1(ans, gold),
                "predicted_answer": ans, "gold_answers": gold}


# --------------------------------------------------------------------------- #
def _truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(context) <= max_chars:
        return context
    docs = context.split("[DOC]")
    result = ""
    for doc in docs:
        candidate = result + "[DOC]" + doc if result else doc
        if len(candidate) > max_chars:
            break
        result = candidate
    if not result:
        result = context[:max_chars] + "\n...[truncated]"
    return result


class SearchQATask:
    name = "searchqa"

    def __init__(self, root: str | os.PathLike | None = None,
                 split_dir: str | os.PathLike | None = None):
        root = Path(root) if root else data_root()
        self.split_dir = Path(split_dir or root / "data" / "searchqa_split")
        self.items: dict = {}
        for sp in SPLITS:
            f = self.split_dir / sp / "items.json"
            if not f.exists():
                raise FileNotFoundError(
                    f"SearchQA split file missing: {f}\n"
                    "Data files are prepared separately; see README "
                    "(or pass split_dir explicitly).")
            self.items[sp] = json.loads(f.read_text())
        self.scorer = _Scorer()

    # -- data ---------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.items["test"])

    def get_split(self, split: str) -> list:
        return list(self.items[split])

    # -- prompt -------------------------------------------------------- #
    @staticmethod
    def build_system(skill_content: str = "") -> str:
        section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
        return _SYSTEM_TEMPLATE.format(skill_section=section)

    @staticmethod
    def build_user(item: dict) -> str:
        return ("## Context\n" + _truncate_context(item.get("context", ""))
                + "\n\n## Question\n" + item["question"])

    def build_messages(self, item: dict, skill_content: str = "") -> list:
        return [{"role": "system", "content": self.build_system(skill_content)},
                {"role": "user", "content": self.build_user(item)}]

    def build_prompt(self, tokenizer, item: dict, skill_content: str = "") -> str:
        """Return the full prompt text, ready for vLLM / HF."""
        msgs = self.build_messages(item, skill_content)
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        if isinstance(text, list):
            text = text[0]
        return text

    # -- metric -------------------------------------------------------- #
    def metric(self, text: str, item: dict) -> float:
        return float(self.scorer.score(text, item)["em"])

    def reward(self, text: str, item: dict) -> float:
        """GRPO reward = EM (+ optional F1 auxiliary term, mirroring the
        baseline method's reward_aux_w)."""
        return self.metric(text, item)


TASKS = {"searchqa": SearchQATask}


def get_task(name: str, **kw):
    return TASKS[name](**kw)
