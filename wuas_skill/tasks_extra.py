"""Extra task types: DocVQA (multimodal / ANLS) and STaRK-Prime (10-way EM).

This file only *adds* tasks; ``wuas_skill/tasks.py`` stays untouched. At
runtime, :func:`register_extra` injects the tasks into the
``wuas_skill.tasks.TASKS`` registry and teaches :func:`wuas_skill.tasks.gold_target`
the new task names, so existing code and the new tasks stay fully decoupled.

Protocols:

* **DocVQA**
  * Data: ``<split_dir>/{split}/{split}.csv`` (train 107 / val 53 / test 374)
    with columns questionId/question/answer (list literal)/image_path/topic.
    The split file format originates from the baseline method's released data.
  * System prompt: constant below (``skill_section`` empty).
  * User turn: ``question + "\\n\\nReturn the final answer inside <answer>...</answer>."``
    + image.
  * Metric: ANLS (threshold 0.5, local implementation below).

* **STaRK-Prime**
  * Data: ``<split_dir>/{split}/items.json`` (train 1992 / val 299 / test 500).
  * System prompt: ``_STARK_DOMAIN`` + the shared ``_MCQ_SYSTEM_TEMPLATE``.
  * User turn: ``## Question`` + ``## Choices`` (one ``A. text`` per line).
  * Metric: 10-way EM (local implementation below).
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
from pathlib import Path

from .probe_paths import data_root

__all__ = ["DocVQATask", "StarkPrimeTask", "register_extra", "gold_target_extra"]

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Common utilities
# --------------------------------------------------------------------------- #
def _cut_at_answer(text: str) -> str:
    cut = text.find("</answer>")
    return text[: cut + len("</answer>")] if cut != -1 else text


def _extract_label(text: str) -> str:
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def _norm_label(text) -> str:
    return str(text).strip().upper().rstrip(".):")


# --------------------------------------------------------------------------- #
# DocVQA
# --------------------------------------------------------------------------- #
# Inlined verbatim from the baseline method's DocVQA rollout system prompt
# (English original); ``{skill_section}`` stays empty in practice.
_DOCVQA_SYSTEM_TEMPLATE = """You are an expert visual document question answering agent.

{skill_section}You will receive a document image and a question about the document.
Read the visual evidence carefully and answer concisely.

Rules:
- Ground the answer in the visible document content.
- Prefer exact spans, numbers, dates, and names from the document.
- Do not invent content that is not visible.
- If multiple near-matches exist, choose the one best supported by the document.

Return the final answer inside <answer>...</answer>.
"""

_ANSWER_INSTRUCTION = "\n\nReturn the final answer inside <answer>...</answer>."


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm_text(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _score_anls(pred: str, gold: str, threshold: float = 0.5) -> float:
    a, b = _norm_text(pred), _norm_text(gold)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    d = _levenshtein(a, b) / max(len(a), len(b))
    return 0.0 if d >= threshold else 1.0 - d


class _DocVQAScorer:
    """Self-contained ANLS scorer (threshold 0.5)."""

    def score(self, text: str, item: dict) -> dict:
        answers = item.get("answers") or []
        pred = _extract_label(text)
        best = max((_score_anls(pred, g) for g in answers), default=0.0)
        return {"anls": best, "em": best, "f1": best,
                "predicted_answer": pred, "gold_answers": answers}


class DocVQATask:
    """DocVQA (multimodal): inference goes through the HF path only (no vLLM server)."""

    name = "docvqa"
    multimodal = True

    def __init__(self, root: str | os.PathLike | None = None,
                 split_dir: str | os.PathLike | None = None,
                 strict_counts: bool = True):
        self.root = Path(root) if root else data_root()
        self.split_dir = Path(split_dir or os.environ.get(
            "DOCVQA_SPLIT_DIR", self.root / "data" / "docvqa" / "splits"))
        self.images_dir = Path(os.environ.get(
            "DOCVQA_IMAGES", self.root / "data" / "docvqa_images"))
        self.strict_counts = strict_counts
        self.items = self._load_via_csv()
        self.scorer = _DocVQAScorer()

    # -- image path self-healing ----------------------------------------- #
    def _resolve_image(self, raw: str) -> str:
        """``image_path`` entries in the CSV are absolute and break when the
        data directory moves; if the original path no longer exists, fall back
        to ``<root>/data/docvqa_images/<basename>``."""
        if raw and os.path.exists(raw):
            return raw
        if raw:
            cand = self.images_dir / Path(raw).name
            if cand.exists():
                return str(cand)
        return raw

    # -- data ---------------------------------------------------------- #
    def _load_via_csv(self) -> dict:
        out = {}
        for s in SPLITS:
            f = self.split_dir / s / f"{s}.csv"
            if not f.exists():
                raise FileNotFoundError(f"docvqa split file missing: {f}")
            with f.open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            items = []
            for r in rows:
                raw = r.get("answer") or ""
                try:
                    answers = [str(x).strip() for x in ast.literal_eval(raw)]
                except Exception:
                    answers = [raw.strip()]
                img = self._resolve_image(r.get("image_path") or "")
                items.append({"id": str(r.get("questionId") or r.get("id") or "").strip(),
                              "questionId": str(r.get("questionId") or "").strip(),
                              "question": r.get("question") or "",
                              "answers": answers,
                              "answer": answers[0] if answers else "",
                              "image_path": img,
                              "task_type": r.get("topic") or "docvqa"})
            out[s] = items
        if self.strict_counts:
            want = {"train": 107, "val": 53, "test": 374}
            got = {s: len(out[s]) for s in SPLITS}
            if got != want:
                raise RuntimeError(f"docvqa split counts {got} != {want}")
        return out

    def get_split(self, split: str) -> list:
        return list(self.items[split])

    def __len__(self) -> int:
        return len(self.items["test"])

    # -- prompt -------------------------------------------------------- #
    def build_system(self, skill_content: str = "") -> str:
        section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
        return _DOCVQA_SYSTEM_TEMPLATE.format(skill_section=section)

    @staticmethod
    def build_user_text(item: dict) -> str:
        return str(item["question"]) + _ANSWER_INSTRUCTION

    def build_messages(self, item: dict, skill_content: str = "") -> list:
        """Message format with an image content block
        (``{"type": "image", "image": path}``)."""
        return [
            {"role": "system", "content": self.build_system(skill_content)},
            {"role": "user", "content": [
                {"type": "text", "text": self.build_user_text(item)},
                {"type": "image", "image": item["image_path"]},
            ]},
        ]

    def build_prompt(self, tokenizer, item: dict, skill_content: str = "") -> str:
        raise NotImplementedError(
            "DocVQA is a multimodal task with no text-only prompt; "
            "use build_messages(...) + processor.apply_chat_template(...)")

    # -- metric -------------------------------------------------------- #
    @staticmethod
    def gold(item: dict) -> str:
        answers = item.get("answers") or []
        return answers[0] if answers else str(item.get("answer", ""))

    def metric(self, text: str, item: dict) -> float:
        return float(self.scorer.score(_cut_at_answer(text), item)["anls"])

    def reward(self, text: str, item: dict) -> float:
        return self.metric(text, item)


# --------------------------------------------------------------------------- #
# STaRK-Prime
# --------------------------------------------------------------------------- #
# System-prompt domain string for STaRK-Prime
_STARK_DOMAIN = ("You are identifying which entity of a biomedical knowledge graph "
                 "satisfies a natural-language query. Each candidate is shown as its "
                 "name followed by its entity type in parentheses.")

# Shared MCQ system-prompt template
_MCQ_SYSTEM_TEMPLATE = """{domain}

{skill_section}## Task Format
You will receive one multiple-choice question and its answer choices.
Exactly one choice is correct.

## Answer Format
Think step by step, then provide your final answer inside <answer>...</answer> tags.
Inside the tags, output only the single choice label, such as A or C.

Example:
<answer>B</answer>
"""


class _McqScorer:
    """Self-contained MCQ EM scorer (label extraction + normalization)."""

    def score(self, text: str, item: dict) -> dict:
        text = _cut_at_answer(text)
        choices = item["choices"]
        pred = _parse_label_fallback(text, choices)
        ok = float(pred == _norm_label(item["correct_choice"].get("label", "")))
        return {"em": ok, "f1": ok, "predicted_answer": pred}


def _parse_label_fallback(text: str, choices: list) -> str:
    ans = _extract_label(text)
    label = _norm_label(ans)
    valid = {_norm_label(c.get("label", "")) for c in choices}
    if label in valid:
        return label
    for c in choices:
        t = str(c.get("text", "")).strip().lower()
        if t and t in str(ans).strip().lower():
            return _norm_label(c.get("label", ""))
    return label


class StarkPrimeTask:
    """STaRK-Prime (10-way EM)."""

    name = "stark_prime"

    def __init__(self, root: str | os.PathLike | None = None,
                 split_dir: str | os.PathLike | None = None):
        self.root = Path(root) if root else data_root()
        self.split_dir = Path(split_dir or self.root / "data" / "stark_prime_split")
        self.items = {}
        for sp in SPLITS:
            f = self.split_dir / sp / "items.json"
            if not f.exists():
                raise FileNotFoundError(
                    f"STaRK-Prime split file missing: {f}\n"
                    "Data files are prepared separately; see README "
                    "(or pass split_dir explicitly).")
            self.items[sp] = json.loads(f.read_text())
        self.scorer = _McqScorer()

    def get_split(self, split: str) -> list:
        return list(self.items[split])

    def __len__(self) -> int:
        return len(self.items["test"])

    def build_system(self, skill_content: str = "") -> str:
        section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
        return _MCQ_SYSTEM_TEMPLATE.format(domain=_STARK_DOMAIN, skill_section=section)

    @staticmethod
    def build_user(item: dict) -> str:
        choices = "\n".join(f"{c['label']}. {c['text']}" for c in item["choices"])
        return f"## Question\n{item['question']}\n\n## Choices\n{choices}"

    def build_messages(self, item: dict, skill_content: str = "") -> list:
        return [{"role": "system", "content": self.build_system(skill_content)},
                {"role": "user", "content": self.build_user(item)}]

    def build_prompt(self, tokenizer, item: dict, skill_content: str = "") -> str:
        msgs = self.build_messages(item, skill_content)
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        return text[0] if isinstance(text, list) else text

    @staticmethod
    def gold(item: dict) -> str:
        return str(item["correct_choice"]["label"]).strip()

    def metric(self, text: str, item: dict) -> float:
        return float(self.scorer.score(text, item)["em"])

    def reward(self, text: str, item: dict) -> float:
        return self.metric(text, item)


# --------------------------------------------------------------------------- #
# Runtime registration (tasks.py itself stays untouched)
# --------------------------------------------------------------------------- #
_EXTRA_TASKS = {
    "docvqa": DocVQATask,
    "stark_prime": StarkPrimeTask,
}


def _gold_in_store(name: str) -> None:  # pragma: no cover - placeholder
    raise KeyError(name)


def gold_target_extra(task_name: str, item: dict) -> str:
    """BC supervision target for the extra tasks (filled into the
    ``<answer>{gold}</answer>`` template)."""
    if task_name == "docvqa":
        return DocVQATask.gold(item)
    if task_name == "stark_prime":
        return StarkPrimeTask.gold(item)
    raise KeyError(f"no gold_target rule for task {task_name!r}")


def register_extra(verbose: bool = True) -> list:
    """Register the extra tasks in ``wuas_skill.tasks``'s registry and its
    ``gold_target`` rules.

    Returns the list of newly registered task names. Idempotent.
    """
    from . import tasks as tasks_mod

    added = []
    for name, cls in _EXTRA_TASKS.items():
        tasks_mod.TASKS[name] = cls
        added.append(name)

    if not getattr(tasks_mod, "_EXTRA_GOLD_REGISTERED", False):
        original = tasks_mod.gold_target

        def gold_target(task_name: str, item: dict) -> str:
            if task_name in _EXTRA_TASKS:
                return gold_target_extra(task_name, item)
            return original(task_name, item)

        gold_target.__wrapped__ = original          # lets callers check the wrap
        tasks_mod.gold_target = gold_target
        tasks_mod._EXTRA_GOLD_REGISTERED = True

    if verbose:
        print(f"[tasks_extra] registered {added}; TASKS now "
              f"{tasks_mod.list_tasks()}")
    return added


if __name__ == "__main__":
    register_extra()
