"""MCQ task family (CommonsenseQA / OpenBookQA / ARC-Challenge).

* **Data**: ``<split_dir>/{train,val,test}/items.json``
  (csqa 400/200/1221, openbookqa 400/200/500, arc 400/200/1172).
  The split file format originates from the baseline method's released data.
* **System prompt**: ``{domain}\n\n{skill_section}## Task Format…`` — the
  domain strings are fixed below; ``skill_section`` is empty (skills act on
  hidden states, not on the prompt text).
* **User turn**: ``## Question`` + ``## Choices`` (one ``A. text`` per line).
* **Metric**: ``<answer>LABEL</answer>`` extraction + label normalization
  (self-contained implementation below).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .data_searchqa import cut_at_answer
from .probe_paths import data_root

__all__ = ["McqTask", "build_mcq_tasks"]

SPLITS = ("train", "val", "test")

# System-prompt domain strings (fixed per task)
_DOMAIN = {
    "csqa": ("You are answering commonsense multiple-choice questions about everyday "
             "situations, objects, and social norms."),
    "openbookqa": ("You are answering elementary-science multiple-choice questions "
                   "that combine a core science fact with everyday reasoning."),
    "arc": ("You are answering grade-school to middle-school science multiple-choice "
            "questions (physics, biology, earth science, chemistry)."),
}

# Shared MCQ system-prompt template
_SYSTEM_TEMPLATE = """{domain}

{skill_section}## Task Format
You will receive one multiple-choice question and its answer choices.
Exactly one choice is correct.

## Answer Format
Think step by step, then provide your final answer inside <answer>...</answer> tags.
Inside the tags, output only the single choice label, such as A or C.

Example:
<answer>B</answer>
"""


# --------------------------------------------------------------------------- #
# Scoring: self-contained local implementation
# --------------------------------------------------------------------------- #
def _extract_label(text: str) -> str:
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def _norm_label(text) -> str:
    return str(text).strip().upper().rstrip(".):")


def _fallback_parse_label(text: str, choices: list) -> str:
    """Extract the predicted choice label from a generation."""
    ans = _extract_label(text)
    label = _norm_label(ans)
    valid = {_norm_label(c.get("label", "")) for c in choices}
    if label in valid:
        return label
    low = ans.lower()
    for c in choices:
        if str(c.get("text", "")).strip().lower() == low and low:
            return _norm_label(c.get("label", ""))
    first = _norm_label(ans.split()[0]) if ans.split() else ""
    return first if first in valid else label


class _McqScorer:
    """Self-contained MCQ scorer (label extraction + normalization)."""

    def score(self, text: str, item: dict) -> dict:
        text = cut_at_answer(text)
        choices = item["choices"]
        pred = _fallback_parse_label(text, choices)
        ok = float(pred == _norm_label(item["correct_choice"].get("label", "")))
        return {"em": ok, "f1": ok, "predicted_answer": pred,
                "predicted_label": pred}


# --------------------------------------------------------------------------- #
class McqTask:
    """One MCQ task (csqa / openbookqa / arc)."""

    def __init__(self, name: str, root: str | os.PathLike | None = None,
                 split_dir: str | os.PathLike | None = None):
        if name not in _DOMAIN:
            raise KeyError(f"unknown MCQ task {name!r}; have {sorted(_DOMAIN)}")
        self.name = name
        root = Path(root) if root else data_root()
        self.split_dir = Path(split_dir or root / "data" / f"{name}_split")
        self.items: dict = {}
        for sp in SPLITS:
            f = self.split_dir / sp / "items.json"
            if not f.exists():
                raise FileNotFoundError(
                    f"{name} split file missing: {f}\n"
                    "Data files are prepared separately; see README "
                    "(or pass split_dir explicitly).")
            self.items[sp] = json.loads(f.read_text())
        self.scorer = _McqScorer()

    # -- data ---------------------------------------------------------- #
    def get_split(self, split: str) -> list:
        return list(self.items[split])

    # -- prompt -------------------------------------------------------- #
    def build_system(self, skill_content: str = "") -> str:
        section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
        return _SYSTEM_TEMPLATE.format(domain=_DOMAIN[self.name], skill_section=section)

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

    # -- metric -------------------------------------------------------- #
    @staticmethod
    def gold(item: dict) -> str:
        """Gold answer for the BC target (the correct choice label)."""
        return str(item["correct_choice"]["label"]).strip()

    def metric(self, text: str, item: dict) -> float:
        return float(self.scorer.score(text, item)["em"])

    def reward(self, text: str, item: dict) -> float:
        return self.metric(text, item)


def build_mcq_tasks() -> dict:
    return {n: (lambda n=n, **kw: McqTask(n, **kw)) for n in _DOMAIN}
