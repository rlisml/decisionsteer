"""LiveMath (LiveMathematicianBench) — the paper's main task; metric Acc. (= MCQ EM).

* **Data**: ``<data_dir>/qa_*_final.json``, normalized by :func:`_load_items`
  (item id ``"{month}:{no}"`` when a month is present, choices coerced to
  ``[{"label", "text"}]``, correct-choice label normalized; rows without a
  question + choices + correct label are skipped).
* **Splits**: ``<manifest_dir>/{train,val,test}/items.json`` id manifests
  (train 35 / val 18 / **test 124**).
* **Choice shuffle**: deterministic per item via ``sha256("42:" + id)``
  (``SHUFFLE_SEED=42``), identical to the baseline method's behavior.
* **System prompt**: fixed template below; ``skill_section`` is empty.
* **Metric**: shares the MCQ EM scorer.

Generation budget: the baseline method used ``max_new=8000`` (20% of the base
model's context); this repo defaults to 2048 — adjust via ``--max-new-tokens``.
Prompts can reach several thousand tokens because of long LaTeX theorem
statements, so ``--max-model-len`` must be sized accordingly (the default
scripts use 16384).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from pathlib import Path

from .data_mcq import _McqScorer
from .data_searchqa import cut_at_answer
from .probe_paths import data_root

__all__ = ["LiveMathTask"]

SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 35, "val": 18, "test": 124}
SHUFFLE_SEED = 42
_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]

_SYSTEM_TEMPLATE = """You are an expert mathematical reasoning agent solving multiple-choice questions.

{skill_section}## Task Format
You will receive one mathematics multiple-choice question and its answer choices.
Reason carefully about quantifiers, hypotheses, extremal wording, and exact equality conditions.

## Answer Format
Think step by step, then provide your final answer inside <answer>...</answer> tags.
Inside the tags, output only the single choice label, such as A or C.

Example:
<answer>B</answer>
"""


def _norm_label(text) -> str:
    return str(text).strip().upper().rstrip(".):")


# --------------------------------------------------------------------------- #
# Data loading: normalize the raw monthly qa_*_final.json files
# --------------------------------------------------------------------------- #
def _iter_monthly_files(data_dir: str) -> list:
    """Collect qa_*_final.json files (a single file or a directory tree)."""
    if not data_dir:
        return []
    if os.path.isfile(data_dir):
        return [data_dir]
    if os.path.isdir(data_dir):
        nested = glob.glob(os.path.join(data_dir, "**", "qa_*_final.json"),
                           recursive=True)
        flat = glob.glob(os.path.join(data_dir, "qa_*_final.json"))
        return sorted(set(nested + flat))
    return []


def _coerce_choices(raw) -> list:
    """Coerce raw choice data into a list of ``{"label", "text"}`` dicts."""
    if isinstance(raw, list):
        choices: list = []
        for idx, entry in enumerate(raw):
            if isinstance(entry, dict):
                label = str(entry.get("label") or _CHOICE_LABELS[idx]).strip()
                text = str(entry.get("text") or entry.get("content") or "").strip()
            else:
                label = _CHOICE_LABELS[idx]
                text = str(entry).strip()
            if text:
                choices.append({"label": label, "text": text})
        return choices
    if isinstance(raw, dict):
        return [{"label": str(k).strip(), "text": str(raw[k]).strip()}
                for k in sorted(raw) if str(raw[k]).strip()]
    return []


def _coerce_theorem_types(raw) -> list:
    """Coerce theorem_type into a list of non-empty strings."""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    return [text] if text else []


def _normalize_item(item: dict, row_idx: int) -> dict:
    mcq = item.get("mcq") if isinstance(item.get("mcq"), dict) else {}
    question = str(mcq.get("question") or item.get("question") or "").strip()
    choices = _coerce_choices(mcq.get("choices") or item.get("choices") or [])
    correct = mcq.get("correct_choice") or item.get("correct_choice") or {}
    if isinstance(correct, dict):
        correct_label = _norm_label(correct.get("label", ""))
        correct_text = str(correct.get("text") or "").strip()
    else:
        correct_label = _norm_label(correct)
        correct_text = ""

    by_label = {_norm_label(c["label"]): c["text"] for c in choices}
    if correct_label and not correct_text:
        correct_text = by_label.get(correct_label, "")
    if correct_label and correct_text and correct_label not in by_label:
        choices.append({"label": correct_label, "text": correct_text})
        choices.sort(key=lambda c: _CHOICE_LABELS.index(c["label"])
                     if c["label"] in _CHOICE_LABELS else len(_CHOICE_LABELS))

    month = str(item.get("month") or "").strip()
    no = item.get("no", row_idx + 1)
    return {
        "id": f"{month}:{no}" if month else str(no),
        "month": month,
        "no": no,
        "theorem": str(item.get("theorem") or "").strip(),
        "sketch": str(item.get("sketch") or "").strip(),
        "theorem_type": _coerce_theorem_types(item.get("theorem_type")),
        "question": question,
        "choices": choices,
        "correct_choice": {"label": correct_label, "text": correct_text},
    }


def _load_items(data_dir: str) -> list:
    """Load and normalize LiveMathematicianBench items from qa_*_final.json files."""
    files = _iter_monthly_files(data_dir)
    if not files:
        raise ValueError(
            "livemath requires data_dir to be a qa_*_final.json file or a "
            "directory containing monthly qa_*_final.json files.")
    items: list = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            raise ValueError(f"Expected a JSON array in {path}")
        for row_idx, row in enumerate(raw):
            norm = _normalize_item(row, row_idx=row_idx)
            if norm["question"] and norm["choices"] and norm["correct_choice"]["label"]:
                items.append(norm)
    if not items:
        raise ValueError(f"No valid livemath items loaded from {data_dir}")
    return items


def _shuffle_item_choices(item: dict, seed: int = SHUFFLE_SEED) -> dict:
    """Deterministic per-item choice shuffle (sha256 of ``"{seed}:{id}"``)."""
    digest = hashlib.sha256(f"{seed}:{item['id']}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    shuffled = [dict(c) for c in item["choices"]]
    rng.shuffle(shuffled)
    original_correct = _norm_label(item["correct_choice"]["label"])
    remapped, new_correct = [], dict(item["correct_choice"])
    for idx, choice in enumerate(shuffled):
        new_label = _CHOICE_LABELS[idx]
        remapped.append({"label": new_label, "text": choice["text"]})
        if _norm_label(choice["label"]) == original_correct:
            new_correct = {"label": new_label, "text": choice["text"]}
    return {**item, "choices": remapped, "correct_choice": new_correct}


class LiveMathTask:
    name = "livemath"

    def __init__(self, root: str | os.PathLike | None = None,
                 data_dir: str | os.PathLike | None = None,
                 manifest_dir: str | os.PathLike | None = None,
                 split_dir: str | os.PathLike | None = None):
        # ``split_dir`` is the parameter name used by the other tasks
        # (mcq/searchqa) for a directory holding ``{train,val,test}/items.json``;
        # here it serves as an alias for ``manifest_dir`` so that
        # train_bc/eval_vllm --split-dir can pass it uniformly (it is also used
        # to swap manifests when extending the training set).
        root = Path(root) if root else data_root()
        self.data_dir = Path(data_dir or root / "data" / "livemath")
        self.manifest_dir = Path(split_dir or manifest_dir
                                 or root / "third_party" / "SoftSkill"
                                 / "data" / "livemathematicianbench_id_split")
        self.items = {sp: self._load(sp) for sp in SPLITS}
        self.scorer = _McqScorer()

    def _load(self, split: str) -> list:
        mpath = self.manifest_dir / split / "items.json"
        if not mpath.exists():
            raise FileNotFoundError(f"LiveMath manifest file missing: {mpath}")
        by_id = {it["id"]: it for it in _load_items(str(self.data_dir))}
        ids = [row["id"] for row in json.loads(mpath.read_text())]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise RuntimeError(
                f"{len(missing)} manifest ids missing from the local data "
                f"(e.g. {missing[:3]})")
        out = [_shuffle_item_choices(by_id[i]) for i in ids]
        if len(out) != EXPECTED_COUNTS[split]:
            print(f"[livemath] warning: split {split!r} has {len(out)} items, "
                  f"expected {EXPECTED_COUNTS[split]}")
        return out

    # -- data ---------------------------------------------------------- #
    def get_split(self, split: str) -> list:
        return list(self.items[split])

    # -- prompt -------------------------------------------------------- #
    def build_system(self, skill_content: str = "") -> str:
        section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
        return _SYSTEM_TEMPLATE.format(skill_section=section)

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
        return str(item["correct_choice"]["label"]).strip()

    def metric(self, text: str, item: dict) -> float:
        return float(self.scorer.score(cut_at_answer(text), item)["em"])

    def reward(self, text: str, item: dict) -> float:
        return self.metric(text, item)
