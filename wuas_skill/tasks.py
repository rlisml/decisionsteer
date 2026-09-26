"""Unified task registry — trainers/evaluators depend only on this module;
adding a task only requires editing this file.

Each task object must implement:
  * ``get_split(split) -> list[dict]``
  * ``build_prompt(tokenizer, item) -> str``
  * ``metric(text, item) -> float``
  * ``scorer.score(text, item) -> dict`` (with ``em``/``f1``/``predicted_answer``)

BC targets are obtained uniformly through :func:`gold_target` (the first gold
answer for SearchQA, the correct choice label for MCQ).
"""

from __future__ import annotations

from .data_livemath import LiveMathTask
from .data_mcq import McqTask
from .data_searchqa import SearchQATask

__all__ = ["TASKS", "get_task", "gold_target", "list_tasks"]


def _mk_searchqa(**kw):
    return SearchQATask(**kw)


def _mk_mcq(name):
    return lambda **kw: McqTask(name, **kw)


# 4 of the 5 main tasks in the paper's Table 1 are supported here
# (the 5th, DocVQA, is multimodal and registered separately via tasks_extra)
TASKS = {
    "searchqa": _mk_searchqa,          # EM
    "livemath": LiveMathTask,          # Acc. (the paper's headline task, test n=124)
    **{name: _mk_mcq(name) for name in ("csqa", "openbookqa", "arc")},
}


def list_tasks() -> list:
    return sorted(TASKS)


def get_task(name: str, **kw):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; have {list_tasks()}")
    return TASKS[name](**kw)


def gold_target(task_name: str, item: dict) -> str:
    """Supervision target text for BC (filled into the ``<answer>{gold}</answer>`` template)."""
    if task_name == "searchqa":
        ans = item.get("answers")
        return ans[0] if isinstance(ans, list) and ans else str(ans)
    if task_name == "livemath":
        return LiveMathTask.gold(item)
    if isinstance(task_name, str) and task_name in ("csqa", "openbookqa", "arc"):
        return McqTask.gold(item)
    # No gold_target rule for this task name
    raise KeyError(f"no gold_target rule for task {task_name!r}")
