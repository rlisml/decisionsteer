"""Fast vLLM evaluation — injects WUAS-Skill operators into the engine and measures EM.

Requires a vLLM build exposing the ``vllm.steer_vectors`` steer-vector API.

Three modes:
  * ``--scale 0`` or no ``--bundle``: base model (same-engine control)
  * ``--bundle ... --scale 1``: load the WUAS-Skill operator for online steering
  * ``--bundle ... --scale s``: strength sweep (s in [0, 2], controllability check)

Outputs: ``<out-dir>/<tag>.jsonl`` (per item) + ``<tag>.metrics.json`` (summary).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .tasks import get_task

DEFAULT_ENV = "vllm-steer"  # informational name of the vLLM env with steer-vector support


def _gold_of(item: dict):
    """Generic gold field (SearchQA: answers; MCQ: correct_choice)."""
    if "answers" in item:
        return item["answers"]
    if "correct_choice" in item:
        return item["correct_choice"]
    return None


def _apply_prompt_suffix(task, tokenizer, item: dict, suffix: str) -> str:
    """Append a suffix instruction to the last user message (task template untouched)."""
    if hasattr(task, "build_messages"):
        msgs = [dict(m) for m in task.build_messages(item)]
        msgs[-1]["content"] = msgs[-1]["content"].rstrip() + "\n\n" + suffix
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        return text[0] if isinstance(text, list) else text
    return task.build_prompt(tokenizer, item) + "\n\n" + suffix


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="vLLM eval for WUAS-Skill")
    p.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    p.add_argument("--task", default="searchqa")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--split-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--bundle", default=None)
    p.add_argument("--scale", type=float, default=1.0)
    # Protocol: greedy with a generous budget, stopping at the answer delimiter so
    # deliberation before answering is not penalized. Default 1024 suffices for MCQ;
    # LiveMath needs 4096+.
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--stop-strings", default="</answer>",
                   help="stop when this string appears (comma-separated). "
                        "Empty string disables stop strings")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-prompt-tokens", type=int, default=3072)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--algorithm", default="wuas_skill")
    p.add_argument("--prompt-suffix", default=None,
                   help="instruction appended to the last user message (e.g. forced-direct-answer control)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="sampling temperature (0=greedy; >0 with --n-samples for pass@k)")
    p.add_argument("--n-samples", type=int, default=1,
                   help="samples per prompt (vLLM n; >1 emits pass@k stats)")
    p.add_argument("--inject-phases", choices=["both", "generation"], default="both",
                   help="injection phases: both = prompt+generation (current); "
                        "generation = only the generation phase (apply.prompt=null)")
    p.add_argument("--force", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()

    import wuas_skill_plugin  # noqa: F401  # register in the parent process (engine process uses the entry point)

    wuas_skill_plugin.register()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    task = get_task(args.task, **(dict(split_dir=args.split_dir) if args.split_dir else {}))
    items = task.get_split(args.split)[args.offset:]
    if args.limit:
        items = items[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.prompt_suffix:
        prompts = [_apply_prompt_suffix(task, tokenizer, it, args.prompt_suffix)
                   for it in items]
    else:
        prompts = [task.build_prompt(tokenizer, it) for it in items]

    # Overlong prompts raise an error; never silently truncate
    lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    over = [i for i, n in enumerate(lens) if n + args.max_new_tokens > args.max_model_len]
    if over:
        raise SystemExit(f"{len(over)} prompts exceed max_model_len={args.max_model_len}; "
                         "increase --max-model-len (no silent truncation)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale_tag = f"scale{args.scale:g}"
    out_path = out_dir / f"{args.task}__{args.run_tag}__{args.split}__n{len(items)}__{scale_tag}.jsonl"

    if out_path.exists() and not args.force:
        print(f"[eval-vllm] {out_path} exists, skipping (use --force to rerun)")
        return 0

    steering = None
    manifest = None
    if args.bundle:
        with open(os.path.join(args.bundle, "manifest.json")) as f:
            manifest = json.load(f)
        # Injection phases: both (default) injects into prompt and generation;
        # generation-only leaves prompt=None, so SelectSpec selects no tokens there.
        if args.inject_phases == "generation":
            apply_clause = {"prompt": None, "generation": "all"}
        else:
            apply_clause = {"prompt": "all", "generation": "all"}
        vector_spec = {
            "source": os.path.abspath(args.bundle),
            "algorithm": args.algorithm,
            "scale": args.scale,
            "layers": list(manifest["depths"]),
            "apply": apply_clause,
            "name": f"{args.run_tag}-{scale_tag}",
        }
        steering = {"vectors": [vector_spec], "conflict": "priority"}

    # steer_multi_vector=True: multi-phase bundles attach several vectors to one slot
    # (with prompt / generation_window selectors); the engine must declare this at startup.
    kwargs = dict(model=args.model_name, dtype=args.dtype,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  seed=args.seed, enable_steer_vector=True,
                  steer_multi_vector=True,
                  steer_algorithms=[args.algorithm])
    if steering is not None:
        kwargs["steering_config"] = json.dumps(steering)
    llm = LLM(**kwargs)

    # include_stop_str_in_output=True keeps the ``</answer>`` tail; otherwise
    # answer extraction degrades to last-line heuristics.
    stops = [s for s in (args.stop_strings or "").split(",") if s]
    sampling = SamplingParams(temperature=args.temperature,
                              n=args.n_samples,
                              max_tokens=args.max_new_tokens,
                              stop=stops or None, include_stop_str_in_output=True)
    t_gen = time.time()
    outputs = llm.generate(prompts, sampling)
    gen_seconds = time.time() - t_gen

    n_em = 0.0
    n_f1 = 0.0
    n_pass = 0
    n_sample = 0
    n_sample_em = 0.0
    with open(out_path, "w") as f:
        for it, out in zip(items, outputs):
            cands = out.outputs
            if len(cands) == 1:
                text = cands[0].text.strip()
                r = task.scorer.score(text, it)
                n_em += float(r["em"])
                n_f1 += float(r["f1"])
                f.write(json.dumps({
                    "id": it.get("id"),
                    "em": float(r["em"]),
                    "f1": float(r["f1"]),
                    "pred": r["predicted_answer"],
                    "gold": _gold_of(it),
                    "gen_tokens": len(cands[0].token_ids),
                    "finish": cands[0].finish_reason,
                    "output": text[:500],
                }) + "\n")
            else:
                recs = [task.scorer.score(c.text.strip(), it) for c in cands]
                ems = [float(r["em"]) for r in recs]
                toks = [len(c.token_ids) for c in cands]
                n_pass += int(any(ems))
                n_sample += len(ems)
                n_sample_em += sum(ems)
                best = max(range(len(recs)), key=lambda i: recs[i]["em"])
                f.write(json.dumps({
                    "id": it.get("id"),
                    "ems": ems,
                    "pass": int(any(ems)),
                    "mean_em": sum(ems) / len(ems),
                    "gold": _gold_of(it),
                    "preds": [r["predicted_answer"] for r in recs],
                    "gen_tokens": toks,
                    "finish": [c.finish_reason for c in cands],
                    "output": recs[best] and cands[best].text.strip()[:500],
                }) + "\n")
    n = max(1, len(items))
    metrics = {
        "temperature": args.temperature,
        "n_samples": args.n_samples,
        "run_tag": args.run_tag, "model": args.model_name, "task": args.task,
        "split": args.split, "n": len(items), "offset": args.offset,
        "scale": args.scale, "bundle": args.bundle,
        "inject_phases": args.inject_phases,
        "algorithm": args.algorithm if args.bundle else None,
        "depths": manifest["depths"] if manifest else None,
        "n_phases": int(manifest.get("n_phases", 1)) if manifest else None,
        "phase_edges": manifest.get("phase_edges") if manifest else None,
        "params_trainable": manifest.get("params_trainable") if manifest else 0,
        "em": (n_em / n) if args.n_samples == 1 else (n_sample_em / n_sample),
        "f1": n_f1 / n,
        "per_sample_acc": (n_sample_em / n_sample) if n_sample else None,
        "pass_at_k": (n_pass / n) if args.n_samples > 1 else None,
        "predicted_pass_at_k": (1 - (1 - n_sample_em / n_sample) ** args.n_samples)
                               if n_sample else None,
        "max_new_tokens": args.max_new_tokens,
        "prompt_suffix": args.prompt_suffix,
        "mean_prompt_tokens": round(sum(lens) / n, 1),
        "wall_total_s": round(time.time() - t0, 1),
        "wall_generate_s": round(gen_seconds, 1),
        "examples_per_s": round(len(items) / gen_seconds, 2),
        "output_file": str(out_path),
    }
    (out_dir / (out_path.stem + ".metrics.json")).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
