"""HuggingFace multimodal probe evaluation (DocVQA path).

vLLM's continuous batching breaks multimodal decoding, so DocVQA is evaluated with a
plain HF rollout. The protocol matches the vLLM fast path: greedy decoding
(``do_sample=False``), a generous token budget, and stopping at ``</answer>`` with the
stop string kept in the output (equivalent to ``include_stop_str_in_output=True``),
implemented via a ``StoppingCriteria``. Scoring uses
``wuas_skill.tasks_extra.DocVQATask.scorer`` (ANLS).

Outputs ``<out-dir>/<task>__<tag>__<split>__n<N>.{jsonl,metrics.json}``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import tasks_extra  # noqa: F401  # ensure extra tasks are registered


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HF multimodal probe eval")
    p.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    p.add_argument("--task", default="docvqa")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--split-dir", default=None)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--stop-string", default="</answer>")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="cap per-image pixels to control visual token count; "
                        "default uses the transformers default")
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--out-dir", default="results/wuas_skill/probe_tasks")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--print-every", type=int, default=5)
    # ---- operator: without --bundle this is the plain base model (scale=0 is not
    # assumed equivalent to no bundle) ----
    p.add_argument("--bundle", default=None, help="wuas_skill bundle directory")
    p.add_argument("--scale", type=float, default=1.0,
                   help="operator strength (linear scaling of cfg.alpha; 1.0 = as trained)")
    return p


class StopOnString:
    """Stop generation once ``stop_text`` appears."""

    def __init__(self, tokenizer, n_prompt: int, stop_text: str, window: int = 64):
        self.tok = tokenizer
        self.n_prompt = n_prompt
        self.stop_text = stop_text
        self.window = window

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0]
        start = max(self.n_prompt, int(seq.shape[-1]) - self.window)
        return self.stop_text in self.tok.decode(seq[start:], skip_special_tokens=False)


def load_model(args, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model_name)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name, dtype=torch.bfloat16, attn_implementation=args.attn_impl)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_impl)
    model.to(device)
    model.eval()
    return proc, model


def check_layer_stack(model, n_layers: int) -> dict:
    """Check that the VLM layer path matches ``find_layer_stack`` (the injection point)."""
    from .steerer import find_layer_stack

    try:
        stack = find_layer_stack(model, n_layers)
        return {"ok": True, "n_layers": n_layers, "layers_found": len(stack)}
    except Exception as e:
        return {"ok": False, "n_layers": n_layers,
                "error": f"{type(e).__name__}: {e}"}


def _apply_chat(proc, tokenizer, messages):
    try:
        return proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t_start = time.time()

    from PIL import Image
    from transformers import StoppingCriteriaList

    from .tasks import get_task

    tasks_extra.register_extra(verbose=False)
    task = get_task(args.task,
                    **(dict(split_dir=args.split_dir) if args.split_dir else {}))
    items = task.get_split(args.split)[args.offset:]
    if args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}__{args.run_tag}__{args.split}__n{len(items)}.jsonl"
    if out_path.exists() and not args.force:
        print(f"[eval-hf-probe] {out_path} exists, skipping (use --force to rerun)")
        return 0

    import torch

    device = torch.device("cuda:0")
    t_load = time.time()
    proc, model = load_model(args, device)
    model_load_s = round(time.time() - t_load, 1)

    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None)
    n_layers = int(text_cfg.num_hidden_layers) if text_cfg is not None else 0
    layer_info = check_layer_stack(model, n_layers) if n_layers else {
        "ok": False, "error": "no text_config"}
    print(f"[eval-hf-probe] model loaded {model_load_s}s; layer-stack {layer_info}",
          flush=True)

    # ---- optional: attach the WUAS-Skill operator to the HF model (no bundle => plain base) ----
    gen_model, bundle_meta = model, None
    if args.bundle:
        from .bundle import load_operator_from_bundle
        from .steerer import SkillSteerer

        op = load_operator_from_bundle(args.bundle)
        if args.scale != 1.0:
            op.cfg.alpha = float(op.cfg.alpha) * float(args.scale)
        steerer = SkillSteerer(model, op.cfg)
        steerer.operator.load_state_dict(op.state_dict())
        steerer.operator.eval()
        steerer.enable()
        steerer.set_phase_ctx(None)
        gen_model = steerer
        bundle_meta = {"bundle": args.bundle, "scale": args.scale,
                       "depths": list(op.cfg.depths), "skill_dim": op.cfg.skill_dim,
                       "n_phases": int(op.n_phases),
                       "params_trainable": int(steerer.num_trainable())}
        print(f"[eval-hf-probe] bundle attached: {bundle_meta}", flush=True)

    img_kwargs = {}
    if args.max_pixels:
        img_kwargs["max_pixels"] = args.max_pixels
    if args.min_pixels:
        img_kwargs["min_pixels"] = args.min_pixels

    tokenizer = proc.tokenizer
    img_token_id = getattr(cfg, "image_token_id", None)

    rows, n_anls, n_hard, gen_s = [], 0.0, 0, 0.0
    for i, item in enumerate(items):
        image = Image.open(item["image_path"]).convert("RGB")
        text = _apply_chat(proc, tokenizer, task.build_messages(item))
        try:
            inputs = proc(text=[text], images=[image], return_tensors="pt", **img_kwargs)
        except TypeError:
            inputs = proc(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        n_visual = int((inputs["input_ids"] == img_token_id).sum()) \
            if img_token_id is not None else -1
        n_prompt = int(inputs["input_ids"].shape[-1])

        stopping = StoppingCriteriaList(
            [StopOnString(tokenizer, n_prompt, args.stop_string)])
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, stopping_criteria=stopping)
        dt = time.time() - t0
        gen_s += dt

        gen_ids = out[0][n_prompt:]
        text_out = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        r = task.scorer.score(text_out, item)
        anls = float(r["anls"])
        n_anls += anls
        n_hard += int(anls >= 0.999)
        rows.append({
            "id": item.get("id"), "anls": anls, "em": anls,
            "pred": r.get("predicted_answer"),
            "gold": item.get("answers") or item.get("answer"),
            "gen_tokens": int(gen_ids.shape[-1]), "prompt_tokens": n_prompt,
            "visual_tokens": n_visual, "image_size": list(image.size),
            "seconds": round(dt, 3), "hit_stop": args.stop_string in text_out,
            "output": text_out[:500],
        })
        if (i + 1) % max(1, args.print_every) == 0:
            print(f"  [{i+1}/{len(items)}] mean_ANLS={n_anls/(i+1):.4f} "
                  f"last={anls:.3f} vis_tok={n_visual} {dt:.2f}s", flush=True)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = max(1, len(items))
    metrics = {
        "run_tag": args.run_tag, "model": args.model_name, "task": args.task,
        "split": args.split, "n": len(items), "offset": args.offset,
        "backend": "hf-multimodal", "metric": "anls",
        "anls": n_anls / n, "acc_strict": n_hard / n,
        "max_new_tokens": args.max_new_tokens, "stop_string": args.stop_string,
        "hit_stop_rate": sum(bool(r["hit_stop"]) for r in rows) / n,
        "mean_gen_tokens": round(sum(r["gen_tokens"] for r in rows) / n, 1),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / n, 1),
        "mean_visual_tokens": round(sum(r["visual_tokens"] for r in rows) / n, 1),
        "max_pixels": args.max_pixels, "min_pixels": args.min_pixels,
        "bundle_meta": bundle_meta,
        "layer_stack_check": layer_info,
        "model_load_s": model_load_s,
        "wall_total_s": round(time.time() - t_start, 1),
        "wall_generate_s": round(gen_s, 1),
        "seconds_per_example": round(gen_s / n, 3),
        "output_file": str(out_path),
    }
    (out_dir / (out_path.stem + ".metrics.json")).write_text(
        json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
