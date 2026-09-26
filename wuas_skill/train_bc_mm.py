"""Multimodal behavior-cloning trainer (DocVQA) — HF-only counterpart of ``train_bc.py``.

``train_bc.py`` is text-only (``AutoModelForCausalLM`` + ``task.build_prompt`` +
``steerer.generate(input_ids)``), while DocVQA prompts must carry images. This file
swaps in three pieces:

* model: ``AutoModelForImageTextToText``;
* input: ``processor.apply_chat_template`` + ``processor(text=..., images=...)``,
  passing ``pixel_values`` / ``image_grid_thw`` through to ``SkillSteerer``;
* target: ``<answer>{gold}</answer>`` appended after the multimodal prompt;
  loss is computed on that segment only.

The operator/injection machinery is reused unchanged (``SkillOperator`` /
``SkillSteerer`` / ``save_bundle``): hooks attach to the text decoder layers and work
with visual placeholder tokens in the sequence. ``train_bc.py`` is not modified.

Semantics shared with ``train_bc.py``: frozen backbone (operator-only training), the
same optimizer/schedule/grad_accum/grad_clip logic, and the same output layout
(``operator.pt`` + ``bundle/`` + ``summary.json`` + ``metrics.jsonl``).

One intentional difference: cross entropy is computed only at labeled positions
(gather first, then ``.float()``) — mathematically equivalent to ``ignore_index=-100``
but avoids an fp32 copy of the full (B*S, V) logits, which costs several GB on the
~4k-token image-text prompts of DocVQA.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import get_cosine_schedule_with_warmup

from . import tasks_extra
from .bundle import save_bundle
from .skill_operator import SkillOperatorConfig
from .steerer import SkillSteerer, default_depths
from .tasks import get_task


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WUAS-Skill multimodal (DocVQA) BC trainer")
    p.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    p.add_argument("--task", default="docvqa")
    p.add_argument("--split-dir", default=None)
    p.add_argument("--out-dir", required=True)

    # ---- operator structure (same flags as train_bc.py) ----
    p.add_argument("--arm", default="wuas_skill",
                   choices=["wuas_skill", "wuas", "wuas_skill_linear"])
    p.add_argument("--skill-dim", type=int, default=64)
    p.add_argument("--n-depths", type=int, default=8)
    p.add_argument("--depths", default=None)
    p.add_argument("--depth-rank", type=int, default=0)
    p.add_argument("--use-gain", action="store_true")
    p.add_argument("--gate-bias-init", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--no-rms", action="store_true")
    p.add_argument("--no-gate", action="store_true")

    # ---- optimization ----
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gate-lr", type=float, default=None)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-grad-steps", type=int, default=None)

    # ---- data / images ----
    p.add_argument("--max-train-items", type=int, default=None)
    p.add_argument("--max-target-tokens", type=int, default=32)
    p.add_argument("--target-template", default="<answer>{gold}</answer>")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="consistent with eval: default None (transformers default)")
    p.add_argument("--min-pixels", type=int, default=None)
    p.add_argument("--max-seq-tokens", type=int, default=8192,
                   help="skip (and count) image-text samples longer than this: a few "
                        "DocVQA scans reach ~16k tokens and would OOM in the training "
                        "forward; test-time inference handles them, so eval still runs "
                        "all items")

    # ---- run ----
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=16)
    p.add_argument("--eval-max-new", type=int, default=384)
    p.add_argument("--smoke", action="store_true")
    return p


ARM_PRESETS = {
    "wuas_skill": dict(share="shared", use_rms=True, use_gate=True, activation="silu"),
    "wuas": dict(share="per_depth", use_rms=False, use_gate=False, activation="silu"),
    "wuas_skill_linear": dict(share="shared", use_rms=True, use_gate=True, activation="linear"),
}


def resolve_cfg(args, hidden_size: int, n_layers: int) -> SkillOperatorConfig:
    preset = dict(ARM_PRESETS[args.arm])
    if args.no_rms:
        preset["use_rms"] = False
    if args.no_gate:
        preset["use_gate"] = False
    depths = ([int(x) for x in args.depths.split(",") if x.strip()]
              if args.depths else default_depths(n_layers, args.n_depths))
    return SkillOperatorConfig(
        hidden_size=hidden_size, depths=tuple(depths), skill_dim=args.skill_dim,
        activation=preset["activation"], use_rms=preset["use_rms"],
        use_gate=preset["use_gate"], gate_bias_init=args.gate_bias_init,
        share=preset["share"], depth_rank=args.depth_rank,
        use_gain=args.use_gain, alpha=args.alpha)


# --------------------------------------------------------------------------- #
def load_model_and_processor(args, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model_name)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name, dtype=dtype, attn_implementation="sdpa")
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name, torch_dtype=dtype, attn_implementation="sdpa")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return proc, model


def _img_kwargs(args) -> dict:
    kw = {}
    if args.max_pixels:
        kw["max_pixels"] = args.max_pixels
    if args.min_pixels:
        kw["min_pixels"] = args.min_pixels
    return kw


def encode_example(proc, tokenizer, task, item, args, device):
    """Multimodal prompt + ``<answer>gold</answer>``; returns (enc, prompt_len, target_ids)."""
    messages = task.build_messages(item)
    try:
        text = proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
    except Exception:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    image = Image.open(item["image_path"]).convert("RGB")
    try:
        enc = proc(text=[text], images=[image], return_tensors="pt", **_img_kwargs(args))
    except TypeError:
        enc = proc(text=[text], images=[image], return_tensors="pt")
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}

    target = args.target_template.format(gold=tasks_extra.gold_target_extra(args.task, item))
    t_ids = tokenizer(target, add_special_tokens=False)["input_ids"][: args.max_target_tokens]
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    t_ids = t_ids + [eos]
    return enc, int(enc["input_ids"].shape[1]), t_ids


def collate(batch, pad_id, device):
    """Pack (enc, prompt_len, t_ids) into a right-padded batch (for bs>1).

    Two pitfalls:

    * the processor's ``attention_mask`` covers only the prompt; passing it through
      would override the mask recomputed after appending the target segment — and
      ``Qwen3_5Model.get_rope_index`` indexes ``input_ids`` with ``attention_mask``,
      so mismatched lengths raise IndexError.
    * ``mm_token_type_ids`` (0=text / 1=image) is required by M-RoPE and must be
      extended with 0s alongside the target segment (same length-matching issue).
    """
    seqs, labels, attn, tok_types = [], [], [], []
    extra = {}
    for enc, plen, t_ids in batch:
        ids = enc["input_ids"][0].tolist() + t_ids
        seqs.append(ids)
        labels.append([-100] * plen + t_ids)
        attn.append([1] * len(ids))
        if "mm_token_type_ids" in enc:
            tok_types.append(enc["mm_token_type_ids"][0].tolist() + [0] * len(t_ids))
        for k, v in enc.items():
            if k in ("input_ids", "attention_mask", "mm_token_type_ids"):
                continue
            extra.setdefault(k, []).append(v)

    S, B = max(len(s) for s in seqs), len(seqs)
    ids = torch.full((B, S), pad_id, dtype=torch.long, device=device)
    lab = torch.full((B, S), -100, dtype=torch.long, device=device)
    att = torch.zeros((B, S), dtype=torch.long, device=device)
    for i, (s, l, a) in enumerate(zip(seqs, labels, attn)):
        n = len(s)
        ids[i, :n] = torch.tensor(s, device=device)
        lab[i, :n] = torch.tensor(l, device=device)
        att[i, :n] = torch.tensor(a, device=device)

    out = {"input_ids": ids, "attention_mask": att, "labels": lab}
    if tok_types:
        tt = torch.zeros((B, S), dtype=torch.long, device=device)
        for i, t in enumerate(tok_types):
            tt[i, : len(t)] = torch.tensor(t, device=device)
        out["mm_token_type_ids"] = tt
    for k, vs in extra.items():
        out[k] = torch.cat(vs, dim=0) if vs[0].ndim > 1 else torch.stack(vs)
    return out, [b[1] for b in batch]


def masked_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """CE at aligned positions only (equivalent to ignore_index=-100, but avoids the fp32 (B*S,V) copy)."""
    tgt = labels[:, 1:]
    lg = logits[:, :-1, :]
    mask = tgt != -100
    n = int(mask.sum())
    if n == 0:
        return logits.sum() * 0.0
    sel = lg.reshape(-1, lg.size(-1))[mask.reshape(-1)]
    return F.cross_entropy(sel.float(), tgt.reshape(-1)[mask.reshape(-1)])


@torch.no_grad()
def mm_eval(steerer, proc, task, items, device, args, stop_string="</answer>") -> float:
    """Quick in-training HF multimodal eval (greedy, stops at ``</answer>``, ANLS)."""
    from transformers import StoppingCriteriaList

    from .eval_hf_probe import StopOnString

    tokenizer = proc.tokenizer
    steerer.eval()
    total = 0.0
    for item in items:
        try:
            text = proc.apply_chat_template(task.build_messages(item), tokenize=False,
                                            add_generation_prompt=True)
        except Exception:
            text = tokenizer.apply_chat_template(task.build_messages(item),
                                                 tokenize=False,
                                                 add_generation_prompt=True)
        image = Image.open(item["image_path"]).convert("RGB")
        try:
            enc = proc(text=[text], images=[image], return_tensors="pt",
                       **_img_kwargs(args))
        except TypeError:
            enc = proc(text=[text], images=[image], return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        n_prompt = int(enc["input_ids"].shape[-1])
        steerer.set_phase_ctx(None)
        out = steerer.generate(
            **enc, max_new_tokens=args.eval_max_new, do_sample=False,
            stopping_criteria=StoppingCriteriaList(
                [StopOnString(tokenizer, n_prompt, stop_string)]))
        text_out = tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
        total += float(task.scorer.score(text_out, item)["anls"])
    steerer.train()
    return total / max(1, len(items))


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.max_train_items = args.max_train_items or 8
        args.max_grad_steps = args.max_grad_steps or 3
        args.eval_every = args.eval_every or 3
        args.eval_limit = args.eval_limit or 4
        args.epochs = 1.0

    tasks_extra.register_extra(verbose=False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    task = get_task(args.task, **(dict(split_dir=args.split_dir) if args.split_dir else {}))
    train_items = task.get_split("train")
    if args.max_train_items:
        train_items = train_items[: args.max_train_items]
    val_items = task.get_split("val")[: args.eval_limit] if args.eval_every else []

    proc, model = load_model_and_processor(args, device)
    tokenizer = proc.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_cfg = model.config.get_text_config()
    hidden_size, n_layers = int(text_cfg.hidden_size), int(text_cfg.num_hidden_layers)
    cfg = resolve_cfg(args, hidden_size, n_layers)
    steerer = SkillSteerer(model, cfg)
    print(f"[wuas-skill-mm] model={args.model_name} d_m={hidden_size} L={n_layers} "
          f"depths={list(cfg.depths)} trainable={steerer.num_trainable():,}", flush=True)

    groups = steerer.operator.param_groups(args.lr, args.gate_lr)
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    print(f"[wuas-skill-mm] encoding {len(train_items)} image-text samples "
          f"(max_seq_tokens={args.max_seq_tokens}) ...", flush=True)
    t_enc = time.time()
    encoded, skipped = [], []
    for it in train_items:
        e = encode_example(proc, tokenizer, task, it, args, device)
        if e[1] > args.max_seq_tokens:
            skipped.append((it.get("id"), e[1]))
            continue
        encoded.append(e)
    lens = [e[1] for e in encoded]
    print(f"[wuas-skill-mm] encoded in {time.time()-t_enc:.1f}s; usable {len(encoded)}, "
          f"skipped {len(skipped)} overlong {[s for s in skipped][:5]}; "
          f"prompt tokens mean={sum(lens)/len(lens):.0f} "
          f"max={max(lens)} min={min(lens)}", flush=True)
    if not encoded:
        raise SystemExit("no usable training samples (--max-seq-tokens too small)")

    steps_per_epoch = math.ceil(len(encoded) / args.bs / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_grad_steps:
        total_steps = min(total_steps, args.max_grad_steps)
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * args.warmup_ratio), max(1, total_steps))
    print(f"[wuas-skill-mm] items={len(encoded)} epochs={args.epochs} "
          f"bs={args.bs}×accum{args.grad_accum} total_steps={total_steps}", flush=True)

    rng = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    step, micro, run_loss, run_n = 0, 0, 0.0, 0
    history, done = [], False
    metrics_path = out / "metrics.jsonl"
    steerer.train()
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        order = torch.randperm(len(encoded), generator=rng).tolist()
        for i in range(0, len(order), args.bs):
            if done:
                break
            batch = [encoded[j] for j in order[i: i + args.bs]]
            pack, plens = collate(batch, tokenizer.pad_token_id, device)
            labels = pack.pop("labels")
            steerer.set_phase_ctx(None)
            outp = steerer(**pack)
            loss = masked_ce(outp.logits, labels)
            (loss / args.grad_accum).backward()
            run_loss += float(loss.detach()) * int((labels[:, 1:] != -100).sum())
            run_n += int((labels[:, 1:] != -100).sum())
            micro += 1

            if micro % args.grad_accum == 0:
                gmax = steerer.operator.clip_grads_(args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "epoch": epoch, "loss": run_loss / max(1, run_n),
                       "grad_max": gmax, "lr": sched.get_last_lr()[0],
                       "wall_s": round(time.time() - t0, 1),
                       "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                run_loss, run_n = 0.0, 0
                print(f"[step {step}/{total_steps}] " +
                      " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
                history.append(rec)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if args.eval_every and step % args.eval_every == 0 and val_items:
                    anls = mm_eval(steerer, proc, task, val_items, device, args)
                    print(f"[eval] step={step} val_anls={anls:.4f} "
                          f"n={len(val_items)}", flush=True)
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps({"step": step, "val_anls": anls}) + "\n")
                if step >= total_steps:
                    done = True
                    break

    torch.save(steerer.operator.state_dict(), out / "operator.pt")
    report = save_bundle(steerer.operator, str(out / "bundle"),
                         model_name=args.model_name, task=args.task,
                         extra_meta={
                             "arm": args.arm,
                             "train": {"lr": args.lr, "epochs": args.epochs, "bs": args.bs,
                                       "grad_accum": args.grad_accum,
                                       "n_train_items": len(encoded), "steps": step,
                                       "seed": args.seed, "smoke": bool(args.smoke)},
                             "modality": "multimodal",
                         })
    summary = {
        "model": args.model_name, "task": args.task, "arm": args.arm,
        "cfg": cfg.to_dict(), "params_trainable": steerer.num_trainable(),
        "steps": step, "seed": args.seed, "train_seconds": round(time.time() - t0, 1),
        "prompt_tokens_mean": round(sum(lens) / len(lens), 1),
        "loadback": report,
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
