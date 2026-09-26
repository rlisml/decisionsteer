"""WUAS-Skill trainer (step 1: behavior cloning).

Behavior cloning on answer tokens only: gold answers from the train split are
rendered as ``<answer>...</answer>`` targets and loss is computed solely on those
tokens. The backbone stays frozen (``requires_grad_(False)``); only the
:class:`SkillOperator` parameters (~0.3M) are trained. GRPO-style RL is a possible
follow-up (``--arm`` reserved).

Arms (ablation)
---------------
``wuas_skill``  shared trunk + RMS norm + gating (this method)
``wuas``        plain per-depth low-rank WUAS adapters (same depths/rank, no
                sharing/norm/gating), i.e. ``share=per_depth, use_rms=False, use_gate=False``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .bundle import save_bundle
from .skill_operator import SkillOperator, SkillOperatorConfig
from .steerer import SkillSteerer, default_depths, find_layer_stack
from .tasks import get_task, gold_target

# Arm presets: structural switches (explicit CLI args override)
ARM_PRESETS = {
    "wuas_skill": dict(share="shared", use_rms=True, use_gate=True, activation="silu"),
    "wuas": dict(share="per_depth", use_rms=False, use_gate=False, activation="silu"),
    "wuas_skill_linear": dict(share="shared", use_rms=True, use_gate=True, activation="linear"),
}


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WUAS-Skill behavior-cloning trainer")
    p.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    p.add_argument("--task", default="searchqa")
    p.add_argument("--split-dir", default=None)
    p.add_argument("--out-dir", required=True)

    # ---- operator structure ----
    p.add_argument("--arm", default="wuas_skill", choices=list(ARM_PRESETS))
    p.add_argument("--skill-dim", type=int, default=64, help="d_s (low-rank bottleneck)")
    p.add_argument("--n-depths", type=int, default=8, help="number of injection depths")
    p.add_argument("--depths", default=None, help="explicit depth list, e.g. 7,15,23,31")
    p.add_argument("--activation", default=None)
    p.add_argument("--share", default=None, choices=["shared", "per_depth"])
    p.add_argument("--depth-rank", type=int, default=0, help="r_d: per-depth low-rank residual")
    p.add_argument("--use-gain", action="store_true")
    p.add_argument("--gate-bias-init", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--no-rms", action="store_true")
    p.add_argument("--no-gate", action="store_true")
    # ---- position/phase conditioning ----
    p.add_argument("--n-phases", type=int, default=1,
                   help="number of position phases; 1 = global operator. For P>=2, "
                        "phase0=prompt and phase1..P-1 are split by --phase-edges")
    p.add_argument("--phase-edges", default=None,
                   help="generation-phase boundaries (0-based decode steps), "
                        "comma-separated; count must be n_phases-2")
    p.add_argument("--phase-mode", default="gate", choices=["bias", "gate", "gate_gain"],
                   help="form of phase increments: gating bias only / + gate weight / + log gain")

    # ---- optimization ----
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gate-lr", type=float, default=None)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-grad-steps", type=int, default=None, help="smoke run: cap optimizer steps")

    # ---- data ----
    p.add_argument("--max-train-items", type=int, default=None)
    p.add_argument("--max-prompt-tokens", type=int, default=3072)
    p.add_argument("--max-target-tokens", type=int, default=32)
    p.add_argument("--target-template", default="<answer>{gold}</answer>")
    # Rejection-sampling BC (RFT): targets are the model's own sampled completions that
    # scored correct, preserving CoT (gold-only targets hurt CSQA/OBQA in practice).
    p.add_argument("--rft-file", default=None,
                   help="JSONL produced by rft_sample.py ({prompt, completion}); "
                        "when given, --target-template is ignored")
    p.add_argument("--max-rft-tokens", type=int, default=1024,
                   help="token cap for RFT targets (must fit the full CoT)")

    # ---- run ----
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=0, help="if >0, quick HF eval every N steps")
    p.add_argument("--eval-limit", type=int, default=50)
    p.add_argument("--eval-max-new", type=int, default=64)
    p.add_argument("--limit-eval", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p


# --------------------------------------------------------------------------- #
def resolve_cfg(args, hidden_size: int, n_layers: int) -> SkillOperatorConfig:
    preset = dict(ARM_PRESETS[args.arm])
    if args.activation is not None:
        preset["activation"] = args.activation
    if args.share is not None:
        preset["share"] = args.share
    if args.no_rms:
        preset["use_rms"] = False
    if args.no_gate:
        preset["use_gate"] = False
    depths = ([int(x) for x in args.depths.split(",") if x.strip()]
              if args.depths else default_depths(n_layers, args.n_depths))
    edges = ([int(x) for x in args.phase_edges.split(",") if x.strip()]
             if args.phase_edges else [])
    return SkillOperatorConfig(
        hidden_size=hidden_size,
        depths=tuple(depths),
        skill_dim=args.skill_dim,
        activation=preset["activation"],
        use_rms=preset["use_rms"],
        use_gate=preset["use_gate"],
        gate_bias_init=args.gate_bias_init,
        share=preset["share"],
        depth_rank=args.depth_rank,
        use_gain=args.use_gain,
        alpha=args.alpha,
        n_phases=args.n_phases,
        phase_edges=tuple(edges),
        phase_mode=args.phase_mode,
    )


def encode_example(tokenizer, prompt: str, target: str, max_prompt_tokens: int,
                   max_target_tokens: int, device):
    """Concatenate (prompt + target + eos); return ids / labels / prompt length.

    ``prompt_len`` feeds the phase context (generation phase = target segment).
    """
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tokenizer(target, add_special_tokens=False)["input_ids"][:max_target_tokens]
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id
    if len(p_ids) > max_prompt_tokens:      # left-truncate context, keep question and template tail
        p_ids = p_ids[-max_prompt_tokens:]
    ids = p_ids + t_ids + [eos]
    labels = [-100] * len(p_ids) + t_ids + [eos]
    return (torch.tensor([ids], device=device), torch.tensor([labels], device=device),
            len(p_ids))


@torch.no_grad()
def hf_eval(steerer, tokenizer, task, items, max_new, device, max_prompt_tokens: int,
            max_new_tokens: int) -> float:
    """Small HF greedy eval (quick in-training signal; official eval runs on vLLM)."""
    steerer.eval()
    hits = 0.0
    for item in items:
        prompt = task.build_prompt(tokenizer, item)
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > max_prompt_tokens:
            ids = ids[-max_prompt_tokens:]
        x = torch.tensor([ids], device=device)
        # Single-sequence greedy, no padding: generation phase starts at len(ids)
        steerer.set_phase_ctx(x.shape[1])
        out = steerer.generate(x, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][x.shape[1]:], skip_special_tokens=True)
        hits += task.metric(text, item)
    steerer.train()
    return hits / max(1, len(items))


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.max_train_items = args.max_train_items or 8
        args.max_grad_steps = args.max_grad_steps or 3
        args.eval_limit = args.eval_limit or 4
        args.epochs = 1.0

    torch.manual_seed(args.seed)
    device = "cuda"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    task = get_task(args.task, **(dict(split_dir=args.split_dir) if args.split_dir else {}))
    train_items = task.get_split("train")
    if args.max_train_items:
        train_items = train_items[: args.max_train_items]
    val_items = task.get_split("val")
    if args.limit_eval:
        val_items = val_items[: args.limit_eval]

    # ---- model ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype).to(device)
    model.eval()
    model.requires_grad_(False)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    text_cfg = model.config.get_text_config()
    hidden_size = int(text_cfg.hidden_size)
    n_layers = int(text_cfg.num_hidden_layers)

    cfg = resolve_cfg(args, hidden_size, n_layers)
    steerer = SkillSteerer(model, cfg)
    print(f"[wuas-skill] model={args.model_name} d_m={hidden_size} L={n_layers}")
    print(f"[wuas-skill] arm={args.arm} cfg={json.dumps(cfg.to_dict())}")
    print(f"[wuas-skill] depths={list(cfg.depths)} "
          f"trainable={steerer.num_trainable():,} ({steerer.num_trainable()/1e6:.4f}M)")

    # Parameter-count reference: per-layer WUAS at the same rank vs. the baseline
    # method's config (print only, not trained)
    wuas_all = 2 * args.skill_dim * hidden_size * n_layers
    baseline = 2 * 256 * hidden_size + 2 * 256 * 32 + hidden_size + 4
    print(f"[wuas-skill] reference param counts: WUAS rank{args.skill_dim}x{n_layers} "
          f"layers={wuas_all:,} | baseline d_s256/r32/D4={baseline:,}")

    # ---- optimizer ----
    groups = steerer.operator.param_groups(args.lr, args.gate_lr)
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    # ---- data encoding ----
    if args.rft_file:
        # RFT targets: prompt + model-sampled completions that scored correct (with CoT)
        pairs = []
        with open(args.rft_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    pairs.append((r["prompt"], r["completion"]))
        if not pairs:
            raise SystemExit(f"--rft-file {args.rft_file} has no usable samples")
        print(f"[wuas-skill] RFT targets: {len(pairs)} (max_rft_tokens={args.max_rft_tokens})")
        encoded = [encode_example(tokenizer, p, t, args.max_prompt_tokens,
                                  args.max_rft_tokens, device) for p, t in pairs]
    else:
        encoded = []
        for item in train_items:
            prompt = task.build_prompt(tokenizer, item)
            target = args.target_template.format(gold=gold_target(args.task, item))
            encoded.append(encode_example(tokenizer, prompt, target, args.max_prompt_tokens,
                                          args.max_target_tokens, device))

    steps_per_epoch = math.ceil(len(encoded) / args.bs / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_grad_steps:
        total_steps = min(total_steps, args.max_grad_steps)
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * args.warmup_ratio), max(1, total_steps))

    print(f"[wuas-skill] train_items={len(encoded)} epochs={args.epochs} "
          f"bs={args.bs}×accum{args.grad_accum} total_steps={total_steps}")

    # ---- training loop ----
    rng = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    step = 0
    micro = 0
    run_loss, run_n = 0.0, 0
    metrics_path = out / "metrics.jsonl"
    history = []
    done = False
    steerer.train()
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        order = torch.randperm(len(encoded), generator=rng).tolist()
        for i in range(0, len(order), args.bs):
            if done:
                break
            batch = [encoded[j] for j in order[i: i + args.bs]]
            # right padding
            S = max(b[0].shape[1] for b in batch)
            ids = torch.full((len(batch), S), tokenizer.pad_token_id, dtype=torch.long,
                             device=device)
            lab = torch.full((len(batch), S), -100, dtype=torch.long, device=device)
            att = torch.zeros((len(batch), S), dtype=torch.long, device=device)
            for bi, (x, y, _plen) in enumerate(batch):
                n = x.shape[1]
                ids[bi, :n], lab[bi, :n], att[bi, :n] = x[0], y[0], 1
            # Each row's generation phase starts at its own prompt length (right padding)
            steerer.set_phase_ctx(torch.tensor([b[2] for b in batch], device=device))

            outp = steerer(input_ids=ids, attention_mask=att)
            logits = outp.logits[:, :-1, :]
            tgt = lab[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
                ignore_index=-100)
            (loss / args.grad_accum).backward()
            run_loss += loss.detach().float().item() * int((tgt != -100).sum())
            run_n += int((tgt != -100).sum())
            micro += 1

            if micro % args.grad_accum == 0:
                gmax = steerer.operator.clip_grads_(args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "epoch": epoch,
                       "loss": run_loss / max(1, run_n), "grad_max": gmax,
                       "lr": sched.get_last_lr()[0],
                       "wall_s": round(time.time() - t0, 1)}
                run_loss, run_n = 0.0, 0
                print(f"[step {step}/{total_steps}] " +
                      " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
                history.append(rec)
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if args.eval_every and step % args.eval_every == 0:
                    em = hf_eval(steerer, tokenizer, task, val_items[: args.eval_limit],
                                 args.eval_max_new, device, args.max_prompt_tokens,
                                 args.eval_max_new)
                    print(f"[eval] step={step} val_em={em:.4f} n={min(len(val_items), args.eval_limit)}",
                          flush=True)
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps({"step": step, "val_em": em}) + "\n")
                if step >= total_steps:
                    done = True
                    break

    # ---- save ----
    torch.save(steerer.operator.state_dict(), out / "operator.pt")
    report = save_bundle(
        steerer.operator, str(out / "bundle"), model_name=args.model_name,
        task=args.task,
        extra_meta={
            "arm": args.arm,
            "train": {"lr": args.lr, "epochs": args.epochs, "bs": args.bs,
                      "grad_accum": args.grad_accum, "n_train_items": len(encoded),
                      "steps": step, "seed": args.seed, "smoke": args.smoke},
            "params_reference": {"wuas_rank_d_all_layers": wuas_all,
                                 "baseline_ds256_r32_D4": baseline},
        })
    summary = {
        "model": args.model_name, "task": args.task, "arm": args.arm,
        "cfg": cfg.to_dict(), "params_trainable": steerer.num_trainable(),
        "n_phases": cfg.n_phases, "phase_edges": list(cfg.phase_edges),
        "phase_mode": cfg.phase_mode, "phase_stats": steerer.operator.phase_stats(),
        "steps": step, "train_seconds": round(time.time() - t0, 1),
        "loadback": report,
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
