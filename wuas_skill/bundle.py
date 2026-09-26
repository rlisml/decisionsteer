"""WUAS-Skill ↔ vLLM sidecar bundle: export / load / self-check.

Bundle layout (standard sidecar bundle layout):

``manifest.json``
    schema_version / kind=``wuas_skill_operator`` / structural hyperparameters
    (depths, skill_dim, use_silu, use_rms, use_gate, alpha, rms_eps) /
    ``params_trainable`` / provenance and sha256.
``adapter.safetensors``
    One set of **materialized** tensors per injection depth,
    ``layer{i}.{down,up,gate_w,gate_b,gain}``. The vLLM side implements a
    single expression; no shared-trunk matmuls in the engine.
``train_state.safetensors``
    Compact training parameters (A/B/wg/bl/gl plus optional Ua/Va/Ub/Vb), for
    resuming training and parameter auditing.

After export, two load-back self-checks run (either failing raises an assert):
  1. Tensor level: reloaded vs source tensors max|diff| == 0;
  2. Functional level: ``skill_delta(reloaded)`` vs ``SkillOperator.delta`` on
     fixed random inputs, max|diff| <= 1e-5 (fp32).

Multi-phase (Step 3)
--------------------
Phase conditioning is realized engine-side via **multiple steer vectors +
where-clause windows** (the engine's ``store.py`` caches payloads by path, so a
single directory can hold only one operator). Hence: ``bundle/`` root = phase 0
(prompt), ``bundle/phase{p}/`` = phase p (generation window p-1).
:func:`phase_specs` expands the manifest into ``[(source_dir, apply_clause), ...]``.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import torch

from .skill_operator import SkillOperator, skill_delta

__all__ = ["save_bundle", "load_bundle", "load_operator_from_bundle", "sha256_file",
           "phase_specs", "phase_layout"]

SCHEMA_VERSION = 2
KIND = "wuas_skill_operator"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_bundle(
    operator: SkillOperator,
    out_dir: str,
    *,
    model_name: str,
    task: str,
    extra_meta: dict | None = None,
    seed: int = 0,
    verify: bool = True,
) -> dict:
    """Export the operator to ``out_dir``; returns the load-back self-check report.

    ``verify=False`` skips the load-back checks (GRPO exports a bundle to the
    vLLM engine every step; per-step self-check overhead is not worth it; final
    artifacts keep ``verify=True``).
    """
    from safetensors.torch import load_file, save_file

    os.makedirs(out_dir, exist_ok=True)
    cfg = operator.cfg
    tensors = operator.materialize()

    st_path = os.path.join(out_dir, "adapter.safetensors")
    save_file({k: v.contiguous() for k, v in tensors.items()}, st_path)

    # ---- Multi-phase: phase 0 at the root, phase p>=1 under phase{p}/ ----
    # The engine store caches payloads by path; one directory cannot hold two
    # tensor sets, hence separate directories.
    phase_paths = {0: out_dir}
    for p in range(1, operator.n_phases):
        d = os.path.join(out_dir, f"phase{p}")
        os.makedirs(d, exist_ok=True)
        save_file({k: v.contiguous() for k, v in operator.materialize(p).items()},
                  os.path.join(d, "adapter.safetensors"))
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump({"schema_version": SCHEMA_VERSION, "kind": KIND,
                       "depths": list(cfg.depths), "n_depths": len(cfg.depths),
                       "hidden_size": cfg.hidden_size, "skill_dim": cfg.skill_dim,
                       "use_silu": (cfg.activation or "linear").lower() == "silu",
                       "use_rms": bool(cfg.use_rms), "use_gate": bool(cfg.use_gate),
                       "alpha": float(cfg.alpha), "rms_eps": float(cfg.rms_eps),
                       "phase": p, "parent": os.path.basename(os.path.abspath(out_dir))},
                      f, indent=2)
        phase_paths[p] = d

    train_state = {k: v.detach().float().cpu().contiguous()
                   for k, v in operator.state_dict().items()}
    ts_path = os.path.join(out_dir, "train_state.safetensors")
    save_file(train_state, ts_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        # ---- Structural hyperparameters (the vLLM plugin rebuilds the operator from these) ----
        "depths": list(cfg.depths),
        "n_depths": len(cfg.depths),
        "hidden_size": cfg.hidden_size,
        "skill_dim": cfg.skill_dim,
        "use_silu": (cfg.activation or "linear").lower() == "silu",
        "activation": cfg.activation,
        "use_rms": bool(cfg.use_rms),
        "use_gate": bool(cfg.use_gate),
        "alpha": float(cfg.alpha),
        "rms_eps": float(cfg.rms_eps),
        "share": cfg.share,
        "depth_rank": int(cfg.depth_rank),
        "use_gain": bool(cfg.use_gain),
        # ---- Step 3: phase conditioning ----
        "n_phases": int(operator.n_phases),
        "phase_edges": list(cfg.phase_edges),
        "phase_mode": cfg.phase_mode,
        # ---- Auditing ----
        "params_trainable": int(operator.num_trainable()),
        "params_materialized": int(sum(t.numel() for t in tensors.values())),
        "params_total_all_phases": int(sum(
            sum(t.numel() for t in operator.materialize(p).values())
            for p in range(operator.n_phases))),
        "storage_dtype": "float32",
        "model_name": model_name,
        "task": task,
        "adapter_sha256": sha256_file(st_path),
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "converted_by": "wuas_skill/bundle.py",
    }
    if extra_meta:
        manifest.update(extra_meta)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    report = (_loadback_check(operator, st_path, seed, phase_paths) if verify
              else {"skipped": True, "tensor_check_pass": None,
                    "functional_check_pass": None})
    if verify:
        with open(os.path.join(out_dir, "loadback_check.json"), "w") as f:
            json.dump(report, f, indent=2)
        assert report["tensor_check_pass"], f"tensor load-back failed: {report}"
        assert report["functional_check_pass"], f"functional load-back failed: {report}"
    return report


def _loadback_check(operator: SkillOperator, st_path: str, seed: int,
                    phase_paths: dict) -> dict:
    from safetensors.torch import load_file

    # Random inputs must live on the operator's device (the operator trains on CUDA)
    dev = next(operator.parameters()).device
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(32, operator.d_m, generator=g, dtype=torch.float32).to(dev)
    cfg = operator.cfg
    use_silu = (cfg.activation or "linear").lower() == "silu"

    tensor_diff = 0.0
    func_diff = 0.0
    operator.eval()
    with torch.no_grad():
        for p, d in phase_paths.items():
            reloaded = load_file(os.path.join(d, "adapter.safetensors"))
            src = operator.materialize(p)
            for k, v in src.items():
                tensor_diff = max(tensor_diff, float((reloaded[k] - v).abs().max()))
            for li in range(operator.n_depths):
                ref = operator.delta(h, li, phase=torch.full((h.shape[0],), p,
                                                             dtype=torch.long, device=dev))
                payload = {
                    "down": reloaded[f"layer{li}.down"].to(dev),
                    "up": reloaded[f"layer{li}.up"].to(dev),
                    "gate_w": reloaded[f"layer{li}.gate_w"].to(dev),
                    "gate_b": float(reloaded[f"layer{li}.gate_b"]),
                    "gain": float(reloaded[f"layer{li}.gain"]),
                }
                ours = skill_delta(payload, h, use_silu=use_silu, use_rms=cfg.use_rms,
                                   use_gate=cfg.use_gate, alpha=cfg.alpha,
                                   rms_eps=cfg.rms_eps)
                func_diff = max(func_diff, float((ours - ref).abs().max()))
    return {
        "n_phases": int(operator.n_phases),
        "tensor_max_abs_diff": tensor_diff,
        "tensor_check_pass": tensor_diff == 0.0,
        "functional_max_abs_diff": func_diff,
        "functional_check_pass": func_diff <= 1e-5,
        "seed": seed,
    }


def phase_layout(n_phases: int, phase_edges) -> list:
    """Language-neutral description of phases → generation windows (shared by
    eval / rollout / docs).

    Returns ``[(phase, gen_window), ...]``; ``gen_window`` is a half-open
    interval ``(lo, hi)`` (``hi=None`` means unbounded); phase 0's
    ``gen_window`` is ``None`` (prompt phase).

    ``n_phases==1`` ⇒ ``[(0, None)]`` (global operator, no phase concept).
    """
    n = int(n_phases)
    if n <= 1:
        return [(0, None)]
    edges = [int(e) for e in (phase_edges or [])]
    out = [(0, None)]
    for p in range(1, n):
        lo = edges[p - 2] if p >= 2 else 0
        hi = edges[p - 1] if p - 1 < len(edges) else None
        out.append((p, (lo, hi)))
    return out


def phase_specs(manifest: dict, bundle_dir: str) -> list:
    """Expand a (multi-phase) bundle into ``[(source_dir, apply_dict), ...]``.

    Single-phase bundle ⇒ length 1 with ``apply = {"prompt": "all",
    "generation": "all"}`` (identical to the Step 1/2 evaluation). Multi-phase
    ⇒ one steer vector per phase, using the ``prompt`` / ``generation_window``
    selectors to dispatch tokens to their operator; the engine caches payloads
    by ``source`` path, so each phase must live in its **own directory**.
    """
    n = int(manifest.get("n_phases", 1) or 1)
    if n <= 1:
        return [(bundle_dir, {"prompt": "all", "generation": "all"})]
    specs = []
    for p, gw in phase_layout(n, manifest.get("phase_edges", [])):
        src = bundle_dir if p == 0 else os.path.join(bundle_dir, f"phase{p}")
        if not os.path.isdir(src):
            raise FileNotFoundError(f"multi-phase bundle missing phase{p} directory: {src}")
        apply = {"prompt": "all"} if p == 0 else {"generation_window": list(gw)}
        specs.append((src, apply))
    return specs


def load_bundle(path: str) -> tuple[dict, dict]:
    """Return ``(manifest, tensors)``."""
    from safetensors.torch import load_file

    with open(os.path.join(path, "manifest.json")) as f:
        manifest = json.load(f)
    if manifest.get("kind") != KIND:
        raise ValueError(f"{path}: expected kind={KIND!r}, got {manifest.get('kind')!r}")
    tensors = load_file(os.path.join(path, "adapter.safetensors"))
    return manifest, tensors


def load_operator_from_bundle(path: str):
    """Rebuild a :class:`SkillOperator` from the bundle's
    ``train_state.safetensors`` (CPU).

    Used for HF-side reproduction (training resume / HF↔vLLM consistency
    diagnostics). For multi-phase bundles, ``n_phases`` / ``phase_edges`` /
    ``phase_mode`` are restored from the manifest.
    """
    from safetensors.torch import load_file

    from .skill_operator import SkillOperator, SkillOperatorConfig

    with open(os.path.join(path, "manifest.json")) as f:
        m = json.load(f)
    cfg = SkillOperatorConfig(
        hidden_size=int(m["hidden_size"]), depths=tuple(m["depths"]),
        skill_dim=int(m["skill_dim"]), activation=m.get("activation", "silu"),
        use_rms=bool(m.get("use_rms", True)), use_gate=bool(m.get("use_gate", True)),
        share=m.get("share", "shared"), depth_rank=int(m.get("depth_rank", 0)),
        use_gain=bool(m.get("use_gain", False)), alpha=float(m.get("alpha", 1.0)),
        rms_eps=float(m.get("rms_eps", 1e-6)),
        n_phases=int(m.get("n_phases", 1) or 1),
        phase_edges=tuple(int(e) for e in (m.get("phase_edges") or [])),
        phase_mode=m.get("phase_mode", "gate"))
    op = SkillOperator(cfg)
    op.load_state_dict(load_file(os.path.join(path, "train_state.safetensors")),
                       strict=True)
    return op
