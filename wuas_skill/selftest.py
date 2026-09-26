"""GPU-free self-test: core operator invariants plus bundle round-trip checks.

Run:
    python -m wuas_skill.selftest

Pure CPU; no GPU, model, or data required (only torch and the local
``bundle`` / ``skill_operator`` modules).

Checks
------
1. **Identity start**: ``up`` is zero-initialized, so ``delta == 0`` for any
   input (the V=0 anchor point).
2. **Math equivalence**: the functional implementation
   :func:`operator.skill_delta` (used by the vLLM plugin) matches
   :class:`SkillOperator.delta` on random inputs (fp32 tolerance 1e-5).
3. **Parameter-count formula**: matches the analytic form
   ``2*d_s*d_m + [d_m + D]`` (with gating); ``D * 2*d_s*d_m`` for
   ``share="per_depth"``.
4. **Bundle round-trip**: save -> load yields bit-identical tensors and
   identical function outputs.
5. **Per-depth low-rank residual**: with ``depth_rank > 0`` the initial
   residual is zero (``Va/Vb`` zero-initialized), preserving identity start.
6. **Phase index/window layout**: prompt/generation-window split agrees with
   ``phase_layout`` (non-overlapping, covers ``[0, inf)``).
7. **Multi-phase identity start**: a phase-conditioned operator still starts
   exactly at identity for every phase.
8. **Phase payload equivalence**: per-phase materialized payloads match the
   module implementation, and phase 0 matches the global path.
9. **Multi-phase bundle round-trip**: phase directories complete, load-back
   self-check passes, and ``phase_specs`` layout is correct.
10. **Phase-config validation**: invalid phase configurations are rejected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .bundle import load_bundle, phase_layout, phase_specs, save_bundle
from .skill_operator import (SkillOperator, SkillOperatorConfig, phase_index_of,
                             skill_delta)

TOL = 1e-5


def _rand(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def check_identity_start() -> None:
    cfg = SkillOperatorConfig(hidden_size=64, depths=(0, 1, 2), skill_dim=8,
                              use_rms=True, use_gate=True)
    op = SkillOperator(cfg)
    h = _rand(4, 64, seed=1)
    for li in range(3):
        d = op.delta(h, li)
        assert torch.all(d == 0), f"depth {li}: delta must be 0 after zero-init, got max={d.abs().max()}"
    print("  [ok] identity start: up=0 => delta == 0")


def check_math_equivalence() -> None:
    cfg = SkillOperatorConfig(hidden_size=64, depths=(0, 1, 2), skill_dim=8,
                              activation="silu", use_rms=True, use_gate=True,
                              use_gain=True, alpha=1.5)
    op = SkillOperator(cfg)
    with torch.no_grad():                       # break zero-init for non-zero delta
        op.B.normal_(std=0.1)
        op.wg.normal_(std=0.1)
        op.gl.normal_(std=0.2)
    mats = op.materialize()
    h = _rand(6, 64, seed=2)
    worst = 0.0
    for li in range(3):
        with torch.no_grad():
            ref = op.delta(h, li)
        p = {k: mats[f"layer{li}.{k}"] for k in ("down", "up", "gate_w")}
        p["gate_b"] = float(mats[f"layer{li}.gate_b"])
        p["gain"] = float(mats[f"layer{li}.gain"])
        ours = skill_delta(p, h, use_silu=True, use_rms=True, use_gate=True, alpha=1.5)
        worst = max(worst, float((ours - ref).abs().max()))
    assert worst <= TOL, f"functional and module implementations disagree, max|diff|={worst}"
    print(f"  [ok] math equivalence: SkillOperator.delta vs skill_delta, max|diff|={worst:.2e}")


def check_param_count() -> None:
    d_m, d_s, D = 128, 16, 4
    shared = SkillOperator(SkillOperatorConfig(hidden_size=d_m, depths=tuple(range(D)),
                                               skill_dim=d_s))
    want = 2 * d_s * d_m + d_m + D                 # A, B + gate w, b
    assert shared.num_trainable() == want, (shared.num_trainable(), want)

    per = SkillOperator(SkillOperatorConfig(hidden_size=d_m, depths=tuple(range(D)),
                                            skill_dim=d_s, share="per_depth"))
    want_per = D * 2 * d_s * d_m + d_m + D
    assert per.num_trainable() == want_per, (per.num_trainable(), want_per)

    nogate = SkillOperator(SkillOperatorConfig(hidden_size=d_m, depths=tuple(range(D)),
                                               skill_dim=d_s, use_gate=False))
    assert nogate.num_trainable() == 2 * d_s * d_m
    print(f"  [ok] parameter count: shared={shared.num_trainable():,} "
          f"per_depth={per.num_trainable():,} no_gate={nogate.num_trainable():,}")


def check_bundle_roundtrip() -> None:
    cfg = SkillOperatorConfig(hidden_size=64, depths=(0, 3), skill_dim=8, use_gain=True)
    op = SkillOperator(cfg)
    with torch.no_grad():
        op.B.normal_(std=0.1)
        op.wg.normal_(std=0.05)
    with tempfile.TemporaryDirectory() as td:
        rep = save_bundle(op, str(Path(td) / "bundle"), model_name="dummy", task="dummy")
        assert rep["tensor_check_pass"] and rep["functional_check_pass"], rep
        manifest, tensors = load_bundle(str(Path(td) / "bundle"))
        assert manifest["depths"] == [0, 3] and manifest["skill_dim"] == 8
        assert manifest["params_trainable"] == op.num_trainable()
    print(f"  [ok] bundle round-trip: tensor_diff={rep['tensor_max_abs_diff']} "
          f"func_diff={rep['functional_max_abs_diff']:.2e}")


def check_depth_rank_identity() -> None:
    cfg = SkillOperatorConfig(hidden_size=64, depths=(0, 1), skill_dim=8, depth_rank=4)
    op = SkillOperator(cfg)
    h = _rand(4, 64, seed=3)
    for li in range(2):
        assert torch.all(op.delta(h, li) == 0), "depth_rank residual must be zero at init"
    print("  [ok] per-depth low-rank residual: Va/Vb zero-init => identity start preserved")


def check_phase_index_of() -> None:
    """Phase-index semantics (prompt/generation_window for the vLLM where-clause)."""
    # P=3, edges=(2,): prompt | gen<2 | gen>=2
    ph = phase_index_of(6, 2, (2,))
    assert torch.equal(ph, torch.tensor([[0, 0, 1, 1, 2, 2]])), ph
    # per-row gen_start: (B, T)
    ph = phase_index_of(5, torch.tensor([1, 2]), (2,))
    assert torch.equal(ph, torch.tensor([[0, 1, 1, 2, 2], [0, 0, 1, 1, 2]])), ph
    # P=2, edges=(): prompt | all of generation
    ph = phase_index_of(4, 2, ())
    assert torch.equal(ph, torch.tensor([[0, 0, 1, 1]])), ph
    # window partition agrees with phase_layout (no overlap, covers [0, inf))
    for n, edges in ((2, ()), (3, (128,)), (4, (64, 256)), (5, (16, 64, 256))):
        lay = phase_layout(n, edges)
        assert [p for p, _ in lay] == list(range(n)), lay
        assert lay[0][1] is None and lay[-1][1][1] is None, lay
        assert [w[0] for _, w in lay[1:]] == [0] + [e for e in edges], lay
    print("  [ok] phase index: prompt/generation windows agree with phase_layout")


def check_phase_identity() -> None:
    """A multi-phase operator must still start at exact identity for every phase."""
    cfg = SkillOperatorConfig(hidden_size=64, depths=(0, 2, 3), skill_dim=8,
                              n_phases=4, phase_edges=(16, 64), use_gain=True,
                              phase_mode="gate_gain")
    op = SkillOperator(cfg)
    h = _rand(10, 64, seed=7)
    ph = phase_index_of(10, 4, cfg.phase_edges)
    for li in range(3):
        d = op.delta(h, li, phase=ph)
        assert torch.all(d == 0), f"delta must be 0 under phase conditioning, depth {li} max={d.abs().max()}"
    # Parameter count: trunk 2*d_s*d_m + global gate (d_m + D) + global gain D
    #                + (P-1)*d_m (per-phase gate weights) + D*(P-1) (depth x phase biases) + D*(P-1) (gains)
    want = 2 * 8 * 64 + 64 + 3 + 3 + (4 - 1) * 64 + 3 * (4 - 1) * 2
    assert op.num_trainable() == want, (op.num_trainable(), want)
    print(f"  [ok] phase-conditioned identity start + parameter count {op.num_trainable():,}")


def check_phase_math_equivalence() -> None:
    """Per-phase module output must match expanded payload + functional form bit-for-bit."""
    cfg = SkillOperatorConfig(hidden_size=64, depths=(0, 1), skill_dim=8,
                              activation="silu", use_rms=True, use_gate=True,
                              use_gain=True, alpha=1.2, n_phases=4,
                              phase_edges=(16, 64), phase_mode="gate_gain")
    op = SkillOperator(cfg)
    with torch.no_grad():
        op.B.normal_(std=0.1)
        op.wg.normal_(std=0.1)
        op.bl.normal_(std=0.1)
        op.gl.normal_(std=0.2)
        op.phase_wg.normal_(std=0.1)
        op.phase_bl.normal_(std=0.1)
        op.phase_gl.normal_(std=0.1)
    h = _rand(9, 64, seed=11)
    worst = 0.0
    for p in range(cfg.n_phases):
        mat = op.materialize(p)
        ph = torch.full((h.shape[0],), p, dtype=torch.long)
        for li in range(2):
            with torch.no_grad():
                ref = op.delta(h, li, phase=ph)
            payload = {
                "down": mat[f"layer{li}.down"], "up": mat[f"layer{li}.up"],
                "gate_w": mat[f"layer{li}.gate_w"],
                "gate_b": float(mat[f"layer{li}.gate_b"]),
                "gain": float(mat[f"layer{li}.gain"]),
            }
            ours = skill_delta(payload, h, use_silu=True, use_rms=True, use_gate=True,
                               alpha=1.2)
            worst = max(worst, float((ours - ref).abs().max()))
    assert worst <= TOL, f"phase payload disagrees with module implementation, max|diff|={worst}"
    # Phase 0 must match the phase-free (global) path numerically (up to float summation order)
    for li in range(2):
        with torch.no_grad():
            a = op.delta(h, li, phase=torch.zeros(h.shape[0], dtype=torch.long))
            b = op.delta(h, li)
        assert float((a - b).abs().max()) <= TOL, "phase0 must match the global path"
    print(f"  [ok] phase payload math equivalence: max|diff|={worst:.2e}; phase0 == global path")


def check_phase_bundle_roundtrip() -> None:
    """Multi-phase bundle: complete phase dirs + load-back self-check + phase_specs layout."""
    cfg = SkillOperatorConfig(hidden_size=64, depths=(0, 3), skill_dim=8,
                              n_phases=3, phase_edges=(32,), use_gain=True,
                              phase_mode="gate_gain")
    op = SkillOperator(cfg)
    with torch.no_grad():
        op.B.normal_(std=0.1)
        op.phase_wg.normal_(std=0.1)
    with tempfile.TemporaryDirectory() as td:
        root = str(Path(td) / "bundle")
        rep = save_bundle(op, root, model_name="dummy", task="dummy")
        assert rep["tensor_check_pass"] and rep["functional_check_pass"], rep
        assert rep["n_phases"] == 3
        assert (Path(root) / "phase1" / "adapter.safetensors").exists()
        assert (Path(root) / "phase2" / "manifest.json").exists()
        manifest, _ = load_bundle(root)
        assert manifest["n_phases"] == 3 and manifest["phase_edges"] == [32]
        specs = phase_specs(manifest, root)
        assert [s[1] for s in specs] == [{"prompt": "all"},
                                        {"generation_window": [0, 32]},
                                        {"generation_window": [32, None]}], specs
        # Single-phase bundle specs must match the non-phase layout exactly
        op1 = SkillOperator(SkillOperatorConfig(hidden_size=64, depths=(0,), skill_dim=8))
        r1 = str(Path(td) / "b1")
        save_bundle(op1, r1, model_name="dummy", task="dummy")
        m1, _ = load_bundle(r1)
        assert phase_specs(m1, r1) == [(r1, {"prompt": "all", "generation": "all"})]
    print(f"  [ok] multi-phase bundle round-trip: tensor_diff={rep['tensor_max_abs_diff']} "
          f"func_diff={rep['functional_max_abs_diff']:.2e}; phase_specs layout correct")


def check_phase_config_validation() -> None:
    """Invalid phase configurations must be rejected (out-of-range / non-increasing / wrong count / missing use_gain)."""
    bad = [
        dict(n_phases=3, phase_edges=()),                      # wrong count
        dict(n_phases=1, phase_edges=(8,)),                    # edges must be empty
        dict(n_phases=4, phase_edges=(64, 16)),                # not increasing
        dict(n_phases=3, phase_edges=(0,)),                    # not positive
        dict(n_phases=3, phase_edges=(8,), phase_mode="gate_gain", use_gain=False),
    ]
    for kw in bad:
        try:
            SkillOperatorConfig(hidden_size=64, depths=(0,), skill_dim=8, **kw).check_phases()
        except ValueError:
            continue
        raise AssertionError(f"invalid phase config not rejected: {kw}")
    print(f"  [ok] phase config validation: all {len(bad)} invalid configs rejected")


def main() -> int:
    print("[wuas-skill selftest]")
    check_identity_start()
    check_math_equivalence()
    check_param_count()
    check_bundle_roundtrip()
    check_depth_rank_identity()
    check_phase_index_of()
    check_phase_identity()
    check_phase_math_equivalence()
    check_phase_bundle_roundtrip()
    check_phase_config_validation()
    print("[wuas-skill selftest] ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
