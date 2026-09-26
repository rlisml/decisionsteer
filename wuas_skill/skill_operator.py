"""WUAS-Skill operator — a gated low-rank residual operator for post-block steering.

Design
------
* **RMS-normalized input** ``rms = x * rsqrt(mean(x^2)+eps)`` — scale-invariant,
  decoupling the operator from activation magnitude.
* **Input-dependent sigmoid gate** ``g = sigmoid(rms @ w + b_l)``.
* **Depth-shared trunk** — A/B are shared across injection depths (the main
  parameter saving).
* **WUAS form** ``Δh = α · W_up(φ(W_down · rms(h))) · gate`` — still a "block
  output + low-rank delta" residual-stream intervention, so it plugs directly
  into the vLLM engine for online steering without touching backbone weights.
* **Zero init** ``up = 0`` ⇒ θ⁰ is an **exact identity** (equivalent to the
  baseline's V=0 identity anchor); the base model's zero-shot scores are
  reproduced through the identity gate.

Per-depth formula (block output h)::

    x   = RMS(h)                  (if use_rms)
    z   = phi(x @ A_l^T)          A_l: (d_s, d_m)
    u   = z @ B_l^T               B_l: (d_m, d_s)
    g   = sigmoid(x @ w + b_l)    (if use_gate)
    h'  = h + alpha * gamma_l * g ⊙ u
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SkillOperatorConfig", "SkillOperator", "skill_delta",
           "phase_index_of", "phase_index_from_positions", "PHASE_MODES"]


@dataclass
class SkillOperatorConfig:
    """Structural hyperparameters (written to the bundle manifest; the vLLM side rebuilds from it)."""

    hidden_size: int = 2560          # d_m
    depths: tuple = ()               # decoder layer indices to inject at
    skill_dim: int = 64              # d_s: low-rank bottleneck (~ the WUAS rank)
    activation: str = "silu"         # silu | gelu | relu | tanh | linear (identity)
    use_rms: bool = True             # input RMS normalization
    use_gate: bool = True            # input-dependent gate
    gate_bias_init: float = 0.0      # b_l init (0 → sigmoid=0.5; -2 → 0.12)
    share: str = "shared"            # shared (across depths) | per_depth (independent per depth)
    depth_rank: int = 0              # r_d>0: per-depth low-rank residual A_l = A + Ua_l @ Va_l^T
    use_gain: bool = False           # per-depth learnable scalar gain gamma_l (init 1)
    alpha: float = 1.0               # global scale (the WUAS alpha)
    rms_eps: float = 1e-6
    # ---- Step 3: position/phase conditioning ----
    # ``n_phases``: number of phases token positions are split into. 1 = no
    # conditioning (the global operator of Step 1/2).
    # For P>=2, phase 0 = prompt (t < gen_start), phases 1..P-1 = generation
    # windows with boundaries from ``phase_edges`` (0-based decode steps,
    # strictly increasing positive integers), requiring
    # ``len(phase_edges) == n_phases - 2``:
    #   P=2, edges=()        ⇒ prompt | all gen
    #   P=3, edges=(128,)    ⇒ prompt | gen<128 | gen>=128
    #   P=4, edges=(64,256)  ⇒ prompt | gen<64 | 64<=gen<256 | gen>=256
    n_phases: int = 1
    phase_edges: tuple = ()
    # Form of the **incremental** phase deltas (zero-init ⇒ every phase starts
    # ≡ the global operator ⇒ identity at init unchanged):
    #   bias      — per-(depth,phase) gate-bias delta only
    #   gate      — per-phase gate-weight delta + per-(depth,phase) bias delta (default)
    #   gate_gain — plus per-(depth,phase) log-gain delta (requires use_gain=True)
    phase_mode: str = "gate"

    def check_phases(self) -> None:
        """Validate the phase configuration (call before building the operator
        or exporting a bundle)."""
        if self.n_phases < 1:
            raise ValueError(f"n_phases must be >= 1, got {self.n_phases}")
        edges = tuple(int(e) for e in self.phase_edges)
        if self.n_phases == 1:
            if edges:
                raise ValueError("phase_edges must be empty when n_phases=1")
        else:
            if len(edges) != self.n_phases - 2:
                raise ValueError(
                    f"expected n_phases-2={self.n_phases - 2} phase_edges, "
                    f"got {len(edges)} ({edges})")
            if list(edges) != sorted(edges) or any(e <= 0 for e in edges):
                raise ValueError(f"phase_edges must be strictly increasing positive integers, got {edges}")
        if self.phase_mode not in PHASE_MODES:
            raise ValueError(f"phase_mode must be one of {PHASE_MODES}, got {self.phase_mode!r}")
        if self.n_phases > 1 and self.phase_mode == "gate_gain" and not self.use_gain:
            raise ValueError('phase_mode="gate_gain" requires use_gain=True')

    def to_dict(self) -> dict:
        d = asdict(self)
        d["depths"] = list(self.depths)
        d["phase_edges"] = list(self.phase_edges)
        return d


PHASE_MODES = ("bias", "gate", "gate_gain")


def phase_index_from_positions(pos: torch.Tensor, gen_start, edges) -> torch.Tensor:
    """Phase index from **absolute positions**, shape ``(B, T)`` (long).

    ``pos``: position tensor of shape ``(T,)`` or ``(B,T)`` (with HF incremental
    decoding, ``hidden_states`` only covers the current chunk, so
    ``position_ids`` is required to obtain absolute positions).
    ``gen_start``: int or ``(B,)``, the per-row absolute start of the generation
    phase (= prompt length; aligned with the vLLM ``generation_window`` 0-based
    decode step).
    ``p < gen_start`` → 0; ``s = p - gen_start`` → ``1 + |{e in edges : s >= e}|``.
    """
    p = pos if pos.ndim == 2 else pos.reshape(1, -1)
    if isinstance(gen_start, torch.Tensor):
        gs = gen_start.to(device=p.device, dtype=torch.long).reshape(-1, 1)
    else:
        gs = torch.tensor(int(gen_start), device=p.device, dtype=torch.long).reshape(1, 1)
    s = p.to(torch.long) - gs
    ph = (s >= 0).long()                            # s>=0 ⇒ at least phase 1 (generation)
    for e in edges:
        ph = ph + (s >= int(e)).long()
    return ph


def phase_index_of(seq_len: int, gen_start, edges, device=None) -> torch.Tensor:
    """Equivalent to ``phase_index_from_positions(arange(seq_len), gen_start, edges)``."""
    return phase_index_from_positions(torch.arange(seq_len, device=device),
                                      gen_start, edges)


def _make_activation(name: str):
    """Map an activation name to a module; ``linear``/``none`` → identity
    (WUAS ``--linear``)."""
    key = (name or "linear").lower()
    if key in ("linear", "none", "identity", ""):
        return None
    if key == "silu":
        return nn.SiLU()
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation {name!r}")


class SkillOperator(nn.Module):
    """Low-rank skill operator: depth-shared trunk + per-depth gate/gain.

    Injected at the output (residual stream) of decoder block ``depths[li]``:

    .. math::
        x   &= \\mathrm{RMS}(h)                       \\quad (\\text{if use\\_rms})
        z   &= \\phi(x A_l^\\top)                     \\quad A_l \\in \\mathbb{R}^{d_s \\times d_m}
        u   &= z B_l^\\top                            \\quad B_l \\in \\mathbb{R}^{d_m \\times d_s}
        g   &= \\sigma(x w + b_l)                     \\quad (\\text{if use\\_gate})
        h'  &= h + \\alpha \\gamma_l \\, g \\odot u
    """

    def __init__(self, cfg: SkillOperatorConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert cfg.skill_dim > 0, "skill_dim (d_s) must be positive"
        assert cfg.share in ("shared", "per_depth")
        assert not (cfg.share == "per_depth" and cfg.depth_rank), (
            "share=per_depth and depth_rank are mutually exclusive "
            "(both mean one matrix set per depth)"
        )
        self.cfg = cfg
        d_m, d_s, D = cfg.hidden_size, cfg.skill_dim, len(cfg.depths)
        self.d_m, self.d_s, self.n_depths = d_m, d_s, D
        self.act = _make_activation(cfg.activation)

        # ---- Trunk: down projection A (d_s, d_m), up projection B (d_m, d_s) ----
        # share=shared    → 2-D, shared by all injection depths (main parameter saving)
        # share=per_depth → 3-D, independent per depth (ablation)
        a_shape = (D, d_s, d_m) if cfg.share == "per_depth" else (d_s, d_m)
        b_shape = (D, d_m, d_s) if cfg.share == "per_depth" else (d_m, d_s)
        g = torch.Generator().manual_seed(0)
        self.A = nn.Parameter(torch.randn(*a_shape, generator=g) / math.sqrt(d_m))
        # WUAS convention: zero-init up ⇒ exact identity mapping at the start of training
        self.B = nn.Parameter(torch.zeros(*b_shape))

        # ---- Optional per-depth low-rank residual: A_l = A + Ua_l @ Va_l^T ----
        self.Ua = self.Va = self.Ub = self.Vb = None
        if cfg.depth_rank:
            r_d = cfg.depth_rank
            mk = lambda *s: nn.Parameter(torch.randn(*s, generator=g) / math.sqrt(s[-2]))
            zk = lambda *s: nn.Parameter(torch.zeros(*s))
            self.Ua, self.Vb = mk(D, d_s, r_d), zk(D, d_s, r_d)
            self.Va, self.Ub = zk(D, d_m, r_d), zk(D, d_m, r_d)

        # ---- Gate: shared weight w (d_m,), per-depth bias b_l ----
        # When gating is off these tensors stay out of the forward pass and are
        # non-trainable (keeps parameter counts honest)
        self.wg = nn.Parameter(torch.zeros(d_m), requires_grad=cfg.use_gate)
        self.bl = nn.Parameter(torch.full((D,), float(cfg.gate_bias_init)),
                               requires_grad=cfg.use_gate)

        # ---- Per-depth scalar gain gamma_l = exp(gl_l), init 1 ----
        self.gl = nn.Parameter(torch.zeros(D)) if cfg.use_gain else None

        # ---- Phase deltas (Step 3): zero-init ⇒ every phase starts equal to
        # the global operator ----
        cfg.check_phases()
        self.n_phases = int(cfg.n_phases)
        P1 = self.n_phases - 1                       # number of delta columns (phase >= 1)
        mkp = nn.Parameter
        self.phase_wg = (mkp(torch.zeros(P1, d_m), requires_grad=cfg.use_gate)
                         if P1 and cfg.phase_mode in ("gate", "gate_gain") else None)
        self.phase_bl = (mkp(torch.zeros(D, P1), requires_grad=cfg.use_gate)
                         if P1 and cfg.use_gate else None)
        self.phase_gl = (mkp(torch.zeros(D, P1), requires_grad=cfg.use_gain)
                         if P1 and cfg.use_gain and cfg.phase_mode == "gate_gain"
                         else None)

    # ------------------------------------------------------------------ #
    # effective matrices
    # ------------------------------------------------------------------ #
    def A_eff(self, li: int) -> torch.Tensor:
        """Down projection (d_s, d_m) at the li-th injection depth."""
        A = self.A[li] if self.cfg.share == "per_depth" else self.A
        if self.Ua is not None:
            A = A + self.Ua[li] @ self.Va[li].T
        return A

    def B_eff(self, li: int) -> torch.Tensor:
        """Up projection (d_m, d_s) at the li-th injection depth."""
        B = self.B[li] if self.cfg.share == "per_depth" else self.B
        if self.Ub is not None:
            B = B + self.Ub[li] @ self.Vb[li].T
        return B

    # ------------------------------------------------------------------ #
    # phase tables (Step 3)
    # ------------------------------------------------------------------ #
    def phase_tables(self):
        """Expand into (wg_tab, bl_tab, gl_tab): ``(P,d_m)`` / ``(D,P)`` / ``(D,P)|None``.

        Zero-init deltas ⇒ ``phase_tables()[:, 0]`` matches the original global
        parameters exactly.
        """
        if self.n_phases == 1:
            return (self.wg[None], self.bl[:, None],
                    None if self.gl is None else self.gl[:, None])
        wg_tab = (torch.cat([self.wg[None], self.wg[None] + self.phase_wg], 0)
                  if self.phase_wg is not None else self.wg[None].expand(self.n_phases, -1))
        bl_tab = (torch.cat([self.bl[:, None], self.bl[:, None] + self.phase_bl], 1)
                  if self.phase_bl is not None else self.bl[:, None].expand(-1, self.n_phases))
        if self.gl is None:
            gl_tab = None
        else:
            gl_tab = (torch.cat([self.gl[:, None], self.gl[:, None] + self.phase_gl], 1)
                      if self.phase_gl is not None
                      else self.gl[:, None].expand(-1, self.n_phases))
        return wg_tab, bl_tab, gl_tab

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def delta(self, h: torch.Tensor, li: int, phase: torch.Tensor | None = None) -> torch.Tensor:
        """Return Δh (same dtype/shape as h); the caller adds it to the residual stream.

        ``phase`` has shape ``(T,)`` or ``(B,T)`` (long, values 0..P-1);
        ``None`` means no conditioning (equivalent to phase 0, taking the exact
        same numerical path as Step 1/2).
        """
        cfg = self.cfg
        x = h.float()
        if cfg.use_rms:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_eps)
        z = F.linear(x, self.A_eff(li))          # (..., d_s)
        if self.act is not None:
            z = self.act(z)
        u = F.linear(z, self.B_eff(li))          # (..., d_m)
        if cfg.use_gate:
            if phase is None:
                u = u * torch.sigmoid(x @ self.wg + self.bl[li]).unsqueeze(-1)
            else:
                wg_tab, bl_tab, _ = self.phase_tables()
                g = (x * wg_tab[phase]).sum(-1) + bl_tab[li][phase]
                u = u * torch.sigmoid(g).unsqueeze(-1)
        if self.gl is not None:
            if phase is None:
                u = u * self.gl[li].exp()
            else:
                _, _, gl_tab = self.phase_tables()
                u = u * gl_tab[li][phase].exp().unsqueeze(-1)
        if cfg.alpha != 1.0:
            u = u * cfg.alpha
        return u.to(h.dtype)

    def forward(self, h: torch.Tensor, li: int, phase: torch.Tensor | None = None) -> torch.Tensor:
        return h + self.delta(h, li, phase)

    # ------------------------------------------------------------------ #
    # bookkeeping
    # ------------------------------------------------------------------ #
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def phase_stats(self) -> dict:
        """Magnitude of the phase deltas (all 0 ⇒ the phase degrees of freedom
        are unused, i.e. degenerate to the global operator)."""
        def _mx(p):
            return round(float(p.detach().abs().max()), 6) if p is not None else 0.0

        def _nm(p):
            return round(float(p.detach().norm()), 6) if p is not None else 0.0

        return {"phase_wg_norm": _nm(self.phase_wg),
                "phase_bl_absmax": _mx(self.phase_bl),
                "phase_gl_absmax": _mx(self.phase_gl)}

    def param_groups(self, lr: float, gate_lr: float | None = None):
        """Group trunk vs gate+gain parameters (the baseline method also uses a
        separate gate lr)."""
        gate_lr = gate_lr if gate_lr is not None else lr
        gate = [self.wg, self.bl] + ([self.gl] if self.gl is not None else [])
        gate += [p for p in (self.phase_wg, self.phase_bl, self.phase_gl)
                 if p is not None]
        gate = [p for p in gate if p.requires_grad]
        trunk = [p for p in self.parameters()
                 if p.requires_grad and not any(p is q for q in gate)]
        groups = [{"params": trunk, "lr": lr}]
        if gate:
            groups.append({"params": gate, "lr": gate_lr})
        return groups

    @torch.no_grad()
    def clip_grads_(self, max_norm: float) -> float:
        """Per-tensor gradient clipping (equivalent of the baseline method's
        row_clip); returns the max pre-clip norm."""
        mx = 0.0
        for p in self.parameters():
            if p.grad is not None:
                n = float(p.grad.norm())
                mx = max(mx, n)
                scale = torch.as_tensor(max_norm / (n + 1e-12),
                                        device=p.grad.device, dtype=p.grad.dtype)
                p.grad.mul_(scale.clamp(max=1.0))
        return mx

    # ------------------------------------------------------------------ #
    # export
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def materialize(self, phase: int = 0) -> dict:
        """Expand the shared trunk into explicit per-depth down/up tensors for
        direct consumption by the vLLM side.

        The vLLM side only implements the single expression
        ``gate * up(act(down(rms(h))))`` and ignores sharing, avoiding extra
        matmuls in the engine. Trainable parameter count is still reported by
        ``num_trainable()``.

        For ``phase>0``, take that phase's ``gate_w/gate_b/gain`` (down/up are
        shared across phases); phase conditioning is realized engine-side via
        **multiple steer vectors + where-clause windows**, so each phase is
        exported separately.
        """
        # ``.clone()`` is required: with a shared trunk, down/up at each depth
        # are the same Parameter; without cloning, safetensors refuses to save
        # due to shared tensor memory (always happens on CPU).
        wg_tab, bl_tab, gl_tab = self.phase_tables()
        out = {}
        for li in range(self.n_depths):
            out[f"layer{li}.down"] = self.A_eff(li).detach().float().cpu().clone()
            out[f"layer{li}.up"] = self.B_eff(li).detach().float().cpu().clone()
            out[f"layer{li}.gate_w"] = wg_tab[phase].detach().float().cpu().clone()
            out[f"layer{li}.gate_b"] = torch.as_tensor(
                float(bl_tab[li, phase]), dtype=torch.float32)
            out[f"layer{li}.gain"] = torch.as_tensor(
                float(gl_tab[li, phase].exp()) if gl_tab is not None else 1.0,
                dtype=torch.float32)
        return out


def skill_delta(payload: dict, h: torch.Tensor, *, use_silu: bool, use_rms: bool,
                use_gate: bool, alpha: float, rms_eps: float = 1e-6) -> torch.Tensor:
    """Functional implementation for vLLM / verification (line-by-line match
    with ``SkillOperator.delta``).

    ``payload`` holds ``down`` / ``up`` / ``gate_w`` / ``gate_b`` / ``gain``.
    """
    x = h.float()
    if use_rms:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_eps)
    r = torch.matmul(x, payload["down"].T)
    if use_silu:
        r = F.silu(r)
    d = torch.matmul(r, payload["up"].T)
    if use_gate:
        d = d * torch.sigmoid(torch.matmul(x, payload["gate_w"])
                              + payload["gate_b"]).unsqueeze(-1)
    d = d * payload["gain"]
    if alpha != 1.0:
        d = d * alpha
    return d
