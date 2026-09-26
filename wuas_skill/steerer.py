"""SkillSteerer — attaches :class:`SkillOperator` to the decoder block outputs of a frozen backbone.

A post-block residual-stream hook injector (same pattern as a generic adapter
steerer); the only difference is the per-layer operator (``SkillOperator``).
Therefore:

* Training side: ``SkillSteerer`` is an ``nn.Module`` with only operator
  parameters trainable, plugging into ``Trainer`` / a custom optimizer;
  ``forward``/``generate`` pass through to the base model and the hooks fire
  automatically;
* Inference side: the operator can be exported by :mod:`wuas_skill.bundle` into
  a vLLM sidecar bundle for online steering inside the engine (no hook sites
  needed).

Injection semantics (aligned with the baseline method's hook): Δh is added at
**all token positions** — no "only the last token" shortcut.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from .skill_operator import (SkillOperator, SkillOperatorConfig,
                             phase_index_from_positions, phase_index_of)

__all__ = ["SkillSteerer", "find_layer_stack", "default_depths"]


def find_layer_stack(model: nn.Module, n_layers: int) -> nn.Module:
    """Return the ``nn.ModuleList`` holding the decoder layers (not its parent).

    Qwen3.5 (``Qwen3_5ForCausalLM``) uses ``model.model.layers`` where
    ``model.model`` is ``Qwen3_5TextModel``; LLaMA is analogous; Gemma uses
    ``model.model.language_model.layers``. A generic search is the fallback,
    avoiding hardcoded paths.
    """
    def ok(m: nn.Module) -> bool:
        st = getattr(m, "layers", None)
        return isinstance(st, nn.ModuleList) and len(st) == n_layers

    inner = getattr(model, "model", None)
    if inner is not None and ok(inner):
        return inner.layers
    for _, m in model.named_modules():
        if ok(m):
            return m.layers
    raise RuntimeError(
        f"{type(model).__name__}: no submodule with a {n_layers}-layer `.layers` found; "
        "specify --layer-path explicitly")


def default_depths(n_layers: int, n_depths: int = 8) -> List[int]:
    """Evenly pick ``n_depths`` injection depths, including the last layer
    (aligned with the baseline method's rho ∈ {.25, .5, .75, 1}).

    The baseline method uses ``int(rho*L)-1``; here generalized to ``n_depths``
    equidistant points + the last layer.
    """
    n_depths = max(1, min(n_depths, n_layers))
    return sorted({int(round((i + 1) / n_depths * n_layers)) - 1 for i in range(n_depths)})


class SkillSteerer(nn.Module):
    """Frozen backbone + several ``SkillOperator`` injection points."""

    def __init__(
        self,
        base_model: nn.Module,
        cfg: SkillOperatorConfig,
        layer_path: str | None = None,
        apply_to: str = "all",     # "all" (aligned with the baseline method) | "last"
    ):
        super().__init__()
        assert apply_to in ("all", "last")
        self.base_model = base_model
        self.cfg = cfg
        self.apply_to = apply_to

        self.layers = find_layer_stack(base_model, base_model.config.get_text_config().num_hidden_layers) \
            if layer_path is None else _get_by_path(base_model, layer_path)
        n_layers = len(self.layers)
        bad = [d for d in cfg.depths if not 0 <= d < n_layers]
        if bad:
            raise ValueError(f"injection depths {bad} out of range: the model has {n_layers} layers")

        param_dtype = next(base_model.parameters()).dtype
        self.operator = SkillOperator(cfg, dtype=torch.float32)
        # Operator params fixed to fp32 (even if the backbone is bf16), matching the
        # baseline method's fp32 master weights; moved explicitly to the backbone's
        # device (the constructor does not follow automatically)
        self.operator.to(device=next(base_model.parameters()).device, dtype=torch.float32)

        self._handles: list = []
        self._enabled = True
        # Step 3: phase-conditioning context. ``_gen_start`` None ⇒ phase 0
        # (equivalent to Step 1/2).
        self._gen_start: torch.Tensor | int | None = None
        self._phase_warned = False
        for li, layer_idx in enumerate(cfg.depths):
            self._handles.append(
                self.layers[layer_idx].register_forward_hook(
                    self._make_hook(li, layer_idx), with_kwargs=True))

        # no-ops for GRPO/Trainer compatibility
        self.warnings_issued: dict = {}
        self.is_gradient_checkpointing = None
        self._param_dtype = param_dtype

    # ------------------------------------------------------------------ #
    def _make_hook(self, li: int, layer_idx: int):
        def fn(module, args, kwargs, output):
            if not self._enabled:
                return None
            is_t = torch.is_tensor(output)
            h = output if is_t else output[0]
            if not torch.is_tensor(h):
                raise TypeError(
                    f"SkillSteerer hook @ layer {layer_idx}: expected a residual-stream "
                    f"tensor, got {type(output).__name__}")
            if h.ndim != 3:
                return None
            delta = self.operator.delta(h, li, self._phase_of(h, kwargs))
            if self.apply_to == "last":
                delta = _mask_last(delta)
            out = h + delta
            if is_t:
                return out
            return (out,) + tuple(output[1:])
        return fn

    # ------------------------------------------------------------------ #
    # Step 3: phase context
    # ------------------------------------------------------------------ #
    def set_phase_ctx(self, gen_start) -> None:
        """Declare the start **absolute position** of the generation phase (int or ``(B,)`` tensor).

        * Right-padded training (prompt+target concatenated, right padding): pass
          each row's prompt token count;
        * Left-padded generation: pass the **padded** prompt length (same for all
          rows);
        * ``None``: clear the context (phases degrade to 0).

        Aligned with the vLLM side's ``generation_window`` 0-based decode step: the
        hidden state at position ``gen_start + j`` corresponds to decode step ``j``.
        """
        self._gen_start = gen_start

    def _phase_of(self, h: torch.Tensor, kwargs: dict | None):
        """Return the (B,T) phase indices per the current context; None when
        ``n_phases==1`` (fast path).

        Must use ``position_ids`` (not ``h.shape[1]``): with HF incremental
        decoding, ``hidden_states`` only holds the current chunk (usually 1
        token), so local indices cannot express phases.
        """
        if self.cfg.n_phases == 1:
            return None
        gs = self._gen_start
        if gs is None:
            if not self._phase_warned:
                self._phase_warned = True
                print("[steerer] n_phases>1 but set_phase_ctx() was not called: treating all as phase 0")
            return None
        pos = (kwargs or {}).get("position_ids")
        if torch.is_tensor(pos):
            if pos.ndim == 3:                     # (4, B, T) → text positions
                pos = pos[0]
            if pos.shape[-1] == h.shape[1]:
                return phase_index_from_positions(pos, gs, self.cfg.phase_edges)
        if not self._phase_warned:
            self._phase_warned = True
            print("[steerer] position_ids unavailable; falling back to local indices "
                  "(correct only for full-sequence forward)")
        return phase_index_of(h.shape[1], gs, self.cfg.phase_edges, device=h.device)

    # ------------------------------------------------------------------ #
    def enable(self):
        self._enabled = True
        return self

    def disable(self):
        """Disable the intervention (equivalent to the base model) — for the
        identity gate and ablations."""
        self._enabled = False
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        return [p for p in self.operator.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return self.operator.num_trainable()

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    # GRPO/Trainer compatibility no-ops
    def add_model_tags(*args, **kwargs):
        pass

    def gradient_checkpointing_enable(*args, **kwargs):
        pass

    @torch.no_grad()
    def probe(self) -> dict:
        """Diagnostic: per-layer Δh relative norms (equivalent of the baseline
        method's branch.probe)."""
        return {}


def _mask_last(delta: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(delta)
    z[:, -1:, :] = delta[:, -1:, :]
    return z


def _get_by_path(model: nn.Module, path: str) -> nn.Module:
    m = model
    for part in path.split("."):
        m = getattr(m, part) if not part.isdigit() else m[int(part)]
    return m
