"""The ``wuas_skill`` steering algorithm for vLLM (steer-vector API, vLLM 0.26).

Math (line-by-line equivalent of ``wuas_skill/operator.py::SkillOperator.delta``):

    x = RMSNorm(h)                      (use_rms)
    r = x @ down.T                      (d_m -> d_s)
    r = silu(r)                         (use_silu)
    d = r @ up.T                        (d_s -> d_m)
    d = d * sigmoid(x @ gate_w + gate_b)  (use_gate)
    h' = h + scale * (alpha * gain * d)

Relative to a plain post-block low-rank adapter
(``delta = alpha*up(silu(down(h)))``), WUAS-Skill adds three components:
RMS normalization, input-dependent gating, and a cross-depth shared trunk
(the shared trunk is expanded into one down/up pair per depth at export
time, so the engine-side expression is uniform).

Bundle index -> model layer number mapping is given by ``manifest["depths"]``
(the bundle stores ``layer0..layer{D-1}``, corresponding to
``depths[0..D-1]``).
"""

import json
import os

import torch
import torch.nn.functional as F

from vllm.steer_vectors.algorithms.base import BaseSteerVectorAlgorithm
from vllm.steer_vectors.algorithms.factory import register_algorithm


@register_algorithm("wuas_skill")
class WuasSkillAlgorithm(BaseSteerVectorAlgorithm):
    """WUAS-Skill post-block skill operator (see module docstring)."""

    @classmethod
    def load_from_path(cls, path, device, *, config, target_layers=None, **kwargs):
        from safetensors.torch import load_file

        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        if manifest.get("kind") != "wuas_skill_operator":
            raise ValueError(
                f"{path}: expected kind 'wuas_skill_operator', got "
                f"{manifest.get('kind')!r}")
        depths = list(manifest["depths"])
        if target_layers is not None:
            depths = [d for d in depths if d in target_layers]
        tensors = load_file(os.path.join(path, "adapter.safetensors"))
        dtype = config.adapter_dtype

        base = {
            "use_silu": bool(manifest["use_silu"]),
            "use_rms": bool(manifest["use_rms"]),
            "use_gate": bool(manifest["use_gate"]),
            "alpha": float(manifest["alpha"]),
            "rms_eps": float(manifest.get("rms_eps", 1e-6)),
        }
        layer_payloads = {}
        for i, layer in enumerate(depths):
            p = dict(base)
            p["down"] = tensors[f"layer{i}.down"].to(device=device, dtype=dtype)
            p["up"] = tensors[f"layer{i}.up"].to(device=device, dtype=dtype)
            p["gate_w"] = tensors[f"layer{i}.gate_w"].to(device=device, dtype=dtype)
            p["gate_b"] = float(tensors[f"layer{i}.gate_b"])
            p["gain"] = float(tensors[f"layer{i}.gain"])
            layer_payloads[int(layer)] = p
        if not layer_payloads:
            raise ValueError(
                f"bundle {path} targets no requested layer "
                f"(manifest depths {manifest['depths']}, target_layers {target_layers})")
        return {"layer_payloads": layer_payloads}

    def _transform(self, hidden_state: torch.Tensor, params: dict) -> torch.Tensor:
        scale_factor = params.get("scale_factor", 1.0)
        if scale_factor == 0.0:
            return hidden_state                      # exact no-op (base-model reference)

        x = hidden_state
        if params.get("use_rms", True):
            x = x * torch.rsqrt(
                x.pow(2).mean(-1, keepdim=True) + params.get("rms_eps", 1e-6))
        r = torch.matmul(x, params["down"].T)
        if params.get("use_silu", True):
            r = F.silu(r)
        d = torch.matmul(r, params["up"].T)
        if params.get("use_gate", True):
            d = d * torch.sigmoid(
                torch.matmul(x, params["gate_w"]) + params["gate_b"]).unsqueeze(-1)
        d = d * params.get("gain", 1.0) * params.get("alpha", 1.0)
        return hidden_state + scale_factor * d
