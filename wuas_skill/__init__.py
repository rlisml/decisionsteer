"""wuas_skill — WUAS skill operator via post-block low-rank steering.

Modules
----
``skill_operator`` :class:`SkillOperator` — depth-shared trunk + RMS norm + input-dependent gating
``steerer``        :class:`SkillSteerer` — attaches the operator to a frozen backbone via forward hooks
``bundle``         Export / load / load-back self-check for the vLLM sidecar bundle
``data_searchqa``  SearchQA task (reuses the baseline's data and EM scoring)
``train_bc``       Behavior-cloning trainer (first step: quick validation)
``eval_vllm``      Fast evaluation with the vLLM fork
``plugin/``        ``wuas_skill`` steering algorithm for the vLLM fork

Core formula (at the l-th injection depth, block output h)::

    x  = RMS(h)
    z  = phi(x @ A_l^T)          A_l: (d_s, d_m), shared across depths (+ optional per-depth low-rank residual)
    u  = z @ B_l^T               B_l: (d_m, d_s), zero-init ⇒ exact identity at the start of training
    g  = sigmoid(x @ w + b_l)
    h' = h + alpha * gamma_l * g ⊙ u
"""

__version__ = "0.1.0"

__all__ = [
    "operator",
    "steerer",
    "bundle",
    "data_searchqa",
    "train_bc",
    "eval_vllm",
]
