"""wuas-skill-plugin — registers the ``wuas_skill`` steering algorithm with
vLLM through the general-plugins entry point. Purely additive: it does not
modify any existing files. The entry point is loaded by vLLM in every
engine/worker process.
"""

__all__ = ["register"]


def register():
    from . import wuas_skill  # noqa: F401  (import performs the registration)
