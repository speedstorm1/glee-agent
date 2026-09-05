"""GLEE competition agent (NeurIPS 2026 IAB workshop).

Deterministic game-theoretic solvers behind a budget-aware transport layer.
No LLM is in the decision path: every numeric choice comes from a closed form,
which is faster, cheaper, and cannot time out or emit malformed JSON.
"""

__all__ = ["api", "config", "safety", "telemetry", "transport", "solvers"]
