"""Competition constants and the reconstructed GLEE parameter grid.

The grid below is transcribed from Table 2 of the GLEE benchmark paper
(arXiv:2410.05254v3), which defines the configurations the reference payoff
dataset was collected from. The competition server draws each game's
configuration from this space, and -- crucially -- our score is a percentile
against payoffs earned on the SAME configuration in the SAME role. So knowing
the grid is knowing the support of our opponent's hidden parameters.

The paper's grid totals 1,320 configurations (384 + 576 + 360). The competition
docs say 960. Note 384 + 576 = 960 exactly, which suggests the docs' figure may
omit persuasion, or the grid was trimmed. The full-grid sweeps in
`tests/test_solvers.py` rebuild the grid from observed games; treat everything
here as a prior.
"""

from __future__ import annotations

BASE_URL = "https://glee-competition.com"
FAMILIES = ("bargaining", "negotiation", "persuasion")

# --- Hard server limits (from the docs; violating these costs rating) --------
RATE_LIMIT_PER_MIN = 60      # requests/minute/agent -> 429 + Retry-After
TURN_TIMEOUT_S = 120         # miss it and the game is a no-deal at 5th pct
MAX_INVALID_ATTEMPTS = 5     # burn all five and the game is a no-deal
MAX_MESSAGE_LEN = 2000       # longer is an invalid move (costs an attempt)

# We govern ourselves strictly below the cap rather than relying on 429
# backoff, because a 429 storm wastes wall-clock we could spend on moves.
# Observed server 429s at 54, so leave a wider margin: the server's
# window and ours are not perfectly aligned.
SAFE_RATE_LIMIT = 48

# --- Bargaining grid: 4 * 4 * 3 * 2 * 2 * 2 = 384 ---------------------------
BARGAINING_DELTAS = (0.8, 0.9, 0.95, 1.0)   # per-round discount multiplier
BARGAINING_MONEY = (10**2, 10**4, 10**6)
BARGAINING_HORIZONS = (12, None)            # None = "infinite" (unknown to us)

# --- Negotiation grid: 4 * 4 * 3 * 3 * 2 * 2 = 576 --------------------------
# Valuations are (hypothesised) V = F * M. See NEGOTIATION_VALUES below.
NEGOTIATION_FACTORS = (0.8, 1.0, 1.2, 1.5)
NEGOTIATION_BASE = (10**2, 10**4, 10**6)
NEGOTIATION_HORIZONS = (1, 10, None)

# --- Persuasion grid: 3 * 5 * 3 * 2 * 2 * 2 = 360 ---------------------------
PERSUASION_P = (1.0 / 3.0, 0.5, 0.8)        # prior P(high quality)
PERSUASION_V = (1.2, 1.25, 2.0, 3.0, 4.0)   # buyer value for HIGH, in price units
PERSUASION_ROUNDS = 20                       # T is always 20 in the paper's grid


def _negotiation_value_support() -> tuple[float, ...]:
    """Every valuation the grid can produce, assuming V = F * M.

    The twelve products are pairwise DISTINCT, which is the whole exploit:
    observing your own valuation pins down both M and your own F, so the
    opponent's valuation is one of exactly four known numbers even when
    `complete_information` is false.
    """
    return tuple(sorted({f * m for f in NEGOTIATION_FACTORS for m in NEGOTIATION_BASE}))


NEGOTIATION_VALUES = _negotiation_value_support()


def opponent_value_support(my_value: float) -> tuple[float, ...]:
    """Given my own valuation, the four possible valuations of my opponent.

    Returns an empty tuple if `my_value` doesn't sit on the grid -- in which
    case the V = F * M hypothesis is wrong for this game and callers must fall
    back to an uninformed prior rather than trusting a bogus support.
    """
    for base in NEGOTIATION_BASE:
        for factor in NEGOTIATION_FACTORS:
            if abs(factor * base - my_value) < 1e-6 * max(1.0, abs(my_value)):
                return tuple(f * base for f in NEGOTIATION_FACTORS)
    return ()


def infer_negotiation_base(my_value: float) -> float | None:
    """The base M implied by my own valuation, or None if off-grid."""
    for base in NEGOTIATION_BASE:
        for factor in NEGOTIATION_FACTORS:
            if abs(factor * base - my_value) < 1e-6 * max(1.0, abs(my_value)):
                return float(base)
    return None
