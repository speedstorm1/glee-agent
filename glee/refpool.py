"""Reference payoff distributions, and scoring a decision by expected percentile.

Why this exists
---------------
GLEE pays the PERCENTILE of your payoff within the pool of players who played
the identical configuration in the same role, not the payoff itself. Percentile
is monotone in payoff, so a payoff-maximizer is never badly wrong, but it is
not smooth, and the bends are what this module exists to price.

The bend that matters is an atom at exactly zero: reference buyers who refused
for twenty rounds and banked nothing. Measured on our own field (41,771 logged
persuasion games), the atom is

    p = 1/3  ->  0.357        p = 0.5  ->  0.265        p = 0.8  ->  0.115

so at a low prior banking zero already scores above a third of the pool, while
at a high prior refusing lands in the bottom eighth. Two decisions with
identical expected payoff can therefore want opposite actions.

This module only scores. It answers what percentile a projected payoff
distribution earns; the caller decides what to do with that.

Provenance and its limits
-------------------------
The pool is built from the OPPONENT's payoff in our own games: in every game we
played, the counterpart's payoff is one draw from the field's distribution for
that (cell, role). That makes it a sample of the live field rather than of some
frozen reference corpus.

Two caveats are carried in the data and enforced below:

  * `v` is hidden from us in the games where we were the seller and
    `is_seller_know_cv` is false, so 20.4% of BUYER rows had their v-bucket
    imputed from the uniform-grid shortfall. Strata that lean on that
    imputation are refused by `lookup` -- see `MAX_IMPUTED`.
  * A pool is a snapshot of a moving field. Measured drift over our log is
    +0.0094 (z = +1.09), not significant, but that is a bound rather than a
    guarantee, so callers should treat a percentile as a tie-break and not as
    a precise quantity.
"""

from __future__ import annotations

import bisect
import gzip
import json
import math
import os

#: Minimum reference samples before a stratum is allowed to influence a
#: decision. Below this the atom estimate is worth less than the EV it would
#: override: at n = 30 an atom near 0.35 carries a standard error of ~8.7pp.
MIN_SAMPLES = 30

#: Maximum fraction of a stratum that may come from v-imputed rows. The
#: imputation assigns a v-bucket by grid shortfall rather than observation, so
#: a stratum built mostly from it describes the right (p, price, msg_type)
#: block but the wrong column of it. 18 of 180 buyer strata exceed this.
MAX_IMPUTED = 0.5

_PATH = os.path.join(os.path.dirname(__file__), "data",
                     "persuasion_refpool.json.gz")

_POOL: dict | None = None


class Stratum:
    """One (configuration, role) reference distribution, as sorted payoffs."""

    __slots__ = ("n", "atom0", "neg", "mean", "vals", "imputed")

    def __init__(self, raw: dict):
        self.n = int(raw["n"])
        self.atom0 = float(raw["atom0"])
        self.neg = float(raw["neg"])
        self.mean = float(raw["mean"])
        self.vals = raw["vals"]
        self.imputed = float(raw.get("n_imputed", 0)) / max(self.n, 1)

    def percentile(self, x: float) -> float:
        """Mid-rank percentile of `x`, the convention the server scores with.

        Ties take half credit, so a payoff that lands exactly on the atom
        scores the middle of it rather than the top or the bottom. This is
        also what makes `expected_percentile` continuous as sd goes to zero:
        the atom needs no special case anywhere in this module.
        """
        lo = bisect.bisect_left(self.vals, x)
        hi = bisect.bisect_right(self.vals, x)
        return (lo + 0.5 * (hi - lo)) / self.n

    def expected_percentile(self, mean: float, sd: float) -> float:
        """E[percentile] of a Normal(mean, sd) projected payoff.

        E[pct] = mean_j P(X > y_j) over reference payoffs y_j, which is the
        expectation of the mid-rank statistic under the projection. Degenerate
        sd falls through to the exact mid-rank so the two agree at the limit.
        """
        if sd <= 1e-12:
            return self.percentile(mean)
        acc = 0.0
        inv = 1.0 / (sd * math.sqrt(2.0))
        for y in self.vals:
            # P(X > y) for X ~ Normal(mean, sd)
            acc += 0.5 * math.erfc((y - mean) * inv)
        return acc / self.n


def _load() -> dict:
    global _POOL
    if _POOL is None:
        try:
            with gzip.open(_PATH, "rt", encoding="utf-8") as fh:
                raw = json.load(fh)
            _POOL = {k: Stratum(v) for k, v in raw["strata"].items()}
        except (OSError, ValueError, KeyError):
            # A missing or corrupt pool must never take the agent down; every
            # caller has an expected-value fallback.
            _POOL = {}
    return _POOL


#: The published configuration grid. Runtime values are snapped onto it before
#: a key is formed: `v / price` is a float division, so the 1.2 ratio arrives
#: as 1.1999999999999997 at some price scales and would miss its own stratum.
_V_RATIOS = (1.2, 1.25, 2.0, 3.0, 4.0)
_PRIORS = (1.0 / 3.0, 0.5, 0.8)
_PRICES = (100.0, 10000.0, 1000000.0)


def _snap(x: float, grid: tuple[float, ...]) -> float | None:
    best = min(grid, key=lambda g: abs(g - x))
    return best if abs(best - x) <= 1e-6 * max(1.0, abs(best)) else None


def _key(p: float, v_ratio: float, price: float, msg_type: str,
         know_cv: bool, role: str) -> str | None:
    p_s = _snap(p, _PRIORS)
    v_s = _snap(v_ratio, _V_RATIOS)
    pr_s = _snap(price, _PRICES)
    if p_s is None or v_s is None or pr_s is None:
        return None
    return "%s|%s|%s|%s|%d|%s" % (
        _grid(p_s), _grid(v_s), _grid(pr_s), msg_type,
        1 if know_cv else 0, role)


def _grid(x: float) -> str:
    """Format a grid value the way the builder did, so keys match exactly.

    `%g` alone is wrong: it renders 1000000 as `1e+06`, which is not the key
    the builder wrote. Integers go through `int` and only genuine fractions
    take `%g`, whose six significant digits reproduce `0.333333` for 1/3.
    """
    return str(int(x)) if float(x).is_integer() else ("%g" % x)


def lookup(p: float, v: float, price: float, msg_type: str,
           know_cv: bool, role: str) -> Stratum | None:
    """The reference stratum for this cell, or None if it is not usable.

    Returns None rather than a poor stratum on purpose: the caller's fallback
    is expected value, which is less precise but better behaved than a thin or
    heavily imputed stratum.
    """
    if price <= 0:
        return None
    key = _key(p, v / price, price, msg_type, know_cv, role)
    if key is None:
        return None
    st = _load().get(key)
    if st is None or st.n < MIN_SAMPLES or st.imputed > MAX_IMPUTED:
        return None
    return st


def normalise(payoff: float, price: float, total_rounds: int) -> float:
    """Put a raw payoff on the pool's axis: payoff per round, in price units."""
    denom = price * max(total_rounds, 1)
    return payoff / denom if denom else 0.0
