"""Per-game randomness: unpredictable across games, reproducible within one.

Our policy was a pure strategy. Measured over real play, the persuasion seller
made an identical choice in 91% of repeated (config, round, quality, history
length) states, and its lie placement was literally a function of the round
index. Against a field that is growing fast -- 17 agents to 42 in a day, with
the leaders playing thousands of games and therefore meeting us often -- a
deterministic policy is a standing invitation to be learned and exploited.
Mixed strategies are the textbook answer, and this is the cheapest form of one.

The seed is derived from the game id, which buys two properties at once:

  * ACROSS games the draw is effectively unpredictable, so there is no fixed
    schedule for an opponent to key on;
  * WITHIN a game it is stable, so our stateless solvers still replay
    identically from `history` and any logged game can be reconstructed
    exactly. That matters more than it sounds: the percentile-recovery loop is
    what found every real bug in this project, and randomness that broke
    replay would blind it.

Jitter is deliberately small and always centred, so it costs no expected value
-- we are buying unpredictability, not trading payoff for it.
"""

from __future__ import annotations

import hashlib
import random


def game_rng(game_id: str | None, salt: str = "") -> random.Random:
    """A stable pseudo-random stream for one game and one purpose."""
    key = f"{game_id or 'nogame'}|{salt}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return random.Random(seed)


def jitter(game_id: str | None, salt: str, spread: float) -> float:
    """A centred multiplicative wobble in [1-spread, 1+spread]."""
    if spread <= 0.0:
        return 1.0
    return 1.0 + game_rng(game_id, salt).uniform(-spread, spread)


def offset(game_id: str | None, salt: str, spread: float) -> float:
    """A centred additive wobble in [-spread, +spread]."""
    if spread <= 0.0:
        return 0.0
    return game_rng(game_id, salt).uniform(-spread, spread)


def unit(game_id: str | None, salt: str) -> float:
    """A stable draw in [0, 1) for this game and purpose."""
    return game_rng(game_id, salt).random()
