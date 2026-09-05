"""Bargaining (alternating offers / divide-the-dollar).

Theory (Rubinstein 1982, as used by the GLEE paper). Accepting at stage t gives
utilities M * (d_A^(t-1) * p, d_B^(t-1) * (1-p)). With an infinite horizon and
common knowledge of the discount factors, the unique SPE is immediate agreement
with proposer share

    p* = (1 - d_responder) / (1 - d_proposer * d_responder)

which over the paper's grid d in {0.8, 0.9, 0.95, 1.0} means patience is
everything: d_me = 1.0 against any d_opp < 1.0 is a theoretical 100% share,
while d_me = 0.8 against d_opp = 0.95 is worth only 20.8%.

Finite horizons (T = 12) are solved here by exact backward induction rather
than approximated, because the endgame dominates: whoever proposes last has a
take-it-or-leave-it.

Two departures from textbook SPE, both deliberate:

  1. We never demand the theoretical maximum. SPE assumes a rational opponent;
     a real one walks away or stonewalls, and a no-deal pays zero. So we cap
     demands and converge to a closable number.
  2. Under incomplete information the opponent's delta is hidden, but it is one
     of only four values. We carry an explicit posterior over those types,
     replayed from `history` each turn (so the solver stays stateless), and
     pick the offer maximizing expected payoff against it.

Speed matters more than greed whenever delta is low: at d = 0.8, taking half
the pot in round 1 (0.500) beats taking 70% in round 5 (0.8^4 * 0.7 = 0.287).
"""

from __future__ import annotations

import logging
import os
import re

from ..config import BARGAINING_DELTAS
from ..rng import jitter, offset
from ..safety import bargaining_offer

logger = logging.getLogger("glee.bargaining")

# Cap: an opponent facing a 97/3 split tends to walk, and a walkaway pays zero.
MAX_DEMAND = 0.90
# Floor we will not concede below while we still have rounds in hand.
MIN_TARGET = 0.42
#: How fast the patient side's accept floor relaxes toward MIN_TARGET as the
#: deadline nears. 1.0 is linear; higher holds firm for longer. Tuned against
#: the simulated no-deal rate, since a no-deal pays zero.
URGENCY_EXP = 2.0
#: How far from the safe closable bar toward the full SPE share we push when
#: we are the patient side. 0 is the cautious extreme.
STRONG_PUSH = 0.45
#: Centred multiplicative wobble on the demand we name, per game: an anti-
#: modeling measure, and the only exogenous read on the acceptance curve.
DEMAND_JITTER = 0.04
#: Multiplier on the demand we name, applied ONLY where delay costs us
#: (d_me < FREE_WAITING_DELTA). Set from a randomized probe over 3,503 games,
#: discounted payoff for a low multiplier (<0.95) against a high one (>=1.05):
#:   stratum              n      low      high     diff       t
#:   II, d_me < 1.0    1343   0.4338   0.4066   +0.0272    4.07
#:   CI, d_me < 1.0    1253   0.4038   0.3696   +0.0343    3.40
#:   II, d_me = 1.0     447   0.5661   0.5445   +0.0216    1.11
#:   CI, d_me = 1.0     462   0.6181   0.6487   -0.0307   -1.38
#: Ask for less where delay costs us, and leave the patient side alone. The
#: curve peaks at 0.82-0.94 and is flat across it, so 0.88 is the midpoint.
DEMAND_SCALE = 0.88
#: Our own discount factor at or above which waiting is costless, so any share
#: we extract by holding out is kept in full. On the grid {0.8, 0.9, 0.95, 1.0}
#: the measured benefit of holding out is a cliff at 1.0, not a gradient (see
#: the `post_floor_cap` table).
FREE_WAITING_DELTA = 0.999

#: Bargaining policy arm. "current" is the empirical continuation, a field
#: prior for round one, and calibrated cave and hazard rates; "legacy" restores
#: the policy that preceded it, so the two can be compared in the field.
#: Read at CALL time, never bound at import: run_agent.py sets the env var
#: after these modules load.
def legacy() -> bool:
    return os.environ.get("GLEE_BARG_ARM", "current") == "legacy"


#: One knob per independent change in the current arm, so the bundle can be
#: bisected instead of reverted. `legacy()` flips all of them at once;
#: GLEE_BARG_KNOBS="field_prior=off,post_floor_cap=off" then overrides
#: individual knobs on top of whichever arm is selected. Defaults reproduce
#: that arm exactly, and are CALLABLES wherever a module-level constant exists,
#: which keeps the constant authoritative. Read at CALL time, as `legacy()`.
_KNOB_DEFAULTS = {
    # Allow a NEGATIVE concession rate; off clamps at >= 0, which is false here.
    "rate_clamp":     {"current": "off",  "legacy": "on"},
    # Beta prior on the chance they accept one of OUR offers in a cycle.
    "cave_prior":     {"current": "0.02", "legacy": "0.10"},
    # Per-cycle chance a game we keep alive dies anyway.
    "hazard":         {"current": lambda: HAZARD_PER_CYCLE, "legacy": "0.08"},
    # Fall back to the measured field offer curve when we have no history.
    "field_prior":    {"current": "on",   "legacy": "off"},
    # Re-apply the achievability cap AFTER the patient-side floor, which is
    # what can make that floor dead code when a projection exists.
    #   on           always re-apply -> the floor never binds
    #   off          never re-apply  -> the floor binds whenever d_me >= d_opp
    #   free_waiting re-apply UNLESS waiting is costless to us (d_me = 1.0)
    # Discounted payoff, legacy minus current, by our discount factor (765
    # live games):
    #   0.80  +0.005 (t  0.22)      0.90  -0.022 (t -1.38)
    #   0.95  -0.012 (t -0.74)      1.00  +0.074 (t  4.18)
    # Holding out buys share and pays for it in delay, so the whole effect sits
    # at d_me = 1.0: gate on absolute patience, not relative. `free_waiting`
    # beat both other settings (t = +2.36, n = 1,124), so it is the default.
    "post_floor_cap": {"current": "free_waiting", "legacy": "off"},
    # How far toward the full SPE share we push when we are the patient side.
    "strong_push":    {"current": lambda: STRONG_PUSH, "legacy": "0.0"},
    # Spread of the per-game demand wobble; mean-preserving, so raising it
    # probes the acceptance curve without trading payoff for the information.
    "demand_jitter":  {"current": lambda: DEMAND_JITTER,
                       "legacy": lambda: DEMAND_JITTER},
    # Multiplier on the demand where delay costs us; 1.0 is the unscaled
    # schedule and DEMAND_SCALE is what the randomized probe selected.
    "demand_scale":   {"current": lambda: DEMAND_SCALE, "legacy": "1.0"},
}


def knob(name: str) -> str:
    """Effective value of one policy knob, read at CALL time."""
    arm = "legacy" if legacy() else "current"
    value = _KNOB_DEFAULTS[name][arm]
    if callable(value):
        value = repr(float(value()))
    raw = os.environ.get("GLEE_BARG_KNOBS", "")
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, _, val = item.partition("=")
        if key.strip() == name:
            value = val.strip()
    return value


def knob_float(name: str) -> float:
    return float(knob(name))


def knob_on(name: str) -> bool:
    return knob(name) == "on"
#: Per-cycle chance a game we keep alive ends without agreement anyway: the
#: opponent walks, stalls out, or an unknown horizon runs out. Without it the
#: maths makes waiting free at delta_me = 1.0 and the agent holds out forever.
#: Calibrated on 487 uncapped games, which produced zero no-deals, so the true
#: rate is far below the 0.08 first guessed. Some no-deal risk is optimal.
HAZARD_PER_CYCLE = 0.02
#: What this field's opponents actually leave us, by round. Measured over the
#: 36,628 games in which a counterpart made us an offer, with no conditioning
#: on our response, which would bias the slope downward:
#:   r1 0.436  r3 0.435  r5 0.468  r7 0.483  r9 0.475
#: Paired within a game their offers IMPROVE: +0.027 by round 3 (t = 22.7,
#: n = 6,209) and +0.046 by round 7 (t = 22.5, n = 3,654). Values below shade
#: those means ~4% for the walk-away risk `HAZARD_PER_CYCLE` prices separately;
#: the odd/even sawtooth tracks which side is proposing. Load-bearing:
#: `field_continuation` feeds a cap that overrides the patient-side floor.
FIELD_OFFER_BY_ROUND = {1: 0.418, 2: 0.378, 3: 0.418, 4: 0.389, 5: 0.449,
                        6: 0.404, 7: 0.464, 8: 0.399, 9: 0.456, 10: 0.399}


def field_continuation(rnd: int, d_me: float) -> float:
    """Best discounted share the field will still offer us after round `rnd`.

    A prior, not a law: it is replaced by the opponent's own observed sequence
    as soon as we have two of their offers.
    """
    best = 0.0
    for r, share in FIELD_OFFER_BY_ROUND.items():
        if r <= rnd:
            continue
        best = max(best, share * (d_me ** (r - rnd)))
    return best
#: Posterior mass placed on a discount factor the opponent states themselves.
_STATED_CONFIDENCE = 0.88


def _other(player: str) -> str:
    return "player_2" if player == "player_1" else "player_1"


def spe_infinite(d_me: float, d_opp: float) -> float:
    """Proposer's SPE share with an infinite horizon."""
    denom = 1.0 - d_me * d_opp
    if denom <= 1e-12:
        # Both perfectly patient: the folk theorem bites and every split is an
        # equilibrium, so treat it as even and let the message layer fight.
        return 0.5
    return max(0.0, min(1.0, (1.0 - d_opp) / denom))


def spe_finite(t_now: int, horizon: int, proposer_now: str,
               me: str, d_me: float, d_opp: float) -> dict[int, float]:
    """Backward induction over a capped horizon.

    Returns g[t] = the share secured by whoever PROPOSES at round t.
    At the final round the proposer takes everything (the responder's
    alternative is a no-deal worth zero). Earlier, the proposer must leave the
    responder exactly what waiting is worth to them: d_responder * g[t+1],
    since the responder becomes the proposer next round.
    """
    g: dict[int, float] = {horizon: 1.0}
    delta = {me: d_me, _other(me): d_opp}
    for t in range(horizon - 1, t_now - 1, -1):
        # Proposers alternate, so the responder at t proposes at t+1.
        proposer_t = proposer_now if (t - t_now) % 2 == 0 else _other(proposer_now)
        responder_t = _other(proposer_t)
        g[t] = max(0.0, min(1.0, 1.0 - delta[responder_t] * g[t + 1]))
    return g


def _my_share_as_proposer(state: dict, me: str, d_me: float, d_opp: float) -> float:
    rnd = int(state.get("round") or 1)
    horizon = state.get("max_rounds")
    if horizon and state.get("horizon_known"):
        proposer_now = state.get("proposer") or me
        g = spe_finite(rnd, int(horizon), proposer_now, me, d_me, d_opp)
        return g.get(rnd, spe_infinite(d_me, d_opp))
    return spe_infinite(d_me, d_opp)


def _continuation_if_i_reject(state: dict, me: str,
                              d_me: float, d_opp: float) -> float:
    """What rejecting is worth to me right now, in current-round units.

    Rejecting makes me the proposer next round, where I secure some share, but
    one round later, so it is scaled by my own discount factor. A flat
    percentage threshold is wrong whenever delta or the horizon varies.
    """
    rnd = int(state.get("round") or 1)
    horizon = state.get("max_rounds")
    if horizon and state.get("horizon_known"):
        horizon = int(horizon)
        if rnd >= horizon:
            return 0.0  # no next round: rejecting means a no-deal, worth zero
        proposer_now = state.get("proposer") or _other(me)
        g = spe_finite(rnd, horizon, proposer_now, me, d_me, d_opp)
        return d_me * g.get(rnd + 1, 0.0)
    return d_me * spe_infinite(d_me, d_opp)


# -- opponent type inference -------------------------------------------------

#: Phrasings in which a counterpart states THEIR OWN per-round loss. Of 390
#: real opponent messages, 26 quoted a rate and 13 of the 14 we could check
#: quoted it correctly, so under incomplete information this identifies the
#: hidden parameter outright.
_SELF_RATE_PATTERNS = (
    r"costs?\s+me\s+(?:about\s+)?(\d{1,2})\s*%",
    r"i\s+lose\s+(?:about\s+)?(\d{1,2})\s*%",
    r"i'?m\s+losing\s+(?:about\s+)?(\d{1,2})\s*%",
    r"my\s+(?:money|value|share|payoff)\s+(?:drops|shrinks|decays|loses)\s+(?:by\s+)?(\d{1,2})\s*%",
    r"(\d{1,2})\s*%\s+(?:per|each)\s+round\s+for\s+me",
    r"my\s+(?:discount|inflation)\s+(?:rate\s+)?is\s+(\d{1,2})\s*%",
)


def parse_stated_delta(text: object) -> float | None:
    """Extract the opponent's OWN discount factor if they state it.

    Only first-person phrasings count. "you lose 20% per round" is a claim
    about US, and is very often our own message quoted back, so matching it
    would poison the posterior with our own beliefs.

    The stated rate must land on the published grid; anything else is noise or
    a rhetorical flourish rather than a real parameter.
    """
    if not text:
        return None
    body = " ".join(str(text).lower().split())
    for pattern in _SELF_RATE_PATTERNS:
        m = re.search(pattern, body)
        if not m:
            continue
        pct = int(m.group(1))
        for delta in BARGAINING_DELTAS:
            if abs(round((1.0 - delta) * 100) - pct) < 1e-9:
                return delta
        return None
    return None


def _opponent_messages(state: dict, me: str) -> list[str]:
    opp = _other(me)
    out = []
    last = state.get("last_offer") or {}
    if last.get("proposer") == opp and last.get("message"):
        out.append(str(last["message"]))
    for entry in (state.get("history") or []):
        if entry.get("proposer") != opp:
            continue
        msg = (entry.get("offer") or {}).get("message")
        if msg:
            out.append(str(msg))
    return out


def opponent_delta_posterior(state: dict, me: str) -> dict[float, float]:
    """Posterior over the opponent's hidden discount factor.

    Under `complete_information` we just read it. Otherwise it is one of the
    four grid values, and their observed behavior is informative: a player who
    rejects a generous offer is revealing patience (high delta), because
    rejection is only worth it if waiting is cheap for them.

    Replayed from `history` on every call so the solver holds no state.
    """
    opp = _other(me)
    opp_delta = state.get("delta_1" if opp == "player_1" else "delta_2")
    if opp_delta is not None:
        return {float(opp_delta): 1.0}

    post = {d: 1.0 / len(BARGAINING_DELTAS) for d in BARGAINING_DELTAS}
    money = float(state.get("money_to_divide") or 1.0) or 1.0

    # Believe a stated rate, but not absolutely: 13 of 14 checkable claims were
    # true, so concentrate the posterior rather than collapse it and let one
    # liar capture it.
    for msg in _opponent_messages(state, me):
        stated = parse_stated_delta(msg)
        if stated is not None:
            spread = (1.0 - _STATED_CONFIDENCE) / (len(BARGAINING_DELTAS) - 1)
            return {d: (_STATED_CONFIDENCE if d == stated else spread)
                    for d in BARGAINING_DELTAS}

    d_me = float(state.get("delta_1" if me == "player_1" else "delta_2") or 1.0)

    # Their OFFERS are the strong signal: what a player demands for themselves
    # is a direct read on how much they think waiting is worth, since a patient
    # counterpart asks for more and holds it.
    for entry in (state.get("history") or []) + [
            {"proposer": (state.get("last_offer") or {}).get("proposer"),
             "offer": state.get("last_offer") or {}, "decision": None}]:
        offer = entry.get("offer") or {}
        proposer = entry.get("proposer")
        decision = str(entry.get("decision") or "").lower()

        if proposer == opp and offer.get(f"{opp}_gain") is not None:
            demand = max(0.0, min(1.0, float(offer[f"{opp}_gain"]) / money))
            for d in list(post):
                predicted = spe_infinite(d, d_me)
                # Soft, never zero: counterparts anchor on focal splits, so an
                # exact-match likelihood would rule out every type at 50/50.
                err = abs(demand - predicted)
                post[d] *= max(0.08, 1.0 - err)

        elif proposer == me and decision and offer.get(f"{opp}_gain") is not None:
            frac = max(0.0, min(1.0, float(offer[f"{opp}_gain"]) / money))
            for d in list(post):
                # Waiting is worth their SPE share next round, discounted once.
                cont = d * spe_infinite(d, d_me)
                if decision.startswith("reject"):
                    post[d] *= 0.8 if cont > frac else 0.2
                elif decision.startswith("accept"):
                    post[d] *= 0.8 if cont <= frac else 0.2

    total = sum(post.values()) or 1.0
    return {d: w / total for d, w in post.items()}


def _their_offers_to_me(state: dict, me: str) -> list[tuple[int, float]]:
    """The opponent's offers to us so far, as (round, my share of the pot)."""
    opp = _other(me)
    money = float(state.get("money_to_divide") or 1.0) or 1.0
    seq: list[tuple[int, float]] = []
    for entry in (state.get("history") or []):
        if entry.get("proposer") != opp:
            continue
        offer = entry.get("offer") or {}
        mine = offer.get(f"{me}_gain")
        if mine is None:
            continue
        seq.append((int(entry.get("round") or 0), float(mine) / money))
    last = state.get("last_offer") or {}
    if last.get("proposer") == opp and last.get(f"{me}_gain") is not None:
        rnd = int(last.get("round") or 0)
        if not seq or seq[-1][0] != rnd:
            seq.append((rnd, float(last[f"{me}_gain"]) / money))
    seq.sort()
    return seq


def _cave_probability(state: dict, me: str) -> float:
    """Chance the opponent accepts one of OUR offers in a given cycle.

    Estimated from how often they have folded so far, shrunk toward a modest
    prior early on. Without this term the continuation value is just "their
    next offer", which badly understates waiting: the whole point of holding
    firm is that they might come to us.
    """
    opp = _other(me)
    ours = caved = 0
    for entry in (state.get("history") or []):
        if entry.get("proposer") != me:
            continue
        decision = str(entry.get("decision") or "").lower()
        if not decision:
            continue
        ours += 1
        caved += decision.startswith("accept")
    # Measured, not guessed: across 1,024 completed agreements, opponents
    # accepted an offer of ours exactly zero times. Every deal we closed, we
    # closed by accepting theirs.
    prior_p = knob_float("cave_prior")
    prior_w = 5.0 if legacy() else 6.0
    return (caved + prior_w * prior_p) / (ours + prior_w)


def realistic_continuation(state: dict, me: str, d_me: float,
                           d_opp_hint: float = 0.9) -> float | None:
    """What holding out is ACTUALLY worth, given how fast they are conceding.

    The SPE threshold assumes a rational opponent who caves immediately. Real
    counterparts inch: in one observed game the opponent moved from 28.1% to
    30.6% over eight rounds while our delta of 0.95 cost us 34% of the pot, so
    "winning" the argument turned 281k into 203k.

    We therefore extrapolate their concession curve and price the wait
    honestly: accepting now is worth `share`; waiting n of their offers is
    worth d_me^(2n) * projected, since a full proposal cycle is two stages.
    Returns None until we have enough of their offers to see a trend.
    """
    seq = _their_offers_to_me(state, me)
    if len(seq) < 2:
        return None
    current = seq[-1][1]
    steps = [seq[i][1] - seq[i - 1][1] for i in range(1, len(seq))]
    # Allow a NEGATIVE rate: clamping at zero assumes a counterpart never moves
    # backwards, which the paired data behind FIELD_OFFER_BY_ROUND contradicts.
    rate = sum(steps) / len(steps)
    if knob_on("rate_clamp"):
        rate = max(0.0, rate)

    horizon = state.get("max_rounds") if state.get("horizon_known") else None
    rnd = int(state.get("round") or 1)
    cycles_left = ((int(horizon) - rnd) // 2) if horizon else 6
    cycles_left = max(0, min(6, cycles_left))

    # Waiting is worth more than "their next offer": each cycle they might
    # instead accept OURS. Without that term the rule collapses at d_me = 1.0,
    # where the SPE threshold is 1.0 and this projection decides everything.
    p_cave = _cave_probability(state, me)
    # If they fold, they fold to what we are asking for, not to the SPE ideal.
    target = current_demand(state, me, d_me, d_opp_hint)

    best = current
    for n in range(1, cycles_left + 1):
        their_next = min(MAX_DEMAND, current + rate * n)
        folded = 1.0 - (1.0 - p_cave) ** n
        value = folded * target + (1.0 - folded) * their_next
        # Discount for our own impatience and for the risk the game dies while
        # we wait, which is what stops a patient agent holding out forever.
        haz = knob_float("hazard")
        survives = (1.0 - haz) ** n
        best = max(best, (d_me ** (2 * n)) * survives * value)
    return best


def _acceptance_threshold_for_opponent(state: dict, me: str,
                                       d_me: float, d_opp: float) -> float:
    """The smallest share the opponent should rationally accept, as a fraction."""
    opp = _other(me)
    rnd = int(state.get("round") or 1)
    horizon = state.get("max_rounds")
    if horizon and state.get("horizon_known"):
        horizon = int(horizon)
        if rnd >= horizon:
            return 0.0
        proposer_now = state.get("proposer") or me
        g = spe_finite(rnd, horizon, proposer_now, opp, d_opp, d_me)
        return d_opp * g.get(rnd + 1, 0.0)
    return d_opp * spe_infinite(d_opp, d_me)


def _best_offer_under_uncertainty(state: dict, me: str, d_me: float,
                                  posterior: dict[float, float]) -> float:
    """Choose my share to maximize expected payoff over opponent types.

    Each candidate type accepts iff their share clears their own continuation
    value, so acceptance is a step function in my demand. We evaluate my demand
    exactly at each type's indifference point (the most I can extract while
    still being accepted by that type and every less patient one) and take the
    best expected value. Rejection is priced at its true cost: my continuation,
    discounted one round.
    """
    candidates = []
    for d_opp in posterior:
        thresh = _acceptance_threshold_for_opponent(state, me, d_me, d_opp)
        # Leave a sliver above indifference so a fairness-minded opponent agrees.
        candidates.append(max(0.0, min(MAX_DEMAND, 1.0 - thresh - 0.01)))
    candidates.append(0.5)
    candidates = sorted({round(c, 4) for c in candidates if c > 0})

    best_share, best_ev = 0.5, -1.0
    for share in candidates:
        ev = 0.0
        for d_opp, w in posterior.items():
            thresh = _acceptance_threshold_for_opponent(state, me, d_me, d_opp)
            if (1.0 - share) >= thresh - 1e-9:
                ev += w * share
            else:
                ev += w * _continuation_if_i_reject(state, me, d_me, d_opp) * 0.95
        if ev > best_ev:
            best_share, best_ev = share, ev
    return best_share


# -- the policy --------------------------------------------------------------

def solve(game: dict) -> dict:
    state = game["game_state"]
    gid = game.get("game_id")
    me = game.get("your_player") or state.get("current_player")
    money = float(state["money_to_divide"])
    rnd = int(state.get("round") or 1)
    horizon = int(state["max_rounds"]) if (state.get("max_rounds")
                                           and state.get("horizon_known")) else None

    d_me = float(state.get("delta_1" if me == "player_1" else "delta_2") or 1.0)
    posterior = opponent_delta_posterior(state, me)
    d_opp_mean = sum(d * w for d, w in posterior.items())

    atype = game["valid_actions"]["type"]

    if atype == "offer":
        share = current_demand(state, me, d_me, d_opp_mean, posterior, gid)
        msg = _offer_message(state, me, money, share, d_me, d_opp_mean, rnd, horizon)
        return bargaining_offer(state, me, share * money, msg)

    return _decide(game, state, me, money, rnd, horizon, d_me, posterior, gid)


def current_demand(state: dict, me: str, d_me: float, d_opp_mean: float,
                   posterior: dict[float, float] | None = None,
                   gid: str | None = None) -> float:
    """The share we ask for this round: one schedule, used by both branches.

    The accept rule needs this too: the value of holding out depends on what
    we would actually demand if the opponent folded, so the two must agree.
    """
    if posterior is None:
        posterior = {d_opp_mean: 1.0}
    rnd = int(state.get("round") or 1)
    horizon = int(state["max_rounds"]) if (state.get("max_rounds")
                                           and state.get("horizon_known")) else None
    if True:
        share = _best_offer_under_uncertainty(state, me, d_me, posterior)

        # Anchor above target early: the GLEE paper measures a 0.63 correlation
        # between an LLM's first offer and its final payoff against a human.
        if rnd <= 2:
            share = min(MAX_DEMAND, share + (0.10 if d_me >= 0.95 else 0.05))

        # Closing schedule: delay burns the pot, so converge to something
        # acceptable rather than ride an uncapped game into a zero.
        if horizon:
            progress = (rnd - 1) / max(1, horizon - 1)
        else:
            progress = min(1.0, (rnd - 1) / 10.0)
        floor = max(MIN_TARGET, 1.0 - MAX_DEMAND)

        # Converge toward the largest demand a TRUE argument can defend:
        # leaving them just over 0.5 * d_opp means accepting now beats winning
        # an even split next round, by their own arithmetic. Blended toward the
        # SPE share, since that bar alone is safe but soft.
        safe = 1.0 - (0.5 * d_opp_mean + 0.02)
        spe = spe_infinite(d_me, d_opp_mean)
        push = knob_float("strong_push")
        closable = max(floor, min(MAX_DEMAND, safe + push * max(0.0, spe - safe)))

        if d_me < 1.0:
            share = max(floor, share * (1.0 - 0.35 * progress ** 2))
            share = max(share, closable) if share > closable else share
        if horizon:
            # A cap means no agreement pays zero, so blend to the closable
            # number as the deadline nears whatever our patience.
            blend = progress ** 2
            share = (1.0 - blend) * share + blend * closable
        elif d_me >= 1.0:
            # Uncapped and costless to wait: concede only very slowly.
            share = max(closable, share * (1.0 - 0.15 * progress ** 3))
        share = max(floor, min(MAX_DEMAND, share))

        # Delay expensive for me and cheap for them: close early.
        if d_me < 0.95 and d_me < d_opp_mean:
            share = min(share, 0.60)

        # Ground the demand in what they have shown they will pay: asking 90%
        # of a counterpart sitting at 28% only buys more rounds of discounting.
        seq = _their_offers_to_me(state, me)
        if seq and d_me < 1.0:
            share = min(share, max(0.55, seq[-1][1] + 0.25))

        # Centred wobble so no counterpart can key on a fixed schedule, applied
        # last so it survives the clamps above. Only with a real game id:
        # `realistic_continuation` calls this with gid=None to price what the
        # opponent would fold to, and jitter(None, ...) is one fixed draw off
        # the string "nogame", not a wobble.
        if gid is not None:
            # Scale the demand where delay costs us, under the same `gid`
            # guard as the jitter: that is what the probe measured, so the
            # accept path stays unscaled and disagrees slightly, by design.
            if d_me < FREE_WAITING_DELTA:
                share *= knob_float("demand_scale")
            share *= jitter(gid, "barg-anchor", knob_float("demand_jitter"))
        return max(1.0 - MAX_DEMAND, min(MAX_DEMAND, share))


def _decide(game: dict, state: dict, me: str, money: float, rnd: int,
            horizon: int | None, d_me: float,
            posterior: dict[float, float], gid: str | None = None) -> dict:
    d_opp_mean = sum(d * w for d, w in posterior.items())
    offer = state.get("last_offer") or {}
    my_gain = offer.get(f"{me}_gain")
    if my_gain is None:
        return {"decision": "accept"}
    frac = float(my_gain) / money if money else 0.0

    threshold = max(
        sum(w * _continuation_if_i_reject(state, me, d_me, d) for d, w in posterior.items()),
        0.0,
    )

    # Reality check: the SPE threshold is what a rational opponent SHOULD
    # concede, the projection is what this one actually will. Take the smaller
    # of the two once we can see the trend.
    projected = realistic_continuation(state, me, d_me, d_opp_mean)
    if projected is None and knob_on("field_prior"):
        # No history yet, but "no evidence" is not "assume the textbook": fall
        # back to what opponents measurably do. It is a GLOBAL average, so it
        # runs too cheap when we are patient against an impatient opponent, and
        # with `post_floor_cap` below that can leave the floor dead code.
        projected = field_continuation(rnd, d_me)
    if projected is not None:
        threshold = min(threshold, projected)

    # Floor for the PATIENT side. When delay costs us less than it costs them,
    # accepting below the split we are ourselves offering hands back the one
    # weapon we have: without it we realized 0.40 at delta_me = 1.0, where SPE
    # says we take essentially everything. The floor is the closable target,
    # relaxed as a known deadline nears because a no-deal pays zero.
    if d_me >= d_opp_mean - 1e-9:
        safe = 1.0 - (0.5 * d_opp_mean + 0.02)
        spe = spe_infinite(d_me, d_opp_mean)
        push = knob_float("strong_push")
        closable = max(0.0, min(MAX_DEMAND, safe + push * max(0.0, spe - safe)))
        if horizon:
            urgency = min(1.0, max(0.0, (rnd - 1) / max(1, horizon - 1))) ** URGENCY_EXP
        else:
            urgency = min(1.0, (rnd - 1) / 16.0) ** URGENCY_EXP
        floor = closable * (1.0 - urgency) + 0.42 * urgency
        threshold = max(threshold, floor)

    # ...but never hold out for more than we can actually get: a threshold
    # above the achievable continuation just refuses the best offer we will
    # ever see. Applied AFTER the floor it overrides that floor rather than
    # bounding the SPE threshold, which is what `post_floor_cap` controls.
    cap = knob("post_floor_cap")
    if cap == "free_waiting":
        cap_applies = d_me < FREE_WAITING_DELTA
    else:
        cap_applies = cap == "on"
    if projected is not None and cap_applies:
        threshold = min(threshold, projected)

    # Rejecting risks the opponent walking or stalling, so demand a real
    # improvement before turning money down. Sized to the measured risk: over
    # 42,358 logged games, 0.371% ended in anything other than an agreement,
    # which a 1% haircut still covers about threefold.
    threshold *= 0.99
    # ...and wobble it, so repeated probing cannot binary search our exact
    # acceptance point.
    threshold += offset(gid, f"barg-accept-{rnd}", 0.02)

    # The wobble is anti-exploitation noise, not a change of mind: it must
    # never push us into refusing an offer that already beats the continuation
    # we project, which would be a strict loss taken on purpose. Re-clamp after
    # the noise so the wobble can only move the threshold DOWN.
    if projected is not None and cap_applies:
        threshold = min(threshold, projected)

    if horizon and rnd >= horizon:
        # Last round: it is this offer or zero. Take anything non-negative.
        return {"decision": "accept"}

    if frac >= threshold:
        return {"decision": "accept"}
    return {"decision": "reject"}


def beats_their_next_round_even_split(their_share: float, d_opp: float) -> bool:
    """Is our offer worth more to them than a 50/50 one round later?

    Accepting `s` at round t pays them s * d_opp^(t-1); a 50/50 at t+1 pays
    0.5 * d_opp^t. So our offer wins exactly when s > 0.5 * d_opp: against an
    opponent inflating at 20% per round, even 41% beats them waiting for half.

    It is arithmetic the counterpart can check and it is true whenever we
    assert it, so it justifies a split in our favor without bluffing.
    """
    return their_share > 0.5 * d_opp + 1e-9


def _offer_message(state: dict, me: str, money: float, share: float,
                   d_me: float, d_opp: float, rnd: int,
                   horizon: int | None) -> str | None:
    """Arithmetic-forward justification. Every claim here must be TRUE.

    The GLEE paper finds messages improve both efficiency and fairness in
    bargaining, and a checkable numeric argument is what actually moves an LLM
    counterpart. A false claim the opponent can verify costs us the credibility
    that makes the true ones work.
    """
    if not state.get("messages_allowed"):
        return None

    their_share = 1.0 - share
    their_cash = money * their_share
    known_opp = state.get("complete_information") and d_opp is not None
    parts = [f"My proposal: you take {their_cash:,.0f} ({their_share:.0%}), "
             f"I take {money - their_cash:,.0f}."]

    # The strongest true claim, when it holds: waiting for an even split next
    # round leaves them with strictly less than this offer does.
    if d_opp < 1.0 and beats_their_next_round_even_split(their_share, d_opp):
        next_even = 0.5 * d_opp * money
        if known_opp:
            parts.append(
                f"Run the numbers: your money loses {1 - d_opp:.0%} of its value every "
                f"round. Rejecting this to win an even split NEXT round would be worth "
                f"{next_even:,.0f} to you — less than the {their_cash:,.0f} on the "
                f"table right now. Holding out cannot beat accepting."
            )
        else:
            # Same argument, stated as a conditional we can actually defend.
            parts.append(
                f"I cannot see your inflation rate, so check this yourself: if your "
                f"money loses even {1 - d_opp:.0%} a round, then an even split next "
                f"round is worth about {next_even:,.0f} to you — less than the "
                f"{their_cash:,.0f} you can take right now. The worse your rate, the "
                f"more this favours you."
            )
    elif d_opp < 1.0 and d_me >= d_opp:
        if known_opp:
            loss = their_cash * (1.0 - d_opp ** 2)
            parts.append(
                f"The clock is asymmetric: you lose {1 - d_opp:.0%} per round, I lose "
                f"{1 - d_me:.0%}. Two more rounds of this costs you about {loss:,.0f} "
                f"and costs me almost nothing, so delay only moves value from you to me."
            )
        else:
            # Never quote a rate we cannot see: the posterior mean is not even
            # on the grid (the only rates are 20/10/5/0%), and an impossible
            # number told to someone who knows their own rate reads as a bluff.
            parts.append(
                f"I do not know your inflation rate, but I lose {1 - d_me:.0%} per "
                f"round, so I can afford to wait longer than most. If your money is "
                f"losing value at all, this offer is worth more to you now than a "
                f"better split several rounds from now."
            )
    elif d_me < 1.0:
        parts.append("Inflation is eating both of us. A deal now beats a better "
                     "split three rounds from now for both sides.")

    if horizon:
        left = max(0, horizon - rnd)
        parts.append(f"{left} round{'s' if left != 1 else ''} left before we both get nothing.")
    return " ".join(parts)
