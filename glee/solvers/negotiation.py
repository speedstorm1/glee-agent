"""Negotiation (bilateral trade over the price of one indivisible good).

Payoffs: seller gets P - V_seller, buyer gets V_buyer - P, and no deal pays
both zero. Unlike bargaining, this family has no discounting: utility is
constant over time, so delay costs nothing except deadline and walkaway risk.
It is a war of attrition against a deadline, not a Rubinstein game, so
bargaining logic does not transfer.

Valuations come from V = F * M with F in {0.8, 1, 1.2, 1.5}. In 10 of the 16
(F_seller, F_buyer) pairs there is no positive surplus, so the correct play is
to take the zero. The twelve possible valuations (4 factors x 3 bases) are
pairwise distinct, so our own valuation pins down M and the opponent's
valuation is one of four known numbers even when `complete_information` is
false; their own offers narrow that support further.

Concession follows the time-dependent (Boulware) family (Faratin, Sierra &
Jennings 1998): hold near the anchor, concede sharply near the deadline.
"""

from __future__ import annotations

import logging
import os

from ..config import opponent_value_support
from ..rng import jitter

logger = logging.getLogger("glee.negotiation")

BOULWARE_E = 0.18          # <1 => concede late; smaller is more stubborn
INFINITE_HORIZON_PLAN = 14  # rounds we pretend a no-limit game will last
WALK_AWAY_AFTER = 10        # hopeless uncapped games: free the slot


def boulware_e() -> float:
    """Concession exponent, read at call time so it can be overridden.

    Defaults to BOULWARE_E, so an unset environment reproduces the shipped
    policy exactly. 0.18 gives an exponent of 1/0.18 = 5.56, which holds the
    anchor nearly flat until the final proposal; larger values concede earlier.
    """
    raw = os.environ.get("GLEE_NEGO_BOULWARE")
    if raw:
        try:
            v = float(raw)
            if 0.0 < v <= 2.0:
                return v
            logger.warning("GLEE_NEGO_BOULWARE=%r out of range; using %.2f",
                           raw, BOULWARE_E)
        except ValueError:
            logger.warning("GLEE_NEGO_BOULWARE=%r unparseable; using %.2f",
                           raw, BOULWARE_E)
    return BOULWARE_E


def _other(player: str) -> str:
    return "player_2" if player == "player_1" else "player_1"


# --- opponent valuation inference ------------------------------------------

def opponent_value_beliefs(state: dict, me: str) -> tuple[list[float], bool]:
    """Possible opponent valuations, plus whether we know it exactly.

    Returns an empty list when our own valuation is off-grid, which signals
    that the V = F * M hypothesis does not hold for this game; the caller then
    reasons without a support rather than trusting a bogus one.
    """
    opp = _other(me)
    known = state.get(f"{opp}_value")
    if known is not None:
        return [float(known)], True

    my_value = state.get(f"{me}_value")
    if my_value is None:
        return [], False
    support = list(opponent_value_support(float(my_value)))
    if not support:
        return [], False

    my_role = state.get(f"{me}_role")
    opp_is_seller = my_role != "seller"

    # Narrow using the opponent's own offers: a seller naming price P has value
    # <= P, a buyer naming P has value >= P.
    lo, hi = float("-inf"), float("inf")
    for entry in (state.get("history") or []):
        offer = entry.get("offer") or {}
        if offer.get("from_player") != opp:
            continue
        price = offer.get("price")
        if price is None:
            continue
        price = float(price)
        if opp_is_seller:
            hi = min(hi, price)
        else:
            lo = max(lo, price)
    last = state.get("last_offer") or {}
    if last.get("from_player") == opp and last.get("price") is not None:
        price = float(last["price"])
        if opp_is_seller:
            hi = min(hi, price)
        else:
            lo = max(lo, price)

    narrowed = [v for v in support if lo - 1e-9 <= v <= hi + 1e-9]
    return (narrowed or support), False


#: Fraction of the counterpart's surplus we hand back so that accepting is
#: strictly better for them than refusing. Small, but it must not be zero.
SURPLUS_SLIVER = 0.04

#: Exponent on profit when choosing between prices. 1.0 maximizes expected
#: payoff; below 1.0 trades margin for reliability.
PAYOFF_CONCAVITY = 0.5

#: Share of the surplus we are willing to have conceded by our final offer
#: under complete information.
FAIR_SPLIT = 0.5


def shade_inside(price: float, my_value: float, role: str) -> float:
    """Move a price off the indifference point and into the surplus band.

    Theory puts the optimal posted price at a support point, but at exactly the
    counterpart's valuation their surplus is zero, so they refuse. Measured:
    73% of our seller quotes landed on a grid support point and closed zero
    deals across 121 seller games.
    """
    if role == "seller":
        margin = max(0.0, price - my_value) * SURPLUS_SLIVER
        return max(my_value, price - margin)
    margin = max(0.0, my_value - price) * SURPLUS_SLIVER
    return min(my_value, price + margin)


def jittered_anchor(price: float, my_value: float, role: str,
                    gid: str | None) -> float:
    """Shade inside the band, then wobble within the surplus we are conceding.

    The wobble moves how much surplus we hand over, never whether we profit.
    """
    base = shade_inside(price, my_value, role)
    wobble = jitter(gid, "nego-anchor", 0.03)
    if role == "seller":
        return max(my_value, my_value + (base - my_value) * wobble)
    return min(my_value, my_value - (my_value - base) * wobble)


def optimal_posted_price(my_value: float, role: str,
                         support: list[float]) -> float:
    """Take-it-or-leave-it price against a discrete valuation distribution.

    With a prior over the counterpart's valuation a fixed posted price is
    optimal (Harris & Raviv 1981; Riley & Zeckhauser 1983), and over a discrete
    support the optimum is always at a support point. We enumerate, then shade
    just inside it so the type we target actually gains by saying yes.
    """
    if not support:
        return my_value * (1.35 if role == "seller" else 0.7)
    n = len(support)
    best_price, best_score = None, -1.0
    for candidate in sorted(support):
        if role == "seller":
            prob = sum(1 for v in support if v >= candidate - 1e-9) / n
            profit = candidate - my_value
        else:
            prob = sum(1 for v in support if v <= candidate + 1e-9) / n
            profit = my_value - candidate
        if profit <= 0:
            continue
        # Score expected utility, not expected payoff: a seller worth 100
        # facing {80,100,120,150} maximizes EV at 150 (0.25 * 50 = 12.5 beats
        # 0.5 * 20 = 10) but takes a zero three times in four.
        score = prob * (profit ** PAYOFF_CONCAVITY)
        if score > best_score:
            best_price, best_score = candidate, score
    if best_price is None or best_score <= 0:
        return my_value  # nothing profitable exists: quote our own value
    return shade_inside(float(best_price), my_value, role)


# --- the policy -------------------------------------------------------------

def _horizon(state: dict) -> int | None:
    if state.get("horizon_known") and state.get("max_rounds"):
        return int(state["max_rounds"])
    return None


def _progress(state: dict) -> float:
    """How far through our concession schedule we are, in [0, 1].

    Measured against our last chance to propose, not against the round cap.
    Offers alternate, so in a 10-round game round 10 belongs to the opponent's
    decision and round 9 is the final price we ever name.
    """
    rnd = int(state.get("round") or 1)
    horizon = _horizon(state) or INFINITE_HORIZON_PLAN
    if horizon <= 1:
        return 1.0
    last_proposal = max(1, horizon - 1)
    if last_proposal <= 1:
        return 1.0
    return max(0.0, min(1.0, (rnd - 1) / (last_proposal - 1)))


#: Probability a counterpart accepts our last named price. Break-even against
#: conceding to an even split is 51.2% and measured acceptance runs 89.5-96.5%,
#: so this sits conservatively below both.
ULTIMATUM_ACCEPT_P = 0.85


def is_last_proposal(state: dict) -> bool:
    """Is the price we are about to name the last one anybody will name?

    Offers alternate and the final round is a pure accept-or-refuse, so in a
    horizon-T game the last price uttered belongs to whoever proposes on round
    T-1. After it the counterpart chooses between our number and nothing, which
    is an ultimatum however politely it is phrased.
    """
    horizon = _horizon(state)
    if horizon is None:
        return False
    return int(state.get("round") or 1) >= max(1, horizon - 1)


def concession_floor(my_value: float, role: str, support: list[float]) -> float:
    """The worst price still worth trading at, given the discrete support.

    Conceding to our own valuation earns exactly zero, which is what refusing
    earns anyway. Because the counterpart's valuation lies on a known
    four-point grid, the floor is the nearest support point that leaves us a
    positive margin.
    """
    if role == "seller":
        better = [v for v in support if v > my_value + 1e-9]
        # Shade inside the point: at exactly that valuation no buyer gains.
        return shade_inside(min(better), my_value, role) if better else my_value
    better = [v for v in support if v < my_value - 1e-9]
    return shade_inside(max(better), my_value, role) if better else my_value


def _target_price(state: dict, me: str, my_value: float, role: str,
                  support: list[float], certain: bool,
                  gid: str | None = None) -> float:
    """Where we want the price, given how far into the negotiation we are."""
    # Our last named price is a take-it-or-leave-it, so it should be priced
    # like one rather than as the end of a concession schedule.
    if is_last_proposal(state):
        return optimal_posted_price(my_value, role, support)

    floor = concession_floor(my_value, role, support)
    # Vary the shape of the concession curve per game: a fixed exponent lets a
    # counterpart who has met us before predict our ask on every round.
    boulware = boulware_e() * jitter(gid, "nego-curve", 0.25)

    if role == "seller":
        if certain and support:
            anchor = jittered_anchor(support[0], my_value, role, gid)
            reservation = max(my_value, my_value + FAIR_SPLIT * (support[0] - my_value))
        else:
            anchor = jittered_anchor(max(support), my_value, role, gid) if support \
                else my_value * 1.6
            reservation = max(my_value, floor)
        anchor = max(anchor, reservation)
        conceded = anchor - (anchor - reservation) * _progress(state) ** (1.0 / boulware)
        return max(reservation, conceded)

    if certain and support:
        anchor = jittered_anchor(support[0], my_value, role, gid)
        reservation = min(my_value, my_value - FAIR_SPLIT * (my_value - support[0]))
    else:
        anchor = jittered_anchor(min(support), my_value, role, gid) if support \
            else my_value * 0.5
        reservation = min(my_value, floor)
    anchor = min(anchor, reservation)
    conceded = anchor + (reservation - anchor) * _progress(state) ** (1.0 / boulware)
    return min(reservation, conceded)


def _stalemated(state: dict, me: str, my_value: float, role: str) -> bool:
    """Has the counterpart frozen at a price we can never accept?

    Uncapped games have no deadline to force a resolution, so a counterpart
    that repeats the same unprofitable number can burn our concurrency slot
    indefinitely. A walkaway and an endless stalemate both pay zero, so we
    would rather free the slot for a winnable game.
    """
    opp = _other(me)
    prices = []
    for entry in (state.get("history") or []):
        offer = entry.get("offer") or {}
        if offer.get("from_player") == opp and offer.get("price") is not None:
            prices.append(round(float(offer["price"]), 6))
    if len(prices) < 4:
        return False
    recent = prices[-4:]
    if len(set(recent)) > 1:
        return False
    return _profit(role, recent[-1], my_value) <= 0


def _surplus_possible(my_value: float, role: str, support: list[float]) -> bool:
    if not support:
        return True  # unknown: assume a deal might exist rather than stonewall
    if role == "seller":
        return any(v > my_value + 1e-9 for v in support)
    return any(v < my_value - 1e-9 for v in support)


def solve(game: dict) -> dict:
    state = game["game_state"]
    gid = game.get("game_id")
    me = game.get("your_player") or state.get("current_player")
    role = state.get(f"{me}_role") or ("seller" if me == "player_1" else "buyer")
    my_value = float(state[f"{me}_value"])
    atype = game["valid_actions"]["type"]
    rnd = int(state.get("round") or 1)
    horizon = _horizon(state)
    final_round = horizon is not None and rnd >= horizon

    support, certain = opponent_value_beliefs(state, me)
    hopeful = _surplus_possible(my_value, role, support)

    # --- no surplus exists: refuse to book a loss --------------------------
    if not hopeful:
        if atype == "offer":
            msg = ("My reservation price is firm at this number. Below it I am better "
                   "off keeping the item, so there is no version of this deal I can "
                   "improve on." if role == "seller" else
                   "This is the most the item is worth to me. Above it I would be "
                   "paying more than I gain, so I cannot go higher.")
            return _offer(state, my_value, msg)
        offered = state.get("last_offer") or {}
        price = offered.get("price")
        if price is not None and _profit(role, float(price), my_value) > 0:
            return {"decision": "AcceptOffer"}   # our belief was wrong; take it
        if horizon is None and rnd >= WALK_AWAY_AFTER:
            # Uncapped and hopeless. Every outcome here pays zero, so a last
            # offer is a free option: a counterpart optimizing for agreement
            # occasionally takes it. Walk only once it has been declined.
            if not _liquidation_offered(state, me):
                return _offer(state, _liquidation_price(my_value, role),
                              "Final offer. I am at my limit — this is the only "
                              "price that beats walking away for me, and I would "
                              "rather close than have us both take nothing.")
            return {"decision": "WalkAway"}
        if final_round:
            return {"decision": "RejectOffer"}
        return {"decision": "RejectOffer", "product_price": my_value}

    # --- take-it-or-leave-it ------------------------------------------------
    if horizon == 1 and atype == "offer":
        price = optimal_posted_price(my_value, role, support)
        return _offer(state, price, _pitch(role, price, final=True))

    # --- normal play --------------------------------------------------------
    target = _target_price(state, me, my_value, role, support, certain, gid)

    # Frozen counterpart in an uncapped game: nothing left to negotiate over.
    if (horizon is None and atype == "decision"
            and rnd >= WALK_AWAY_AFTER and _stalemated(state, me, my_value, role)):
        return {"decision": "WalkAway"}

    if atype == "offer":
        return _offer(state, target, _pitch(role, target, final=False))

    offered = state.get("last_offer") or {}
    price = offered.get("price")
    if price is None:
        return {"decision": "RejectOffer", "product_price": target}
    price = float(price)
    profit = _profit(role, price, my_value)

    if final_round:
        # Last chance: any positive profit beats the zero we get by refusing.
        return {"decision": "AcceptOffer"} if profit > 0 else {"decision": "RejectOffer"}

    # Accept once their offer is at least as good as what we would ask for next.
    if profit > 0 and _at_least_as_good(role, price, target):
        return {"decision": "AcceptOffer"}

    if profit > 0 and is_last_proposal(state):
        # The last price in the game: weigh their standing offer against the
        # ultimatum discounted by the chance they refuse it.
        if profit >= ULTIMATUM_ACCEPT_P * _profit(role, target, my_value):
            return {"decision": "AcceptOffer"}

    # Counter at our scheduled target rather than clamping to their offer.
    counter = max(my_value, target) if role == "seller" else min(my_value, target)
    return {"decision": "RejectOffer", "product_price": round(counter, 2),
            "message": _pitch(role, counter, final=False)
            if state.get("messages_allowed") else None}


def _liquidation_price(my_value: float, role: str) -> float:
    """The smallest price still strictly better than no deal, for us."""
    step = max(1.0, abs(my_value) * 0.01)
    return my_value + step if role == "seller" else my_value - step


def _liquidation_offered(state: dict, me: str) -> bool:
    """Have we already put the take-it-or-leave-it on the table?"""
    target = _liquidation_price(float(state[f"{me}_value"]),
                               state.get(f"{me}_role") or "seller")
    for entry in (state.get("history") or []):
        offer = entry.get("offer") or {}
        if offer.get("from_player") == me and offer.get("price") is not None:
            if abs(float(offer["price"]) - target) < 1e-6:
                return True
    return False


def _profit(role: str, price: float, my_value: float) -> float:
    return price - my_value if role == "seller" else my_value - price


def _at_least_as_good(role: str, price: float, target: float) -> bool:
    return price >= target - 1e-9 if role == "seller" else price <= target + 1e-9


def _offer(state: dict, price: float, message: str | None) -> dict:
    action = {"product_price": round(max(0.0, float(price)), 2)}
    if message and state.get("messages_allowed"):
        action["message"] = message
    return action


def _pitch(role: str, price: float, final: bool) -> str:
    # A specific figure reads as informed and is harder to counter-anchor.
    if role == "seller":
        base = f"I can do {price:,.2f}."
        why = ("That is what the item is worth against my alternative use for it. "
               "I would rather keep it than go under.")
    else:
        base = f"I can pay {price:,.2f}."
        why = ("That is the most it is worth on my side. Past that I am paying "
               "for the privilege of buying, which I will not do.")
    if final:
        return f"{base} This is a single take-it-or-leave-it offer. {why}"
    return f"{base} {why}"
