"""Offline validation across the whole parameter grid.

Every (family, phase, configuration) combination is exercised here, and every
produced action goes through the same sanitize() the transport uses. Five
invalid moves forfeit a live game, so nothing untested should reach the server.

Run:  python -m pytest tests/ -q      (or)   python tests/test_solvers.py
"""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glee.config import (BARGAINING_DELTAS, BARGAINING_MONEY,
                         NEGOTIATION_BASE, NEGOTIATION_FACTORS,
                         PERSUASION_P, PERSUASION_V, opponent_value_support)
from glee.safety import sanitise
from glee.solvers import bargaining, negotiation, persuasion, strategy

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


# --------------------------------------------------------------- helpers ----

def bargaining_game(delta_1, delta_2, money, horizon, ci, ma, phase,
                    me="player_1", rnd=1, last_offer=None, history=None):
    state = {
        "phase": phase, "current_player": me,
        "proposer": me if phase == "offer" else ("player_2" if me == "player_1" else "player_1"),
        "round": rnd, "money_to_divide": money,
        "delta_1": delta_1, "delta_2": delta_2 if ci else None,
        "horizon_known": horizon is not None,
        "messages_allowed": ma, "complete_information": ci,
        "last_offer": last_offer, "history": history or [],
    }
    if not ci:
        state["delta_1" if me == "player_1" else "delta_2"] = (
            delta_1 if me == "player_1" else delta_2)
        state["delta_2" if me == "player_1" else "delta_1"] = None
    else:
        state["delta_1"], state["delta_2"] = delta_1, delta_2
    if horizon is not None:
        state["max_rounds"] = horizon
    va = ({"type": "offer", "fields": {}} if phase == "offer"
          else {"type": "decision", "fields": {}})
    return {"game_id": "t", "game_family": "bargaining", "your_player": me,
            "phase": phase, "game_state": state, "valid_actions": va, "prompt": ""}


def negotiation_game(v_seller, v_buyer, horizon, ci, ma, phase, me="player_1",
                     rnd=1, last_offer=None, history=None):
    state = {
        "phase": phase, "current_player": me,
        "player_1_role": "seller", "player_2_role": "buyer",
        "round": rnd, "horizon_known": horizon is not None,
        "messages_allowed": ma, "complete_information": ci,
        "last_offer": last_offer, "history": history or [],
    }
    state["player_1_value"] = v_seller
    state["player_2_value"] = v_buyer
    if not ci:
        other = "player_2" if me == "player_1" else "player_1"
        state[f"{other}_value"] = None
    if horizon is not None:
        state["max_rounds"] = horizon
    va = ({"type": "offer", "fields": {}} if phase == "offer"
          else {"type": "decision", "fields": {}})
    return {"game_id": "t", "game_family": "negotiation", "your_player": me,
            "phase": phase, "game_state": state, "valid_actions": va, "prompt": ""}


def persuasion_game(p, v, price, rnd, total, atype, me, quality="high",
                    history=None, seller_msg=None, know_cv=True):
    state = {
        "phase": "seller_message" if "seller" in atype else "buyer_decision",
        "current_player": me, "product_price": price, "p": p, "u": 0.0,
        "round": rnd, "total_rounds": total, "history": history or [],
        "seller_message": seller_msg,
        "seller_message_type": "text" if atype == "seller_message" else "binary",
        "is_seller_know_cv": know_cv,
    }
    if me == "player_2" or know_cv:
        state["v"] = v
    if me == "player_1":
        state["current_quality"] = quality
    return {"game_id": "t", "game_family": "persuasion", "your_player": me,
            "phase": state["phase"], "game_state": state,
            "valid_actions": {"type": atype, "fields": {}}, "prompt": ""}


def validate(game, action):
    """Check the action in the form the server would receive it.

    Asserts on sanitise() output, so an action the sanitizer replaces is
    checked in its replaced form.
    """
    clean = sanitise(game, action)
    family = game["game_family"]
    atype = game["valid_actions"]["type"]
    state = game["game_state"]

    if atype == "offer" and family == "bargaining":
        total = clean["alice_gain"] + clean["bob_gain"]
        money = state["money_to_divide"]
        check(abs(total - money) < 1e-9,
              f"bargaining gains {clean} sum to {total}, need exactly {money}")
        check(clean["alice_gain"] >= -1e-9 and clean["bob_gain"] >= -1e-9,
              f"negative gain in {clean}")
    if atype == "offer" and family == "negotiation":
        check(clean.get("product_price") is not None and clean["product_price"] >= 0,
              f"bad negotiation price {clean}")
    if atype == "decision" and family == "negotiation":
        if clean["decision"] == "RejectOffer":
            final = (state.get("max_rounds") is not None
                     and state["round"] >= state["max_rounds"])
            check(final or "product_price" in clean,
                  f"non-final RejectOffer needs a counteroffer: {clean}")
    for key in ("message",):
        if key in clean:
            check(len(clean[key]) <= 2000, "message too long")
    return clean


# ------------------------------------------------------------- economics ----

def test_spe_formulas():
    # Computed independently from p* = (1-dB)/(1-dA*dB).
    expected = {(0.8, 0.8): 0.5556, (0.9, 0.8): 0.7143, (0.95, 0.8): 0.8333,
                (0.8, 0.9): 0.3571, (0.9, 0.9): 0.5263, (0.8, 0.95): 0.2083,
                (1.0, 0.8): 1.0, (1.0, 0.95): 1.0}
    for (d_me, d_opp), want in expected.items():
        got = bargaining.spe_infinite(d_me, d_opp)
        check(abs(got - want) < 1e-3,
              f"spe_infinite({d_me},{d_opp}) = {got:.4f}, expected {want}")

    # Finite horizon: the last proposer takes all; one round earlier they must
    # leave the responder exactly d_responder.
    g = bargaining.spe_finite(1, 2, "player_1", "player_1", 0.9, 0.8)
    check(abs(g[2] - 1.0) < 1e-9, f"final-round proposer should take all, got {g[2]}")
    check(abs(g[1] - (1 - 0.8 * 1.0)) < 1e-9,
          f"g[1] should be 1 - d_opp = 0.2, got {g[1]}")


def test_lie_rate():
    # q* = p(v-pi)/((1-p)pi); with pi = 1 this is p(v-1)/(1-p).
    for p, v, want in [(1 / 3, 1.2, 0.1), (1 / 3, 2.0, 0.5), (0.5, 1.2, 0.2),
                       (0.5, 1.25, 0.25), (0.8, 1.2, 0.8)]:
        got = persuasion.optimal_lie_rate(p, v, 1.0)
        check(abs(got - want) < 1e-6,
              f"optimal_lie_rate({p:.3f},{v}) = {got:.4f}, expected {want}")
    # p*v >= price => the prior alone sells it, so recommend everything.
    for p, v in [(1 / 3, 4.0), (0.5, 3.0), (0.8, 2.0)]:
        check(persuasion.optimal_lie_rate(p, v, 1.0) == 1.0,
              f"p*v>=price should give q*=1 for p={p},v={v}")
    check(abs(persuasion.optimal_lie_rate(0.5, 2.4, 2.0)
              - persuasion.optimal_lie_rate(0.5, 1.2, 1.0)) < 1e-9,
          "lie rate should depend on v/price only")


def test_value_support_is_distinct():
    values = [f * m for f in NEGOTIATION_FACTORS for m in NEGOTIATION_BASE]
    check(len(set(values)) == 12,
          f"the 12 grid valuations must be distinct, got {len(set(values))}")
    for f, m in itertools.product(NEGOTIATION_FACTORS, NEGOTIATION_BASE):
        support = opponent_value_support(f * m)
        check(len(support) == 4, f"support for V={f*m} should have 4 points")
        check(all(abs(s - g * m) < 1e-6
                  for s, g in zip(support, NEGOTIATION_FACTORS)),
              f"support for V={f*m} should be the same base: {support}")
    check(opponent_value_support(12345.0) == (),
          "off-grid valuation must yield an empty support, not a guess")


# ------------------------------------------------------------ grid sweeps ----

def test_bargaining_grid():
    n = 0
    for d1, d2, money, horizon, ci, ma in itertools.product(
            BARGAINING_DELTAS, BARGAINING_DELTAS, BARGAINING_MONEY,
            (12, None), (True, False), (True, False)):
        for me in ("player_1", "player_2"):
            for rnd in (1, 2, 6, 12):
                if horizon and rnd > horizon:
                    continue
                g = bargaining_game(d1, d2, money, horizon, ci, ma, "offer", me, rnd)
                validate(g, strategy(g))
                n += 1

                offer = {"player_1_gain": money * 0.3, "player_2_gain": money * 0.7,
                         "proposer": "player_2" if me == "player_1" else "player_1",
                         "round": rnd, "message": None}
                g = bargaining_game(d1, d2, money, horizon, ci, ma, "decision",
                                    me, rnd, last_offer=offer)
                validate(g, strategy(g))
                n += 1
    print(f"  bargaining: {n} states exercised")


def test_negotiation_grid():
    n = 0
    for fa, fb, base, horizon, ci, ma in itertools.product(
            NEGOTIATION_FACTORS, NEGOTIATION_FACTORS, NEGOTIATION_BASE,
            (1, 10, None), (True, False), (True, False)):
        vs, vb = fa * base, fb * base
        for me in ("player_1", "player_2"):
            for rnd in (1, 5, 10):
                if horizon and rnd > horizon:
                    continue
                g = negotiation_game(vs, vb, horizon, ci, ma, "offer", me, rnd)
                validate(g, strategy(g))
                n += 1

                probe = (vs + vb) / 2
                offer = {"price": probe, "message": None,
                         "from_player": "player_2" if me == "player_1" else "player_1",
                         "round": rnd}
                g = negotiation_game(vs, vb, horizon, ci, ma, "decision", me,
                                     rnd, last_offer=offer)
                validate(g, strategy(g))
                n += 1
    print(f"  negotiation: {n} states exercised")


def test_persuasion_grid():
    n = 0
    for p, v, price in itertools.product(PERSUASION_P, PERSUASION_V, (1.0, 100.0)):
        for rnd in (1, 4, 10, 20):
            for quality in ("high", "low"):
                for atype in ("seller_recommendation", "seller_message"):
                    g = persuasion_game(p, v * price, price, rnd, 20, atype,
                                        "player_1", quality)
                    validate(g, strategy(g))
                    n += 1
            for msg in ("yes", "no", "I recommend this one.",
                        "I would pass on this one.", None):
                g = persuasion_game(p, v * price, price, rnd, 20,
                                    "buyer_decision", "player_2",
                                    seller_msg=msg)
                validate(g, strategy(g))
                n += 1
    print(f"  persuasion: {n} states exercised")


# ------------------------------------------------------- behavioral checks --

def test_negotiation_refuses_losses():
    """Never accept a trade that books a negative payoff."""
    # Seller value 15000, buyer value 8000 -> no surplus exists.
    g = negotiation_game(15000.0, 8000.0, 10, True, True, "decision", "player_1",
                         rnd=5, last_offer={"price": 9000.0, "from_player": "player_2",
                                            "round": 5, "message": None})
    a = strategy(g)
    check(a["decision"] != "AcceptOffer",
          f"seller must not accept 9000 when its value is 15000: {a}")

    g = negotiation_game(8000.0, 8000.0, 10, True, True, "decision", "player_2",
                         rnd=5, last_offer={"price": 12000.0, "from_player": "player_1",
                                            "round": 5, "message": None})
    a = strategy(g)
    check(a["decision"] != "AcceptOffer",
          f"buyer must not pay 12000 when its value is 8000: {a}")

    g = negotiation_game(8000.0, 15000.0, 10, True, True, "decision", "player_1",
                         rnd=10, last_offer={"price": 12000.0, "from_player": "player_2",
                                             "round": 10, "message": None})
    a = strategy(g)
    check(a["decision"] == "AcceptOffer",
          f"seller must accept 12000 > 8000 on the final round: {a}")


def test_bargaining_never_walks_away():
    for d1, d2, money in itertools.product(BARGAINING_DELTAS, BARGAINING_DELTAS,
                                           BARGAINING_MONEY):
        offer = {"player_1_gain": money * 0.05, "player_2_gain": money * 0.95,
                 "proposer": "player_2", "round": 3}
        g = bargaining_game(d1, d2, money, 12, True, True, "decision",
                            "player_1", 3, last_offer=offer)
        a = strategy(g)
        check(a["decision"] != "walkaway",
              "walking away pays zero and is never right in bargaining")


def test_bargaining_accepts_on_last_round():
    money = 10000
    offer = {"player_1_gain": money * 0.1, "player_2_gain": money * 0.9,
             "proposer": "player_2", "round": 12}
    g = bargaining_game(0.9, 0.9, money, 12, True, True, "decision",
                        "player_1", 12, last_offer=offer)
    a = strategy(g)
    check(a["decision"] == "accept",
          f"last round: 10% beats the zero from refusing, got {a}")


def test_bargaining_patience_is_used():
    """d_me = 1.0 vs d_opp = 0.8 should demand far more than the reverse."""
    money = 10000
    greedy = strategy(bargaining_game(1.0, 0.8, money, None, True, False,
                                      "offer", "player_1", 1))
    meek = strategy(bargaining_game(0.8, 1.0, money, None, True, False,
                                    "offer", "player_1", 1))
    check(greedy["alice_gain"] > meek["alice_gain"] + 0.15 * money,
          f"patient Alice should demand much more: {greedy} vs {meek}")


def test_persuasion_buyer_uses_the_signal():
    """The buyer reacts to the seller's record, not just the prior."""
    p, v, price = 1 / 3, 1.2, 1.0          # p*v = 0.4 < price: trust is scarce
    # Eight honest calls, not five: five puts the posterior at 0.8327 against a
    # break-even of 0.8333, too close to assert either way.
    honest = [{"round": i, "seller_message": "yes", "bought": True,
               "quality": "high"} for i in range(1, 9)]
    g = persuasion_game(p, v, price, 9, 20, "buyer_decision", "player_2",
                        history=honest, seller_msg="yes")
    check(strategy(g)["decision"] == "yes",
          "buyer should buy from a seller with a demonstrated honest record")

    liar = [{"round": i, "seller_message": "yes", "bought": True,
             "quality": "low"} for i in range(1, 6)]
    g = persuasion_game(p, v, price, 6, 20, "buyer_decision", "player_2",
                        history=liar, seller_msg="yes")
    check(strategy(g)["decision"] == "no",
          "buyer must stop buying from a seller caught lying five times")

    # Trivial regime: p*v > price, so buy regardless of what the seller says.
    g = persuasion_game(0.8, 4.0, 1.0, 6, 20, "buyer_decision", "player_2",
                        history=liar, seller_msg="yes")
    check(strategy(g)["decision"] == "yes",
          "p*v > price makes buying profitable on the prior alone")


def test_persuasion_seller_protects_reputation():
    p, v, price = 1 / 3, 1.2, 1.0
    g = persuasion_game(p, v, price, 1, 20, "seller_recommendation",
                        "player_1", quality="low")
    check(strategy(g)["decision"] == "no",
          "seller must not burn a low unit in round 1 of a 20-round game")

    g = persuasion_game(p, v, price, 1, 20, "seller_recommendation",
                        "player_1", quality="high")
    check(strategy(g)["decision"] == "yes", "seller must always push a high unit")

    g = persuasion_game(p, v, price, 20, 20, "seller_recommendation",
                        "player_1", quality="low")
    check(strategy(g)["decision"] == "yes",
          "final round: reputation has no continuation value, so spend it")


# ------------------------------------- regressions from observed live games --

def test_regression_takes_the_early_deal():
    """Accept early when the opponent's concession rate cannot beat discounting.

    They concede about 1% of the pot per offer; delta 0.95 costs 5% a round.
    """
    money = 1_000_000
    seq = [(1, 0.281), (3, 0.290), (5, 0.297), (7, 0.302)]
    history = [{"round": r, "proposer": "player_1",
                "offer": {"player_1_gain": money * (1 - s),
                          "player_2_gain": money * s},
                "decision": "reject"} for r, s in seq[:-1]]
    # Our own counteroffers, all rejected, so the cave-probability estimate has
    # evidence to work from rather than sitting on its prior.
    history += [{"round": r, "proposer": "player_2",
                 "offer": {"player_1_gain": money * 0.45,
                           "player_2_gain": money * 0.55},
                 "decision": "reject"} for r in (2, 4, 6)]
    history.sort(key=lambda e: e["round"])
    last = {"player_1_gain": money * (1 - seq[-1][1]),
            "player_2_gain": money * seq[-1][1],
            "proposer": "player_1", "round": 7}
    g = bargaining_game(None, 0.95, money, None, False, False, "decision",
                        "player_2", 7, last_offer=last, history=history)
    g["game_state"]["delta_2"] = 0.95
    g["game_state"]["delta_1"] = None
    a = strategy(g)
    check(a["decision"] == "accept",
          "must accept when the opponent concedes ~1%/offer and delta is 0.95 "
          f"(waiting is strictly worse), got {a}")

    # With no discounting at all, waiting is free, so keep pushing.
    g2 = bargaining_game(None, 1.0, money, None, False, False, "decision",
                         "player_2", 7, last_offer=last, history=history)
    g2["game_state"]["delta_2"] = 1.0
    g2["game_state"]["delta_1"] = None
    check(strategy(g2)["decision"] == "reject",
          "with delta = 1.0 waiting costs nothing, so a conceding opponent "
          "should be pushed further")


def test_regression_no_zero_margin_offers():
    """The concession floor must pay strictly more than our own value: a price
    at exactly our value is a trade worth nothing, identical to no deal."""
    for base in NEGOTIATION_BASE:
        for fa in NEGOTIATION_FACTORS:
            my_value = fa * base
            support = list(opponent_value_support(my_value))
            floor = negotiation.concession_floor(my_value, "seller", support)
            profitable = [v for v in support if v > my_value]
            if profitable:
                target = min(profitable)
                check(floor > my_value,
                      f"seller floor {floor} must beat own value {my_value}")
                # Strictly inside the band, not on the support point: at the
                # point itself the only buyer who could take it gains zero.
                check(floor < target,
                      f"seller floor {floor} must sit below the support point "
                      f"{target} so the buyer gains something")
                check(floor > my_value + 0.9 * (target - my_value),
                      f"seller floor {floor} concedes too much of the "
                      f"{my_value}->{target} band")
            floor_b = negotiation.concession_floor(my_value, "buyer", support)
            cheaper = [v for v in support if v < my_value]
            if cheaper:
                target_b = max(cheaper)
                check(floor_b < my_value,
                      f"buyer floor {floor_b} must beat own value {my_value}")
                check(floor_b > target_b,
                      f"buyer floor {floor_b} must sit above the support point "
                      f"{target_b} so the seller gains something")

    g = negotiation_game(100.0, None, None, False, True, "offer", "player_1", rnd=14)
    g["game_state"]["player_2_value"] = None
    price = strategy(g)["product_price"]
    check(price > 100.0, f"seller worth 100 must not quote {price}")


def test_regression_walks_away_from_stalemate():
    """Walk away from a counterpart frozen at an unprofitable price in an
    uncapped game, rather than burn a slot for a guaranteed zero."""
    history = [{"round": r, "offer": {"price": 67.15, "from_player": "player_2"},
                "decision": "reject"} for r in (2, 4, 6, 8, 10)]
    g = negotiation_game(100.0, None, None, False, True, "decision", "player_1",
                         rnd=12,
                         last_offer={"price": 67.15, "from_player": "player_2",
                                     "round": 12, "message": None},
                         history=history)
    g["game_state"]["player_2_value"] = None
    a = strategy(g)
    check(a["decision"] == "WalkAway",
          f"a frozen unprofitable counterpart in an uncapped game -> walk, got {a}")

    moving = [{"round": r, "offer": {"price": 60.0 + r, "from_player": "player_2"},
               "decision": "reject"} for r in (2, 4, 6, 8, 10)]
    g = negotiation_game(100.0, None, None, False, True, "decision", "player_1",
                         rnd=12,
                         last_offer={"price": 70.0, "from_player": "player_2",
                                     "round": 12, "message": None},
                         history=moving)
    g["game_state"]["player_2_value"] = None
    check(strategy(g)["decision"] != "WalkAway",
          "must keep negotiating while the counterpart is still conceding")


def test_seller_keeps_buyer_above_breakeven():
    """Realized P(high | recommended) must stay at or above the buyer's price/v.

    Kamenica-Gentzkow is the most dishonest schedule preserving that. The final
    round is excluded: reputation has no continuation value by then.
    """
    import random
    for p, v in [(1 / 3, 1.2), (1 / 3, 1.25), (1 / 3, 2.0), (0.5, 1.2),
                 (0.5, 1.25), (0.8, 1.2), (1 / 3, 3.0), (0.5, 4.0)]:
        q = persuasion.optimal_lie_rate(p, v, 1.0)
        breakeven = 1.0 / v
        if q >= 1.0:
            # p*v >= price: recommending everything is the K-G optimum and
            # realized P(high|rec) is just p. Nothing to ration, so assert that.
            g = persuasion_game(p, v, 1.0, 5, 20, "seller_recommendation",
                                "player_1", quality="low")
            check(strategy(g)["decision"] == "yes",
                  f"p*v >= price (p={p}, v={v}): should push every unit")
            continue
        for seed in range(25):
            rng = random.Random(seed)
            highs = lows = 0
            history: list[dict] = []
            for rnd in range(1, 21):
                is_high = rng.random() < p
                if is_high:
                    rec = True
                elif rnd <= persuasion.TRUST_BUILDING_ROUNDS:
                    rec = False
                elif rnd >= 20:
                    rec = True
                else:
                    rec = persuasion._within_lie_budget(history, q, v, 1.0, rnd, 20)
                if rec and rnd < 20:
                    highs += is_high
                    lows += not is_high
                history.append({"round": rnd, "bought": rec,
                                "seller_message": "yes" if rec else "no",
                                "quality": "high" if is_high else "low"})
            if highs + lows:
                posterior = highs / (highs + lows)
                check(posterior >= breakeven - 1e-6,
                      f"p={p:.3f} v={v} seed={seed}: P(high|rec)={posterior:.3f} "
                      f"fell below the buyer's break-even {breakeven:.3f}")


def test_lows_per_high_identity():
    """lows_per_high must equal v/price - 1 (the prior cancels out)."""
    for v, price in [(1.2, 1.0), (2.0, 1.0), (4.0, 1.0), (240.0, 100.0)]:
        got = persuasion.lows_per_high(v, price, margin=1.0)
        want = v / price - 1.0
        check(abs(got - want) < 1e-9,
              f"lows_per_high({v},{price}) = {got}, expected {want}")
    # Cross-check against the K-G rate: (1-p)q*/p should equal v/price - 1.
    for p in (1 / 3, 0.5, 0.8):
        for v in (1.2, 1.25, 2.0):
            q = persuasion.optimal_lie_rate(p, v, 1.0)
            if q < 1.0:
                check(abs((1 - p) * q / p - (v - 1.0)) < 1e-9,
                      f"K-G identity broken at p={p}, v={v}")


def test_regression_buyer_probe_is_priced():
    """One probe is fine; probing into a hopeless prior is not."""
    # Deep scarce-trust (p*v = 0.4) with almost no rounds left: never probe.
    g = persuasion_game(1 / 3, 1.2, 1.0, 19, 20, "buyer_decision", "player_2",
                        seller_msg="yes")
    check(strategy(g)["decision"] == "no",
          "no information value left in the last rounds -> do not probe")

    burned = [{"round": i, "seller_message": "yes", "bought": True,
               "quality": "low"} for i in range(1, 4)]
    g = persuasion_game(1 / 3, 1.2, 1.0, 4, 20, "buyer_decision", "player_2",
                        history=burned, seller_msg="yes")
    check(strategy(g)["decision"] == "no",
          "three burns should end exploration")


def test_regression_buyer_respects_a_refusal():
    """Skip a unit the seller declines to recommend: a refusal costs them a
    sale, so it is credible."""
    for p, v, price in [(0.5, 200.0, 100.0), (1 / 3, 400.0, 100.0),
                        (0.8, 250.0, 100.0)]:
        g = persuasion_game(p, v, price, 6, 20, "buyer_decision", "player_2",
                            seller_msg="I do not recommend this one.")
        check(strategy(g)["decision"] == "no",
              f"must skip a unit the seller disowns (p={p}, v={v}): "
              f"got {strategy(g)}")
        g = persuasion_game(p, v, price, 6, 20, "buyer_decision", "player_2",
                            seller_msg="I recommend this one.")
        check(strategy(g)["decision"] == "yes",
              f"must take the guaranteed surplus when urged (p={p}, v={v})")


def test_seller_reads_regime_off_the_buyer():
    """With is_seller_know_cv false we cannot see v, but a buyer who buys units
    we declined to recommend has told us p*v >= price."""
    ignoring = [{"round": i, "seller_message": "no", "bought": True,
                 "quality": "low"} for i in (1, 2)]
    check(persuasion._buyer_ignores_our_advice(ignoring),
          "two buys against an explicit refusal should reveal the regime")
    obedient = [{"round": i, "seller_message": "no", "bought": False,
                 "quality": "low"} for i in (1, 2)]
    check(not persuasion._buyer_ignores_our_advice(obedient),
          "a buyer who heeds refusals reveals nothing about v")

    # End to end: hidden v, and the buyer has shown it buys regardless.
    g = persuasion_game(0.5, 200.0, 100.0, 8, 20, "seller_recommendation",
                        "player_1", quality="low", history=ignoring,
                        know_cv=False)
    g["game_state"].pop("v", None)
    check(strategy(g)["decision"] == "yes",
          "hidden v + a buyer who ignores refusals -> recommend everything")


def test_regression_patient_side_does_not_capitulate():
    """At delta_me = 1.0 waiting is costless, so a lowball must be refused.

    Measured over 294 live games we took a mean share of 0.377 there, against
    an SPE that says we should extract nearly everything.
    """
    money = 1_000_000
    seq = [(1, 0.28), (3, 0.29), (5, 0.30)]
    history = [{"round": r, "proposer": "player_1",
                "offer": {"player_1_gain": money * (1 - x),
                          "player_2_gain": money * x},
                "decision": "reject"} for r, x in seq[:-1]]
    last = {"player_1_gain": money * 0.70, "player_2_gain": money * 0.30,
            "proposer": "player_1", "round": 5}

    g = bargaining_game(0.8, 1.0, money, None, True, False, "decision",
                        "player_2", 5, last_offer=last, history=history)
    check(strategy(g)["decision"] == "reject",
          "delta_me=1.0 vs impatient opponent: 30% must not be accepted")

    g = bargaining_game(0.8, 1.0, money, 12, True, False, "offer",
                        "player_2", 8, history=history)
    share = strategy(g)["bob_gain"] / money
    check(share >= 0.62,
          f"patient side should still be demanding a lot at round 8, got {share:.3f}")

    g = bargaining_game(1.0, 0.8, money, None, True, False, "decision",
                        "player_2", 5, last_offer=last, history=history)
    check(strategy(g)["decision"] == "accept",
          "delta_me=0.8 vs patient opponent: take the 30% rather than bleed")


def test_variant_switch_is_read_at_call_time():
    """The A/B arm must be read at call time: run_agent.py sets GLEE_VARIANT
    after importing the solvers."""
    import os
    from glee import solvers

    def offer():
        return strategy(bargaining_game(1.0, 0.8, 10000, 12, True, True,
                                        "offer", "player_1", rnd=3))

    saved = os.environ.get("GLEE_VARIANT")
    try:
        os.environ["GLEE_VARIANT"] = "full"
        full = offer()
        os.environ["GLEE_VARIANT"] = "nomsg"
        nomsg = offer()
        check("message" in full, "full arm should carry a message")
        check("message" not in nomsg, "nomsg arm must strip the message")
        # Numeric play must be identical, or the comparison is meaningless.
        check(full["alice_gain"] == nomsg["alice_gain"],
              f"arms must differ ONLY in text: {full} vs {nomsg}")
        # Persuasion text mode is the move itself and must survive stripping.
        g = persuasion_game(0.5, 300.0, 100.0, 4, 20, "seller_message",
                            "player_1", quality="low")
        check("message" in strategy(g),
              "seller_message is the move; nomsg must not strip it")
    finally:
        if saved is None:
            os.environ.pop("GLEE_VARIANT", None)
        else:
            os.environ["GLEE_VARIANT"] = saved


def test_regression_seller_offers_are_acceptable():
    """Every price we name must leave some grid buyer type strictly better off.

    73% of our live quotes landed exactly on a support point, where the only
    buyer who could take them gains precisely zero.
    """
    bad = []
    for base in NEGOTIATION_BASE:
        support = [f * base for f in NEGOTIATION_FACTORS]
        for fs in NEGOTIATION_FACTORS:
            vs = fs * base
            possible = [v for v in support if v > vs + 1e-9]
            if not possible:
                continue          # no surplus exists; refusing is correct
            for horizon in (1, 10, None):
                for ci in (True, False):
                    for rnd in ([1] if horizon == 1 else [1, max(1, (horizon or 14) - 1)]):
                        g = negotiation_game(vs, None, horizon, ci, True,
                                             "offer", "player_1", rnd=rnd)
                        g["game_state"]["player_2_value"] = None
                        q = strategy(g)["product_price"]
                        if not [v for v in support if v > q + 1e-9]:
                            bad.append((base, fs, horizon, ci, rnd, q))
    check(not bad,
          f"{len(bad)} seller quotes left NO buyer type any surplus, "
          f"e.g. {bad[:3]}")


def test_seller_concedes_before_its_last_turn():
    """Our last quote must drop below the next buyer type on the grid.

    Offers alternate, so in a T-round game our final proposal is round T-1;
    scheduling concession against T itself arrives one round too late.
    """
    vs = 100.0
    for horizon in (10, None):
        last = (horizon - 1) if horizon else 13
        g = negotiation_game(vs, None, horizon, False, True, "offer",
                             "player_1", rnd=last)
        g["game_state"]["player_2_value"] = None
        q = strategy(g)["product_price"]
        check(q < 120.0,
              f"T={horizon}: final quote {q:.1f} must drop below the 120 buyer "
              f"type to capture the reliable deal")
        check(q > vs,
              f"T={horizon}: final quote {q:.1f} must still beat our own value")


def test_pricing_prefers_reliability_over_margin():
    """Posted price targets the reliable buyer type, not the EV-optimal one.

    Seller worth 100 facing {80,100,120,150}: 150 has the higher expected
    payoff (0.25*50 > 0.5*20) but pays zero three times in four.
    """
    p = negotiation.optimal_posted_price(100.0, "seller", [80, 100, 120, 150])
    check(100.0 < p < 120.0,
          f"should target the reliable 120 type (shaded), got {p}")
    p = negotiation.optimal_posted_price(120.0, "buyer", [80, 100, 120, 150])
    check(100.0 < p < 120.0,
          f"buyer should target the reliable 100 type (shaded), got {p}")


def test_patient_side_holds_out_but_not_forever():
    """delta_me = 1.0 must neither capitulate nor hold out forever.

    Both failure modes are guarded: letting a stubborn opponent set the price,
    and running every uncapped game to the cap for a zero.
    """
    money = 1_000_000
    hist = [{"round": k, "proposer": "player_1",
             "offer": {"player_1_gain": money * (1 - 0.28 - 0.01 * k),
                       "player_2_gain": money * (0.28 + 0.01 * k)},
             "decision": "reject"} for k in (1, 3)]

    def accepts(share, rnd, horizon):
        lo = {"player_1_gain": money * (1 - share),
              "player_2_gain": money * share,
              "proposer": "player_1", "round": rnd}
        g = bargaining_game(0.8, 1.0, money, horizon, True, True, "decision",
                            "player_2", rnd, last_offer=lo, history=hist)
        return strategy(g)["decision"] == "accept"

    check(not accepts(0.30, 5, 12),
          "patient side must not take 30% with rounds still in hand")
    # The floor must relax toward the deadline (URGENCY_EXP), and the final
    # round must take anything positive rather than bank a zero. A 0.45 offer
    # at round 11 of 12 is not required to be accepted under free_waiting.
    early = accepts(0.45, 3, 12)
    late = accepts(0.45, 11, 12)
    check(not early, "0.45 must be refused early with the pot intact")
    check(accepts(0.45, 12, 12),
          "the FINAL round must take a positive offer over a certain zero")
    check(late or not early,
          "the floor must be monotonically easier to clear as the cap nears")
    # Uncapped, opponent stalled: waiting is free in the arithmetic, so only
    # the hazard term makes us ever settle.
    stalled = [{"round": k, "proposer": "player_1",
                "offer": {"player_1_gain": money * 0.56,
                          "player_2_gain": money * 0.44},
                "decision": "reject"} for k in (13, 15, 17)]
    lo = {"player_1_gain": money * 0.55, "player_2_gain": money * 0.45,
          "proposer": "player_1", "round": 19}
    g = bargaining_game(0.8, 1.0, money, None, True, True, "decision",
                        "player_2", 19, last_offer=lo, history=stalled)
    check(strategy(g)["decision"] == "accept",
          "a stalled opponent in an uncapped game must be taken, not outwaited")


def test_hazard_makes_waiting_costly():
    from glee.solvers import bargaining as B
    check(0.0 < B.HAZARD_PER_CYCLE < 0.5,
          f"hazard rate {B.HAZARD_PER_CYCLE} should be a small positive number")
    money = 1_000_000
    seq = [(1, 0.30), (3, 0.31)]
    hist = [{"round": r, "proposer": "player_1",
             "offer": {"player_1_gain": money * (1 - x),
                       "player_2_gain": money * x}, "decision": "reject"}
            for r, x in seq]
    st = bargaining_game(0.8, 1.0, money, None, True, True, "decision",
                         "player_2", 5,
                         last_offer={"player_1_gain": money * 0.69,
                                     "player_2_gain": money * 0.31,
                                     "proposer": "player_1", "round": 5},
                         history=hist)["game_state"]
    saved = B.HAZARD_PER_CYCLE
    try:
        B.HAZARD_PER_CYCLE = 0.0
        free = B.realistic_continuation(st, "player_2", 1.0, 0.8)
        B.HAZARD_PER_CYCLE = 0.15
        risky = B.realistic_continuation(st, "player_2", 1.0, 0.8)
    finally:
        B.HAZARD_PER_CYCLE = saved
    check(risky < free,
          f"a hazard must lower the value of waiting: {risky} vs {free}")


def test_frequency_inference_recovers_posterior():
    """P(high|rec) = p / P(rec), recoverable without buying anything.

    Any seller worth modeling recommends every high unit, so P(rec) = p+(1-p)q
    and the prior cancels. The signal is visible even on rounds we pass.
    """
    import random
    for p in (1 / 3, 0.5, 0.8):
        for q in (0.0, 0.25, 0.5, 1.0):
            rng = random.Random(4)
            hist = []
            for i in range(600):
                high = rng.random() < p
                rec = high or (rng.random() < q)
                hist.append({"round": i + 1, "seller_message": "yes" if rec else "no"})
            got = persuasion.signal_frequency_estimate(hist, p, True)
            check(got is not None, f"should estimate at p={p}, q={q}")
            truth = p / (p + (1 - p) * q)
            check(abs(got[0] - truth) < 0.15,
                  f"p={p} q={q}: estimated {got[0]:.3f}, true {truth:.3f}")
    # Too little history to say anything.
    check(persuasion.signal_frequency_estimate(
        [{"round": 1, "seller_message": "yes"}], 0.5, True) is None,
        "must not estimate from a single observation")


def test_no_probing_before_free_information():
    """Never pay for a sample that arrives free a few rounds later.

    A probe costs full price in the scarce regime; the recommendation rate
    identifies the seller by round 4 for free.
    """
    p, v, price = 1 / 3, 200.0, 100.0        # p*v = 66.7 < price: scarce
    for rnd in (1, 2, 3):
        g = persuasion_game(p, v, price, rnd, 20, "buyer_decision", "player_2",
                            history=[{"round": k, "seller_message": "yes",
                                      "bought": False, "quality": None}
                                     for k in range(1, rnd)],
                            seller_msg="yes")
        check(strategy(g)["decision"] == "no",
              f"round {rnd}: must not probe before frequency data exists")

    # A seller who recommends everything carries no information, so the prior
    # decides, and in the scarce regime the prior says pass.
    spam = [{"round": k, "seller_message": "yes", "bought": False,
             "quality": None} for k in range(1, 13)]
    g = persuasion_game(p, v, price, 13, 20, "buyer_decision", "player_2",
                        history=spam, seller_msg="yes")
    check(strategy(g)["decision"] == "no",
          "a seller who recommends every round should never be bought from "
          "in the scarce regime")


def test_liquidation_offer_before_walking():
    """In a hopeless config a walkaway pays zero, so one final offer is a free
    option and must be tabled before walking."""
    history = [{"round": r, "offer": {"price": 60.0, "from_player": "player_2"},
                "decision": "reject"} for r in (2, 4, 6, 8, 10)]
    g = negotiation_game(150.0, None, None, False, True, "decision", "player_1",
                         rnd=12,
                         last_offer={"price": 60.0, "from_player": "player_2",
                                     "round": 12, "message": None},
                         history=history)
    g["game_state"]["player_2_value"] = None
    a = strategy(g)
    check(a.get("product_price") is not None and a.get("decision") != "WalkAway",
          f"should table a final offer before walking, got {a}")
    check(a["product_price"] > 150.0,
          f"the liquidation price must still beat our own value: {a}")

    history2 = history + [{"round": 13,
                           "offer": {"price": 151.5, "from_player": "player_1"},
                           "decision": "reject"}]
    g = negotiation_game(150.0, None, None, False, True, "decision", "player_1",
                         rnd=14,
                         last_offer={"price": 60.0, "from_player": "player_2",
                                     "round": 14, "message": None},
                         history=history2)
    g["game_state"]["player_2_value"] = None
    check(strategy(g)["decision"] == "WalkAway",
          "after the free option is declined, free the slot")


def test_messages_never_assert_hidden_parameters():
    """Never state the opponent's discount rate as fact when it is hidden.

    9% per round is not even on the grid {20,10,5,0}%, so such a claim is
    checkable and false.
    """
    for d_opp in BARGAINING_DELTAS:
        for rnd in (1, 3, 7):
            g = bargaining_game(1.0, d_opp, 10000, 12, False, True, "offer",
                                "player_1", rnd)
            g["game_state"]["delta_2"] = None
            msg = strategy(g).get("message") or ""
            check("The clock is asymmetric: you lose" not in msg,
                  f"unhedged claim about a hidden delta: {msg[:110]}")
            if msg:
                check("I do not know your inflation rate" in msg
                      or "I cannot see your inflation rate" in msg
                      or "Inflation is eating both of us" in msg
                      or "rounds left" in msg,
                      f"expected a hedged message, got: {msg[:110]}")


def test_parses_self_stated_discount_rate():
    """Parse a counterpart's own stated discount rate out of their message.

    Of 390 real opponent messages 75 stated a rate; in the 31 cases where the
    true delta was visible the parse was correct 31 times.
    """
    B = bargaining
    for text, want in [
        ("Every extra round costs me 20% of whatever I end up with", 0.8),
        ("Every extra round costs me 5% of whatever I end up with", 0.95),
        ("I lose 10% per round, so let's close now.", 0.9),
        ("my money shrinks by 10% each round", 0.9),
        ("My discount rate is 5%.", 0.95),
    ]:
        got = B.parse_stated_delta(text)
        check(got == want, f"expected {want} from {text!r}, got {got}")

    # Must not fire on claims about us, often our own message quoted back.
    for text in ("Note the asymmetry: your value drops 20% per round",
                 "you lose 9% per round",
                 "inflation is eating both of us"):
        check(B.parse_stated_delta(text) is None,
              f"must ignore a claim about our side: {text!r}")
    # Off-grid rates are rhetoric, not parameters.
    check(B.parse_stated_delta("costs me 7% per round") is None,
          "off-grid rate must be rejected")


def test_posterior_identifies_opponent_type():
    """The opponent's offer identifies their discount factor: what they demand
    reveals how much they think waiting is worth."""
    money = 1_000_000
    d_me = 0.9
    for true_d in BARGAINING_DELTAS:
        share = bargaining.spe_infinite(true_d, d_me)
        hist = [{"round": r, "proposer": "player_1",
                 "offer": {"player_1_gain": money * share,
                           "player_2_gain": money * (1 - share)},
                 "decision": "reject"} for r in (1, 3, 5)]
        st = {"money_to_divide": money, "delta_1": None, "delta_2": d_me,
              "history": hist,
              "last_offer": {"player_1_gain": money * share,
                             "player_2_gain": money * (1 - share),
                             "proposer": "player_1", "round": 5}}
        post = bargaining.opponent_delta_posterior(st, "player_2")
        best = max(post, key=post.get)
        check(best == true_d,
              f"opponent playing delta={true_d} should be identified, got {best}")
        check(post[true_d] > 0.25,
              f"posterior mass on the true type should beat uniform, "
              f"got {post[true_d]:.2f}")
        check(all(w > 0.0 for w in post.values()),
              "no type may be ruled out entirely -- real players use focal "
              "splits that match no equilibrium")

    st = {"money_to_divide": money, "delta_1": None, "delta_2": 0.9,
          "history": [],
          "last_offer": {"player_1_gain": money * 0.6,
                         "player_2_gain": money * 0.4, "proposer": "player_1",
                         "round": 1,
                         "message": "Every extra round costs me 20% of what I get."}}
    post = bargaining.opponent_delta_posterior(st, "player_2")
    check(post[0.8] > 0.8, f"stated rate should dominate: {post}")
    check(all(w > 0.0 for w in post.values()),
          "a liar must not be able to fully capture the posterior")


def test_randomisation_is_replayable_but_unpredictable():
    """Draws replay within a game and differ across games.

    Replay matters because the solvers are stateless; variation matters
    because we made an identical choice in 91% of repeated states.
    """
    from glee import rng as R

    # Same game id, same draw.
    for salt in ("a", "b", "lie-7"):
        vals = {R.unit("game-xyz", salt) for _ in range(5)}
        check(len(vals) == 1, f"replay broken for salt {salt}: {vals}")

    spread = {R.unit(f"game-{i}", "lie-7") for i in range(50)}
    check(len(spread) > 40, f"draws should differ across games, got {len(spread)}")

    # Jitter is centred, so it costs no expected value.
    xs = [R.jitter(f"g{i}", "anchor", 0.04) for i in range(4000)]
    mean = sum(xs) / len(xs)
    check(abs(mean - 1.0) < 0.004, f"jitter must be centred on 1.0, got {mean:.4f}")
    check(all(0.96 - 1e-9 <= x <= 1.04 + 1e-9 for x in xs), "jitter out of bounds")
    offs = [R.offset(f"g{i}", "acc", 0.02) for i in range(4000)]
    check(abs(sum(offs) / len(offs)) < 0.002, "offset must be centred on zero")

    def offer(gid):
        g = bargaining_game(0.95, 0.9, 10000, 12, True, True, "offer",
                            "player_1", 3)
        g["game_id"] = gid
        return strategy(g)["alice_gain"]

    check(offer("same") == offer("same"), "same game must replay identically")
    shares = {offer(f"g{i}") for i in range(30)}
    check(len(shares) > 20,
          f"anchors should vary across games, got {len(shares)} distinct")


def test_lie_placement_varies_but_rate_holds():
    """Randomise which low units we push, not how many: the buyer's posterior
    only cares about the realized rate, so varying the timing is free."""
    import random as _r
    for p, v in [(1 / 3, 2.0), (0.5, 1.25)]:
        q = persuasion.optimal_lie_rate(p, v, 1.0)
        schedules = []
        highs_rec = lows_rec = 0
        for game in range(120):
            gid = f"game-{game}"
            rng = _r.Random(game)
            hist, pushed = [], set()
            for rnd in range(1, 21):
                high = rng.random() < p
                if high:
                    rec = True
                elif rnd <= persuasion.TRUST_BUILDING_ROUNDS:
                    rec = False
                elif rnd >= 20:
                    rec = True
                else:
                    rec = persuasion._within_lie_budget(hist, q, v, 1.0, rnd, 20, gid)
                if rec and rnd < 20:
                    highs_rec += high
                    lows_rec += not high
                    if not high:
                        pushed.add(rnd)
                hist.append({"round": rnd, "seller_message": "yes" if rec else "no",
                             "bought": rec, "quality": "high" if high else "low"})
            schedules.append(frozenset(pushed))
        posterior = highs_rec / max(1, highs_rec + lows_rec)
        check(posterior >= 1.0 / v - 1e-6,
              f"p={p} v={v}: P(high|rec)={posterior:.3f} below break-even {1/v:.3f}")
        most_common = max(schedules.count(x) for x in set(schedules))
        check(most_common / len(schedules) < 0.6,
              f"p={p} v={v}: {most_common}/{len(schedules)} games share one "
              f"lie schedule -- still predictable")


def test_detects_behave_then_defect():
    """A seller honest early and recommending everything late must be cut off.

    A cumulative average hides it: eight honest rounds then all-yes still reads
    as P(high|rec) = 0.52 against a 0.50 break-even.
    """
    p, v, price = 1 / 3, 200.0, 100.0        # break-even 0.50
    honest_phase = [{"round": r, "seller_message": "yes" if r % 3 == 0 else "no",
                     "bought": r % 3 == 0,
                     "quality": "high" if r % 3 == 0 else None}
                    for r in range(1, 9)]
    defect_phase = [{"round": r, "seller_message": "yes", "bought": False,
                     "quality": None} for r in range(9, 17)]
    g = persuasion_game(p, v, price, 17, 20, "buyer_decision", "player_2",
                        history=honest_phase + defect_phase, seller_msg="yes")
    check(strategy(g)["decision"] == "no",
          "a seller who switched to recommending everything must be cut off, "
          "however good their early record was")

    steady = [{"round": r, "seller_message": "yes" if r % 3 == 0 else "no",
               "bought": r % 3 == 0, "quality": "high" if r % 3 == 0 else None}
              for r in range(1, 17)]
    g = persuasion_game(p, v, price, 17, 20, "buyer_decision", "player_2",
                        history=steady, seller_msg="yes")
    check(strategy(g)["decision"] == "yes",
          "a consistently selective seller should still be bought from")


def test_punishment_tolerates_equilibrium_lying():
    """Punishment is calibrated to q*, not to zero: a seller lying at the
    Kamenica-Gentzkow rate is playing the equilibrium, so only material excess
    over q* counts as exploitation."""
    P = persuasion
    q = P.optimal_lie_rate(1 / 3, 2.0, 1.0)      # q* = 0.5
    at_rate = [{"round": r, "seller_message": "yes", "bought": True,
                "quality": "high" if r % 2 else "low"} for r in range(1, 7)]
    check(P.punishment_rounds(at_rate, q, scarce=True) == 0,
          "lying at exactly q* is equilibrium play and must not be punished")

    all_lies = [{"round": r, "seller_message": "yes", "bought": True,
                 "quality": "low"} for r in range(1, 7)]
    check(P.punishment_rounds(all_lies, q, scarce=True) > 0,
          "burning us on every purchase must trigger a sit-out")
    check(P.punishment_rounds(all_lies, q, scarce=True)
          >= P.punishment_rounds(all_lies, q, scarce=False),
          "the scarce regime should punish at least as hard -- there is no "
          "floor under us there")
    check(P.punishment_rounds([], q, scarce=True) == 0,
          "no purchases yet means nothing to punish")


def test_takes_the_opening_offer_when_the_field_hardens():
    """Take an average opening where delay costs us; hold out where it does not.

    In 1,024 completed bargaining agreements an opponent never once accepted an
    offer of ours, so every deal we closed came from taking theirs.
    """
    money = 1_000_000

    def decide(share, rnd, d_me, d_opp, hist):
        lo = {"player_1_gain": money * (1 - share),
              "player_2_gain": money * share,
              "proposer": "player_1", "round": rnd}
        g = bargaining_game(d_opp, d_me, money, None, True, False, "decision",
                            "player_2", rnd, last_offer=lo, history=hist)
        return strategy(g)["decision"]

    # Take a round-one average offer only where delay costs us. At
    # delta_me = 1.0 holding out measured +0.074 discounted payoff
    # (t = 4.18, n = 193), so that case must reject instead.
    for d_me in (0.8, 0.9, 0.95):
        check(decide(0.426, 1, d_me, 0.9, []) == "accept",
              f"d_me={d_me}: must take a field-average opening offer rather "
              f"than wait for offers that measurably get worse")
    check(decide(0.426, 1, 1.0, 0.9, []) == "reject",
          "d_me=1.0: waiting is costless, so the field-average opening is "
          "NOT good enough -- measured +0.074 for holding out")

    for d_me in (0.9, 1.0):
        check(decide(0.08, 1, d_me, 0.9, []) == "reject",
              f"d_me={d_me}: an 8% opening is worse than the continuation")

    declining = [{"round": r, "proposer": "player_1",
                  "offer": {"player_1_gain": money * (1 - x),
                            "player_2_gain": money * x},
                  "decision": "reject"}
                 for r, x in ((1, 0.42), (3, 0.39))]
    check(decide(0.36, 5, 0.95, 0.9, declining) == "accept",
          "a declining offer sequence means now is the best it gets")


def test_cave_probability_reflects_reality():
    """The cave prior must be small: no opponent has ever accepted our offer."""
    from glee.solvers import bargaining as B
    money = 1_000_000
    # No history: the prior itself must be small.
    p = B._cave_probability({"history": []}, "player_2")
    check(p < 0.06,
          f"cave prior {p:.3f} is too high -- 0 of 1,024 agreements came from "
          f"an opponent accepting our offer")
    hist = [{"round": r, "proposer": "player_2",
             "offer": {"player_1_gain": money * 0.4, "player_2_gain": money * 0.6},
             "decision": "reject"} for r in (2, 4, 6, 8)]
    check(B._cave_probability({"history": hist}, "player_2") < p,
          "repeated rejections of our offers must lower the cave estimate")


def test_bargaining_arm_is_read_at_call_time():
    """The A/B arm must be read at call time: run_agent.py sets GLEE_BARG_ARM
    after the solvers load, and binding it at import makes both arms equal."""
    import os
    money = 1_000_000
    offer = {"player_1_gain": money * 0.574, "player_2_gain": money * 0.426,
             "proposer": "player_1", "round": 1}

    def decide():
        # d_me = 0.9 against d_opp = 0.8: the strong side, but delay still
        # costs us, so the two arms actually differ here. At d_me = 1.0 both
        # hold out and the test could not tell them apart.
        g = bargaining_game(0.8, 0.9, money, None, True, False, "decision",
                            "player_2", 1, last_offer=offer)
        return strategy(g)["decision"]

    saved = os.environ.get("GLEE_BARG_ARM")
    try:
        os.environ["GLEE_BARG_ARM"] = "current"
        cur = decide()
        os.environ["GLEE_BARG_ARM"] = "legacy"
        leg = decide()
        check(cur != leg,
              f"arms must differ on a field-average opening offer: "
              f"current={cur} legacy={leg}")
        check(cur == "accept",
              "the current arm takes the opening offer (offers decline)")
        check(leg == "reject",
              "the legacy arm holds out, which is what we are testing")
    finally:
        if saved is None:
            os.environ.pop("GLEE_BARG_ARM", None)
        else:
            os.environ["GLEE_BARG_ARM"] = saved


def test_demand_jitter_is_a_knob_and_spares_the_accept_path():
    """The demand wobble is tunable, and must not leak into the accept rule.

    realistic_continuation calls current_demand with no game id, and
    jitter(None, ...) is one fixed draw off "nogame" that scales with the
    spread (0.978 at 0.04, 0.889 at 0.20), so widening it could move the
    accept threshold too.
    """
    import os
    import statistics
    from glee.solvers import bargaining as B

    saved_arm = os.environ.get("GLEE_BARG_ARM")
    saved_knobs = os.environ.get("GLEE_BARG_KNOBS")
    try:
        os.environ["GLEE_BARG_ARM"] = "current"
        os.environ.pop("GLEE_BARG_KNOBS", None)
        check(B.knob_float("demand_jitter") == B.DEMAND_JITTER,
              "the knob must track DEMAND_JITTER, not a frozen copy")

        state = bargaining_game(0.9, 0.9, 1_000_000, None, True, False,
                                "offer", "player_1", 1)["game_state"]
        post = {0.9: 1.0}

        # No game id => jitter-free, and therefore insensitive to the spread.
        base = B.current_demand(state, "player_1", 0.9, 0.9, post, None)
        for spread in ("0.04", "0.20", "0.40"):
            os.environ["GLEE_BARG_KNOBS"] = "demand_jitter=%s" % spread
            got = B.current_demand(state, "player_1", 0.9, 0.9, post, None)
            check(abs(got - base) < 1e-12,
                  f"accept-path demand moved with spread {spread}: "
                  f"{got} vs {base}")

        gids = ["g%04d" % i for i in range(400)]
        vals = {}
        for spread in ("0.04", "0.25"):
            os.environ["GLEE_BARG_KNOBS"] = "demand_jitter=%s" % spread
            vals[spread] = [B.current_demand(state, "player_1", 0.9, 0.9,
                                             post, g) for g in gids]
        sd_small = statistics.pstdev(vals["0.04"])
        sd_big = statistics.pstdev(vals["0.25"])
        check(sd_big > 3 * sd_small,
              f"0.25 must widen the demand materially: sd {sd_big:.4f} "
              f"vs {sd_small:.4f}")
        m_small, m_big = statistics.mean(vals["0.04"]), statistics.mean(vals["0.25"])
        check(abs(m_big - m_small) < 0.05 * m_small,
              f"the probe must stay roughly centred: {m_big:.4f} vs {m_small:.4f}")
    finally:
        _restore("GLEE_BARG_ARM", saved_arm)
        _restore("GLEE_BARG_KNOBS", saved_knobs)


def test_demand_scale_only_where_delay_costs_us():
    """Ask for less only when waiting is expensive for us.

    Set by a randomized probe: over 3,503 games a low demand multiplier beat a
    high one by +0.0272 (t=4.07) under incomplete information and +0.0343
    (t=3.40) under complete, but only at d_me < 1.0. Like the jitter, the scale
    sits under the gid guard, so the accept path (gid=None) is untouched.
    """
    import os
    from glee.solvers import bargaining as B

    saved_arm = os.environ.get("GLEE_BARG_ARM")
    saved_knobs = os.environ.get("GLEE_BARG_KNOBS")
    try:
        os.environ["GLEE_BARG_ARM"] = "current"
        os.environ.pop("GLEE_BARG_KNOBS", None)
        check(B.knob_float("demand_scale") == B.DEMAND_SCALE,
              "the knob must track DEMAND_SCALE, not a frozen copy")
        check(0.5 < B.DEMAND_SCALE < 1.0,
              f"DEMAND_SCALE {B.DEMAND_SCALE} should be a modest reduction")

        post = {0.9: 1.0}

        def demand(d_me, gid):
            st = bargaining_game(d_me, 0.9, 1_000_000, None, True, False,
                                 "offer", "player_1", 1)["game_state"]
            return B.current_demand(st, "player_1", d_me, 0.9, post, gid)

        for d_me in (0.8, 0.9, 0.95):
            os.environ["GLEE_BARG_KNOBS"] = "demand_scale=1.0"
            off = demand(d_me, "g1")
            os.environ["GLEE_BARG_KNOBS"] = "demand_scale=0.88"
            on = demand(d_me, "g1")
            check(abs(on / off - 0.88) < 1e-6,
                  f"d_me={d_me}: expected a 0.88x demand, got {on / off:.4f}")
        os.environ["GLEE_BARG_KNOBS"] = "demand_scale=1.0"
        off = demand(1.0, "g1")
        os.environ["GLEE_BARG_KNOBS"] = "demand_scale=0.88"
        on = demand(1.0, "g1")
        check(abs(on - off) < 1e-9,
              f"d_me=1.0 must be untouched: {on} vs {off}")

        for d_me in (0.8, 0.9, 0.95, 1.0):
            os.environ["GLEE_BARG_KNOBS"] = "demand_scale=1.0"
            off = demand(d_me, None)
            os.environ["GLEE_BARG_KNOBS"] = "demand_scale=0.5"
            on = demand(d_me, None)
            check(abs(on - off) < 1e-9,
                  f"accept-path demand moved at d_me={d_me}: {on} vs {off}")
    finally:
        _restore("GLEE_BARG_ARM", saved_arm)
        _restore("GLEE_BARG_KNOBS", saved_knobs)


def test_shipped_defaults_are_the_proven_policy():
    """An unset environment must run the proven defaults, not flags.

    As flags, one dropped --barg-knobs argument would restore post_floor_cap.
    """
    import os
    from glee.solvers import bargaining as B

    saved_arm = os.environ.get("GLEE_BARG_ARM")
    saved_knobs = os.environ.get("GLEE_BARG_KNOBS")
    try:
        os.environ.pop("GLEE_BARG_ARM", None)
        os.environ.pop("GLEE_BARG_KNOBS", None)
        check(B.knob("post_floor_cap") == "free_waiting",
              f"default post_floor_cap is {B.knob('post_floor_cap')!r}, "
              "should be the proven free_waiting")
        check(B.knob_float("demand_scale") == B.DEMAND_SCALE,
              "default demand_scale should be the probed value")
        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=on,demand_scale=1.0"
        check(B.knob("post_floor_cap") == "on"
              and B.knob_float("demand_scale") == 1.0,
              "the pre-change policy must remain reproducible by flag")
    finally:
        _restore("GLEE_BARG_ARM", saved_arm)
        _restore("GLEE_BARG_KNOBS", saved_knobs)


def test_boulware_knob_is_call_time_and_inert_by_default():
    """The negotiation concession exponent is A/B-able and defaults to shipped.

    At BOULWARE_E = 0.18 the T=10 seller ask runs 0.946/0.945/0.935/0.850/0.500
    over rounds 1-9, nothing acceptable mid-game. A larger exponent must concede
    earlier while leaving both endpoints fixed.
    """
    import os
    from glee.solvers import negotiation as NG

    saved = os.environ.get("GLEE_NEGO_BOULWARE")
    v_s, v_b = 10000.0, 15000.0

    def ask(rnd):
        g = negotiation_game(v_s, v_b, 10, True, False, "offer", "player_1", rnd)
        price = strategy(g)["product_price"]
        return (price - v_s) / (v_b - v_s)

    try:
        os.environ.pop("GLEE_NEGO_BOULWARE", None)
        check(NG.boulware_e() == NG.BOULWARE_E,
              "unset env must reproduce the shipped exponent")
        base = {r: ask(r) for r in (1, 5, 7, 9)}

        os.environ["GLEE_NEGO_BOULWARE"] = "0.50"
        check(NG.boulware_e() == 0.50,
              "the override must be read at call time, not bound at import")
        alt = {r: ask(r) for r in (1, 5, 7, 9)}

        # Endpoints pinned: same anchor, same reservation.
        check(abs(alt[1] - base[1]) < 1e-6,
              f"round-1 anchor must not move: {alt[1]:.4f} vs {base[1]:.4f}")
        check(abs(alt[9] - base[9]) < 1e-6,
              f"last-proposal reservation must not move: {alt[9]:.4f} vs {base[9]:.4f}")
        for r in (5, 7):
            check(alt[r] < base[r] - 0.02,
                  f"round {r}: 0.50 must concede earlier than 0.18 "
                  f"({alt[r]:.4f} vs {base[r]:.4f})")

        # Garbage must fall back rather than crash a live turn.
        for bad in ("", "abc", "-1", "99"):
            os.environ["GLEE_NEGO_BOULWARE"] = bad
            check(NG.boulware_e() == NG.BOULWARE_E,
                  f"{bad!r} must fall back to the shipped exponent")
    finally:
        _restore("GLEE_NEGO_BOULWARE", saved)


def test_seller_always_yes_arm():
    """The always_yes seller arm recommends every unit, from round one.

    Always recommending makes the signal uninformative, so P(high | rec) = p
    and a Bayesian buyer buys iff p*v >= price. Pins mechanics, not merits.
    """
    import os
    from glee.solvers import persuasion as PS

    saved = os.environ.get("GLEE_PERS_SELLER")
    try:
        os.environ.pop("GLEE_PERS_SELLER", None)
        check(PS.seller_policy() == "kg", "default seller policy must be kg")

        # A low unit inside the trust-building window, scarce regime: kg has no
        # lie budget to spend, always_yes recommends anyway.
        g = persuasion_game(1 / 3, 1.2, 1.0, 1, 20, "seller_recommendation",
                            "player_1", quality="low")
        check(strategy(g)["decision"] == "no",
              "kg must not push a low unit while building trust")

        os.environ["GLEE_PERS_SELLER"] = "always_yes"
        check(PS.seller_policy() == "always_yes",
              "the arm must be read at CALL time, not bound at import")
        g = persuasion_game(1 / 3, 1.2, 1.0, 1, 20, "seller_recommendation",
                            "player_1", quality="low")
        check(strategy(g)["decision"] == "yes",
              "always_yes must recommend a low unit in round one")

        burned = [{"round": i, "quality": "low", "seller_message": "yes",
                   "buyer_decision": "no", "bought": False}
                  for i in range(1, 12)]
        for regime_v in (1.2, 4.0):
            for rnd in (1, 5, 12, 20):
                g = persuasion_game(1 / 3, regime_v, 1.0, rnd, 20,
                                    "seller_recommendation", "player_1",
                                    quality="low", history=burned[:rnd - 1])
                check(strategy(g)["decision"] == "yes",
                      f"always_yes must recommend at v={regime_v} round {rnd}")

        # Under always_yes, claiming selectivity would be checkably false.
        g = persuasion_game(1 / 3, 1.2, 1.0, 5, 20, "seller_message",
                            "player_1", quality="low", history=burned[:4])
        msg = strategy(g).get("message", "")
        check(msg, "text mode must still produce a message")
        check("stand behind" not in msg and "check that against" not in msg,
              f"always_yes must not claim selectivity: {msg!r}")
    finally:
        _restore("GLEE_PERS_SELLER", saved)


def test_seller_arm_does_not_leak_into_the_buyer():
    """The seller arm must not change buyer play, so any persuasion difference
    is attributable to the seller arm alone."""
    import os

    saved = os.environ.get("GLEE_PERS_SELLER")
    try:
        hist = [{"round": i, "quality": "high" if i % 2 else "low",
                 "seller_message": "yes", "buyer_decision": "yes",
                 "bought": True, "buyer_payoff": 0.2 if i % 2 else -1.0}
                for i in range(1, 8)]
        cases = []
        for v in (1.2, 2.0, 4.0):
            for rnd in (1, 4, 9):
                cases.append(persuasion_game(1 / 3, v, 1.0, rnd, 20,
                                             "buyer_decision", "player_2",
                                             history=hist[:rnd - 1],
                                             seller_msg="yes"))
        os.environ.pop("GLEE_PERS_SELLER", None)
        base = [strategy(g)["decision"] for g in cases]
        os.environ["GLEE_PERS_SELLER"] = "always_yes"
        alt = [strategy(g)["decision"] for g in cases]
        check(base == alt,
              f"buyer decisions must be identical across seller arms: "
              f"{base} vs {alt}")
    finally:
        _restore("GLEE_PERS_SELLER", saved)


def _restore(name, value) -> None:
    import os
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def test_knob_defaults_match_the_module_constants():
    """Knob defaults must match the module constants they duplicate.

    Drift between STRONG_PUSH = 0.45 and the knob default "0.45" would make the
    A/B arm and the shipped policy disagree while every other test passes.
    """
    import os
    from glee.solvers import bargaining as B

    saved_arm = os.environ.get("GLEE_BARG_ARM")
    saved_knobs = os.environ.get("GLEE_BARG_KNOBS")
    try:
        os.environ.pop("GLEE_BARG_ARM", None)
        os.environ.pop("GLEE_BARG_KNOBS", None)
        check(B.knob_float("strong_push") == B.STRONG_PUSH,
              f"strong_push knob {B.knob_float('strong_push')} != "
              f"STRONG_PUSH {B.STRONG_PUSH}")
        check(B.knob_float("hazard") == B.HAZARD_PER_CYCLE,
              f"hazard knob {B.knob_float('hazard')} != "
              f"HAZARD_PER_CYCLE {B.HAZARD_PER_CYCLE}")
        check(not B.legacy(), "unset GLEE_BARG_ARM must mean current")
        for name in B._KNOB_DEFAULTS:
            want = B._KNOB_DEFAULTS[name]["current"]
            want = repr(float(want())) if callable(want) else want
            check(B.knob(name) == want,
                  f"unset env must give the current default for {name}: "
                  f"{B.knob(name)!r} != {want!r}")

        # A knob backed by a module constant must follow that constant rather
        # than a frozen copy, or the constant becomes decorative.
        saved_hazard = B.HAZARD_PER_CYCLE
        try:
            B.HAZARD_PER_CYCLE = 0.123
            check(B.knob_float("hazard") == 0.123,
                  "the hazard knob must track HAZARD_PER_CYCLE, not a copy")
        finally:
            B.HAZARD_PER_CYCLE = saved_hazard
        saved_push = B.STRONG_PUSH
        try:
            B.STRONG_PUSH = 0.321
            check(B.knob_float("strong_push") == 0.321,
                  "the strong_push knob must track STRONG_PUSH, not a copy")
        finally:
            B.STRONG_PUSH = saved_push
    finally:
        _restore("GLEE_BARG_ARM", saved_arm)
        _restore("GLEE_BARG_KNOBS", saved_knobs)


def test_knobs_override_the_arm_at_call_time():
    """GLEE_BARG_KNOBS layers on top of whichever arm is selected, and is read
    at call time: run_agent.py sets it after the solver modules import."""
    import os
    from glee.solvers import bargaining as B

    saved_arm = os.environ.get("GLEE_BARG_ARM")
    saved_knobs = os.environ.get("GLEE_BARG_KNOBS")
    try:
        os.environ["GLEE_BARG_ARM"] = "current"
        os.environ.pop("GLEE_BARG_KNOBS", None)
        check(B.knob("post_floor_cap") == "free_waiting",
              "current now defaults to the proven free_waiting")
        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=off"
        check(B.knob("post_floor_cap") == "off",
              "the knob override must be read at call time, not at import")
        os.environ["GLEE_BARG_ARM"] = "legacy"
        os.environ["GLEE_BARG_KNOBS"] = "strong_push=0.45"
        check(B.knob_float("strong_push") == 0.45,
              "an explicit override must win over the legacy default")
        check(not B.knob_on("post_floor_cap"),
              "an unmentioned knob must keep its arm default")
    finally:
        _restore("GLEE_BARG_ARM", saved_arm)
        _restore("GLEE_BARG_KNOBS", saved_knobs)


def test_patient_side_floor_is_not_dead_code():
    """The floor must be reachable, not that holding out is correct.

    With post_floor_cap=on the achievability cap runs after the floor and
    erases it: at delta_me = 1.0 against an opponent losing 20% a round we
    compute a demand of 0.880 and a floor of 0.769, then accept 0.420.
    """
    import os

    money = 1_000_000
    # 0.50 sits above the continuation cap and well below the 0.769 floor, so
    # it separates "cap overrides floor" from "floor binds".
    offer = {"player_1_gain": money * 0.50, "player_2_gain": money * 0.50,
             "proposer": "player_2", "round": 1}

    def decide():
        g = bargaining_game(1.0, 0.8, money, None, True, False, "decision",
                            "player_1", 1, last_offer=offer)
        return strategy(g)["decision"]

    saved_arm = os.environ.get("GLEE_BARG_ARM")
    saved_knobs = os.environ.get("GLEE_BARG_KNOBS")
    try:
        os.environ["GLEE_BARG_ARM"] = "current"
        # free_waiting is the default now, so ask for the old behavior by name.
        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=on"
        check(decide() == "accept",
              "documents the defect: post_floor_cap=on accepts 0.50 as the "
              "patient side despite computing a floor of 0.769")

        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=on,strong_push=0.0"
        no_push = decide()
        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=on,strong_push=0.9"
        max_push = decide()
        check(no_push == max_push == "accept",
              "while the cap overrides the floor, strong_push cannot matter")

        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=off"
        check(decide() == "reject",
              "with the cap off, the patient side must defend its floor")
        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=off,strong_push=0.0"
        check(decide() == "reject",
              "even the legacy floor of 0.58 beats an offer of 0.50")
    finally:
        _restore("GLEE_BARG_ARM", saved_arm)
        _restore("GLEE_BARG_KNOBS", saved_knobs)


def test_floor_binds_only_where_waiting_is_free():
    """free_waiting gates the floor on absolute, not relative, patience.

    Discounted payoff, legacy minus current: +0.005 at d_me 0.80, -0.022 at
    0.90, -0.012 at 0.95, +0.074 (t = 4.18) at 1.00. So the floor binds at
    d_me = 1.0 but not at 0.95, still the relatively more patient side.
    """
    import os

    money = 1_000_000
    # Above the continuation cap and below the 0.769 floor, so it separates
    # "cap overrides floor" from "floor binds".
    offer = {"player_1_gain": money * 0.50, "player_2_gain": money * 0.50,
             "proposer": "player_2", "round": 1}

    def decide(d_me, d_opp):
        g = bargaining_game(d_me, d_opp, money, None, True, False, "decision",
                            "player_1", 1, last_offer=offer)
        return strategy(g)["decision"]

    saved_arm = os.environ.get("GLEE_BARG_ARM")
    saved_knobs = os.environ.get("GLEE_BARG_KNOBS")
    try:
        os.environ["GLEE_BARG_ARM"] = "current"
        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=free_waiting"
        check(decide(1.0, 0.8) == "reject",
              "waiting is free at d_me=1.0, so the floor must bind")
        check(decide(1.0, 0.95) == "reject",
              "still free at d_me=1.0 even against a patient opponent")
        # Strong side, but delay is not free: the extra share would be eaten.
        check(decide(0.95, 0.8) == "accept",
              "at d_me=0.95 the floor must NOT bind: measured wash (t=-0.74)")
        check(decide(0.9, 0.8) == "accept",
              "at d_me=0.90 the floor must NOT bind: measured negative")

        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=on"
        check(decide(1.0, 0.8) == "accept", "`on` never lets the floor bind")
        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=off"
        check(decide(0.95, 0.8) == "reject",
              "`off` binds on relative patience, which is what we narrowed")
    finally:
        _restore("GLEE_BARG_ARM", saved_arm)
        _restore("GLEE_BARG_KNOBS", saved_knobs)


def test_free_waiting_still_respects_a_deadline():
    """A capped game must still close, since no deal pays zero: the URGENCY_EXP
    ramp decays the floor toward 0.42 and must survive the free_waiting gate."""
    import os

    money = 1_000_000

    def decide(rnd, frac):
        offer = {"player_1_gain": money * frac,
                 "player_2_gain": money * (1 - frac),
                 "proposer": "player_2", "round": rnd}
        g = bargaining_game(1.0, 0.8, money, 12, True, False, "decision",
                            "player_1", rnd, last_offer=offer)
        return strategy(g)["decision"]

    saved_arm = os.environ.get("GLEE_BARG_ARM")
    saved_knobs = os.environ.get("GLEE_BARG_KNOBS")
    try:
        os.environ["GLEE_BARG_ARM"] = "current"
        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=free_waiting"
        check(decide(12, 0.05) == "accept",
              "final round: anything beats a zero")
        early, late = decide(2, 0.50), decide(11, 0.50)
        check(early == "reject",
              "early in a capped game the floor should still hold at 0.50")
        check(late == "accept",
              "near the deadline the urgency ramp must let us close")
    finally:
        _restore("GLEE_BARG_ARM", saved_arm)
        _restore("GLEE_BARG_KNOBS", saved_knobs)


def test_weak_side_is_untouched_by_the_floor_knobs():
    """The floor applies only when we are at least as patient as they are, so
    the floor knobs must leave weak-side play untouched."""
    import os

    money = 1_000_000
    offer = {"player_1_gain": money * 0.42, "player_2_gain": money * 0.58,
             "proposer": "player_2", "round": 1}

    def decide():
        # delta_me = 0.8 against delta_opp = 1.0: strictly the weak side.
        g = bargaining_game(0.8, 1.0, money, None, True, False, "decision",
                            "player_1", 1, last_offer=offer)
        return strategy(g)["decision"]

    saved_arm = os.environ.get("GLEE_BARG_ARM")
    saved_knobs = os.environ.get("GLEE_BARG_KNOBS")
    try:
        os.environ["GLEE_BARG_ARM"] = "current"
        os.environ.pop("GLEE_BARG_KNOBS", None)
        base = decide()
        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=off"
        check(decide() == base,
              "the floor knobs must not change weak-side play")
        os.environ["GLEE_BARG_KNOBS"] = "post_floor_cap=off,strong_push=0.9"
        check(decide() == base,
              "strong_push must not reach the weak side either")
    finally:
        _restore("GLEE_BARG_ARM", saved_arm)
        _restore("GLEE_BARG_KNOBS", saved_knobs)


# ----------------------------------------------- persuasion: buyer rules ----


def test_a_refusal_is_an_absolute_veto():
    """A seller refusal vetoes the buy, however good their record looks.

    We bought 4,280 units against a refusal; 2.31% were high, realizing -0.93
    each. Break-even needs price/v, 25% at the richest grid cell.
    """
    for p, v, price in ((1 / 3, 400.0, 100.0), (0.8, 125.0, 100.0),
                        (0.5, 400.0, 100.0)):
        # A record designed to drive the posterior as high as it will go.
        hot = [{"round": r, "seller_message": "no", "bought": True,
                "quality": "high"} for r in range(1, 13)]
        g = persuasion_game(p, v, price, 13, 20, "buyer_decision", "player_2",
                            history=hot, seller_msg="no")
        check(strategy(g)["decision"] == "no",
              f"p={p} v={v}: a refusal must veto however good the record looks")


def test_endgame_falls_back_to_the_prior_not_to_refusal():
    """Late recommendations carry almost nothing, but they are not poison.

    P(high | recommended) falls from 72.7% over rounds 1-18 to 60.8% at 20, so
    the record dies; round-20 recommend-buys still realized +0.54.
    """
    spotless = [{"round": r, "seller_message": "yes", "bought": True,
                 "quality": "high"} for r in range(1, 19)]
    # p*v > price: the prior alone pays, so buy even though the signal is dead.
    g = persuasion_game(0.8, 300.0, 100.0, 20, 20, "buyer_decision",
                        "player_2", history=spotless, seller_msg="yes")
    check(strategy(g)["decision"] == "yes",
          "endgame with p*v > price must still buy on the prior")

    # p*v < price: a spotless record must NOT carry us over the line at 20.
    g = persuasion_game(1 / 3, 200.0, 100.0, 20, 20, "buyer_decision",
                        "player_2", history=spotless, seller_msg="yes")
    check(strategy(g)["decision"] == "no",
          "endgame with p*v < price must fall back to the prior and pass, "
          "however clean the record is")


def test_seller_never_withholds_when_the_prior_alone_sells():
    """p*v >= price means there is no reputation to protect.

    The buyer profits from buying blind, so selectivity costs sales even during
    trust-building and after trust breaks.
    """
    for rnd in (1, 2, 3, 10):
        g = persuasion_game(0.5, 200.0, 100.0, rnd, 20, "seller_recommendation",
                            "player_1", quality="low")
        check(strategy(g)["decision"] == "yes",
              f"round {rnd}: p*v >= price must recommend, even while young")

    cold = [{"round": r, "seller_message": "yes", "bought": False,
             "quality": "low"} for r in range(1, 8)]
    g = persuasion_game(0.5, 200.0, 100.0, 8, 20, "seller_recommendation",
                        "player_1", quality="low", history=cold)
    check(strategy(g)["decision"] == "yes",
          "p*v >= price: a low trust reading must not stop us recommending")

    g = persuasion_game(1 / 3, 120.0, 100.0, 2, 20, "seller_recommendation",
                        "player_1", quality="low")
    check(strategy(g)["decision"] == "no",
          "p*v < price must still buy credibility before spending it")


def test_hidden_v_is_inferred_from_burn_tested_behaviour():
    """With v hidden, infer the regime from behavior after the buyer is burned.

    An honest seller earns a high buy rate in every regime, so only a rate
    sustained after the buyer eats a lemon is evidence about v.
    """
    P = persuasion
    check(P.infer_v_ratio([], 0.8) < 1.25,
          "with no evidence the opening estimate must stay below the trivial "
          "threshold at every prior")

    trusting = [{"round": r, "seller_message": "yes", "bought": True,
                 "quality": "high"} for r in range(1, 9)]
    check(P.infer_v_ratio(trusting, 0.8) < 1.25,
          "a high buy rate with no burn is trust, not evidence about v")

    burned_on = [{"round": 1, "seller_message": "yes", "bought": True,
                  "quality": "low"}] + [
        {"round": r, "seller_message": "yes", "bought": True, "quality": "high"}
        for r in range(2, 10)]
    check(P.infer_v_ratio(burned_on, 0.8) >= 1.25,
          "still buying after a lemon identifies the trivial regime")

    burned_off = [{"round": 1, "seller_message": "yes", "bought": True,
                   "quality": "low"}] + [
        {"round": r, "seller_message": "yes", "bought": False, "quality": None}
        for r in range(2, 10)]
    check(P.infer_v_ratio(burned_off, 0.8) <= 1.3,
          "a buyer who stops after a lemon is telling us v is small")

    defied = [{"round": r, "seller_message": "no", "bought": True,
               "quality": "low"} for r in (1, 2)]
    check(P.infer_v_ratio(defied, 1 / 3) >= 3.0,
          "buying against an explicit refusal proves p*v >= price")


def test_percentile_tiebreak_is_bounded():
    """The percentile tie-break may only speak near indifference, with
    evidence, and off a real reference pool."""
    P = persuasion
    st = {"p": 0.8, "v": 125.0, "product_price": 100.0, "u": 0.0,
          "seller_message_type": "binary", "is_seller_know_cv": True,
          "buyer_total_payoff": 0.0}
    check(P._percentile_verdict(st, 0.8, 125.0, 0.0, 100.0, 20, 21, 0.8, []) is None,
          "must abstain once no rounds remain")
    check(P._percentile_verdict(st, 0.8, 125.0, 0.0, 0.0, 20, 5, 0.8, []) is None,
          "must abstain on a degenerate price rather than divide by it")
    off_grid = P.refpool.lookup(0.61, 137.0, 100.0, "binary", True, "buyer")
    check(off_grid is None, "a configuration off the published grid has no pool")

    # Banked winnings must be read from history when the server field is absent.
    hist = [{"round": r, "seller_message": "yes", "bought": True,
             "quality": "high"} for r in range(1, 6)]
    bare = {k: val for k, val in st.items() if k != "buyer_total_payoff"}
    check(abs(P._banked_payoff(bare, hist, 125.0, 0.0, 100.0) - 125.0) < 1e-9,
          "banked payoff must be reconstructable from history alone")


def test_family_daily_cap_throttles_without_starving():
    """Cap one family on one agent without starving it.

    Two properties at once: the daily total is bounded, and starts are paced.
    """
    import collections
    import threading
    import time as _time
    from glee.transport import Agent

    a = Agent.__new__(Agent)          # no network, no __init__
    a.family_daily_cap = {"negotiation": 70}
    a.cap_above = None
    a._starts = collections.defaultdict(collections.deque)
    a._counted = set()
    a._inflight_lock = threading.Lock()
    a._ratings = {}
    a._auto_capped = set()
    a.label = "test"

    check(not a._family_capped("negotiation"),
          "an idle capped family must be allowed to start")
    check(not a._family_capped("bargaining"),
          "a family with no cap must never be throttled")

    a._note_start("g0", "bargaining")
    check(len(a._starts["bargaining"]) == 0,
          "an uncapped family must not even be tracked")

    a._note_start("g0", "negotiation")
    check(a._family_capped("negotiation"),
          "pacing: a second game must not start immediately after the first")

    # Fill the day's allowance, all of it comfortably in the past.
    a._starts["negotiation"].clear()
    a._counted.clear()
    old = _time.time() - 3600.0
    for i in range(70):
        a._note_start(f"n{i}", "negotiation")
    for i in range(70):
        a._starts["negotiation"][i] = old
    check(a._family_capped("negotiation"),
          "the daily quota must bind even when pacing would allow a start")

    a._note_start("n0", "negotiation")     # duplicate id
    check(len(a._starts["negotiation"]) == 70,
          "a game must be counted once, however many turns it takes")

    # Age most of them out of the 24h window.
    for i in range(60):
        a._starts["negotiation"][i] = _time.time() - 86400.0 - 60.0
    check(not a._family_capped("negotiation"),
          "the window must roll: yesterday's games cannot block today's")


def test_auto_cap_banks_a_high_rating_and_releases_a_low_one():
    """Engage the cap once a family crosses cap_above, release it on a clear
    fall. The release margin is hysteresis, so a value on the line cannot flap
    between two policies."""
    import collections
    import threading
    from glee.transport import Agent

    a = Agent.__new__(Agent)
    a.family_daily_cap = {}
    a.cap_above = (3000.0, 70)
    a._starts = collections.defaultdict(collections.deque)
    a._counted = set()
    a._inflight_lock = threading.Lock()
    a._ratings = {}
    a._auto_capped = set()
    a.label = "test"

    def scores(**kw):
        return {"scores": {k: {"rating": v, "games_played": 1}
                           for k, v in kw.items()}}

    a._note_ratings(scores(negotiation=2900, persuasion=2100))
    check(a._effective_cap("negotiation") is None,
          "below the threshold nothing is throttled")

    a._note_ratings(scores(negotiation=3010, persuasion=2100))
    check(a._effective_cap("negotiation") == 70,
          "crossing the threshold must engage the cap without being asked")
    check(a._effective_cap("persuasion") is None,
          "the cap is per family, not per agent")

    a._note_ratings(scores(negotiation=2920, persuasion=2100))
    check(a._effective_cap("negotiation") == 70,
          "hysteresis: a small dip must not release the cap and restart the "
          "convergence we are trying to stop")

    a._note_ratings(scores(negotiation=2840, persuasion=2100))
    check(a._effective_cap("negotiation") is None,
          "a clear fall must release it: below its level, volume helps us")

    a._note_ratings(scores(negotiation=2840, persuasion=3200))
    check(a._effective_cap("persuasion") == 70,
          "any family that spikes must be caught, not just the watched one")

    a.family_daily_cap = {"persuasion": 40}
    check(a._effective_cap("persuasion") == 40,
          "the tighter of a standing and an automatic cap must win")
    a.family_daily_cap = {"persuasion": 90}
    check(a._effective_cap("persuasion") == 70,
          "...in whichever direction that happens to be")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        before = len(FAILURES)
        try:
            fn()
        except Exception as e:  # a crash is a failure too
            FAILURES.append(f"{fn.__name__} raised {type(e).__name__}: {e}")
        status = "ok " if len(FAILURES) == before else "FAIL"
        print(f"[{status}] {fn.__name__}")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s):")
        for f in FAILURES[:40]:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
