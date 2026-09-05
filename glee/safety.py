"""Never lose a game to our own bug.

Two failure modes are far more expensive than any strategic mistake:

  * A turn timeout (120 s) ends the game as a no-deal AND is scored at the 5th
    percentile -- the bottom of the scale. Three consecutive self-timeouts also
    trigger a 30-minute queue ban ("crash-loop cooldown").
  * Five invalid moves in one game does the same thing.

So every action leaving this process passes through `sanitize()`, and any
strategy exception is replaced by `fallback_action()` rather than propagating.
It is always better to play a mediocre legal move than to play nothing.
"""

from __future__ import annotations

import logging
import math

from .config import MAX_MESSAGE_LEN

logger = logging.getLogger("glee.safety")

BARGAINING_DECISIONS = {"accept", "reject", "walkaway"}
NEGOTIATION_DECISIONS = {"AcceptOffer", "RejectOffer", "WalkAway"}
YES_NO = {"yes", "no"}


def _clean_message(msg: object) -> str | None:
    if msg is None:
        return None
    text = str(msg).strip()
    if not text:
        return None
    if len(text) > MAX_MESSAGE_LEN:
        # The server rejects an oversized message as an invalid move and never
        # truncates for us, so we must do it here.
        text = text[: MAX_MESSAGE_LEN - 1].rsplit(" ", 1)[0]
    return text


def split_exactly(money: float, my_gain: float) -> tuple[float, float]:
    """Split `money` into two parts that sum to it EXACTLY.

    The server rejects an offer whose two gains don't sum to `money_to_divide`,
    and float arithmetic on values like 1e6 makes that easy to get wrong. We
    round one side and derive the other by subtraction so the identity holds by
    construction.
    """
    my_gain = max(0.0, min(float(money), float(my_gain)))
    if float(money).is_integer():
        mine = int(round(my_gain))
        mine = max(0, min(int(money), mine))
        return float(mine), float(int(money) - mine)
    mine = round(my_gain, 2)
    return mine, round(float(money) - mine, 2)


def bargaining_offer(state: dict, my_player: str, my_gain: float,
                     message: str | None = None) -> dict:
    """Build a legal bargaining offer from *my* desired gain.

    The action keys are always alice_gain/bob_gain (Alice = player_1,
    Bob = player_2) regardless of which side we are, so we map here once.
    """
    money = float(state["money_to_divide"])
    mine, theirs = split_exactly(money, my_gain)
    if my_player == "player_1":
        action = {"alice_gain": mine, "bob_gain": theirs}
    else:
        action = {"alice_gain": theirs, "bob_gain": mine}
    text = _clean_message(message)
    if text and state.get("messages_allowed"):
        action["message"] = text
    return action


def sanitise(game: dict, action: dict) -> dict:
    """Last line of defence: coerce `action` into something the server accepts.

    Returns a legal action for the current phase. Anything unrecognisable is
    replaced wholesale by the fallback.
    """
    if not isinstance(action, dict) or not action:
        return fallback_action(game)

    family = game.get("game_family")
    atype = game.get("valid_actions", {}).get("type")
    state = game.get("game_state", {})
    out: dict = {}

    try:
        if atype == "offer" and family == "bargaining":
            money = float(state["money_to_divide"])
            alice = action.get("alice_gain")
            bob = action.get("bob_gain")
            if alice is None and bob is None:
                return fallback_action(game)
            if alice is None:
                alice = money - float(bob)
            alice = float(alice)
            if not math.isfinite(alice):
                return fallback_action(game)
            a, b = split_exactly(money, alice)
            out = {"alice_gain": a, "bob_gain": b}

        elif atype == "offer" and family == "negotiation":
            price = action.get("product_price")
            if price is None or not math.isfinite(float(price)):
                return fallback_action(game)
            out = {"product_price": max(0.0, round(float(price), 2))}

        elif atype == "decision" and family == "bargaining":
            d = str(action.get("decision", "")).lower()
            if d not in BARGAINING_DECISIONS:
                return fallback_action(game)
            out = {"decision": d}

        elif atype == "decision" and family == "negotiation":
            d = str(action.get("decision", ""))
            match = {k.lower(): k for k in NEGOTIATION_DECISIONS}.get(d.lower())
            if match is None:
                return fallback_action(game)
            out = {"decision": match}
            if match == "RejectOffer":
                price = action.get("product_price")
                # On the final round of a capped game a rejection takes no
                # counteroffer; elsewhere one is required.
                if price is not None and math.isfinite(float(price)):
                    out["product_price"] = max(0.0, round(float(price), 2))
                elif not _is_final_round(state):
                    return fallback_action(game)

        elif atype in ("seller_recommendation", "buyer_decision"):
            d = str(action.get("decision", "")).lower()
            if d not in YES_NO:
                return fallback_action(game)
            out = {"decision": d}

        elif atype == "seller_message":
            text = _clean_message(action.get("message")) or "I recommend this product."
            out = {"message": text}

        else:
            return fallback_action(game)

    except (TypeError, ValueError, KeyError):
        logger.exception("sanitise failed; using fallback")
        return fallback_action(game)

    if atype != "seller_message" and state.get("messages_allowed"):
        text = _clean_message(action.get("message"))
        if text:
            out["message"] = text
    return out


def _is_final_round(state: dict) -> bool:
    mx = state.get("max_rounds")
    rd = state.get("round")
    return mx is not None and rd is not None and int(rd) >= int(mx)


def fallback_action(game: dict) -> dict:
    """A guaranteed-legal, conservative move for any phase.

    Deliberately biased toward *closing* rather than stalling: a no-deal pays
    zero, which is near the bottom of every payoff distribution, so when we
    don't know what to do we take the deal.
    """
    family = game.get("game_family")
    atype = game.get("valid_actions", {}).get("type")
    state = game.get("game_state", {})

    if atype == "offer":
        if family == "bargaining":
            money = float(state.get("money_to_divide", 0) or 0)
            a, b = split_exactly(money, money / 2)
            return {"alice_gain": a, "bob_gain": b}
        me = game.get("your_player") or state.get("current_player")
        my_value = state.get(f"{me}_value")
        role = state.get(f"{me}_role")
        if my_value is None:
            return {"product_price": 1.0}
        # Offer our own valuation: a zero-profit but never-loss-making price.
        return {"product_price": max(0.0, round(float(my_value), 2))}

    if atype == "seller_message":
        return {"message": "This product is available at the listed price."}

    if atype == "seller_recommendation":
        return {"decision": "yes"}

    if atype == "buyer_decision":
        # Only buy when the prior alone makes it profitable; never gamble blind.
        try:
            p = float(state["p"]); v = float(state["v"])
            u = float(state.get("u", 0.0) or 0.0)
            price = float(state["product_price"])
            return {"decision": "yes" if p * v + (1 - p) * u >= price else "no"}
        except (KeyError, TypeError, ValueError):
            return {"decision": "no"}

    if atype == "decision":
        if family == "bargaining":
            return {"decision": "accept"}
        # Negotiation: accept only if the price is actually profitable for us,
        # otherwise reject. Accepting a loss-making trade is worse than no deal.
        try:
            me = game.get("your_player") or state.get("current_player")
            role = state[f"{me}_role"]
            my_value = float(state[f"{me}_value"])
            price = float(state["last_offer"]["price"])
            good = price >= my_value if role == "seller" else price <= my_value
            if good:
                return {"decision": "AcceptOffer"}
            if _is_final_round(state):
                return {"decision": "RejectOffer"}
            return {"decision": "RejectOffer", "product_price": my_value}
        except (KeyError, TypeError, ValueError):
            return {"decision": "AcceptOffer"}

    return {"decision": "accept"}
