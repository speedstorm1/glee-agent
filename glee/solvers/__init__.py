"""Dispatcher: one solver per game family.

One function per family plus a small dispatcher, so each family can be tuned
without touching the others and a single run loop plays all three.
"""

from __future__ import annotations

import logging
import os

from . import bargaining, negotiation, persuasion

logger = logging.getLogger("glee.solvers")

SOLVERS = {
    "bargaining": bargaining.solve,
    "negotiation": negotiation.solve,
    "persuasion": persuasion.solve,
}


#: Optional LLM advisor, installed by run_agent.py. None means pure solver.
ADVISOR = None


def set_advisor(advisor) -> None:
    global ADVISOR
    ADVISOR = advisor


def variant() -> str:
    """Message arm for A/B testing, read at CALL time.

    "full" sends the solvers' generated text; "nomsg" strips every optional
    message while leaving all numeric decisions identical, which isolates what
    the verbal layer is actually worth. Persuasion text-mode messages are never
    stripped -- there the message IS the move.

    Read per call, not at import: run_agent.py sets GLEE_VARIANT after these
    modules are imported, so binding it at import time silently pinned every
    arm to the default and produced an A/B where both sides were identical.
    """
    return os.environ.get("GLEE_VARIANT", "full")


def strategy(game: dict) -> dict:
    family = game.get("game_family")
    solver = SOLVERS.get(family)
    if solver is None:
        raise ValueError(f"unknown game family: {family!r}")
    action = solver(game)
    if variant() == "nomsg" and isinstance(action, dict):
        if game.get("valid_actions", {}).get("type") != "seller_message":
            action.pop("message", None)
        return action

    # Optional LLM pass: wording, plus a clamped nudge to the offer. Only where
    # messages are allowed, and only in the two families where the GLEE paper
    # and our own A/B find that text helps. It degrades persuasion.
    if (ADVISOR is not None and ADVISOR.available() and family != "persuasion"
            and isinstance(action, dict)
            and (game.get("game_state") or {}).get("messages_allowed")):
        try:
            action = _advise(game, family, action)
        except Exception:
            logger.exception("advisor pass failed; keeping solver action")
    return action


def _advise(game: dict, family: str, action: dict) -> dict:
    from ..advisor import summarise_for_advice
    from .bargaining import _opponent_messages

    state = game["game_state"]
    me = game.get("your_player")
    try:
        opp_msgs = _opponent_messages(state, me)
    except Exception:
        opp_msgs = []

    if family == "bargaining" and "alice_gain" in action:
        money = float(state.get("money_to_divide") or 0) or 1.0
        mine = action["alice_gain"] if me == "player_1" else action["bob_gain"]
        plan = f"offer them {(1 - mine / money):.0%} of {money:,.0f}"
    elif family == "negotiation" and action.get("product_price") is not None:
        plan = f"price at {action['product_price']:,.2f}"
    else:
        plan = f"respond with {action.get('decision')}"

    summary = summarise_for_advice(game, plan, opp_msgs,
                                   draft=action.get("message"))
    message, nudge = ADVISOR.advise(game, summary, action.get("message"))
    if message:
        action["message"] = message[:1990]

    # Apply the nudge only to an OFFER, never to accept/reject.
    if nudge:
        if family == "bargaining" and "alice_gain" in action:
            from ..safety import bargaining_offer
            money = float(state.get("money_to_divide") or 0) or 1.0
            mine = action["alice_gain"] if me == "player_1" else action["bob_gain"]
            share = max(0.05, min(0.95, mine / money + nudge))
            action = bargaining_offer(state, me, share * money, action.get("message"))
        elif family == "negotiation" and action.get("product_price") is not None:
            role = state.get(f"{me}_role")
            my_value = state.get(f"{me}_value")
            price = float(action["product_price"]) * (1.0 + nudge)
            if my_value is not None:
                # Never let a nudge push us into a loss-making trade.
                price = (max(float(my_value), price) if role == "seller"
                         else min(float(my_value), price))
            action["product_price"] = round(max(0.0, price), 2)
    return action
