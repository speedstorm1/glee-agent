"""Record everything. We are scored against a distribution we cannot see.

Our game rating is a percentile against every payoff earned on the same
configuration in the same role. That distribution is invisible to us, so the
only way to tune for it is to reconstruct it empirically from our own play:
log every game's configuration, role, action sequence and final payoff, then
later fit "what payoff do I need on this config to clear the 75th percentile?"

The logs are the input to those fits. Two streams:
  moves.jsonl  one row per action we submit (includes the observed state)
  games.jsonl  one row per completed game (config, role, payoff, outcome)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("glee.telemetry")


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


class Telemetry:
    def __init__(self, log_dir: str = "logs", agent_label: str = "main"):
        os.makedirs(log_dir, exist_ok=True)
        self.agent_label = agent_label
        self._moves_path = os.path.join(log_dir, f"{agent_label}.moves.jsonl")
        self._games_path = os.path.join(log_dir, f"{agent_label}.games.jsonl")
        self._ratings_path = os.path.join(log_dir, f"{agent_label}.ratings.jsonl")
        self._advice_path = os.path.join(log_dir, f"{agent_label}.advice.jsonl")
        self._lock = threading.Lock()
        # Per-game scratch space so a completed game can be written with its
        # whole history, not just its last frame.
        self._open: dict[str, dict] = {}

    def _append(self, path: str, row: dict) -> None:
        line = json.dumps(row, default=_jsonable, separators=(",", ":"))
        with self._lock:
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                logger.exception("telemetry write failed")

    def record_move(self, game: dict, action: dict, result: dict | None,
                    error: str | None = None, latency_ms: float | None = None) -> None:
        gid = game.get("game_id", "?")
        state = game.get("game_state", {}) or {}
        row = {
            "ts": time.time(),
            "agent": self.agent_label,
            "game_id": gid,
            "family": game.get("game_family"),
            "your_player": game.get("your_player"),
            "phase": game.get("phase"),
            "action_type": (game.get("valid_actions") or {}).get("type"),
            "round": state.get("round"),
            "state": state,
            "action": action,
            "result": result,
            "error": error,
            "latency_ms": latency_ms,
        }
        self._append(self._moves_path, row)
        with self._lock:
            rec = self._open.setdefault(gid, {"moves": 0, "first_ts": time.time()})
            rec["moves"] += 1
            rec["family"] = game.get("game_family")
            rec["your_player"] = game.get("your_player")
            rec["config"] = extract_config(game.get("game_family"), state)
            rec["last_state"] = state

    def record_rating(self, stats: dict, after_game: str | None = None) -> None:
        """Snapshot the per-family rating and game count.

        Our payoff is converted to a percentile against the reference
        distribution for this configuration and role, which we are never shown.
        The rating update is `delta_R = eta * (game_rating - R)`, so sampling
        the rating either side of a completed game inverts back to the
        percentile that game earned. Paired with the config we logged.
        """
        row = {
            "ts": time.time(),
            "agent": self.agent_label,
            "after_game": after_game,
            "active_games": stats.get("active_games"),
            "scores": stats.get("scores") or {},
        }
        self._append(self._ratings_path, row)

    def record_advice(self, game: dict, message: str | None, nudge: float,
                      reason: object, latency_s: float) -> None:
        """Log every LLM intervention so an advised game can be reconstructed.

        Without this the advisor would be an unlogged source of variation in
        the payoff measurements.
        """
        self._append(self._advice_path, {
            "ts": time.time(), "agent": self.agent_label,
            "game_id": game.get("game_id"), "family": game.get("game_family"),
            "round": (game.get("game_state") or {}).get("round"),
            "message": message, "nudge": nudge, "reason": reason,
            "latency_s": round(latency_s, 3),
        })

    def record_game_end(self, game: dict, result: dict) -> None:
        gid = game.get("game_id", "?")
        with self._lock:
            rec = self._open.pop(gid, {}) or {}
        state = game.get("game_state", {}) or {}
        me = game.get("your_player")
        payoffs = result.get("result") or {}
        my_payoff = payoffs.get(f"{me}_payoff") if isinstance(payoffs, dict) else None
        row = {
            "ts": time.time(),
            "agent": self.agent_label,
            "game_id": gid,
            "family": game.get("game_family") or rec.get("family"),
            "your_player": me,
            "config": rec.get("config") or extract_config(game.get("game_family"), state),
            "moves_made": rec.get("moves"),
            "duration_s": (time.time() - rec["first_ts"]) if rec.get("first_ts") else None,
            "final_round": state.get("round"),
            "outcome": payoffs.get("outcome") if isinstance(payoffs, dict) else None,
            "my_payoff": my_payoff,
            "payoffs": payoffs,
        }
        self._append(self._games_path, row)
        logger.info("game %s (%s as %s) -> payoff %s [%s]", gid[:8],
                    row["family"], me, my_payoff, row["outcome"])


def extract_config(family: str | None, state: dict) -> dict:
    """The configuration key our percentile is computed against.

    Only parameters that define the *environment* belong here -- not the
    realized history -- because the scoring joins on configuration and role.
    """
    if not state:
        return {}
    common = {
        "complete_information": state.get("complete_information"),
        "messages_allowed": state.get("messages_allowed"),
        "horizon_known": state.get("horizon_known"),
        "max_rounds": state.get("max_rounds"),
    }
    if family == "bargaining":
        return {**common,
                "money_to_divide": state.get("money_to_divide"),
                "delta_1": state.get("delta_1"),
                "delta_2": state.get("delta_2")}
    if family == "negotiation":
        return {**common,
                "player_1_value": state.get("player_1_value"),
                "player_2_value": state.get("player_2_value")}
    if family == "persuasion":
        return {**common,
                "p": state.get("p"), "v": state.get("v"), "u": state.get("u"),
                "product_price": state.get("product_price"),
                "total_rounds": state.get("total_rounds"),
                "seller_message_type": state.get("seller_message_type"),
                "is_seller_know_cv": state.get("is_seller_know_cv"),
                # Present in the paper's grid; may or may not be exposed here.
                "buyer_type": state.get("buyer_type")}
    return common
