"""Optional LLM side-role: message wording, plus a tightly bounded nudge.

The hard constraint shapes everything here. A turn we fail to answer within
120 s is scored at the FIFTH PERCENTILE, and three of them in a row earn a
30-minute queue ban. So a network call anywhere near the move path is only
acceptable if it categorically cannot delay a move past the deadline.

The envelope:

  * HARD TIMEOUT of a few seconds, an order of magnitude inside the turn clock.
    On timeout, error, or malformed output we use the deterministic message and
    carry on. There is no path where a slow model costs us a turn.
  * TEXT ONLY, plus a nudge to our OFFER TARGET clamped to a few percent. The
    accept/reject rule stays entirely deterministic: every large loss and gain
    in this project has come from that rule, and it is not somewhere to put a
    nondeterministic component.
  * CIRCUIT BREAKER. Consecutive failures or slow calls disable the advisor for
    a cooling-off period, so a degraded API degrades our text and nothing else.
  * OFF BY DEFAULT, enabled per agent, killable by environment variable.

Everything it returns is logged with the game id so any advised game can be
reconstructed and compared against the deterministic arm.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

logger = logging.getLogger("glee.advisor")

#: Dedicated pool so a stuck advisor call can never occupy a move worker.
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="advisor")

#: Well inside the 120 s turn clock. If the model cannot answer in this long
#: it has nothing useful to add anyway.
TIMEOUT_S = float(os.environ.get("GLEE_ADVISOR_TIMEOUT", "12"))
#: The API refuses deadlines under 10s, so that is the floor we can ask the
#: server for. We enforce our own cap on top, in a worker thread, so that even
#: an SDK that hangs entirely cannot hold a turn open.
_API_DEADLINE_MS = max(10_000, int(TIMEOUT_S * 1000))
#: The most the model may move our target, in either direction.
#:
#: Set to zero by measurement. Over a 15.4-hour A/B the advisor showed no
#: benefit overall (difference-in-differences on opponent-adjusted percentile:
#: -0.020, t = -0.86, using untouched families as a placebo channel), and the
#: subset of games that actually received a nudge scored 0.389 against 0.451
#: for message-only. n = 18 is weak evidence, but it points the same way as the
#: overall estimate, and the nudge is the one component that reaches into the
#: numeric decision. With no measured upside it is set to zero.
#:
#: Message generation stays on: it is neutral in the same test and is our only
#: adaptive response to counterparts we have never seen.
MAX_NUDGE = float(os.environ.get("GLEE_MAX_NUDGE", "0.0"))
#: Consecutive failures before we stop calling it.
BREAKER_TRIPS = 4
BREAKER_COOLDOWN_S = 600.0
#: The advisor sits inside a turn budget, and a turn timeout is scored at the
#: 5th percentile, so latency is a scoring concern rather than a comfort. We
#: therefore ran the smallest, fastest model available rather than the most
#: capable: on our own prompt the small one answered in well under a second
#: against several seconds for the larger one, which
#: moves the worst case from "a third of the timeout" to "noise". Message
#: quality is not a live constraint: the advisor is measured as null on payoff
#: across five outcome measures, so it is bought for adaptivity and speed, not
#: for prose.
MODEL = os.environ.get("GLEE_ADVISOR_MODEL", "")  # set to a small, fast model

_SYSTEM = """You advise a competitive agent in a language-based economic game.

The agent's NUMERIC strategy is already computed by an exact game-theoretic
solver and is almost certainly right. You are not being asked to replace it.
You have two jobs:

1. IMPROVE the agent's drafted message. You will be shown it. It usually
   contains concrete arithmetic -- the counterpart's own inflation rate, what
   waiting actually costs them, exact figures. That arithmetic is the whole
   reason the message works, and it is measurably more persuasive than
   generic goodwill.

   So: KEEP every number and every verifiable claim from the draft. Improve
   the framing, address anything specific the counterpart said, and make it
   read like a person rather than a template. Do NOT replace a concrete
   argument with a vague one such as "let us work together" -- that is
   strictly worse than the draft and you should return the draft unchanged
   before doing it.

   Never assert a value that is hidden from us, and never state a figure that
   is false. Two or three sentences.
2. Optionally suggest a SMALL adjustment to the agent's target, in [-0.05,
   +0.05] as a fraction. Use it only when the counterpart's messages give a
   concrete reason (an explicit deadline, a stated reservation price, obvious
   desperation, a credible final offer). Default to 0.0.

Scoring note: the agent is graded on a percentile against how everyone else
did in its exact seat, not on beating this opponent. A deal slightly worse than
optimal beats no deal, because no deal scores near the bottom.

Reply with JSON only: {"message": str, "adjust": float, "reason": str}
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "adjust": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["message", "adjust"],
}


class Advisor:
    """Thread-safe, fail-open LLM helper. Never raises into the move path."""

    def __init__(self, enabled: bool = False, telemetry=None):
        self.enabled = enabled and bool(os.environ.get("GEMINI_API_KEY"))
        self.telemetry = telemetry
        self._client = None
        self._lock = threading.Lock()
        self._fails = 0
        self._blocked_until = 0.0
        self.calls = 0
        self.failures = 0
        self.nudges_applied = 0
        self.total_latency = 0.0

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client

    def available(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            return time.monotonic() >= self._blocked_until

    def _trip(self, why: str) -> None:
        with self._lock:
            self._fails += 1
            self.failures += 1
            if self._fails >= BREAKER_TRIPS:
                self._blocked_until = time.monotonic() + BREAKER_COOLDOWN_S
                self._fails = 0
                logger.warning("advisor circuit breaker OPEN for %.0fs (%s)",
                               BREAKER_COOLDOWN_S, why)

    def _reset(self) -> None:
        with self._lock:
            self._fails = 0

    def advise(self, game: dict, summary: str,
               fallback_message: str | None) -> tuple[str | None, float]:
        """Return (message, nudge). Falls back silently on any problem."""
        if not self.available():
            return fallback_message, 0.0

        started = time.monotonic()
        try:
            def _call():
                return self._get_client().models.generate_content(
                    model=MODEL,
                    contents=summary,
                    config={
                        "system_instruction": _SYSTEM,
                        "response_mime_type": "application/json",
                        "response_schema": _SCHEMA,
                        "temperature": 1.0,
                        "http_options": {"timeout": _API_DEADLINE_MS},
                    },
                )

            # Belt and braces: the server deadline bounds the request, and this
            # bounds everything else -- DNS, retries inside the SDK, a hung
            # socket. A move must never wait on any of it.
            fut = _POOL.submit(_call)
            try:
                resp = fut.result(timeout=TIMEOUT_S)
            except FuturesTimeout:
                fut.cancel()
                self._trip(f"hard timeout at {TIMEOUT_S}s")
                return fallback_message, 0.0
            elapsed = time.monotonic() - started

            data = json.loads(resp.text or "{}")
            message = str(data.get("message") or "").strip() or fallback_message
            try:
                nudge = float(data.get("adjust") or 0.0)
            except (TypeError, ValueError):
                nudge = 0.0
            # Clamp hard. The model does not get to move our target far, and a
            # NaN or a wild value must not propagate into a live offer.
            if nudge != nudge or abs(nudge) > 1.0:
                nudge = 0.0
            import glee.advisor as _self
            nudge = max(-_self.MAX_NUDGE, min(_self.MAX_NUDGE, nudge))

            self._reset()
            self.calls += 1
            self.total_latency += elapsed
            if nudge:
                self.nudges_applied += 1
            if self.telemetry:
                self.telemetry.record_advice(game, message, nudge,
                                             data.get("reason"), elapsed)
            return message, nudge
        except Exception as e:            # fail open, always
            self._trip(repr(e)[:80])
            logger.debug("advisor failed: %s", e)
            return fallback_message, 0.0


def summarise_for_advice(game: dict, our_target: str,
                         opponent_messages: list[str],
                         draft: str | None = None) -> str:
    """Compact, factual brief. Only what the model can legitimately use."""
    state = game.get("game_state") or {}
    family = game.get("game_family")
    me = game.get("your_player")
    lines = [f"GAME: {family}", f"You are: {me}",
             f"Round: {state.get('round')} of {state.get('max_rounds') or 'unknown'}",
             f"Complete information: {state.get('complete_information')}",
             f"Our solver's plan: {our_target}"]
    if family == "bargaining":
        lines += [f"Money to divide: {state.get('money_to_divide')}",
                  f"Our inflation per round: "
                  f"{state.get('delta_1' if me == 'player_1' else 'delta_2')}",
                  f"Their inflation per round (None = hidden from us): "
                  f"{state.get('delta_2' if me == 'player_1' else 'delta_1')}"]
    elif family == "negotiation":
        lines += [f"Our role: {state.get(f'{me}_role')}",
                  f"Our valuation: {state.get(f'{me}_value')}",
                  f"Their valuation (None = hidden): "
                  f"{state.get('player_2_value' if me == 'player_1' else 'player_1_value')}"]
    if opponent_messages:
        lines.append("Their recent messages:")
        lines += [f"  - {m[:300]}" for m in opponent_messages[-3:]]
    else:
        lines.append("They have sent no messages.")
    if draft:
        lines += ["", "OUR DRAFTED MESSAGE (keep its numbers and claims; "
                      "improve the framing):", draft]
    return "\n".join(lines)
