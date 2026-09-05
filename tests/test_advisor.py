"""Safety envelope for the LLM side-role. No network calls in this suite.

The advisor is the only component that can introduce unbounded latency into
the move path, and a turn we fail to answer inside 120 s is scored at the
FIFTH percentile with three in a row earning a queue ban. So the properties
that matter are not "is the advice good" but "can it ever hurt us":

  * it fails OPEN -- any error, timeout or malformed reply yields the
    deterministic message and a zero nudge;
  * its nudge is CLAMPED, including against NaN and absurd values;
  * a nudge can never push a negotiation price into a loss-making trade;
  * repeated failures trip a breaker so a degraded API degrades our wording
    and nothing else.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glee.advisor import MAX_NUDGE, Advisor  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


class _Boom:
    def __init__(self, exc=None, payload=None, delay=0.0):
        self.exc, self.payload, self.delay = exc, payload, delay
        self.models = self

    def generate_content(self, **kw):
        import time
        if self.delay:
            time.sleep(self.delay)
        if self.exc:
            raise self.exc
        class R:
            text = self.payload
        return R()


def _adv(client) -> Advisor:
    a = Advisor(enabled=True)
    a.enabled = True
    a._client = client
    return a


def test_disabled_by_default():
    a = Advisor()
    check(not a.enabled, "advisor must be off unless explicitly enabled")
    msg, nudge = a.advise({}, "summary", "DETERMINISTIC")
    check(msg == "DETERMINISTIC" and nudge == 0.0,
          "a disabled advisor must pass the deterministic message straight through")


def test_fails_open_on_every_error():
    for exc in (RuntimeError("boom"), ValueError("bad"), KeyError("k")):
        a = _adv(_Boom(exc=exc))
        msg, nudge = a.advise({}, "s", "DETERMINISTIC")
        check(msg == "DETERMINISTIC" and nudge == 0.0,
              f"must fall back on {type(exc).__name__}")
    # Malformed JSON must not propagate either.
    for payload in ("not json", "", "{", '{"adjust": "abc"}', "null"):
        a = _adv(_Boom(payload=payload))
        msg, nudge = a.advise({}, "s", "DETERMINISTIC")
        check(nudge == 0.0, f"malformed payload {payload!r} produced nudge {nudge}")


def test_nudge_disabled_by_measurement():
    """MAX_NUDGE is 0 by default: the A/B found no benefit and the nudged
    subset scored below message-only. Nothing the model returns may move a
    number while that holds."""
    for raw in (0.5, -0.5, 0.02, 1e9, float("nan")):
        a = _adv(_Boom(payload='{"message":"m","adjust":%r}' % raw))
        _, nudge = a.advise({}, "s", "D")
        check(nudge == 0.0, f"adjust={raw} must be neutralised, got {nudge}")


def test_nudge_is_clamped():
    import glee.advisor as A
    saved = A.MAX_NUDGE
    A.MAX_NUDGE = 0.05          # verify the clamp still works if re-enabled
    try:
        _clamp_cases()
    finally:
        A.MAX_NUDGE = saved


def _clamp_cases():
    from glee.advisor import MAX_NUDGE
    for raw, want in [(0.5, MAX_NUDGE), (-0.5, -MAX_NUDGE), (0.02, 0.02),
                      (1e9, 0.0), (float("nan"), 0.0)]:
        a = _adv(_Boom(payload='{"message":"m","adjust":%r}' % raw))
        _, nudge = a.advise({}, "s", "D")
        check(abs(nudge - want) < 1e-9,
              f"adjust={raw} should clamp to {want}, got {nudge}")


def test_hard_timeout_beats_the_turn_clock():
    import time
    a = _adv(_Boom(payload='{"message":"slow","adjust":0}', delay=30.0))
    a_timeout = 2.0
    import glee.advisor as A
    saved = A.TIMEOUT_S
    A.TIMEOUT_S = a_timeout
    try:
        start = time.monotonic()
        msg, nudge = a.advise({}, "s", "DETERMINISTIC")
        elapsed = time.monotonic() - start
    finally:
        A.TIMEOUT_S = saved
    check(elapsed < a_timeout + 1.5,
          f"advise() took {elapsed:.1f}s against a {a_timeout}s cap -- a hung "
          f"call must never approach the 120s turn limit")
    check(msg == "DETERMINISTIC" and nudge == 0.0,
          "a timed-out call must yield the deterministic message")


def test_circuit_breaker_opens():
    a = _adv(_Boom(exc=RuntimeError("down")))
    for _ in range(8):
        a.advise({}, "s", "D")
    check(not a.available(),
          "repeated failures must open the breaker so we stop calling out")


def test_nudge_cannot_create_a_losing_trade():
    """The clamp lives in the dispatcher; verify the invariant it protects."""
    from glee.solvers import _advise as dispatch_advise
    from glee import solvers

    class FakeAdvisor:
        def available(self):
            return True

        def advise(self, game, summary, fallback):
            return "text", -0.05           # push the seller's price DOWN

    saved = solvers.ADVISOR
    solvers.ADVISOR = FakeAdvisor()
    try:
        state = {"round": 2, "max_rounds": 10, "complete_information": False,
                 "messages_allowed": True, "player_1_role": "seller",
                 "player_2_role": "buyer", "player_1_value": 10000.0,
                 "player_2_value": None, "history": [], "last_offer": None}
        game = {"game_id": "g", "game_family": "negotiation",
                "your_player": "player_1", "game_state": state,
                "valid_actions": {"type": "offer", "fields": {}}}
        out = dispatch_advise(game, "negotiation", {"product_price": 10050.0})
        check(out["product_price"] >= 10000.0,
              f"a downward nudge must not price a seller below its own "
              f"valuation: {out['product_price']}")
    finally:
        solvers.ADVISOR = saved


def test_reconciles_games_that_end_on_the_opponents_move():
    """Half of every alternating game ends without us seeing `game_over`.

    In persuasion the buyer acts last, so as SELLER our final recommendation
    returns game_over=False and the game finishes on their decision. Measured:
    1,544 seller games produced ZERO game-end records -- half the family
    invisible, and their rating changes left to be misattributed to whichever
    game happened to end next.
    """
    import time as _t
    from glee.transport import Agent

    class FakeAPI:
        def __init__(self):
            self.governor = type("G", (), {"available": lambda s: 40,
                                           "limit": 48})()
            self.fetched = []

        def game_state(self, gid):
            self.fetched.append(gid)
            return {"game_id": gid, "game_family": "persuasion",
                    "your_player": "player_1", "status": "completed",
                    "game_state": {"phase": "completed", "round": 20},
                    "result": {"outcome": "completed",
                               "player_1_payoff": 1400.0}}

        def stats(self, **kw):
            return {"active_games": 0}

    class FakeTel:
        def __init__(self):
            self.ended = []

        def record_game_end(self, game, result):
            self.ended.append(game.get("game_id"))

        def record_rating(self, *a, **k):
            pass

        def record_move(self, *a, **k):
            pass

    api, tel = FakeAPI(), FakeTel()
    agent = Agent(api, lambda g: {}, telemetry=tel, label="t")
    agent._seen["ghost"] = _t.monotonic() - 200.0     # quiet long enough
    agent._reconcile()
    check(tel.ended == ["ghost"],
          f"a finished opponent-terminated game must be recorded, got {tel.ended}")
    check(agent.completed == 1, "it must count toward completed games")

    # Idempotent: a second pass must not double-record.
    agent._seen["ghost"] = _t.monotonic() - 200.0
    agent._reconcile()
    check(tel.ended == ["ghost"], "must not record the same game twice")

    # A still-active game must be left alone and re-armed.
    class ActiveAPI(FakeAPI):
        def game_state(self, gid):
            return {"game_id": gid, "status": "active", "result": None,
                    "game_state": {"phase": "buyer_decision", "round": 4}}

    api2, tel2 = ActiveAPI(), FakeTel()
    a2 = Agent(api2, lambda g: {}, telemetry=tel2, label="t")
    a2._seen["live"] = _t.monotonic() - 200.0
    a2._reconcile()
    check(tel2.ended == [], "an active game must not be recorded as finished")
    check("live" in a2._seen, "an active game must stay under watch")


def main() -> int:
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        before = len(FAILURES)
        try:
            fn()
        except Exception as e:
            FAILURES.append(f"{name} raised {type(e).__name__}: {e}")
        print(f"[{'ok ' if len(FAILURES) == before else 'FAIL'}] {name}")
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for f in FAILURES[:20]:
            print("  -", f)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
