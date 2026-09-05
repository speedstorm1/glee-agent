"""The run loop: keep games flowing, never time out, never waste the budget.

Budget arithmetic is the whole design. The server allows 60 requests/minute and
only *moves* earn rating, so polling must be as cheap as we can make it:

    poll_interval   polls/min   moves/min left (with SDK-style 16/min top-ups)
        2 s            30            14      <- glee_sdk default
        5 s            12            38
        8 s             8            46

Because `GET /games/pending` returns *every* actionable game in one response,
the fix is high concurrency plus lazy polling: one poll then amortises over
many moves. Turn latency is irrelevant (we have 120 s per turn), so a slow poll
costs nothing but earns a lot of budget.

On top of that:
  * moves always win the budget over polls (a dropped move risks a 5th-pctile
    timeout; a dropped poll costs a few seconds);
  * an invalid move is immediately retried with the guaranteed-legal fallback,
    so a solver bug burns one attempt instead of all five;
  * the queue is always left on exit -- an agent that stays queued but stops
    polling gets matched and loses by timeout.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .api import (CompetitionClosedError, CompetitionNotOpenError, GleeAPI,
                  GleeAPIError, RateLimitedError)
from .config import FAMILIES
from .safety import fallback_action, sanitise
from .telemetry import Telemetry

logger = logging.getLogger("glee.transport")

Strategy = Callable[[dict], dict]


class Agent:
    def __init__(self, api: GleeAPI, strategy: Strategy,
                 telemetry: Telemetry | None = None,
                 families: tuple[str, ...] = FAMILIES,
                 concurrency: int = 16,
                 min_poll: float = 2.0,
                 max_poll: float = 9.0,
                 topup_interval: float = 30.0,
                 family_daily_cap: dict[str, int] | None = None,
                 cap_above: tuple[float, int] | None = None,
                 label: str = "main"):
        self.api = api
        self.strategy = strategy
        self.telemetry = telemetry or Telemetry(agent_label=label)
        self.families = tuple(families)
        # Optional ceiling on games per family per rolling 24h. A cap only ever
        # limits play; it never creates games.
        self.family_daily_cap = dict(family_daily_cap or {})
        self._starts: dict[str, collections.deque] = collections.defaultdict(
            collections.deque)
        self._counted: set[str] = set()
        # (rating threshold, games/24h). Applies the same ceiling to any family
        # whose rating is above the threshold.
        self.cap_above = cap_above
        self._ratings: dict[str, float] = {}
        self._auto_capped: set[str] = set()
        self.concurrency = concurrency
        # Games in flight are limited by the REQUEST budget, not by threads.
        # A persuasion game needs ~20 moves and bargaining ~4, so a fixed high
        # concurrency can demand more moves/minute than 60 req/min allows --
        # and a move that cannot get a slot becomes a turn timeout, scored at
        # the 5th percentile. We therefore discover the sustainable level:
        # back off hard whenever a move is starved, creep back up when calm.
        self.target_concurrency = max(3, min(concurrency, 8))
        self._last_drop = 0.0
        self._last_raise = 0.0
        self.moves_dropped = 0
        self.min_poll = min_poll
        self.max_poll = max_poll
        self.topup_interval = topup_interval
        self.label = label

        # Two-stage shutdown. A drain stops queueing but KEEPS PLAYING until
        # every in-flight game is finished; only a second signal exits now.
        # This is not politeness: exiting with games live abandons them, each
        # abandoned game is scored at the 5th percentile, and three
        # self-timeouts in a row trigger a 30-minute crash-loop cooldown. An
        # early build of this agent lost exactly that way -- it shut down in
        # 0.3s with six games active and was locked out for half an hour.
        self._drain_requested = threading.Event()
        self._stop = threading.Event()
        self._inflight: set[str] = set()
        # (game_id, round, action_type) turns whose move the server has
        # ACCEPTED. A poll issued before our move lands returns the same turn
        # again; without this we submit twice and waste a request from a
        # 60/min budget. Recorded only on success -- marking a turn answered
        # before the move lands would silently retire a failed move and hand
        # the game a turn timeout, which is exactly the 5th-percentile outcome
        # this whole layer exists to prevent.
        self._answered: set[tuple] = set()
        self._inflight_lock = threading.Lock()
        self._rating_lock = threading.Lock()
        self._last_rating_sample = 0.0
        # Games we have moved in, and when we last saw them pending. A game
        # that ends on the OPPONENT's move never returns game_over to us, so
        # without reconciliation it is never recorded at all. That was silently
        # dropping every persuasion game we played as seller -- 1,544 of them,
        # half the family -- and, worse, leaving their rating changes to be
        # misattributed to whichever game happened to end next.
        self._seen: dict[str, float] = {}
        self._recorded: set[str] = set()
        self.completed = 0
        self.invalid_moves = 0
        self.strategy_errors = 0
        self.move_errors = 0
        self._recent_timeouts = 0

    # -- one game's turn ----------------------------------------------------
    def _play_turn(self, game: dict, turn: tuple | None = None) -> None:
        gid = game["game_id"]
        started = time.monotonic()
        try:
            try:
                action = self.strategy(game)
            except Exception:
                self.strategy_errors += 1
                logger.exception("strategy raised for %s (%s); using fallback",
                                 gid[:8], game.get("game_family"))
                action = fallback_action(game)

            action = sanitise(game, action)

            try:
                result = self.api.move(gid, action)
            except RateLimitedError:
                # This is the expensive failure: no slot means no move means a
                # turn timeout. Shrink the number of games we carry so it
                # stops happening, rather than logging and repeating.
                self.moves_dropped += 1
                self.move_errors += 1
                self._last_drop = time.monotonic()
                old = self.target_concurrency
                self.target_concurrency = max(3, int(self.target_concurrency * 0.75))
                logger.error("move STARVED on %s (budget exhausted); "
                             "concurrency %d -> %d", gid[:8], old,
                             self.target_concurrency)
                return
            except GleeAPIError as e:
                # 400 "not your turn" means the opponent raced us or the game
                # already advanced -- harmless. Anything else is worth a look.
                self.move_errors += 1
                if e.status_code not in (400, 403, 404):
                    logger.warning("move error on %s: %s", gid[:8], e)
                self.telemetry.record_move(game, action, None, error=str(e))
                return

            latency = (time.monotonic() - started) * 1000.0
            self.telemetry.record_move(game, action, result, latency_ms=latency)
            if turn is not None:
                with self._inflight_lock:
                    self._answered.add(turn)
                    if len(self._answered) > 20000:
                        self._answered.clear()

            if result.get("valid") is False:
                self.invalid_moves += 1
                left = result.get("attempts_left")
                logger.warning("INVALID move on %s: %r (attempts_left=%s) action=%s",
                               gid[:8], result.get("error"), left, action)
                # Spend one attempt on a move we know is legal rather than
                # letting a solver bug burn all five and forfeit the game.
                if left is None or int(left) > 1:
                    safe = fallback_action(game)
                    if safe != action:
                        try:
                            result = self.api.move(gid, safe)
                            self.telemetry.record_move(game, safe, result,
                                                       error="retry_after_invalid")
                        except GleeAPIError as e:
                            logger.warning("fallback move also failed on %s: %s",
                                           gid[:8], e)
                            return

            if result.get("game_over"):
                self.completed += 1
                with self._inflight_lock:
                    self._recorded.add(gid)
                    self._seen.pop(gid, None)
                self.telemetry.record_game_end(game, result)
                self._sample_rating(gid)
            else:
                with self._inflight_lock:
                    self._seen[gid] = time.monotonic()
        finally:
            with self._inflight_lock:
                self._inflight.discard(gid)

    def _note_start(self, gid: str, family: str | None) -> None:
        """Record the first sighting of a game, for the per-family cap."""
        # Track every family we might ever cap, not just the ones capped now:
        # an automatic cap that engages mid-run needs the preceding 24 hours of
        # starts already counted, or it would allow a full extra day of games
        # before it began to bind.
        if not family or (family not in self.family_daily_cap
                          and not self.cap_above):
            return
        with self._inflight_lock:
            if gid in self._counted:
                return
            self._counted.add(gid)
            self._starts[family].append(time.time())

    #: Rating a family must fall BELOW to be let back up to full speed, as a
    #: margin under the engage threshold. Without hysteresis a rating sitting
    #: on the line would throttle and unthrottle every few games, which is a
    #: worse policy than either state.
    AUTO_CAP_RELEASE = 150.0

    def _note_ratings(self, stats: dict) -> None:
        """Track per-family rating, and engage or release the automatic cap.

        Sticky in both directions on purpose. We throttle a family once it is
        above the threshold and keep throttling it while it stays near there,
        because the point is to stop testing a good estimate. We release only
        once it has fallen a clear margin below, because at that point the
        estimate is running LOW and more games pull it back up -- the same
        mechanism, pointed the other way.
        """
        scores = (stats or {}).get("scores") or {}
        for fam, v in scores.items():
            try:
                self._ratings[fam] = float(v["rating"])
            except (KeyError, TypeError, ValueError):
                continue
        if not self.cap_above:
            return
        threshold, cap = self.cap_above
        for fam, rating in self._ratings.items():
            if fam not in self._auto_capped and rating >= threshold:
                self._auto_capped.add(fam)
                logger.info("[%s] %s hit %.0f (>= %.0f): auto-throttling to "
                            "%d games/24h to bank it", self.label, fam,
                            rating, threshold, cap)
            elif fam in self._auto_capped and rating < threshold - self.AUTO_CAP_RELEASE:
                self._auto_capped.discard(fam)
                logger.info("[%s] %s fell to %.0f: releasing the auto-throttle",
                            self.label, fam, rating)

    def _effective_cap(self, family: str) -> int | None:
        """The tighter of the standing cap and any automatic one."""
        caps = [c for c in (self.family_daily_cap.get(family),
                            self.cap_above[1] if (self.cap_above and
                                                  family in self._auto_capped)
                            else None) if c]
        return min(caps) if caps else None

    def _family_capped(self, family: str) -> bool:
        """Should we sit this family out right now?

        Two conditions, and the second matters as much as the first. A bare
        24-hour quota is satisfiable by playing the whole allowance in the
        first two hours and then going dark for twenty-two, which is poor
        behavior toward a matchmaker that is trying to pair us with someone.
        So we also pace: no faster than the cap implies, with a little slack so
        a burst of concurrent matches is not penalised.
        """
        cap = self._effective_cap(family)
        if not cap:
            return False
        now = time.time()
        with self._inflight_lock:
            q = self._starts[family]
            while q and q[0] < now - 86400.0:
                q.popleft()
            n = len(q)
            last = q[-1] if q else 0.0
        if n >= cap:
            return True
        return (now - last) < (86400.0 / cap) * 0.8

    def _reconcile(self, budget: int = 3) -> None:
        """Record games that ended on the opponent's move.

        Half of every alternating game ends without us getting a `game_over`
        response -- in persuasion the buyer acts last, so as seller we never
        see the end. We fetch the final state for games that have gone quiet,
        a few at a time so this never competes with live moves.
        """
        now = time.monotonic()
        with self._inflight_lock:
            stale = [g for g, t in self._seen.items()
                     if now - t > 90.0 and g not in self._inflight][:budget]
        for gid in stale:
            try:
                state = self.api.game_state(gid)
            except GleeAPIError as e:
                if e.status_code == 404:      # gone for good
                    with self._inflight_lock:
                        self._seen.pop(gid, None)
                continue
            # Verified against the live API: GET /games/{id} returns a
            # top-level `status` of "active" or "completed", with `result`
            # populated once finished. `phase` lives inside game_state.
            status = str(state.get("status") or "").lower()
            result = state.get("result") or {}
            if status == "completed" or result:
                with self._inflight_lock:
                    self._seen.pop(gid, None)
                    if gid in self._recorded:
                        continue
                    self._recorded.add(gid)
                self.completed += 1
                self.telemetry.record_game_end(state, {"result": result,
                                                       "game_over": True})
                self._sample_rating(gid)
            else:
                with self._inflight_lock:
                    if gid in self._seen:
                        self._seen[gid] = now

    def _sample_rating(self, after_game: str | None = None) -> None:
        """Snapshot ratings right after a game ends, to price that game.

        Attribution is only clean when a single game closed since the previous
        sample, which is why this fires immediately on game-over rather than on
        a timer; the analysis keys off games_played advancing by exactly one.
        Serialised and throttled so a burst of simultaneous finishes cannot
        spend the request budget that in-flight moves need.
        """
        with self._rating_lock:
            now = time.monotonic()
            if now - self._last_rating_sample < 1.5:
                return
            self._last_rating_sample = now
        try:
            stats = self.api.stats(blocking=False)
        except (GleeAPIError, RateLimitedError):
            return          # a missed sample costs analysis, never a game
        self.telemetry.record_rating(stats, after_game=after_game)
        self._note_ratings(stats)

    # -- queue upkeep -------------------------------------------------------
    def _topup(self) -> int | None:
        try:
            active = int(self.api.stats().get("active_games", 0))
        except GleeAPIError as e:
            if isinstance(e, (CompetitionClosedError, CompetitionNotOpenError)):
                raise
            logger.debug("stats failed: %s", e)
            return None
        # Creep back up only after a sustained calm spell with real headroom.
        now = time.monotonic()
        if (self.target_concurrency < self.concurrency
                and now - self._last_drop > 300.0
                and now - self._last_raise > 120.0
                and self.api.governor.available() > 10):
            self.target_concurrency += 1
            self._last_raise = now
            logger.info("budget calm; concurrency -> %d", self.target_concurrency)

        if active < self.target_concurrency:
            for family in self.families:
                if self._stop.is_set():
                    break
                if self._family_capped(family):
                    # Also step OUT of the queue: staying in it while capped
                    # would let the server match us anyway, which would make
                    # the cap advisory rather than real.
                    try:
                        self.api.leave_queue(family)
                    except GleeAPIError:
                        pass
                    continue
                try:
                    self.api.queue(family)
                except CompetitionClosedError:
                    raise
                except GleeAPIError as e:
                    if e.code == "agent_cooldown":
                        logger.error("CRASH-LOOP COOLDOWN active: %s", e.message)
                        self._stop.wait(60)
                    else:
                        logger.debug("queue(%s) failed: %s", family, e)
        return active

    # -- main loop ----------------------------------------------------------
    def run(self, max_games: int | None = None, max_time: float | None = None,
            drain_grace: float = 420.0) -> None:
        start = time.monotonic()
        poll = self.min_poll
        last_topup = 0.0
        last_status = 0.0
        last_reconcile = 0.0
        draining = False
        drain_started = 0.0

        try:
            self._topup()
        except CompetitionNotOpenError as e:
            logger.error("competition not open yet (opens %s)",
                         (e.detail or {}).get("competition_open_at"))
            return
        except CompetitionClosedError as e:
            logger.error("competition closed (%s)",
                         (e.detail or {}).get("competition_close_at"))
            return

        logger.info("[%s] running: families=%s concurrency=%d",
                    self.label, ",".join(self.families), self.concurrency)

        with ThreadPoolExecutor(max_workers=self.concurrency,
                                thread_name_prefix="glee") as pool:
            try:
                while not self._stop.is_set():
                    now = time.monotonic()

                    if not draining and (
                        self._drain_requested.is_set()
                        or (max_games is not None and self.completed >= max_games)
                        or (max_time is not None and now - start >= max_time)
                    ):
                        draining = True
                        drain_started = now
                        logger.info("[%s] draining: no new games; playing out the "
                                    "%d already in flight", self.label,
                                    len(self._inflight))
                        self._safe_leave_queue()

                    if now - last_reconcile >= 20.0:
                        last_reconcile = now
                        try:
                            self._reconcile()
                        except Exception:
                            logger.debug("reconcile failed", exc_info=True)

                    if now - last_status >= 120.0:
                        last_status = now
                        with self._inflight_lock:
                            busy_n = len(self._inflight)
                        logger.info("[%s] %d games done | inflight=%d target=%d | "
                                    "budget %d/%d free | dropped=%d invalid=%d "
                                    "strat_err=%d", self.label, self.completed,
                                    busy_n, self.target_concurrency,
                                    self.api.governor.available(),
                                    self.api.governor.limit, self.moves_dropped,
                                    self.invalid_moves, self.strategy_errors)

                    if not draining and now - last_topup >= self.topup_interval:
                        last_topup = now
                        try:
                            self._topup()
                        except CompetitionClosedError:
                            logger.info("competition closed; draining")
                            draining = True
                            drain_started = now

                    # Reserve budget so moves always outrank polls.
                    # Polls must never crowd out moves: only poll when there
                    # is comfortable headroom above what the in-flight games
                    # could still demand this minute.
                    with self._inflight_lock:
                        busy = len(self._inflight)
                    reserve = min(max(busy * 2, 8), 24)

                    games = []
                    try:
                        games = self.api.pending_games(blocking=False, reserve=reserve)
                    except RateLimitedError:
                        pass  # budget is better spent on moves right now
                    except CompetitionClosedError:
                        draining = True
                        drain_started = drain_started or now
                    except GleeAPIError as e:
                        logger.debug("poll failed: %s", e)

                    dispatched = 0
                    for game in games:
                        gid = game.get("game_id")
                        if not gid:
                            continue
                        turn = (gid, (game.get("game_state") or {}).get("round"),
                                (game.get("valid_actions") or {}).get("type"))
                        with self._inflight_lock:
                            if gid in self._inflight or turn in self._answered:
                                continue
                            self._inflight.add(gid)
                        self._note_start(gid, game.get("game_family"))
                        pool.submit(self._play_turn, game, turn)
                        dispatched += 1

                    # Busy -> poll hard; idle -> back off and bank the budget.
                    if dispatched:
                        poll = self.min_poll
                    else:
                        poll = min(self.max_poll, poll * 1.4)

                    if draining:
                        with self._inflight_lock:
                            busy = len(self._inflight)
                        active = None
                        if not games and busy == 0:
                            # Poll the server rather than trust our local view:
                            # a game waiting on the OPPONENT is still ours to
                            # finish, and abandoning it costs us the 5th
                            # percentile. The server closes genuinely stuck
                            # games on their turn timeout, so this terminates.
                            try:
                                active = int(self.api.stats().get("active_games", 0))
                            except GleeAPIError:
                                active = None
                            if active == 0 and now - drain_started > 10.0:
                                logger.info("[%s] drained cleanly; %d games completed",
                                            self.label, self.completed)
                                return
                        if now - drain_started > drain_grace:
                            logger.warning("[%s] drain grace (%.0fs) exceeded with "
                                           "%s games still active; exiting",
                                           self.label, drain_grace,
                                           busy if active is None else active)
                            return

                    self._stop.wait(poll)
            except KeyboardInterrupt:
                logger.info("[%s] interrupted", self.label)
            finally:
                self._safe_leave_queue()
                logger.info("[%s] done: %d games, %d invalid moves, "
                            "%d strategy errors, %d move errors",
                            self.label, self.completed, self.invalid_moves,
                            self.strategy_errors, self.move_errors)

    def stop(self, hard: bool = False) -> None:
        """Request shutdown. The default drains; `hard` abandons live games."""
        if hard:
            self._stop.set()
        else:
            self._drain_requested.set()

    def _safe_leave_queue(self) -> None:
        try:
            self.api.leave_queue()
        except Exception as e:  # never raise on the way out
            logger.warning("leave_queue failed: %s", e)
