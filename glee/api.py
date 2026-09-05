"""Direct REST client for the GLEE competition API, with a rate governor.

We talk to the documented REST endpoints rather than using `glee_sdk` because
the SDK's `run()` loop spends most of the request budget on polling: at its default poll_interval=2 it issues 30 polls/min plus ~16
top-up calls/min, leaving only ~14 of the 60 allowed requests for actual moves.
Since moves are the only requests that earn rating, we schedule the budget
ourselves (see transport.py).

Semantics mirrored deliberately from the SDK source (v0.0.4):
  * POST /move is NOT idempotent, so it is never retried on an ambiguous
    failure (a read timeout may mean the move landed). Callers re-read state.
  * Connection errors never reached the server, so any method may be retried.
  * A missing Authorization header returns 403, not 401.
"""

from __future__ import annotations

import email.utils
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from .config import BASE_URL, SAFE_RATE_LIMIT

logger = logging.getLogger("glee.api")


class GleeAPIError(Exception):
    def __init__(self, status_code: int, message: str, code: str | None = None,
                 detail: Any = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail
        super().__init__(f"[{status_code}] {message}")


class CompetitionNotOpenError(GleeAPIError):
    pass


class CompetitionClosedError(GleeAPIError):
    pass


class RateLimitedError(GleeAPIError):
    pass


class RateGovernor:
    """Sliding-window limiter that keeps us strictly under the server's cap.

    Polls and moves compete for the same 60 requests/minute, but they are not
    equally valuable: a move advances a game (and can prevent a turn timeout,
    which is scored at the 5th percentile), while a poll only discovers work.
    So low-priority callers must leave `reserve` slots free for moves, and
    high-priority callers may consume the budget down to zero.
    """

    def __init__(self, limit_per_min: int = SAFE_RATE_LIMIT):
        self.limit = limit_per_min
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    def available(self) -> int:
        with self._lock:
            self._prune(time.monotonic())
            return self.limit - len(self._events)

    def try_acquire(self, reserve: int = 0) -> bool:
        """Take a slot without blocking, leaving `reserve` slots untouched."""
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._events) + reserve >= self.limit:
                return False
            self._events.append(now)
            return True

    def acquire(self, timeout: float = 90.0) -> bool:
        """Block until a slot frees up. Used for moves, which must go out."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                if len(self._events) < self.limit:
                    self._events.append(now)
                    return True
                wait = max(0.05, self._events[0] + 60.0 - now)
            if time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 1.0))


def _parse_retry_after(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return default
    if when is None:
        return default
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)


class GleeAPI:
    def __init__(self, api_key: str, base_url: str = BASE_URL,
                 timeout: int = 30, governor: RateGovernor | None = None,
                 pool_size: int = 32):
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/agent"
        self.timeout = timeout
        self.governor = governor or RateGovernor()
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # -- plumbing -----------------------------------------------------------
    def _raise_for_response(self, resp: requests.Response) -> None:
        if resp.ok:
            return
        try:
            body: Any = resp.json()
        except ValueError:
            body = resp.text
        detail = body.get("detail") if isinstance(body, dict) else body
        if isinstance(detail, dict):
            code = detail.get("code")
            message = detail.get("message") or str(detail)
            if code == "competition_not_open":
                raise CompetitionNotOpenError(403, message, code, detail)
            if code == "competition_closed":
                raise CompetitionClosedError(403, message, code, detail)
            raise GleeAPIError(resp.status_code, message, code, detail)
        if resp.status_code == 429:
            raise RateLimitedError(429, "rate limited", "rate_limited", detail)
        raise GleeAPIError(resp.status_code, str(detail) if detail else resp.reason)

    def _request(self, method: str, path: str, *, blocking: bool = True,
                 reserve: int = 0, retries: int = 2,
                 acquire_timeout: float = 90.0, **kwargs) -> Any:
        url = f"{self.api_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        resp = None
        for attempt in range(retries + 1):
            # Every HTTP attempt costs the server a request, so every attempt
            # must take its own slot. Charging only the first one let a single
            # governed call issue three real requests and blow past the cap.
            if blocking:
                if not self.governor.acquire(timeout=acquire_timeout):
                    raise RateLimitedError(429, "local rate governor timeout")
            else:
                if not self.governor.try_acquire(reserve=reserve):
                    raise RateLimitedError(429, "local rate governor: budget reserved")
            try:
                resp = self.session.request(method, url, **kwargs)
            except requests.RequestException as e:
                # A ConnectionError never reached the server so any method is
                # safe to replay. Anything else is ambiguous: replaying a POST
                # /move could submit the same move twice, so only GETs retry.
                safe = isinstance(e, requests.ConnectionError) or method == "GET"
                if not safe or attempt == retries:
                    raise
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429 and attempt < retries:
                wait = min(_parse_retry_after(resp.headers.get("Retry-After"),
                                              float(2 ** attempt)), 15.0)
                logger.warning("server 429; sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            break
        self._raise_for_response(resp)
        if resp.status_code == 204:
            return {}
        return resp.json()

    # -- endpoints ----------------------------------------------------------
    def queue(self, family: str) -> dict:
        return self._request("POST", "/queue", json={"game_family": family})

    def leave_queue(self, family: str | None = None) -> dict:
        params = {"game_family": family} if family else None
        try:
            return self._request("DELETE", "/queue", params=params)
        except GleeAPIError as e:
            if e.status_code in (404, 501):
                return {}
            raise

    def pending_games(self, *, blocking: bool = False, reserve: int = 0) -> list[dict]:
        out = self._request("GET", "/games/pending", blocking=blocking, reserve=reserve)
        return out if isinstance(out, list) else []

    def move(self, game_id: str, action: dict) -> dict:
        # A dropped move is a turn timeout, which is scored at the 5th
        # percentile and counts toward the crash-loop cooldown -- far worse
        # than any delay. So moves block for a slot right up to the edge of
        # the 120 s turn clock rather than giving up.
        return self._request("POST", f"/games/{game_id}/move",
                             json={"action": action}, retries=0,
                             acquire_timeout=100.0)

    def game_state(self, game_id: str) -> dict:
        return self._request("GET", f"/games/{game_id}")

    def stats(self, *, blocking: bool = True) -> dict:
        return self._request("GET", "/stats", blocking=blocking)
