#!/usr/bin/env python3
"""Run a GLEE agent.

    python run_agent.py                      # main agent, all families, forever
    python run_agent.py --check              # verify the API key, play nothing
    python run_agent.py --max-games 20       # bounded smoke run
    python run_agent.py --agent lab1 --families persuasion

Stopping cleanly matters: an agent that stays queued but stops polling gets
matched into games it then loses by turn timeout, and each of those is scored
at the 5th percentile. Both the signal handler and the transport's `finally`
leave the queue on every exit path.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from dotenv import load_dotenv

from glee.api import GleeAPI, GleeAPIError, RateGovernor
from glee.config import FAMILIES, SAFE_RATE_LIMIT
from glee.solvers import strategy
from glee.telemetry import Telemetry
from glee.transport import Agent

def key_env_for(agent: str) -> str:
    """Env var holding an agent's key.

    `main` reads GLEE_API_KEY; any other name reads GLEE_API_KEY_<NAME>, so
    adding an agent slot is just a line in .env and a --agent flag.
    """
    return "GLEE_API_KEY" if agent == "main" else f"GLEE_API_KEY_{agent.upper()}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="main",
                        help="agent slot name; picks its key from the environment")
    parser.add_argument("--families", default=",".join(FAMILIES),
                        help="comma-separated subset of the three families")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--family-cap",
                        default=os.environ.get("GLEE_FAMILY_CAP", ""),
                        help="throttle a family, e.g. negotiation=65 -- games "
                             "per rolling 24h. A ceiling only; never creates games.")
    parser.add_argument("--cap-above",
                        default=os.environ.get("GLEE_CAP_ABOVE", ""),
                        help="auto-throttle ANY family whose rating passes a "
                             "threshold, e.g. 3000=70 (rating=games per 24h)")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--rate-limit", type=int, default=SAFE_RATE_LIMIT)
    parser.add_argument("--min-poll", type=float, default=2.0)
    parser.add_argument("--max-poll", type=float, default=9.0)
    parser.add_argument("--check", action="store_true",
                        help="call /stats (not competition-gated) and exit")
    parser.add_argument("--barg", default=None, choices=["current", "legacy"],
                        help="bargaining policy arm for A/B (default: current)")
    parser.add_argument("--barg-knobs", default=None,
                        help="per-knob overrides layered on --barg, e.g. "
                             "'post_floor_cap=off,strong_push=0.45'. Keys are "
                             "_KNOB_DEFAULTS in glee/solvers/bargaining.py")
    parser.add_argument("--nego-boulware", default=None,
                        help="negotiation concession exponent (default 0.18). "
                             "Larger concedes EARLIER; see boulware_e() in "
                             "glee/solvers/negotiation.py")
    parser.add_argument("--pers-seller", default=None,
                        choices=["kg", "always_yes"],
                        help="persuasion seller arm. 'kg' is Kamenica-Gentzkow "
                             "(default); 'always_yes' recommends every unit, "
                             "testing whether the field's buyers do the "
                             "arithmetic")
    parser.add_argument("--advisor", action="store_true",
                        help="enable the LLM side-role for message wording and "
                             "a clamped offer nudge (never the accept rule)")
    parser.add_argument("--variant", default=None,
                        choices=["full", "nomsg"],
                        help="message arm: 'full' sends generated text, 'nomsg' "
                             "strips optional messages (numeric play identical)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
    )
    load_dotenv()
    if args.variant:
        os.environ["GLEE_VARIANT"] = args.variant
    if args.barg:
        # Must be set before the solvers are imported: the arm is read at
        # module load. (An earlier A/B was silently byte-identical on both
        # sides because a flag was bound at import and set afterwards.)
        os.environ["GLEE_BARG_ARM"] = args.barg
    if args.pers_seller:
        os.environ["GLEE_PERS_SELLER"] = args.pers_seller
    if args.nego_boulware:
        os.environ["GLEE_NEGO_BOULWARE"] = args.nego_boulware
    if args.barg_knobs:
        os.environ["GLEE_BARG_KNOBS"] = args.barg_knobs
        # Fail loudly on a typo. A misspelled knob is silently ignored by the
        # override loop, which would produce an experiment whose arms are
        # identical -- the exact failure mode the call-time reads exist to
        # prevent, arriving through a different door.
        from glee.solvers.bargaining import _KNOB_DEFAULTS
        for item in args.barg_knobs.split(","):
            key = item.partition("=")[0].strip()
            if key and key not in _KNOB_DEFAULTS:
                print(f"unknown bargaining knob {key!r}; valid keys are "
                      f"{sorted(_KNOB_DEFAULTS)}", file=sys.stderr)
                return 2

    env_name = key_env_for(args.agent)
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        print(f"No API key: set {env_name} in .env", file=sys.stderr)
        return 2

    api = GleeAPI(api_key, governor=RateGovernor(args.rate_limit))

    # /stats is deliberately not competition-gated, so a successful response
    # proves the key is valid and our requests reach the platform.
    try:
        stats = api.stats()
    except GleeAPIError as e:
        print(f"stats() failed: {e}", file=sys.stderr)
        return 1
    print(f"agent   : {stats.get('agent_name')} ({stats.get('agent_id')})",
          flush=True)
    print(f"active  : {stats.get('active_games')}", flush=True)
    for family, sc in (stats.get("scores") or {}).items():
        print(f"  {family:<12} rating {sc.get('rating')}  "
              f"games {sc.get('games_played')}", flush=True)
    # Echo the RESOLVED policy, not the flags. The flags are what we asked for;
    # this is what the solver will actually read, and the two have silently
    # disagreed twice in this project. Printed before the --check exit so that
    # `run_agent.py --check --barg-knobs ...` is a dry run of the real config.
    #
    # Emitted through the LOGGER, not just print(). stdout is block-buffered
    # when redirected to a file, so a bare print sits in the buffer for the
    # whole run and is flushed only when the process exits -- which is exactly
    # too late for something whose entire purpose is verifying a deploy. The
    # first deploy using this banner appeared to have no banner at all, and
    # what was visible in the log was the PREVIOUS process's buffer draining on
    # exit. Everything below also flushes explicitly.
    from glee.solvers import bargaining as _barg
    from glee.solvers import persuasion as _pers
    resolved = "arm=%s %s" % (
        "legacy" if _barg.legacy() else "current",
        " ".join("%s=%s" % (k, _barg.knob(k))
                 for k in sorted(_barg._KNOB_DEFAULTS)))
    from glee.advisor import TIMEOUT_S as _ADV_T, MODEL as _ADV_M
    adv_line = ("ON (model %s, timeout %ss, fails open)" % (_ADV_M, _ADV_T)
                if args.advisor else "OFF (deterministic messages only)")
    print("barg    : %s" % resolved, flush=True)
    print("pers    : seller=%s" % _pers.seller_policy(), flush=True)
    from glee.solvers import negotiation as _nego
    print("nego    : boulware=%.2f%s" % (
        _nego.boulware_e(),
        "" if _nego.boulware_e() == _nego.BOULWARE_E else "  (A/B arm)"), flush=True)
    logging.getLogger("glee.config").info("resolved negotiation boulware: %.2f",
                                          _nego.boulware_e())
    print("advisor : %s" % adv_line, flush=True)
    logging.getLogger("glee.config").info("resolved advisor: %s", adv_line)
    logging.getLogger("glee.config").info("resolved bargaining policy: %s",
                                          resolved)
    logging.getLogger("glee.config").info("resolved persuasion seller: %s",
                                          _pers.seller_policy())
    # Report the reference pool as RESOLVED, not as configured. `refpool` fails
    # open by design -- a missing or corrupt data file must never take the
    # agent down -- which means a packaging mistake would silently disable the
    # percentile tie-break and look exactly like it working. This is the same
    # discipline as the bargaining knob banner, and for the same reason: the
    # two have silently disagreed before.
    caps = {}
    for item in (args.family_cap or "").split(","):
        item = item.strip()
        if not item:
            continue
        fam, _, n = item.partition("=")
        caps[fam.strip()] = int(n)
    cap_line = (", ".join("%s=%d/24h" % kv for kv in sorted(caps.items()))
                if caps else "none (full rate on every family)")
    print("caps    : %s" % cap_line, flush=True)
    logging.getLogger("glee.config").info("resolved family daily caps: %s",
                                          cap_line)

    cap_above = None
    if args.cap_above.strip():
        thr, _, n = args.cap_above.partition("=")
        cap_above = (float(thr), int(n))
    auto_line = ("any family above %.0f -> %d/24h" % cap_above
                 if cap_above else "off")
    print("autocap : %s" % auto_line, flush=True)
    logging.getLogger("glee.config").info("resolved auto-cap: %s", auto_line)

    from glee import refpool as _refpool
    _usable = sum(1 for s in _refpool._load().values()
                  if s.n >= _refpool.MIN_SAMPLES
                  and s.imputed <= _refpool.MAX_IMPUTED)
    _pool_line = "%d strata loaded, %d usable" % (len(_refpool._load()), _usable)
    if not _usable:
        _pool_line += "  -- PERCENTILE TIE-BREAK INACTIVE"
    print("refpool : %s" % _pool_line, flush=True)
    logging.getLogger("glee.config").info("resolved persuasion refpool: %s",
                                          _pool_line)
    sys.stdout.flush()

    if args.check:
        return 0

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    bad = [f for f in families if f not in FAMILIES]
    if bad:
        print(f"unknown families: {bad}", file=sys.stderr)
        return 2

    # Write our OWN pid, authoritatively. Capturing it in the shell with `$!`
    # is unreliable -- under `setsid nohup ... &` bash records a wrapper
    # subshell instead, and signalling that leaves the real agent orphaned and
    # still playing while the drain never starts.
    pidfile = os.path.join("logs", f"{args.agent}.pid")
    os.makedirs("logs", exist_ok=True)
    with open(pidfile, "w") as fh:
        fh.write(str(os.getpid()))

    telemetry = Telemetry(agent_label=args.agent)
    if args.advisor:
        from glee.advisor import Advisor
        from glee.solvers import set_advisor
        adv = Advisor(enabled=True, telemetry=telemetry)
        if not adv.enabled:
            print("--advisor requested but GEMINI_API_KEY is unset", file=sys.stderr)
            return 2
        set_advisor(adv)

    agent = Agent(api, strategy, telemetry=telemetry,
                  families=families, concurrency=args.concurrency,
                  min_poll=args.min_poll, max_poll=args.max_poll,
                  family_daily_cap=caps, cap_above=cap_above,
                  label=args.agent)

    log = logging.getLogger("run_agent")
    seen_signal = {"n": 0}

    def _shutdown(signum, _frame):
        seen_signal["n"] += 1
        if seen_signal["n"] == 1:
            log.info("signal %s -> draining: no new games, playing out the ones "
                     "in flight. Send again to abandon them (each abandoned game "
                     "scores at the 5th percentile).", signum)
            agent.stop()
        else:
            log.warning("signal %s again -> hard stop, abandoning live games",
                        signum)
            agent.stop(hard=True)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        agent.run(max_games=args.max_games, max_time=args.max_time)
    finally:
        try:
            if os.path.exists(pidfile) and open(pidfile).read().strip() == str(os.getpid()):
                os.remove(pidfile)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
