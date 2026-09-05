#!/usr/bin/env python3
"""Offline decision replay: which policy knob changed OUR action, and where.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
This tool carries no opponent model. It answers one narrow question:

    Given a state we ACTUALLY faced, which knob changes the action we take?

Every state replayed here is a real one, read back from `logs/*.moves.jsonl`,
which records the full observed `game_state` alongside the action we submitted.
The solvers are stateless -- they reconstruct all beliefs from `history` on each
call -- and randomness is seeded from `game_id`, so a replay reproduces the
original decision exactly. That is verified, not assumed: `--verify` replays the
live arm and asserts the reconstructed action matches what we really sent.

It CANNOT tell you what the opponent would have done next, and therefore cannot
tell you the payoff of a counterfactual. Do not ask it to. It answers
"which knob, how often, in which direction, in which configurations" -- which is
what a bisect needs, and it answers that in minutes rather than in days of live
A/B time.

    python analysis/replay.py --bisect              # one knob at a time
    python analysis/replay.py --arms                # current vs legacy
    python analysis/replay.py --knobs field_prior=off,post_floor_cap=off
    python analysis/replay.py --verify              # replay fidelity check
"""

from __future__ import annotations

import argparse
import copy
import glob
import datetime as dt
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glee.solvers import bargaining  # noqa: E402


# --- loading ----------------------------------------------------------------

def iter_moves(pattern: str, family: str, since: float | None,
               limit: int | None):
    """Stream move rows. The move logs are ~150 MB/agent, so never slurp."""
    seen = 0
    for path in sorted(glob.glob(pattern)):
        agent = os.path.basename(path).split(".")[0]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                # Cheap prefilter before the JSON parse: the move logs are
                # ~150 MB/agent and most lines are other families. Telemetry
                # writes compact separators, but tolerate both spacings.
                if family and (f'"family":"{family}"' not in line
                               and f'"family": "{family}"' not in line):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("family") != family:
                    continue
                if since is not None and row.get("ts", 0) < since:
                    continue
                if not row.get("state") or not row.get("action"):
                    continue
                # Only rows the server accepted: a rejected move is not a
                # decision we actually made.
                if not ((row.get("result") or {}).get("valid", True)):
                    continue
                row["_agent"] = agent
                yield row
                seen += 1
                if limit and seen >= limit:
                    return


def to_game(row: dict) -> dict:
    """Rebuild the exact dict the solver was handed."""
    return {
        "game_id": row.get("game_id"),
        "game_family": row.get("family"),
        "your_player": row.get("your_player"),
        "game_state": row.get("state"),
        "valid_actions": {"type": row.get("action_type")},
    }


# --- classification ---------------------------------------------------------

#: How to bucket a state. RELATIVE patience is where the ordering bug lives;
#: ABSOLUTE patience is where holding out actually pays, and the two are not
#: the same axis -- a change can be significant on one and worthless on the
#: other. Measured live: legacy beat current by 0.112 of raw share in STRONG,
#: but on discounted payoff the entire effect sat at d_me = 1.0 and vanished at
#: 0.90-0.95, which are also STRONG.
GROUPING = "leverage"


def leverage(row: dict) -> str:
    """Bucket a state on the axis selected by --group."""
    st = row.get("state") or {}
    me = row.get("your_player")
    d_me = st.get("delta_1") if me == "player_1" else st.get("delta_2")
    d_op = st.get("delta_2") if me == "player_1" else st.get("delta_1")
    if d_me is None:
        return "UNKNOWN"
    if GROUPING == "delta":
        # Our own discount factor: observable in EVERY game, including
        # incomplete-information ones where d_opp is hidden.
        return "d_me=%.2f" % float(d_me)
    if d_op is None:
        return "HIDDEN"
    if d_me > d_op:
        return "STRONG"
    if d_me < d_op:
        return "WEAK"
    return "EVEN"


DELTA_ORDER = ("d_me=0.80", "d_me=0.90", "d_me=0.95", "d_me=1.00", "ALL")


def my_share_of(action: dict, row: dict) -> float | None:
    st = row.get("state") or {}
    money = float(st.get("money_to_divide") or 0) or None
    if money is None:
        return None
    me = row.get("your_player")
    key = "alice_gain" if me == "player_1" else "bob_gain"
    if key in action:
        return float(action[key]) / money
    return None


def offered_share(row: dict) -> float | None:
    """What THEY put on the table on this decision turn."""
    st = row.get("state") or {}
    money = float(st.get("money_to_divide") or 0) or None
    off = st.get("last_offer") or {}
    me = row.get("your_player")
    mine = off.get(f"{me}_gain")
    if mine is None or not money:
        return None
    return float(mine) / money


def summarise(action: dict) -> str:
    if "decision" in action:
        return str(action["decision"])
    return "offer"


# --- the replay -------------------------------------------------------------

def replay(rows: list[dict], arm: str, knobs: str) -> list[dict]:
    """Re-decide every row under one knob configuration."""
    os.environ["GLEE_BARG_ARM"] = arm
    if knobs:
        os.environ["GLEE_BARG_KNOBS"] = knobs
    else:
        os.environ.pop("GLEE_BARG_KNOBS", None)
    out = []
    for row in rows:
        game = to_game(copy.deepcopy(row))
        try:
            action = bargaining.solve(game)
        except Exception as exc:  # a crash IS a finding
            action = {"decision": "EXC:%s" % type(exc).__name__}
        out.append(action)
    return out


def compare(rows, base, alt, label) -> dict:
    """Per-leverage-bucket difference between two replays."""
    buckets = defaultdict(lambda: {
        "n": 0, "flip": 0, "acc_to_rej": 0, "rej_to_acc": 0,
        "base_acc_share": [], "alt_acc_share": [],
        "base_demand": [], "alt_demand": [],
    })
    for row, a, b in zip(rows, base, alt):
        for key in (leverage(row), "ALL"):
            d = buckets[key]
            d["n"] += 1
            sa, sb = summarise(a), summarise(b)
            if sa != sb:
                d["flip"] += 1
                if sa == "accept" and sb == "reject":
                    d["acc_to_rej"] += 1
                elif sa == "reject" and sb == "accept":
                    d["rej_to_acc"] += 1
            off = offered_share(row)
            if off is not None:
                if sa == "accept":
                    d["base_acc_share"].append(off)
                if sb == "accept":
                    d["alt_acc_share"].append(off)
            ma, mb = my_share_of(a, row), my_share_of(b, row)
            if ma is not None:
                d["base_demand"].append(ma)
            if mb is not None:
                d["alt_demand"].append(mb)
    return {"label": label, "buckets": buckets}


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def print_comparison(result, order=("STRONG", "EVEN", "WEAK", "HIDDEN", "ALL")):
    print("\n=== %s" % result["label"])
    print("%-8s %6s %7s %9s %9s   %-19s %-19s" % (
        "bucket", "n", "flip%", "acc->rej", "rej->acc",
        "mean share ACCEPTED", "mean share DEMANDED"))
    for key in order:
        d = result["buckets"].get(key)
        if not d or not d["n"]:
            continue
        print("%-8s %6d %6.1f%% %9d %9d   %6.3f -> %-8.3f  %6.3f -> %-8.3f" % (
            key, d["n"], 100.0 * d["flip"] / d["n"],
            d["acc_to_rej"], d["rej_to_acc"],
            mean(d["base_acc_share"]), mean(d["alt_acc_share"]),
            mean(d["base_demand"]), mean(d["alt_demand"])))


# --- entry points -----------------------------------------------------------

KNOB_ALTERNATIVES = {
    "rate_clamp": "on",
    "cave_prior": "0.10",
    "hazard": "0.08",
    "field_prior": "off",
    "post_floor_cap": "off",
    "strong_push": "0.0",
}


def verify(rows) -> int:
    """Replay each row under the arm it was actually played on and compare.

    This is the load-bearing check: if replay does not reproduce the logged
    action, every other number this tool prints is unreliable. Rows are grouped
    by the arm their agent was running at the time, resolved by `arm_for()`.
    """
    bad = defaultdict(int)
    total = defaultdict(int)
    examples = []
    stale = sum(1 for r in rows if r.get("ts", 0) < _ab_start())
    if stale:
        print("\n  NOTE: %d of %d states predate the last deploy (%s) and were "
              "produced by\n  code that no longer exists. Fidelity is only "
              "meaningful after it." % (stale, len(rows), AB_START_ISO))
        rows = [r for r in rows if r.get("ts", 0) >= _ab_start()]
        if not rows:
            print("  no post-deploy states in this window; widen --hours")
            return 1
    by_agent = defaultdict(list)
    for row in rows:
        # Group by the arm actually in force, not merely by agent -- the arm
        # map changes at the A/B start and a state replayed under the wrong arm
        # reports a mismatch that looks like a harness bug.
        key = (row["_agent"], arm_for(row["_agent"], row.get("ts", 0)))
        by_agent[key].append(row)
    for (agent, arm), group in by_agent.items():
        got = replay(group, arm, "")
        for row, action in zip(group, got):
            total[(agent, arm)] += 1
            want = row.get("action") or {}
            same = summarise(action) == summarise(want)
            if same and "decision" not in want:
                a, b = my_share_of(action, row), my_share_of(want, row)
                same = a is not None and b is not None and abs(a - b) < 1e-6
            if not same:
                bad[(agent, arm)] += 1
                if len(examples) < 5:
                    examples.append((agent, arm, row, want, action))
    print("\n=== REPLAY FIDELITY (does the harness reproduce reality?)")
    worst = 0.0
    for key in sorted(total):
        agent, arm = key
        rate = 100.0 * bad[key] / total[key]
        worst = max(worst, rate)
        print("  %-8s arm=%-8s n=%-6d mismatched %d (%.2f%%)" % (
            agent, arm, total[key], bad[key], rate))
    for agent, arm, row, want, got in examples:
        print("\n  -- mismatch %s/%s round %s type %s" % (
            agent, arm, row.get("round"), row.get("action_type")))
        print("     logged : %s" % json.dumps(want, default=str)[:180])
        print("     replay : %s" % json.dumps(got, default=str)[:180])
    if worst > 1.0:
        print("\n  !! Replay does not reproduce reality. Every other number "
              "from this tool is unreliable until this is fixed.")
        print("""
  Across policy regimes this is expected. The harness replays historical states
  through TODAY'S constants and has no notion of code vintage, so it reports a
  mismatch wherever a constant moved between the logged game and the current
  checkout. Within a single regime fidelity is exact:

      python3 analysis/replay.py --verify --since 2026-08-23

  Use --since, not --hours: --hours is relative to now, so the state set slides
  between runs and the denominator is not reproducible.

  The constants that moved during the run were post_floor_cap (4 Aug),
  demand_jitter 0.04 -> 0.25 (5 Aug), demand_scale -> 0.88 (8 Aug), and the
  acceptance threshold and field table (22 Aug).""")
        return 1
    print("\n  Replay is faithful. Counterfactual knob comparisons are "
          "meaningful (for OUR action only -- never for the opponent's).")
    return 0


#: When the bargaining A/B started. BEFORE this instant both agents ran
#: `current`; after it, Clod ("main") switched to `legacy` and Athena stayed on
#: `current`. Replay fidelity depends on getting this right -- verifying a
#: pre-switch state against the post-switch arm map reports a mismatch that
#: looks like a broken harness but is really a mislabelled experiment.
#:
#: This instant is ALSO the last deploy, which matters more than it looks.
#: Replay can only reproduce states generated by the code currently checked
#: out. Verified against states older than the deploy, fidelity is 16-20% --
#: not because the harness is broken but because the policy that produced those
#: states no longer exists. Counterfactual knob comparisons on pre-deploy states
#: are still well-posed ("would THIS code accept THAT offer?"), but the state
#: distribution is off-policy, so treat share statistics from them as
#: indicative only.
AB_START_ISO = "2026-08-03T23:30:00+00:00"


def _ab_start() -> float:
    import datetime as _dt
    return _dt.datetime.fromisoformat(AB_START_ISO).timestamp()


def arm_for(agent: str, ts: float) -> str:
    """Which arm this agent was actually running at this instant."""
    if ts < _ab_start():
        return "current"
    return {"athena": "current", "main": "legacy"}.get(agent, "current")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/*.moves.jsonl")
    ap.add_argument("--family", default="bargaining")
    ap.add_argument("--hours", type=float, default=None,
                    help="only replay states from the last N hours. RELATIVE TO "
                         "NOW, so the state set slides between runs; prefer "
                         "--since when quoting a number.")
    ap.add_argument("--since", metavar="YYYY-MM-DD", default=None,
                    help="only replay states on or after this UTC date. Fixed, "
                         "so two runs a week apart compare the same states.")
    ap.add_argument("--limit", type=int, default=40000)
    ap.add_argument("--decisions-only", action="store_true",
                    help="accept/reject turns only, where the bug lives")
    ap.add_argument("--bisect", action="store_true",
                    help="toggle each knob singly from `current`")
    ap.add_argument("--arms", action="store_true",
                    help="whole-bundle current vs legacy")
    ap.add_argument("--knobs", default=None,
                    help="explicit knob string to compare against `current`")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--group", default="leverage",
                    choices=["leverage", "delta"],
                    help="bucket by relative patience (leverage) or by our own "
                         "discount factor (delta). Holding out only pays where "
                         "waiting is free, so 'delta' is the decision axis.")
    ap.add_argument("--post-deploy", action="store_true",
                    help="only states generated by the code checked out now "
                         "(on-policy); off-policy states still answer the "
                         "counterfactual but skew the share statistics")
    args = ap.parse_args()

    global GROUPING
    GROUPING = args.group
    order = (DELTA_ORDER if args.group == "delta"
             else ("STRONG", "EVEN", "WEAK", "HIDDEN", "ALL"))

    import time
    since = None
    if args.since:
        since = dt.datetime.strptime(
            args.since, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp()
    elif args.hours:
        since = time.time() - args.hours * 3600
    if args.post_deploy:
        since = max(since or 0.0, _ab_start())

    rows = list(iter_moves(args.logs, args.family, since, args.limit))
    if args.decisions_only:
        rows = [r for r in rows if r.get("action_type") != "offer"]
    if not rows:
        print("no rows matched", file=sys.stderr)
        return 2
    print("replaying %d real states (%s)%s" % (
        len(rows), args.family,
        ", decisions only" if args.decisions_only else ""))
    counts = defaultdict(int)
    for r in rows:
        counts[leverage(r)] += 1
    print("leverage mix: %s" % dict(counts))

    if args.verify:
        return verify(rows)

    base = replay(rows, "current", "")

    if args.arms:
        alt = replay(rows, "legacy", "")
        print_comparison(compare(rows, base, alt, "current -> legacy (all six)"),
                         order=order)

    if args.bisect:
        print("\nSingle-knob toggles from `current`. A knob that moves nothing "
              "is not the cause;\na knob that reproduces the whole-bundle "
              "effect on its own is.")
        for name, value in KNOB_ALTERNATIVES.items():
            alt = replay(rows, "current", "%s=%s" % (name, value))
            print_comparison(compare(rows, base, alt,
                                     "%s -> %s" % (name, value)),
                             order=order)

    if args.knobs:
        alt = replay(rows, "current", args.knobs)
        print_comparison(compare(rows, base, alt, "current + %s" % args.knobs),
                         order=order)

    os.environ["GLEE_BARG_ARM"] = "current"
    os.environ.pop("GLEE_BARG_KNOBS", None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
