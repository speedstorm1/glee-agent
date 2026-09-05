#!/usr/bin/env python3
"""Recover the percentile each game actually earned, by inverting the rating.

The scoring rule hides the thing we most need to optimize. Our payoff is turned
into a percentile against every payoff on the same configuration in the same
role, adjusted for opponent strength -- and we are never shown that
distribution. But the rating update is public:

    game_rating = 2000 + 8000 * (percentile - 0.5)
    delta_R     = eta * (game_rating - R)                   (raw rating)
    displayed   = 1000 + (R - 1000) * g / (g + 30)

So a rating sample either side of one completed game inverts all the way back
to that game's adjusted percentile:

    R_raw        = 1000 + (displayed - 1000) * (g + 30) / g
    game_rating  = R_before + (R_after - R_before) / eta
    percentile   = 0.5 + (game_rating - 2000) / 8000

Attribution is only sound when `games_played` for a family advanced by exactly
one between two samples, so we keep only those pairs and discard the rest.

eta follows the documented schedule (1% decaying to 0.2% by ~120 games); it is
a scale factor on the recovered percentile, so even if the exact functional
form is off, comparisons WITHIN a family at similar game counts stay valid --
which is what A/B-ing a policy change needs.

    python analysis/percentiles.py [--agent athena]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import math
import statistics
import sys
from collections import defaultdict


#: Multiplier on the documented eta schedule, calibrated from our own data.
#: The docs say "1% decaying to 0.2% by ~120 games", but a percentile lies in
#: [0,1], so every observed rating step implies eta >= |dR| / headroom -- and
#: the largest such implication came out ~1.8x above the documented value.
#: Too small an eta inflates every recovered percentile away from 0.5, which
#: is how a bucket ended up reporting an impossible -0.316.
ETA_SCALE = float(os.environ.get("GLEE_ETA_SCALE", "1.84"))


def eta(games_played: int) -> float:
    base = 0.002 if games_played >= 120 else (
        0.01 + (0.002 - 0.01) * games_played / 120.0)
    return base * ETA_SCALE


def raw_from_display(display: float, g: int) -> float:
    if g <= 0:
        return 1000.0
    return 1000.0 + (display - 1000.0) * (g + 30.0) / g


def load(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def implied_percentile(samples: list[dict], family: str,
                       since: float | None = None,
                       until: float | None = None,
                       agent: str | None = None) -> dict | None:
    """Mean percentile over a window, WITHOUT per-game attribution.

    This is the trustworthy estimator and it is the headline number.

    `recover()` pairs each rating change to a specific game, which requires the
    server's rating update to land in the same sample as its games_played
    increment. When that slips, the change is attributed to the wrong game.
    Here we only use the endpoints:

        R1 = R0 + sum_i eta * (G_i - R_i)   =>   G_mean ~ R_mean + (R1-R0)/(N*eta)

    No pairing, no clipping, and the SIGN is unambiguous however wrong eta is:
    a falling raw rating means our games are scoring below our rating, full
    stop. The two estimators disagreed once by 0.06 -- attributed said a
    bargaining change was a +0.07 improvement while the raw rating was visibly
    falling on both agents -- and the attributed one was wrong.
    """
    pts = []
    for row in sorted(samples, key=lambda r: r["ts"]):
        if since is not None and row["ts"] < since:
            continue
        if until is not None and row["ts"] >= until:
            continue
        if agent is not None and row.get("agent") != agent:
            continue
        sc = (row.get("scores") or {}).get(family)
        if not sc:
            continue
        g = int(sc["games_played"])
        if g > 0:
            pts.append((row["ts"], raw_from_display(float(sc["rating"]), g), g))
    if len(pts) < 5:
        return None
    r0, r1 = pts[0][1], pts[-1][1]
    n = pts[-1][2] - pts[0][2]
    if n < 20:
        return None
    e = eta(pts[-1][2])
    game_rating = (r0 + r1) / 2.0 + (r1 - r0) / (n * e)
    return {"family": family, "games": n, "raw_first": r0, "raw_last": r1,
            "game_rating": game_rating,
            "percentile": 0.5 + (game_rating - 2000.0) / 8000.0,
            "rising": r1 > r0}


def recover(samples: list[dict]) -> list[dict]:
    """Percentiles from consecutive samples that differ by exactly one game."""
    out = []
    samples = sorted(samples, key=lambda r: r["ts"])
    # Key on (agent, family), NOT family alone. Globbing logs/*.ratings.jsonl
    # interleaves two agents whose game counts differ, so a family-only key
    # differences one agent's rating against the other's -- producing deltas
    # of up to 31 games in the WRONG DIRECTION and quietly corrupting every
    # attributed percentile, and with them every bucket ranking built on one.
    prev: dict[tuple, tuple[float, int]] = {}
    for row in samples:
        agent = row.get("agent", "?")
        for family, sc in (row.get("scores") or {}).items():
            key = (agent, family)
            disp, g = float(sc["rating"]), int(sc["games_played"])
            if key in prev:
                disp0, g0 = prev[key]
                if g - g0 == 1 and g0 > 0:
                    r0 = raw_from_display(disp0, g0)
                    r1 = raw_from_display(disp, g)
                    e = eta(g0)
                    game_rating = r0 + (r1 - r0) / e
                    pct = 0.5 + (game_rating - 2000.0) / 8000.0
                    # Individual estimates are noisy (attribution slips when
                    # the server's rating update lags its games_played
                    # increment), so clip to the range a percentile can
                    # actually take. Means over many games stay usable; single
                    # games do not.
                    pct = max(0.0, min(1.0, pct))
                    out.append({"ts": row["ts"], "family": family,
                                "after_game": row.get("after_game"),
                                "games_played": g,
                                "game_rating": game_rating,
                                "percentile": pct})
            prev[key] = (disp, g)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="*")
    ap.add_argument("--logs", default="logs")
    args = ap.parse_args()

    samples = load(os.path.join(args.logs, f"{args.agent}.ratings.jsonl"))
    if not samples:
        print("no rating samples yet — they are written as games complete")
        return 0
    games = {g["game_id"]: g
             for g in load(os.path.join(args.logs, f"{args.agent}.games.jsonl"))}

    recovered = recover(samples)
    print(f"{len(samples)} rating samples -> {len(recovered)} cleanly attributed games\n")

    # Headline: attribution-free, computed from rating endpoints alone.
    print("=" * 74)
    print("IMPLIED PERCENTILE (attribution-free -- the trustworthy estimator)")
    print("=" * 74)
    agents = sorted({r.get("agent") for r in samples if r.get("agent")})
    print(f"{'family':<14}{'agent':<9}{'games':>7}{'raw first':>11}"
          f"{'raw now':>10}{'implied pct':>13}")
    free = {}
    for family in ("bargaining", "negotiation", "persuasion"):
        vals = []
        for ag in agents:
            got = implied_percentile(samples, family, agent=ag)
            if not got:
                continue
            vals.append(got["percentile"])
            print(f"{family:<14}{ag:<9}{got['games']:>7}{got['raw_first']:>11.0f}"
                  f"{got['raw_last']:>10.0f}{got['percentile']:>13.3f}")
        if vals:
            free[family] = statistics.mean(vals)
    if not recovered:
        print("none attributable yet (needs consecutive samples one game apart)")
        return 0

    by_family = defaultdict(list)
    for r in recovered:
        by_family[r["family"]].append(r)

    print()
    print("=" * 74)
    print("ATTRIBUTED per-game (ordinal use only -- biased, see divergence check)")
    print("=" * 74)
    print(f"{'family':<14}{'n':>5}{'mean pct':>10}{'median':>9}{'mean rating':>13}"
          f"{'>50th':>8}")
    print("=" * 74)
    for family, rows in sorted(by_family.items()):
        pcts = [r["percentile"] for r in rows]
        above = sum(1 for p in pcts if p > 0.5) / len(pcts)
        print(f"{family:<14}{len(rows):>5}{statistics.mean(pcts):>10.3f}"
              f"{statistics.median(pcts):>9.3f}"
              f"{statistics.mean(r['game_rating'] for r in rows):>13.0f}"
              f"{above:>7.0%}")

    # Divergence alarm: if the two estimators of the same hidden quantity
    # disagree beyond their error bars, one of them is lying and no downstream
    # conclusion is safe. This check would have caught the bargaining call.
    print()
    print("=" * 74)
    print("ESTIMATOR CROSS-CHECK")
    print("=" * 74)
    flagged = False
    for family, rows in sorted(by_family.items()):
        if family not in free:
            continue
        attr = statistics.mean(r["percentile"] for r in rows)
        se = (statistics.stdev([r["percentile"] for r in rows])
              / math.sqrt(len(rows))) if len(rows) > 1 else 1.0
        gap = attr - free[family]
        bad = abs(gap) > max(0.02, 3 * se)
        flagged |= bad
        print(f"  {family:<13} attributed {attr:.3f}  free {free[family]:.3f}  "
              f"gap {gap:+.3f}{'   <-- DISAGREE' if bad else ''}")
    if flagged:
        print("  !! Estimators disagree. Trust the attribution-free column;")
        print("     treat attributed numbers as ordinal within a family only.")

    # eta-independent view. The recovered percentile is only as good as the
    # assumed eta schedule, but the SIGN of a rating change is not: raw R rises
    # exactly when a game scored above our current rating and falls when it
    # scored below, whatever eta is. And because R is an EMA of game_rating, the
    # raw rating IS the quality estimate -- it converges to our mean game
    # rating. Note the DISPLAYED rating climbs on its own as g grows (the
    # g/(g+30) shrinkage relaxes), so a rising display number proves nothing.
    print()
    print("=" * 74)
    print("RAW RATING TREND  (eta-independent: does not depend on the decay model)")
    print("=" * 74)
    per_agent: dict[tuple[str, str], list[tuple[float, float, int]]] = defaultdict(list)
    for row in sorted(samples, key=lambda r: r["ts"]):
        for family, sc in (row.get("scores") or {}).items():
            g = int(sc["games_played"])
            if g > 0:
                per_agent[(row.get("agent", "?"), family)].append(
                    (row["ts"], raw_from_display(float(sc["rating"]), g), g))
    print(f"{'agent/family':<28}{'raw first':>11}{'raw now':>10}{'change':>9}"
          f"{'games':>7}{'up-moves':>10}")
    for (agent, family), series in sorted(per_agent.items()):
        if len(series) < 2:
            continue
        first, now = series[0][1], series[-1][1]
        # Compare each sample against the raw rating the PREVIOUS one implies
        # at the NEW game count. Comparing raw values directly is biased: raw
        # is display unshrunk by (g+30)/g, which falls as g grows even at
        # constant skill, so ordinary games were being miscounted as
        # regressions and this metric read ~47% while ratings clearly climbed.
        ups = moves = 0
        for a, b in zip(series, series[1:]):
            if b[2] == a[2]:
                continue
            disp_a = 1000.0 + (a[1] - 1000.0) * a[2] / (a[2] + 30.0)
            baseline = raw_from_display(disp_a, b[2])
            if abs(b[1] - baseline) <= 1e-9:
                continue
            moves += 1
            ups += b[1] > baseline
        frac = f"{ups}/{moves}" if moves else "-"
        print(f"{agent + '/' + family:<28}{first:>11.0f}{now:>10.0f}"
              f"{now - first:>+9.0f}{series[-1][2]:>7}{frac:>10}")
    print("  up-moves = games scored ABOVE our rating at the time; >50% means improving.")

    # Calibrate eta instead of trusting the documented schedule. A percentile
    # lives in [0,1], so game_rating lies in [-2000, 6000]; every observed step
    # therefore implies eta >= |delta_R| / (bound - R). Taking the largest such
    # implication over all steps gives a hard lower bound on the true eta. If
    # that bound exceeds the assumed value, our percentiles are overstated
    # (a too-small eta inflates delta_R / eta) and should be rescaled.
    print()
    print("=" * 74)
    print("ETA CALIBRATION  (is the assumed decay schedule consistent with reality?)")
    print("=" * 74)
    bound = 0.0
    assumed: list[float] = []
    for (agent, family), series in sorted(per_agent.items()):
        for a, b in zip(series, series[1:]):
            if b[2] - a[2] != 1:
                continue
            d = b[1] - a[1]
            if abs(d) < 1e-9:
                continue
            headroom = (6000.0 - a[1]) if d > 0 else (a[1] + 2000.0)
            bound = max(bound, abs(d) / headroom)
            assumed.append(eta(a[2]))
    if assumed:
        mean_assumed = statistics.mean(assumed)
        print(f"  assumed eta (docs schedule) : {mean_assumed:.5f}")
        print(f"  hard lower bound from data  : {bound:.5f}")
        if bound > mean_assumed:
            print(f"  -> assumed eta is TOO SMALL by at least {bound / mean_assumed:.1f}x; "
                  f"reported percentiles are overstated in magnitude (too far from 0.5).")
        else:
            print("  -> consistent; no contradiction between the data and the schedule.")
        print("  Treat absolute percentiles as provisional until this is tight;")
        print("  the raw-rating trend above is unaffected by eta.")

    # Which configurations are we bad at? That is where tuning pays.
    print()
    print("=" * 74)
    print("WORST CONFIGURATIONS  (mean percentile, n>=2) — where to spend effort")
    print("=" * 74)
    buckets = defaultdict(list)
    for r in recovered:
        g = games.get(r.get("after_game") or "")
        if not g:
            continue
        cfg = g.get("config") or {}
        if r["family"] == "bargaining":
            key = (f"barg d_me={cfg.get('delta_1')}/{cfg.get('delta_2')} "
                   f"T={cfg.get('max_rounds')} CI={cfg.get('complete_information')}")
        elif r["family"] == "negotiation":
            key = (f"nego vs={cfg.get('player_1_value')}/{cfg.get('player_2_value')} "
                   f"T={cfg.get('max_rounds')}")
        else:
            key = (f"pers p={cfg.get('p')} v={cfg.get('v')} price={cfg.get('product_price')} "
                   f"mode={cfg.get('seller_message_type')}")
        buckets[(key, g.get("your_player"))].append(r["percentile"])
    rows = [(statistics.mean(v), len(v), k) for k, v in buckets.items() if len(v) >= 2]
    for mean_p, n, (key, role) in sorted(rows)[:15]:
        print(f"  {mean_p:>6.3f}  n={n:<3} {role:<9} {key}")
    if not rows:
        print("  (need more games per configuration)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, ".")
    raise SystemExit(main())
