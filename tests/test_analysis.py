"""Validate the percentile recovery, which the tuning loop depends on.

If this inversion is wrong we would be optimizing against a fabricated signal,
which is worse than having no signal at all -- so it is checked against
synthetic games whose true percentiles are known by construction.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from percentiles import eta, raw_from_display, recover  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def _display(raw: float, g: int) -> float:
    return 1000.0 + (raw - 1000.0) * g / (g + 30.0)


def test_display_raw_roundtrip():
    for raw in (1000.0, 1400.0, 2600.0, 3800.0):
        for g in (5, 30, 200, 2000):
            check(abs(raw_from_display(_display(raw, g), g) - raw) < 1e-6,
                  f"round-trip failed at raw={raw}, g={g}")
    # The docs' own worked example: raw 1400 at 5 games displays as ~1057.
    check(abs(_display(1400.0, 5) - 1057.14) < 0.1,
          "display formula disagrees with the documented example")


def test_recovers_known_percentiles():
    for start_raw, start_g in [(1400.0, 40), (1000.0, 3), (2600.0, 500)]:
        truth = [0.30, 0.55, 0.62, 0.71, 0.48, 0.80]
        raw, g = start_raw, start_g
        samples = [{"ts": 0, "scores": {"bargaining":
                    {"rating": _display(raw, g), "games_played": g}}}]
        for i, p in enumerate(truth):
            raw += eta(g) * ((2000 + 8000 * (p - 0.5)) - raw)
            g += 1
            samples.append({"ts": i + 1, "after_game": f"g{i}",
                            "scores": {"bargaining":
                                       {"rating": _display(raw, g),
                                        "games_played": g}}})
        got = recover(samples)
        check(len(got) == len(truth),
              f"expected {len(truth)} attributions, got {len(got)}")
        for want, r in zip(truth, got):
            check(abs(want - r["percentile"]) < 1e-6,
                  f"recovered {r['percentile']:.4f}, expected {want}")


def test_skips_ambiguous_attribution():
    """Two games between samples cannot be attributed to either one."""
    samples = [
        {"ts": 0, "scores": {"bargaining": {"rating": 1200.0, "games_played": 40}}},
        {"ts": 1, "scores": {"bargaining": {"rating": 1210.0, "games_played": 42}}},
        {"ts": 2, "after_game": "x",
         "scores": {"bargaining": {"rating": 1215.0, "games_played": 43}}},
    ]
    got = recover(samples)
    check(len(got) == 1,
          f"only the single-game step is attributable, got {len(got)}")
    check(got[0]["after_game"] == "x", "attributed to the wrong game")


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
