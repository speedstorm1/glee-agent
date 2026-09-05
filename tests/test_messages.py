"""Pin the seller-message classifier against hand-written cases.

The persuasion buyer must decide whether a free-text seller message is pushing
this unit or warning us off it. Getting it backwards costs twice: we buy a unit
the seller honestly disowned, and we file that bad outcome under "recommended",
corrupting the Beta posterior every later decision reads from.

The runtime classifier is pure Python, so there is no API call on the hot path.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glee.solvers.persuasion import _classify  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def test_hand_written_cases():
    """The specific phrasings that broke the original keyword classifier."""
    must_pass = [
        "I am passing on this round. That is a sale I am giving up.",
        "I recommend passing on this round. Even though saying so means I lose.",
        "Not this one. I will lose the round by saying so.",
        "I cannot honestly push this unit at the current price. "
        "Skipping preserves your expected value.",
        "No, I am skipping this one at 10000.",
        "I recommend skipping this one. The price is 10000.",
        "Skip this one; I could have pushed it and am telling you not to buy it.",
        "I do not recommend this one. I'd pass.",
        "Low quality this round — I'd skip this one.",
        "no",
    ]
    must_buy = [
        "Buy this one — genuinely high quality and worth more than the asking price.",
        "This is the strongest unit I have put in front of you, and I am not going "
        "to say that every round. Judge me on the rounds I tell you to skip.",
        "Take this one. I would rather lose a sale than spend my word on a weak "
        "unit, and you will see me pass on rounds I could have pushed.",
        "Skip the risk and buy this one now for 10000.",
        "This one I recommend — worth buying.",
        "Yes, buy this one for 100.",
        "yes",
    ]
    for m in must_pass:
        check(not _classify(m), f"should read as PASS: {m[:80]!r}")
    for m in must_buy:
        check(_classify(m), f"should read as BUY: {m[:80]!r}")


def test_binary_mode_is_exact():
    """Binary mode must never be scored heuristically."""
    for yes in ("yes", "YES", " yes ", "true", "1", "recommend", "buy"):
        check(_classify(yes), f"binary {yes!r} should be BUY")
    for no in ("no", "NO", " no ", "false", "0", "pass", "skip"):
        check(not _classify(no), f"binary {no!r} should be PASS")
    # A missing message is not a refusal -- silence is a pitch.
    check(_classify(None), "None should default to BUY")


def main() -> int:
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        before = len(FAILURES)
        print(f"[    ] {name}")
        try:
            fn()
        except Exception as e:
            FAILURES.append(f"{name} raised {type(e).__name__}: {e}")
        print(f"\033[F[{'ok ' if len(FAILURES) == before else 'FAIL'}] {name}")
    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for f in FAILURES[:20]:
            print("  -", f)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
