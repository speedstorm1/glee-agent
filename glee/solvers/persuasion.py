"""Persuasion (repeated strategic information transmission).

Price pi, prior p, buyer value v for HIGH and u for LOW (u = 0 in the
configurations played). Under commitment (Kamenica & Gentzkow 2011) the
seller's optimal policy is: recommend every HIGH item, and recommend a LOW item
with probability

    q* = min{ p(v - pi) / ((1 - p) * pi),  1 }

chosen exactly so the buyer's posterior on a recommendation leaves them
indifferent, hence willing to buy. Without commitment the one-shot game babbles
(Crawford & Sobel 1982); repetition supplies the missing commitment through
reputation (Best & Quigley 2024, cited by the GLEE paper).

Over the paper's grid, p in {1/3, 1/2, 0.8} and v in {1.2, 1.25, 2, 3, 4}
(v in price units), that splits the world into three regimes:
  * p*v > pi  (6/15 cells): the prior alone justifies buying.
  * p*v = pi  (3/15): knife edge.
  * p*v < pi  (6/15): trust is scarce and q* is small.

Both seats replay their beliefs from `history` each turn, so the solver is
stateless and survives a process restart mid-game.
"""

from __future__ import annotations

import logging
import math
import os
import re

from .. import refpool
from ..rng import unit

logger = logging.getLogger("glee.persuasion")

#: P(high | the seller declined to recommend): 4,280 purchases made against a
#: refusal, 99 of them high. Break-even needs price/v, i.e. 25% at the richest
#: cell (v = 4*price), so a refusal is a veto everywhere the grid reaches.
P_HIGH_AFTER_REFUSAL = 0.0231

#: How close to indifference the expected-value call must be before the
#: percentile objective may overrule it, as a fraction of price.
PCT_MARGIN = 0.10

#: Minimum percentile edge required to overrule EV. Reference strata hold ~115
#: samples, so a percentile carries a standard error near 4.7pp. The structure
#: worth acting on, the atom at zero, is 12-36pp.
PCT_DEADBAND = 0.03

#: Grid of v/price ratios, used when the mechanism hides v from the seller.
V_RATIO_GRID = (1.2, 1.25, 2.0, 3.0, 4.0)


def seller_policy() -> str:
    """Which seller policy to run. Read at CALL time, never bound at import.

    "kg"          Kamenica-Gentzkow: recommend every high unit plus a fraction
                  q* of low ones, chosen to leave a Bayesian buyer exactly at
                  break-even.

    "always_yes"  Recommend everything. The signal becomes uninformative, so
                  P(high | rec) = p and a Bayesian buyer buys iff p*v >= price.

    Our own recommendation rate is endogenous: `solve_seller` escalates to
    q* = 1.0 when `_buyer_ignores_our_advice` fires and backs off when trust
    breaks, so observed buyer behavior cannot be read as a response to the
    rate. This flag exists to randomize it.
    """
    return os.environ.get("GLEE_PERS_SELLER", "kg")


# Rounds of unbroken honesty before we start using the lie budget.
TRUST_BUILDING_ROUNDS = 3
#: Rounds of recent behavior weighed alongside the full history when judging a
#: seller, whose conduct can change mid-game.
RECENT_WINDOW = 8
#: Per-observation decay on direct quality evidence, newest first. Mild enough
#: that a consistent seller is still judged on their full record.
RECENCY_DECAY = 0.95
# Reputation has no continuation value in the final round, so it gets spent.
ENDGAME_ROUNDS = 1

# Phrase-level directives, not bare keywords. Against 359 labeled seller
# messages, bare keywords fail on inflected forms ("I am PASSING on this
# round") and on meta-clauses about other rounds said inside a pitch. So we
# match directives about THIS unit and weight the opening sentence.
_BUY_PHRASES = (
    "buy this", "buy it", "take this one", "take it", "i recommend this",
    "recommend this one", "recommend buying", "worth buying", "worth taking",
    "worth securing", "worth it", "i'd take it", "i would take it",
    "you should buy", "yes, buy", "yes. buy", "strongest unit", "high quality",
    "genuinely high", "dependable value", "a reasonable pick",
    "locks the upside", "lean toward yes", "leaning toward yes", "i lean yes",
    "good deal", "decent deal", "strong recommendation", "i recommend it",
    "push this", "worth securing",
)
_PASS_PHRASES = (
    "pass on this", "passing on this", "i am passing", "i'm passing",
    "recommend passing", "i would pass", "i'd pass", "skip this",
    "i'd skip", "i would skip", "do not recommend", "don't recommend",
    "not recommend", "cannot recommend", "can't recommend", "wouldn't recommend",
    "would not recommend", "not this one", "no. skip", "don't buy",
    "do not buy", "avoid this", "hold off", "not worth",
    "low quality", "poor quality", "weak unit", "decline this",
    "give up the sale", "giving up a sale", "give up this sale",
    "lose a sale by saying", "lose the round by saying",
    "i lose a sale", "saying no is the right",
    # "skipping" family: common, and entirely missed by "skip"-prefixed patterns
    "skipping", "recommend skipping", "i am skipping", "i'm skipping",
    "agree with skipping", "no, i am skip",
    # explicit refusals to pitch
    "cannot honestly push", "can't honestly push", "not going to push",
    "cannot push", "won't push",
)

#: Words that invert the phrase immediately following them. Without this,
#: "telling you not to buy it" scores as a recommendation, since it contains
#: "buy it".
_NEGATORS = ("not ", "n't ", "never ", "rather than ", "avoid ", "no ", "cannot ",
             "refuse to ", "won't ", "will not ")
_NEG_WINDOW = 18


def _negated(chunk: str, at: int) -> bool:
    window = chunk[max(0, at - _NEG_WINDOW):at]
    return any(neg in window for neg in _NEGATORS)


def _directive_score(text: str) -> int:
    """Positive means the seller is pushing this unit, negative means warning off.

    The opening sentence carries triple weight: sellers state the instruction
    first and then justify it, and the justification is exactly where words
    about OTHER rounds ("you will see me pass on rounds I could have pushed")
    leak in and invert a naive keyword match.
    """
    text = " ".join(str(text).strip().lower().split())
    if not text:
        return 1
    parts = re.split(r"(?:\. |; |! |\? | — | -- )", text, maxsplit=1)
    head, tail = parts[0], (parts[1] if len(parts) > 1 else "")

    def score(chunk: str) -> int:
        total = 0
        for phrase in _BUY_PHRASES:
            at = chunk.find(phrase)
            if at >= 0:
                total += -1 if _negated(chunk, at) else 1
        for phrase in _PASS_PHRASES:
            at = chunk.find(phrase)
            if at >= 0:
                total += 1 if _negated(chunk, at) else -1
        return total

    return 3 * score(head) + score(tail)


def _is_recommendation(state: dict) -> bool:
    """Did the seller just push this unit? (current round, live state)"""
    return _classify(state.get("seller_message"))


def _entry_recommended(entry: dict) -> bool:
    """Same question, for a past round replayed from history."""
    return _classify(entry.get("seller_message"))


def _classify(msg: object) -> bool:
    """Is this message urging the purchase? Binary mode is exact; text is scored."""
    if msg is None:
        return True
    text = str(msg).strip().lower()
    if text in ("yes", "true", "1", "recommend", "buy"):
        return True
    if text in ("no", "false", "0", "pass", "skip"):
        return False
    score = _directive_score(text)
    # Ties default to "recommending"; asymmetric buyer priors cover downside.
    return score >= 0


def optimal_lie_rate(p: float, v: float, price: float) -> float:
    """Kamenica-Gentzkow optimal probability of recommending a LOW unit."""
    if price <= 0 or p >= 1.0:
        return 1.0
    if p * v >= price:
        return 1.0          # the prior alone sells it; no need to be selective
    return max(0.0, min(1.0, p * (v - price) / ((1.0 - p) * price)))


# ---------------------------------------------------------------- SELLER ----

def _buyer_responsiveness(history: list[dict]) -> tuple[int, int]:
    """(buys, offers) on rounds where we recommended: our live trust meter."""
    buys = offers = 0
    for entry in history or []:
        if not _entry_recommended(entry):
            continue
        offers += 1
        if entry.get("bought"):
            buys += 1
    return buys, offers


def _within_lie_budget(history: list[dict], q_star: float, v: float,
                       price: float, rnd: int, total: int,
                       gid: str | None = None) -> bool:
    """Should we push THIS low-quality unit?

    q* is a conditional rate, P(recommend | LOW), so the budget is counted
    against LOW items seen, not against rounds. Budgeting per round instead
    over-recommends by about 1/(1-p), which is 1.5x the intended rate at
    p = 1/3.

    Counting rather than sampling keeps the realized rate on target: over only
    ~20 rounds, coin flips can hand the buyer an early burn.
    """
    graded = [e for e in history if str(e.get("quality") or "").lower()
              in ("high", "low")]

    if q_star >= 1.0:
        # p*v >= price: the blind bet already pays, so there is nothing to
        # ration and no trust to protect.
        return True

    ratio = lows_per_high(v, price)
    if graded:
        highs_rec = sum(1 for e in graded if str(e["quality"]).lower() == "high"
                        and _entry_recommended(e))
        lows_rec = sum(1 for e in graded if str(e["quality"]).lower() == "low"
                       and _entry_recommended(e))
        # Enforce the commitment policy on realized counts, not an average
        # rate: the buyer buys while P(high | recommended) >= price/v, so h
        # honest pushes afford h * (v/price - 1) dishonest ones, which is q*
        # restated. Self-correcting, and trust-building falls out free since
        # the allowance starts at zero. Which units get pushed is randomized.
        if (lows_rec + 1) > highs_rec * ratio:
            return False
        slack = highs_rec * ratio - lows_rec
        return slack >= 1.0 or unit(gid, f"liepick-{rnd}") < slack

    # Fall back to a schedule at rate q*, keyed on the GAME as well as the
    # round so which rounds we lie on is not predictable from the round number.
    return unit(gid, f"lie-{rnd}") < q_star


def lows_per_high(v: float, price: float, margin: float = 0.85) -> float:
    """Dishonest pushes affordable per honest one.

    Under the K-G policy the expected mix of recommendations is
    (1-p)q*/p lows per high, and substituting q* = p(v-pi)/((1-p)pi) collapses
    that to v/pi - 1, so the prior cancels out entirely. That is the exact
    number of low units we can push per honest high while keeping the buyer's
    posterior at their break-even.

    `margin` lands us a hair under indifference, because a counterpart who is
    even slightly suspicious (or who explores) needs the posterior strictly on
    the buy side, not exactly on the fence.
    """
    if price <= 0:
        return float("inf")
    return max(0.0, (v / price) - 1.0) * margin


def _round_hash01(rnd: int) -> float:
    """A stable pseudo-random value in [0,1) keyed by round index."""
    x = (rnd * 2654435761) % 2147483647
    return (x % 10007) / 10007.0


def _buyer_ignores_our_advice(history: list[dict], need: int = 2) -> bool:
    """Has the buyer bought units we told them to skip?

    Buying against an explicit refusal is only rational when the unconditional
    bet already pays, i.e. p*v >= price. It is therefore a clean, costless
    read on a value we are not allowed to see.
    """
    against = sum(1 for e in history or []
                  if e.get("bought") and not _entry_recommended(e))
    return against >= need


def infer_v_ratio(history: list[dict], p: float) -> float:
    """Estimate v/price on the half of the grid that hides v from the seller.

    Without an estimate, q* would be 0, `lows_per_high` would be 0, and the
    budget test would refuse every low unit for the whole game: half the seller
    grid played as pure honesty by accident rather than by choice.

    The buyer's own behavior identifies the regime for free, because their
    optimal play depends on the value we cannot see. Two reads, in order of
    strength:

      * Buying a unit we explicitly declined is only rational when p*v >= price.
        That is close to proof, so it short-circuits.
      * Otherwise their buy rate is monotone in v, and we map it onto the grid.

    Shaded deliberately low. Over-estimating v inflates the lie budget, and
    spending a budget we have not earned burns a buyer who then stops buying
    for the rest of the game; under-estimating only forgoes marginal sales.
    """
    trivial = max(1.0 / p if p > 0 else 4.0, 2.0)
    if _buyer_ignores_our_advice(history):
        return trivial

    # A raw buy rate is a trap: an honest seller earns a high one in every
    # regime, so it reads our own good behavior back as a large v. What
    # separates the regimes is whether the buyer keeps buying AFTER a lemon;
    # below p*v = price a burned buyer must become selective or lose money.
    after_burn: list[bool] = []
    burned = False
    for entry in history or []:
        if entry.get("bought") is not None and burned:
            after_burn.append(bool(entry.get("bought")))
        if (entry.get("bought")
                and str(entry.get("quality") or "").lower() == "low"):
            burned = True

    if len(after_burn) >= 4:
        rate = sum(after_burn) / len(after_burn)
        if rate >= 0.85:
            return trivial
        for threshold, ratio in ((0.60, 2.0), (0.35, 1.6), (0.20, 1.3)):
            if rate >= threshold:
                return ratio
        return V_RATIO_GRID[0]

    decided = [e for e in history or [] if e.get("bought") is not None]
    if len(decided) < 4:
        # No read yet. Open BELOW the trivial threshold at every prior (1/p is
        # 1.25 at p = 0.8, the tightest), so no evidence cannot commit us.
        return V_RATIO_GRID[0]

    # Some evidence, none of it burn-tested: cap short of the trivial regime,
    # since trust and a large v look alike here. The cap tracks 1/p, only 1.25
    # at p = 0.8, rather than a constant.
    cap = 0.99 / p if p > 0 else 4.0
    rate = sum(1 for e in decided if e.get("bought")) / len(decided)
    ratio = V_RATIO_GRID[0]
    for threshold, candidate in ((0.60, 1.6), (0.35, 1.3)):
        if rate >= threshold:
            ratio = candidate
            break
    return max(V_RATIO_GRID[0], min(ratio, cap))


def solve_seller(game: dict) -> dict:
    state = game["game_state"]
    gid = game.get("game_id")
    rnd = int(state.get("round") or 1)
    total = int(state.get("total_rounds") or 20)
    quality = str(state.get("current_quality") or "").lower()
    price = float(state.get("product_price") or 1.0)
    p = float(state.get("p") or 0.5)
    history = state.get("history") or []
    text_mode = game["valid_actions"]["type"] == "seller_message"

    # v is visible only when `is_seller_know_cv` is set, which is half the
    # grid. Otherwise the buyer's behavior reveals it; see `infer_v_ratio`.
    v = state.get("v")
    if v is not None:
        v_est = float(v)
    else:
        v_est = infer_v_ratio(history, p) * price
    q_star = optimal_lie_rate(p, v_est, price)

    high = quality == "high"
    buys, offers = _buyer_responsiveness(history)
    trust = (buys / offers) if offers else 1.0
    remaining = total - rnd

    if seller_policy() == "always_yes":
        # Everything recommended from round one, no trust-building, no backoff.
        recommend = True
    elif high:
        recommend = True                     # never pass on a genuine winner
    elif q_star >= 1.0:
        # p*v >= price: the buyer profits from buying blind, so no reputation
        # constraint binds. Must sit above the trust-building and trust-broken
        # branches, which would withhold credibility we have no use for.
        recommend = True
    elif remaining <= ENDGAME_ROUNDS:
        # Reputation has no continuation value left, so spend it. From the
        # buyer's seat, P(high | recommended) falls from 72.7% over rounds 1-18
        # to 60.8% at round 20.
        recommend = True
    elif rnd <= TRUST_BUILDING_ROUNDS:
        recommend = False                    # buy credibility before spending it
    elif offers >= 3 and trust < 0.5:
        recommend = False                    # trust is broken; rebuild, don't loot
    else:
        recommend = _within_lie_budget(history, q_star, v_est,
                                       price, rnd, total, gid)

    if not text_mode:
        return {"decision": "yes" if recommend else "no"}

    # The GLEE paper finds free text degrades both efficiency and fairness in
    # persuasion, unlike bargaining, so keep messages short and consistent.
    if not recommend:
        return {"message": "I would pass on this one. I would rather tell you to "
                           "skip a unit than sell you something that disappoints."}

    # When the prior alone clears the price, credibility is beside the point:
    # the buyer only has to do the arithmetic. Saying so converts a buyer who
    # has stopped believing our recommendations.
    if v is not None and p * float(v) > price:
        ev = p * float(v)
        return {"message": (
            f"You have no reason to take my word for it, so don't — just price it. "
            f"{p:.0%} of these are worth {float(v):,.0f} to you, the rest nothing, so a "
            f"blind purchase is worth {ev:,.0f} against a price of {price:,.0f}. "
            f"That is +{ev - price:,.0f} per unit before you weigh anything I say. "
            f"Even if you assume I am lying to you every single round, buying still "
            f"beats passing on the maths alone."
        )}

    if seller_policy() == "always_yes":
        # The default text asserts selectivity, a checkable false claim under
        # always_yes. Drop it, so this arm tests the policy and not a bluff.
        return {"message": "I recommend this one."}

    return {"message": ("I recommend this one. I only push units I would stand "
                        "behind, and you can check that against what you have "
                        "seen from me so far.")}


# ----------------------------------------------------------------- BUYER ----

def signal_frequency_estimate(history: list[dict], p: float,
                              this_round_recommended: bool) -> tuple[float, float] | None:
    """P(high | recommended) inferred from how OFTEN the seller recommends.

    Quality is hidden unless we buy, but the seller's SIGNAL is visible every
    round. Any seller worth modeling recommends every high unit, since passing
    on one is lost revenue, so if they recommend low units at rate q:

        P(rec) = p + (1 - p) q      =>      P(high | rec) = p / P(rec)

    The prior cancels out and the estimate needs zero purchases, which turns a
    pure bandit into something partly observable for free. That matters most
    where probing is expensive (p*v < price).

    Returns (estimated P(high|signal), pseudo-count weight), or None when
    there is not enough signal history to say anything.
    """
    rounds = [e for e in history or [] if e.get("seller_message") is not None]
    n = len(rounds)
    if n < 4:
        return None

    def smoothed_rate(window: list[dict]) -> float:
        return (sum(1 for e in window if _entry_recommended(e)) + 1.0) / (len(window) + 2.0)

    # Take the more PESSIMISTIC of the whole history and the recent window: a
    # seller honest for eight rounds and then recommending everything still
    # shows ~0.64, reading as P(high|rec) = 0.52 and clearing a 0.50 bar.
    rate = smoothed_rate(rounds)
    if n >= RECENT_WINDOW + 2:
        rate = max(rate, smoothed_rate(rounds[-RECENT_WINDOW:]))
    if this_round_recommended:
        if rate <= 1e-9:
            return None
        est = min(1.0, p / rate)
    else:
        # They declined: P(high | no) is 0 under a strict all-highs seller. The
        # floor allows for an imperfect one instead of asserting certainty.
        no_rate = 1.0 - rate
        if no_rate <= 1e-9:
            return None
        est = max(0.0, min(1.0, (p * max(0.0, 1.0 - p / max(rate, 1e-9)))
                           / no_rate)) if rate > 0 else p
    # Laplace-smoothed: a short run of recommendations is not read as certainty.
    return est, min(10.0, n / 1.5)


def _signal_counts(history: list[dict],
                   now: int | None = None) -> dict[bool, tuple[float, float]]:
    """RECENCY-WEIGHTED high/low counts observed per signal value.

    Quality is revealed only on rounds we actually bought, so this is a bandit
    problem: passing is safe but teaches us nothing.

    Observations decay with age, because a counterpart is not a fixed coin:
    behaving early and cashing in late is profitable against a Bayesian buyer,
    and flat counting is blind to it. Decay is mild, so a steady seller is
    still judged on their whole record.
    """
    counts = {True: [0.0, 0.0], False: [0.0, 0.0]}
    graded = [e for e in history or []
              if e.get("bought")
              and str(e.get("quality") or "").lower() in ("high", "low")]
    if not graded:
        return {True: (0.0, 0.0), False: (0.0, 0.0)}
    # Age in ROUNDS ELAPSED, not position in the purchase sequence: by
    # position, stale buys from a seller we have since avoided keep full weight.
    latest = now if now is not None else max(
        int(e.get("round") or 0) for e in graded)
    for entry in graded:
        age = max(0, latest - int(entry.get("round") or latest))
        w = RECENCY_DECAY ** age
        counts[_entry_recommended(entry)][
            0 if str(entry["quality"]).lower() == "high" else 1] += w
    return {k: (a, b) for k, (a, b) in counts.items()}


def recent_betrayals(history: list[dict], window: int = 6) -> tuple[int, int]:
    """(burns, purchases) over our most recent purchases.

    A "burn" is a unit we bought on a recommendation that came back low. The
    Beta posterior reacts to these, but slowly and with the whole game's
    history weighing it down, so a seller who behaved for twelve rounds and
    then started cashing in gets a lot of rope. This looks at the recent
    window only.
    """
    bought = [e for e in reversed(history or []) if e.get("bought")
              and str(e.get("quality") or "").lower() in ("high", "low")]
    recent = bought[:window]
    burns = sum(1 for e in recent
                if _entry_recommended(e) and str(e["quality"]).lower() == "low")
    return burns, len(recent)


def punishment_rounds(history: list[dict], q_star: float, scarce: bool,
                      window: int = 6) -> int:
    """How many rounds to sit out after being burned.

    Calibrated to q*, not to zero. A seller lying at the Kamenica-Gentzkow rate
    is playing the equilibrium, so we tolerate the expected number of burns and
    count only the excess.

    Punishment is purely behavioral: the buyer has no message channel, so the
    seller only ever observes `bought: false`. That transmits noisily, which
    argues for a response sharp enough to be legible rather than a slow drift.

    Harsher in the scarce regime (p*v < price), where a mistake costs the full
    price and the blind bet is negative.
    """
    burns, n = recent_betrayals(history, window)
    if n == 0:
        return 0
    tolerated = q_star * n + 0.5          # what equilibrium play would produce
    excess = burns - tolerated
    if excess <= 0:
        return 0
    per_burn = 4 if scarce else 2
    return min(8, int(round(excess * per_burn)))


def rounds_since_last_burn(history: list[dict]) -> int:
    """Rounds elapsed since the last time a recommendation burned us."""
    seen = 0
    for entry in reversed(history or []):
        if (entry.get("bought") and _entry_recommended(entry)
                and str(entry.get("quality") or "").lower() == "low"):
            return seen
        seen += 1
    return 10 ** 6


def _banked_payoff(state: dict, history: list[dict], v: float, u: float,
                   price: float) -> float:
    """What we have already earned this game.

    The server reports it as `buyer_total_payoff`, but the solver reconstructs
    state from history rather than trust a field to be present. This one is
    load-bearing: a stale or missing zero would score every decision as though
    the game had just started, right against the atom at zero.
    """
    reported = state.get("buyer_total_payoff")
    if isinstance(reported, (int, float)) and math.isfinite(float(reported)):
        return float(reported)
    total = 0.0
    for entry in history or []:
        if not entry.get("bought"):
            continue
        quality = str(entry.get("quality") or "").lower()
        if quality == "high":
            total += v - price
        elif quality == "low":
            total += u - price
    return total


def _percentile_verdict(state: dict, p: float, v: float, u: float, price: float,
                        total: int, rnd: int, mean: float,
                        history: list[dict]) -> bool | None:
    """Buy or stop, scored by expected percentile. None means no opinion.

    GLEE pays the percentile of our payoff within the pool that played this
    exact configuration in this role, and percentile is monotone in payoff but
    not smooth in it. The bend is an atom at exactly zero, buyers who refused
    for twenty rounds, holding 0.357 / 0.265 / 0.115 of the pool at
    p = 1/3 / 0.5 / 0.8. So two decisions with identical expected payoff can
    want opposite actions.

    The comparison is between two whole-game outcomes, not two single rounds.
    We project the terminal payoff as a Normal, buying again whenever they
    recommend again, so n_buys = 1 + remaining * P(recommend). Stopping is a
    point mass at what is already banked. Mid-rank ties make the two agree as
    the variance goes to zero, so the atom needs no special case.

    Deliberately narrow: consulted only near indifference (PCT_MARGIN), only
    when the stratum is deep and not built on imputation, and only when it
    beats the alternative by more than the pool's sampling noise
    (PCT_DEADBAND). Note that the pool is drawn from the counterparts we
    faced, so it inherits any selection our own play imposes.
    """
    if price <= 0 or total <= 0:
        return None
    st = refpool.lookup(p, v, price,
                        str(state.get("seller_message_type") or "binary"),
                        bool(state.get("is_seller_know_cv")), "buyer")
    if st is None:
        return None

    remaining = total - rnd
    if remaining < 1:
        return None

    banked = refpool.normalise(_banked_payoff(state, history, v, u, price),
                               price, total)

    # We only buy on rounds they recommend, so scale the future by that rate.
    seen = [e for e in history if e.get("seller_message") is not None]
    rec_rate = (sum(1 for e in seen if _entry_recommended(e)) / len(seen)
                if seen else max(p, 0.25))
    n_buys = 1.0 + remaining * rec_rate

    gain = (v - price) / price          # per purchase, in price units
    loss = (u - price) / price
    per_buy_mean = mean * gain + (1.0 - mean) * loss
    per_buy_var = mean * (1.0 - mean) * (gain - loss) ** 2

    mu = banked + n_buys * per_buy_mean / total
    sd = math.sqrt(max(n_buys * per_buy_var, 0.0)) / total

    go = st.expected_percentile(mu, sd)
    stop = st.percentile(banked)
    if abs(go - stop) < PCT_DEADBAND:
        return None
    return go > stop


def solve_buyer(game: dict) -> dict:
    state = game["game_state"]
    p = float(state.get("p") or 0.5)
    v = float(state.get("v") or 0.0)
    u = float(state.get("u") or 0.0)
    price = float(state.get("product_price") or 0.0)
    rnd = int(state.get("round") or 1)
    total = int(state.get("total_rounds") or 20)
    history = state.get("history") or []

    recommended = _is_recommendation(state)
    counts = _signal_counts(history, rnd)
    highs, lows = counts.get(recommended, (0, 0))

    # Beta posterior on P(high | this signal), with ASYMMETRIC priors. A
    # refusal is the stronger signal: declining costs the seller a sale, and
    # under K-G they recommend every high unit, so P(high | no) ~ 0. The
    # opening belief for a recommendation is the K-G break-even price/v, nudged
    # just under, since a strategic seller lies at exactly the rate that leaves
    # the buyer indifferent.
    breakeven = (price - u) / (v - u) if v > u else 1.0
    prior_mean = (min((p + 1.0) / 2.0, breakeven * 0.98) if recommended
                  else 0.15 * p)
    weight = 3.0
    # Fold in what the recommendation FREQUENCY already tells us: free, and
    # available from round 4, long before direct evidence accumulates.
    freq = signal_frequency_estimate(history, p, recommended)
    if freq is not None:
        est, w = freq
        # Discount the frequency estimate as DIRECT evidence accumulates: it
        # rests on the assumption that the seller recommends every high unit,
        # whereas an observed outcome does not.
        w /= (1.0 + highs + lows)
        prior_mean = (prior_mean * weight + est * w) / (weight + w)
        weight += w
    a, b = prior_mean * weight + highs, (1.0 - prior_mean) * weight + lows
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
    sd = math.sqrt(max(var, 0.0))

    def ev(prob: float) -> float:
        return prob * v + (1.0 - prob) * u - price

    remaining = max(0, total - rnd)

    # A REFUSAL IS A VETO, not evidence to be weighed: giving up a sale is the
    # one thing the seller cannot fake. Across 4,280 purchases made against a
    # refusal, 2.31% were high, realizing -0.93 each.
    if not recommended and ev(P_HIGH_AFTER_REFUSAL) < 0:
        return {"decision": "no"}

    # ENDGAME: a late recommendation carries almost no information, since any
    # seller spends reputation once it has no continuation value; its lift over
    # the prior at round 20 is +3.7 / +2.2 / +0.8 points at p = 1/3 / 0.5 / 0.8.
    # Fall back on the prior rather than stop buying, since those buys still
    # realized +0.54 per round and the veto above still handles refusals.
    if remaining <= ENDGAME_ROUNDS:
        return {"decision": "yes" if ev(p) >= 0 else "no"}

    # Near indifference, let the scoring rule break the tie; see
    # `_percentile_verdict`. Not before the recommendation rate exists, though:
    # the projection commits us to a view of the whole remaining game, and
    # until then the only belief available is a model, not an observation.
    if (abs(ev(mean)) <= PCT_MARGIN * price
            and signal_frequency_estimate(history, p, recommended) is not None):
        verdict = _percentile_verdict(state, p, v, u, price, total, rnd,
                                      mean, history)
        if verdict is not None:
            return {"decision": "yes" if verdict else "no"}

    # Floor, but only on a POSITIVE signal: if the prior alone makes buying
    # profitable, buying when urged can never do worse than the blind strategy.
    # On a refusal, skipping beats the unconditional bet.
    if recommended and ev(p) >= 0:
        return {"decision": "yes"}

    # TRIGGER: sit out a seller cashing in faster than the equilibrium rate,
    # since the Beta posterior averages over the whole game and gives a late
    # defector too much rope. Cost-gated: sitting out moves a counterpart's
    # push rate by about -5pp for ~3 rounds, and fires on 0.54% of decisions.
    q_star = optimal_lie_rate(p, v, price)
    scarce = p * v < price
    penalty = punishment_rounds(history, q_star, scarce)
    if (penalty and rounds_since_last_burn(history) < penalty and remaining > 1
            and p * v <= 1.05 * price and ev(mean) <= 0.02 * price):
        return {"decision": "no"}

    if ev(mean) >= 0:
        return {"decision": "yes"}

    # Below break-even, so buying now is a PROBE: quality is revealed only on
    # rounds we buy. Worth -ev(mean) if the option value of learning exceeds it.
    if remaining <= 1:
        return {"decision": "no"}

    if not recommended:
        # Do not pay to test a refusal: it costs the seller revenue, so they
        # only make one on genuinely bad units.
        return {"decision": "no"}

    if signal_frequency_estimate(history, p, True) is None:
        # Never buy information that is about to arrive for free: the
        # recommendation RATE identifies P(high | rec) on its own from round 4.
        return {"decision": "no"}

    probe_cost = -ev(mean)
    # Break-even honesty rate, and the posterior probability we are above it.
    theta_star = (price - u) / (v - u) if v > u else 1.0

    freq_now = signal_frequency_estimate(history, p, True)
    if freq_now is not None and freq_now[0] < theta_star:
        # Their recommendation RATE already implies the signal cannot clear
        # our bar even read perfectly. Probing bets our model is wrong.
        return {"decision": "no"}
    p_good = _beta_sf(theta_star, a, b)
    # If the seller turns out honest we capture (v - price) on the remaining
    # rounds they recommend, so the option is priced off the recommendation
    # RATE, not once per round; per-round overstates it by about 1/P(rec).
    # Halved as a hedge, since an honest seller can still defect later.
    rec_rounds = [e for e in history if e.get("seller_message") is not None]
    rec_rate = (sum(1 for e in rec_rounds if _entry_recommended(e)) / len(rec_rounds)
                if rec_rounds else max(p, 0.25))
    upside = p_good * remaining * rec_rate * max(0.0, v - price) * 0.5

    return {"decision": "yes" if upside > probe_cost else "no"}


def _beta_sf(x: float, a: float, b: float) -> float:
    """P(theta > x) for theta ~ Beta(a, b)."""
    if x <= 0.0:
        return 1.0
    if x >= 1.0:
        return 0.0
    # Deliberately NOT scipy: it is not a dependency, and play must be
    # deterministic given a game id across environments, so a conditional
    # import would make the answer depend on what is installed.
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
    sd = math.sqrt(max(var, 1e-12))
    return 0.5 * math.erfc((x - mean) / (sd * math.sqrt(2.0)))


SELLER_ACTIONS = ("seller_message", "seller_recommendation")
BUYER_ACTIONS = ("buyer_decision",)


def solve(game: dict) -> dict:
    """Dispatch to the seller or buyer policy.

    An unrecognised action type resolves by ROLE rather than falling through to
    `solve_buyer`. The two policies return different action shapes and read
    different state, so a seller-side type handled by the buyer path would emit
    a plausible-looking wrong move.
    """
    atype = game["valid_actions"]["type"]
    if atype in SELLER_ACTIONS:
        return solve_seller(game)
    if atype not in BUYER_ACTIONS:
        # Resolve by ROLE rather than guessing, and say so loudly.
        state = game.get("game_state") or {}
        me = game.get("your_player") or state.get("current_player")
        role = state.get(f"{me}_role")
        seller = (role == "seller") if role else (me == "player_1")
        logger.warning("unknown persuasion action type %r; dispatching by role "
                       "to the %s policy", atype, "seller" if seller else "buyer")
        return solve_seller(game) if seller else solve_buyer(game)
    return solve_buyer(game)
