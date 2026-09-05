# A Deterministic Agent for GLEE

The agent behind two entries in the [GLEE competition](https://glee-competition.com),
`Clod` and `Athena`, and the code behind the paper submitted to the GLEE
Competition Paper Track at the 1st Workshop on Interpreting Agent Behavior,
NeurIPS 2026.

GLEE is three two-player economic games played through natural language:
bargaining, negotiation and persuasion. Every action here comes from an exact
solver. There is no model in the decision path, the policy is stateless, and any
logged game replays turn for turn.

```sh
./run_tests.sh      # 72 tests, ~3s, standard library only, no installs
```

## Four ideas

### 1. The declared incomplete information is not incomplete

`glee/solvers/negotiation.py`, `opponent_value_beliefs()`, `_surplus_possible()`

Negotiation declares valuations private. They are drawn as `F x M` from a
published grid of four factors and three magnitudes, and the twelve products are
pairwise distinct. Our own valuation therefore identifies `M`, and the
counterpart's must be one of four known numbers. That four-point support is the
opponent model: there is no belief to update, only an enumeration. Where the
enumeration shows no overlap, no trade exists and the correct play is to quote
our own value.

### 2. The last price is an ultimatum, not the end of a schedule

`glee/solvers/negotiation.py`, `ULTIMATUM_ACCEPT_P`, `optimal_posted_price()`

Whoever speaks last at round `T-1` is making a take-it-or-leave-it offer over a
discrete type space, not the final step of a concession curve. In
complete-information `T=10`, where the last price is ours, captured surplus went
from 0.544 to 0.981. The four cells where we do not speak last moved between
-4.2% and +8.3% over the same days, which is the control.

### 3. Absolute patience decides whether holding out pays

`glee/solvers/bargaining.py`, the demand jitter

The agent's demand carries a seeded per-game jitter. Widening it from +/-4% to
+/-25% makes variation in our own action exogenous within a state, which
identifies the acceptance curve that normal play cannot. The resulting payoff
curve is concave and its sign flips on our own discount factor: where delay
costs us, the optimum sits at 0.979 of the shipped schedule (t = -7.44), and
where waiting is free, asking more pays (t = +3.89).

### 4. Percentile scoring is not expected-value scoring

`glee/refpool.py`, `glee/solvers/persuasion.py`

The leaderboard turns payoff into a percentile within configuration, so the
objective is `P(accept) * profit^0.5`, not expected value. Above the mode that
map is concave and banking a zero is expensive; below it, the same map rewards
risk-seeking. Persuasion scores marginal purchases against a reference
distribution of what the field earns in that exact cell, 360 strata, shipped in
`glee/data/`.

## Layout

| path | what |
|---|---|
| `glee/solvers/negotiation.py` | Four-point type support, Boulware schedule, the `T-1` posted-offer rule |
| `glee/solvers/bargaining.py` | Backward induction and Rubinstein SPE, posterior over discount factors, demand and acceptance |
| `glee/solvers/persuasion.py` | Buyer cascade, seller ordering, lie budget kept on realized counts |
| `glee/transport.py` | Turn dispatch, rate governor, per-family daily caps |
| `glee/safety.py` | The sanitizer every action passes before it is sent |
| `glee/rng.py` | Seeded per-game randomness, `Random(H(game_id ‖ salt))` |
| `glee/refpool.py`, `glee/data/` | Persuasion reference pool, 360 strata over 41,771 games |
| `glee/advisor.py` | Optional language-model advisor for message wording. Off by default |
| `analysis/replay.py` | Pushes a logged state back through the policy and compares the action |
| `analysis/percentiles.py` | Recovers a game's percentile from its displayed rating |

Reading order: `glee/transport.py` for how a turn arrives, then whichever solver
interests you, then `glee/safety.py` for what happens on the way out. Most
constants in the solvers carry the measurement that produced them.

## Reproducing

The tests run against the shipped code and need nothing else. The paper's
replay-fidelity result (527,268 decisions, zero mismatches) is produced by
`analysis/replay.py --verify`, which reads per-turn logs. Those logs are about
8 GB and are not in this repository; point `GLEE_LOG_DIR` at your own if you
have them. Everything under `glee/` and `tests/` runs without them.

## Setup

Python 3.11+. The tests need nothing else. To play live games, see
`requirements.txt`.

```sh
cp .env.example .env    # fill in your competition API key
python run_agent.py --agent main --families bargaining,negotiation,persuasion
```

## License

MIT. See `LICENSE`.
