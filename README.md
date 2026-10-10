# headsup-poker

[![tests](https://github.com/romaf5/headsup-poker/actions/workflows/tests.yml/badge.svg)](https://github.com/romaf5/headsup-poker/actions/workflows/tests.yml)

Heads-up Texas Hold'em research code: a fast engine (Python + C++), Deep CFR / Single Deep CFR /
DREAM / ESCHER, a Pluribus-style tabular blueprint with real-time search, best-response evaluation,
and a browser table to play against the bots.

![Browser table](imgs/web-ui.png)

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # + requirements-rl.txt for the PPO exploiter
python setup.py build_ext --inplace        # C++ kernels
python -m pytest tests -q                  # ~8 min on CPU
python -m headsup.web --opponent pluribus  # play at http://127.0.0.1:8000
```

Opponents: `pluribus` (blueprint + real-time search), `tab` (blueprint), `cfr` (DeepCFR net),
`search:cfr` (net + search), `call` / `random` / `allin`. Keys: `F` `C` `R` `A`, `Enter` next hand,
`S` show the bot's cards, `D` advisor.

## Games

| game | flag | rules |
|---|---|---|
| NL abstraction | default | stacks 100, blinds 1/2, fold / call / min-raise / all-in, 3rd raise → all-in; `--bet-sizes 0.5,1,2` for pot-fraction raises |
| FHP | `--game fhp` | limit, blinds 50/100, bets 100, four bets per round (3 raises pre-flop; a bet and 3 raises on the flop), two rounds, showdown after the flop (DeepCFR paper) |
| HULH | `--game hulh` | limit, bets 100 / 100 / 200 / 200, four bets per round, four rounds |
| Leduc, Kuhn | `headsup.games` | small games with exact exploitability |

## Algorithms

| method | module | run |
|---|---|---|
| Deep CFR, SD-CFR, DREAM, ESCHER (hold'em) | `headsup/deepcfr/`, `sdcfr.py` | `python -m headsup.deepcfr.train --algo both --features history --net paper --out runs/x` |
| same on Leduc / Kuhn | `headsup/algos/deep.py` | `python -m headsup.algos.deep --game leduc --algo sdcfr` |
| Deep DCFR+ / Deep PDCFR+ (Leduc / Kuhn) | `headsup/algos/pdcfr.py` | `python -m headsup.algos.pdcfr --game leduc --variant pdcfr+` |
| NFSP (Leduc / Kuhn) | `headsup/algos/nfsp.py` | `python -m headsup.algos.nfsp --game leduc --preset paper --iterations 3000000` |
| ReBeL: search with a value network on public belief states (Leduc) | `headsup/algos/rebel.py` | `python -m headsup.algos.rebel --game leduc --epochs 300` |
| tabular CFR, CFR+, DCFR, PCFR+, DCFR+, PDCFR+, MCCFR | `headsup/algos/tabular.py` | reference solvers for the small games |
| Pluribus blueprint (Linear MCCFR + pruning) | `headsup/blueprint.py` | `python -m headsup.blueprint --game nlhe --iterations 20000000 --out runs/bp.pt` |
| real-time search (depth-limited, Pluribus mode) | `headsup/search.py` | player spec `search:<blueprint>[@pluribus]` |
| AlphaHoldem (conv-nets on card / action tensors, Trinal-Clip PPO, K-Best self-play) | `headsup/alphaholdem/` | `python -m headsup.alphaholdem.train --out runs/alpha --iterations 5000`; player spec `alpha:runs/alpha/policy.pth` |

Trainer outputs: `policy.pth` (spec `cfr:<path>`), `iterates.pt` (SD-CFR average, spec
`sdcfr:<path>`), `checkpoint.pt` (`--resume`), TensorBoard logs. `--game fhp --preset paper` uses the
paper's hyperparameters and network (`--net deepcfr`: Appendix C with the 3x wider card branch behind its
"98,948 parameters"); `python -m headsup.deepcfr.train -h` lists the network / budget options.

## Evaluation

| tool | measures |
|---|---|
| `python -m headsup.algos.holdem_br --policy <spec> --cards all` | best-response exploitability: exact over hands, boards enumerated (FHP) or sampled per street; SD-CFR averages exactly |
| `python -m headsup.lbr --policy <spec>` | Local Best Response, a lower bound |
| `python -m headsup.compare <spec> <spec> --bots` | head-to-head chips/hand ± SE |
| `python -m headsup.exploit --policy <spec>` | PPO exploiters, a weak lower bound |

**All-in EV.** When a hand goes all-in before the last card, the played-hands tools (`compare`, `lbr`, `exploit`,
`deepcfr.evaluate`) count it with its expectation over the board cards still to come instead of the one runout that
was dealt (exact on the flop and turn, 1,000 sampled runouts pre-flop): the same mean with a smaller standard error.
`--raw` turns it off. In code: `engine.allin_ev()`, `play_hands(..., allin_ev=True)`, `headsup.cards.showdown_equity`.
LBR goes one step further: it tracks the opponent's range anyway, so a showdown counts with its expectation over
that range as well (only with an exact opponent model, i.e. not with `--model-iterates`); standard deviation per
hand against the random bot: 65 chips dealt, 24 with all-in EV, 13 with the range.

**Luck-adjusted results at the browser table.** Besides the runout, a showdown has a second piece of luck: which of
the hands it plays this way the bot happened to hold. The table values every showdown against all of them, weighted
by the probability that the bot takes its actions with each (its strategy for every hand - for `pluribus` from the
round's solve), and all-in hands over the cards to come: "all-in pre-flop with 62 % equity: worth +24.0 on average …
against all the hands the bot plays this way you have 55 %: worth +10.0 on average". Both are expectations given what
the players could see, so the session's luck-adjusted win rate has the same mean as the dealt one. Its standard
deviation per hand is about half (DeepCFR's strategy against the blueprint, 600 hands: 23 chips dealt, 16 with
all-in EV, 12 luck-adjusted), i.e. the same precision with a quarter of the hands.

## Results

### Leduc: reproductions

Exploitability of the average strategy in milli-antes per game (mean over seats, the papers' unit), mean ± sd over
seeds (`python -m headsup.algos.leduc_report`):

| SD-CFR paper setup, by iteration | 500 | 1000 | 2000 | 3000 | 5000 |
|---|---|---|---|---|---|
| **SD-CFR, Steinberger (2019) Fig. 1a** | **96** | **89** | **69** | **67** | **59** |
| SD-CFR, authors' net (3 seeds) | 90 ± 16 | 81 ± 7 | 58 ± 4 | 58 ± 7 | 60 ± 5 |
| SD-CFR, plain MLP + `--masked-loss` (2) | 78 ± 4 | 74 ± 7 | 68 ± 4 | – | – |
| SD-CFR, plain MLP (3) | 76 ± 11 | 78 ± 9 | 105 ± 20 | 80 ± 8 | 80 ± 17 |
| **Deep CFR, Steinberger (2019) Fig. 1a** | **116** | **110** | **93** | **86** | **80** |
| Deep CFR, authors' net (1) | 109 | 110 | 94 | 90 | 101 |
| Deep CFR, plain MLP (3) | 135 ± 28 | 154 ± 14 | 148 ± 17 | 158 ± 6 | 143 ± 10 |

| DREAM paper setup, by nodes touched | 2.6e6 | 3.5e6 | 6.7e6 | 1e7 | 1.3e7 |
|---|---|---|---|---|---|
| **ES-SD-CFR, Steinberger et al. (2020) Fig. 2** | **67** | **57** | **49** | **46** | – |
| ES-SD-CFR, DREAM-code settings (3) | 69 ± 1 | 61 ± 3 | 51 ± 5 | 48 ± 3 | 44 ± 5 |
| ES-SD-CFR, plain MLP (3) | 87 ± 16 | 69 ± 6 | 60 ± 3 | 57 ± 2 | 51 ± 3 |
| **DREAM, Steinberger et al. (2020) Fig. 2** | **78** | **67** | **62** | **57** | **56** |
| DREAM, DREAM-code settings (3) | 90 ± 10 | 74 ± 3 | 71 ± 2 | 60 ± 6 | 56 ± 5 |
| DREAM, plain MLP (3) | 110 ± 15 | 96 ± 9 | 79 ± 8 | – | – |

```bash
# SD-CFR paper setup (Deep CFR: --algo deepcfr --strat-capacity 1000000 --policy-steps 5000)
python -m headsup.algos.deep --game leduc --algo sdcfr --iterations 5000 --traversals 1500 --adv-steps 750 --adv-batch 2048 \
  --warm-start --adv-capacity 1000000 --arch pokerrl --loss-weights normalized --grad-clip 10 --mean-regret --device cuda
# DREAM paper setup (ES-SD-CFR: --algo sdcfr --traversals 346 without the DREAM-only flags)
python -m headsup.algos.deep --game leduc --algo dream --iterations 1500 --traversals 900 --epsilon 0.5 --q-steps 1000 \
  --q-batch 512 --shared-baseline --bootstrap-chance --adv-steps 3000 --adv-batch 2048 --arch deepcfr_dueling \
  --loss-weights normalized --grad-clip 1 --device cuda
```

What it took, beyond the papers' text (all taken from the authors' code):
- **No fitting of illegal actions' outputs.** The SD-CFR / DREAM / ESCHER networks multiply them by the legal mask; a
  plain MLP fitted to zeros there stalls at ~80 mA/g (`--masked-loss` alone fixes it; dueling head and normalisation
  do not matter).
- **The DREAM code's settings differ from the SD-CFR code's:** gradient clipping 1 (not 10), ES regrets not divided by
  the number of legal actions, one baseline net for both players trained once per iteration, with expected-SARSA
  targets bootstrapped after deals.
- **The DREAM plot's x-axis** counts decision and terminal nodes only (ours / 1.23 for ES, / 1.64 for DREAM's
  sampler); the paper values are its 3-seed means (single-seed tails excluded).

**Deep (Predictive) Discounted CFR** (Xu et al. 2025): persistent cumulative-advantage networks that are bootstrapped
from their own discounted, clipped output, outcome sampling with a learned baseline, and an average-policy network.
Paper setup, 10 000 episodes per player and iteration, 3 seeds; the last column is the mean over 9-10 M episodes:

| Leduc, by episodes | 1e6 | 2e6 | 4e6 | 8e6 | 9-10 M |
|---|---|---|---|---|---|
| **VR-DeepDCFR+, Xu et al. (2025)** | **152** | **121** | **114** | **85** | **89** |
| VR-DeepDCFR+ (3) | 101 ± 11 | 79 ± 8 | 69 ± 6 | 68 ± 5 | 69 ± 1 |
| **VR-DeepPDCFR+, Xu et al. (2025)** | **158** | **121** | **115** | **88** | **90** |
| VR-DeepPDCFR+ (3) | 102 ± 6 | 82 ± 5 | 84 ± 6 | 77 ± 6 | 77 ± 5 |

| Kuhn, by episodes | 1e6 | 2e6 | 4e6 | 8e6 | 9-10 M |
|---|---|---|---|---|---|
| **VR-DeepDCFR+, Xu et al. (2025)** | **8.7** | **7.0** | **5.2** | **5.6** | **5.3** |
| VR-DeepDCFR+ (3) | 9.3 ± 2.5 | 7.7 ± 2.3 | 6.3 ± 1.5 | 5.6 ± 0.3 | 5.4 ± 1.0 |
| **VR-DeepPDCFR+, Xu et al. (2025)** | **4.1** | **4.3** | **4.1** | **4.3** | **3.3** |
| VR-DeepPDCFR+ (3) | 4.2 ± 0.6 | 4.7 ± 1.0 | 3.4 ± 0.5 | 2.9 ± 0.3 | 3.0 ± 0.2 |

```bash
python -m headsup.algos.pdcfr --game leduc --variant pdcfr+ --episodes 10000000 --device cuda   # ~35 min on an RTX 3090
```

Both Leduc results lie inside the paper's 95 % bands (51-133 and 63-147 at 10 M episodes) and below its curves
throughout; Kuhn matches. Where the paper and the authors' code differ, the defaults follow the code (discount
constant 1.5 for DCFR+, 1 000 baseline steps, the prediction network is never re-initialised, first-legal-action
fallback); `--discount-offset`, `--q-steps`, `--reinit-prediction` and `--fallback` give the paper's values.

**NFSP** (Heinrich & Silver 2016; `headsup.algos.nfsp`, numpy networks, one CPU core). Exploitability of the average
policy network, mA/g, mean ± sd over 3 seeds:

| iteration (128 steps, 2 SGD steps per network) | 1e4 | 1e5 | 5e5 | 1e6 | 2e6 | 3e6 |
|---|---|---|---|---|---|---|
| **Heinrich & Silver, Fig. 1a (64 units)** | **1040** | **430** | **158** | **128** | **77** | **75** |
| ours, `--preset paper` | 1249 ± 82 | 310 ± 50 | 130 ± 54 | 106 ± 39 | 107 ± 27 | 118 ± 29 |

It follows the paper's curve to 1e6 iterations and then stays near 110 where the paper continues to 75: two of the
three seeds sit at about 80-90, one at 115-145, and single evaluations swing by ±25 % (the average network, not the
data it is fitted to). The preset divides the rewards by 2.6 (the unit of the DREAM authors' NFSP code): with rewards
in antes the same settings rose again after 6e5 iterations (220-300 mA/g); the paper does not give its unit. With the
DREAM paper's NFSP settings (`--preset dream`, one seed) the curve stays at 141-157 mA/g from 3e7 to 3.2e8 nodes,
where that paper's baseline goes on to 58 - not reproduced. Open points and the 17 places where the paper,
the DREAM code and OpenSpiel differ: `docs/superpowers/specs/2026-10-08-nfsp-design.md`.

**ReBeL** (Brown et al. 2020; `headsup.algos.rebel`): CFR search over one betting round with a value network at the
public belief states where the round ends, trained by self-play; at test time it plays a randomly stopped iterate.
The paper has no Leduc experiment, so the reference is the same search with exact leaf values (the rest of the game
solved at every leaf query - what a perfect value network would give). Exact exploitability in mA/g, T = 1024 search
steps, 300 epochs (25-30 min on one core per seed), mean ± sd over 3 seeds:

| | exact leaf values | trained network |
|---|---|---|
| policy played (the mixture over stopping steps) | 22.1 | **27.7 ± 1.4** |
| the paper's protocol: 1,024 sampled playthroughs | 24.8 | 28.2 ± 1.3 |
| "unsafe" search (average strategy) | 54.5 | 53.3 ± 2.0 |

(full-game tabular Linear CFR with the same number of updates: 10.6). The network costs a quarter on top of the
search's own error; what limits it is the fit (value error 0.16 antes on probe states), not the data. Where the paper
and the official Liar's Dice code differ, the defaults follow the code: `docs/superpowers/specs/2026-10-08-rebel-design.md`.

ESCHER's paper has no deep Leduc results. Its Leduc experiment is tabular with oracle history
values (`python -m headsup.algos.oracle`, 500 trajectories per iteration, 1000 iterations):

| NashConv at 1000 iterations | OS-MCCFR | DREAM | ESCHER |
|---|---|---|---|
| **McAleer et al. (2023), Fig. 3** | **≈0.44** | **≈0.37** | **≈0.54** |
| ours, `--average own_is` (3 seeds) | – | 0.51 / 0.48 / 0.48 | 0.87 / 0.28 / 0.51 |
| ours, exact average (default) | 0.44 | 0.15 | 0.10 |

The paper's curves are limited by how the average strategy is accumulated, not by the regret estimator:
OpenSpiel's outcome sampling adds own reach × σ / sampling reach at the update player's sampled infosets
(`--average own_is`), which is unbiased but as noisy as the importance weights; with the exact average ESCHER
is 5× better than its own figure. The regret-estimator variance, pooled over an iteration's estimates as the paper
defines it, is 5.1 for ESCHER and 312 for DREAM (paper: 5.3 and 280).
Tabular references: CFR+ 0.24, DCFR 0.15 mA/g at 1000 iterations.

### NL abstraction (chips/hand, 1 chip = 500 mbb; ± standard errors)

| strategy | LBR | head-to-head |
|---|---|---|
| DeepCFR, paper features + net, 300 it. (`models/deepcfr_policy.pth`) | 1.25 ± 0.07 | ties its SD-CFR average (−0.04 ± 0.05) |
| tabular blueprint, 20 M it. (`models/blueprint_nlhe.pt`) | **1.03 ± 0.10** | **+0.86 ± 0.02** vs the DeepCFR net |
| Pluribus-mode search on the blueprint (`pluribus`) | not re-measured | +0.40 ± 0.29 vs the blueprint (2,000 hands) |
| AlphaHoldem, 0.1e9 samples (15 % of the paper's ablation budget), no value clip | 3.62 ± 0.17 | +0.29 ± 0.05 vs the DeepCFR net, −0.66 ± 0.04 vs the blueprint |
| AlphaHoldem, same, per-state value clip (`--value-clip`) | 3.45 ± 0.15 | +0.10 ± 0.05 vs the DeepCFR net, −1.10 ± 0.05 vs the blueprint |

AlphaHoldem (self-play PPO, no CFR; LBR on 4,000 pairs) after two hours on a shared RTX 3090 already beats the DeepCFR
network in play, loses to the blueprint and is three times as exploitable by LBR as either - the usual profile of
self-play RL. The per-state reading of the paper's value clip biases the advantages (measured: +0.3 to +1 chip for
every decision that is followed by another) and trains the weaker agent, so it is off by default.

LBR: 10,000 duplicate pairs, flop equities enumerated, showdowns averaged over the opponent's range. The numbers shown
here before 2026-10-08 (1.33 / 0.60) came from an estimator that kept the first hands to finish among tables played in
lock-step, which over-represents short hands; the blueprint's value moved with the fix, its lead over the network in
play did not. A blueprint study with the current code (deterministic buckets, pruning threshold scaled with the run)
found nothing clearly better than the shipped file, which therefore stays: the exact per-board abstraction with pruning
wins +0.05 ± 0.02 chips/hand against it after 20 M iterations (LBR 1.14 ± 0.10) and +0.07 ± 0.02 after 200 M (LBR 1.33 ±
0.10) - more iterations in a 200-bucket abstraction win a little more in play and are, if anything, more exploitable.

FHP (DeepCFR paper: 37 mbb/g total exploitability at ~3e8 nodes touched). The paper's game has four bets per round;
until 2026-10-08 our flop stopped at a bet and two raises (Appendix A's "three raises" read literally). Only the
four-bet flop reproduces the abstraction sizes in the paper's Fig. 2 (169 x 21 + buckets x 182 = 39,949 / 367,549 /
3,643,549 infoset-actions, 234,199,693 lossless), and on it our exact tabular Linear CFR (`python -m
headsup.algos.fhp_cfr`) follows the paper's dashed reference line:

| iteration | 20 | 30 | 50 |
|---|---|---|---|
| paper, Linear CFR (digitised) | 118 | 65 | 31 |
| ours, four-bet flop | 115 | 65 | 32 |
| ours, earlier three-bet flop | 116 | 60 | 28 |

(total exploitability, mbb/g). **Deep CFR on FHP is not reproduced yet.** On the four-bet game with the paper's
network and hyperparameters (`--game fhp --preset paper`; policy net by exact best response over all flops, SD-CFR
average on 2,000 sampled flops, which overstates by ~8 %):

| | t = 50 | 100 | 200 | 300 | 450 |
|---|---|---|---|---|---|
| **paper, 10,000 traversals (Fig. 2 / 3-left)** | **154** | **70** | **51** | **40** | **40** |
| paper's SGD-step sweep at 4,000 steps (Fig. 3-middle) | 75 | 82 | 72 | 75 | 65 |
| ours, 10,000 traversals: policy net | | 202 | 113 | 95 | **78** |
| ours, 10,000 traversals: SD-CFR average | 367 | 199 | | | |
| ours, 25,000 traversals: SD-CFR average | 178 | 121 | | | |
| ours, 25,000 traversals: policy net | | 117 | 91 | 69 | **75** |
| ours, 25,000 traversals, tighter fit (below): SD-CFR average | 234 | 134 | | | |
| ours, 25,000 traversals, 16,000 SGD steps (a setting of the paper's sweep; its curve: 55 / 42 / 42 / 37): policy net | | 136 | 87 | 68 | **55.5** |

Two things were measured about the gap. The paper's "10,000 traversals" touch 2.4 times the nodes of ours per
iteration, and with 25,000 traversals the first 50 iterations are close to its curve. And the 4,000-step regret fit is
the weak part afterwards: two refits on the same data disagree by 7 chips at pre-flop infosets where a table of the
same samples is good to 4.5, consecutive iterates play visibly different strategies there, and 16,000 steps lower the
held-out loss by 5 % (most of what is left is sampling noise) - as in the paper's own sweep, whose 4,000-step curve
ends near 65, not 40. `--batch-size 40000 --lr 3e-3 --lr-schedule cosine` gets the 16,000-step fit at 1.1-1.6 times
the paper's cost on that memory (iteration 125) - but it does not help early: with it the 25,000-traversal run is at
234 / 134 after 50 / 100 iterations instead of 178 / 121, because a tighter fit also follows the noise of the small
early memories (our earlier 16,000-step runs were likewise worse before iteration 100 and better from about 200).
The 450-iteration result with the paper's settings, 78, is where the three-bet runs ended too (80-86), and 25,000
traversals end at 75: as in the paper, the final level does not depend on the traversal count - but ours is the level of
the paper's 4,000-step sweep curve (65), not of its headline (40), whose "default" replicates (Fig. 4) coincide with
the sweep's 16,000-32,000-step curves. The fit budget moves our floor the same way: continuing the 25,000-traversal
run from iteration 325 to 450 with 1,000 / 4,000 / 8,000 SGD steps per fit ends at 90 / 75 / 61. A from-scratch run
with 16,000 steps ends at 55.5. Units and game are the paper's (a uniform-random strategy: 3,045 mbb/g total here, about
2,700 at the start of the paper's tabular curves), and stating the regret targets in big blinds instead of our RMS
rescaling changes nothing (370 against 367 after 50 iterations). Plotted against nodes touched, the axis of the paper's
Fig. 2, the 10,000-traversal run is on the paper's curve up to ~3e7 nodes and 1.5 times above it from 1e8 on;
continued to 600 iterations it stays on its plateau (82.8 after 77.7 at 450), which is the level of the paper's own
4,000-step line (65-85). The paper's 16,000-step line (300,000 traversals, 16,000 steps: `runs/fhp4_sweep_k300k_s16k`) is running. `--weight-average` and `--policy-lr` are the related options. Details,
the street split of the error and the fit benchmark: [docs/paper-fidelity.md](docs/paper-fidelity.md).

Every known difference between this code and the papers (fixed, chosen, or still open) is listed in
[docs/paper-fidelity.md](docs/paper-fidelity.md).

## Layout

```
headsup/engine.py game.py cards.py     rules, action trees, observation encoding
headsup/env.py players.py              vectorised envs, batched players, player specs
headsup/model.py numpy_model.py        networks (torch, numpy mirror)
headsup/cpp/headsup_cpp.cpp            C++ kernels: engine, evaluator, MLP, traversals, solvers, blueprint
headsup/deepcfr/ sdcfr.py              hold'em trainer, reservoir memories, SD-CFR player
headsup/blueprint.py search.py         tabular blueprint, real-time search
headsup/alphaholdem/ twoseat.py        AlphaHoldem (end-to-end self-play RL), the two-seat env it trains in
headsup/games/ algos/                  small games, tabular + deep algorithms, best responses
headsup/lbr.py compare.py exploit.py   evaluation tools
headsup/web/                           browser table
models/                                shipped models + README.json (training details)
```

## References

Brown et al., [Deep CFR](https://arxiv.org/abs/1811.00164) (ICML 2019) ·
Steinberger, [Single Deep CFR](https://arxiv.org/abs/1901.07621) (2019) ·
Steinberger, Lerer & Brown, [DREAM](https://arxiv.org/abs/2006.10410) (2020) ·
McAleer et al., [ESCHER](https://arxiv.org/abs/2206.04122) (ICLR 2023) ·
Brown & Sandholm, Pluribus (Science 2019), [nested subgame solving](https://arxiv.org/abs/1705.02955) (NeurIPS 2017),
[DCFR](https://arxiv.org/abs/1809.04040) (AAAI 2019) ·
Brown, Sandholm & Amos, [depth-limited solving](https://arxiv.org/abs/1805.08195) (NeurIPS 2018) ·
Lisý & Bowling, [LBR](https://arxiv.org/abs/1612.07547) (2017) ·
Farina, Kroer & Sandholm, [PCFR+](https://arxiv.org/abs/2007.14358) (AAAI 2021) ·
Xu et al., [DCFR+ / PDCFR+](https://arxiv.org/abs/2404.13891) (IJCAI 2024) and
[Deep (Predictive) Discounted CFR](https://arxiv.org/abs/2511.08174) (2025) ·
Zhao et al., AlphaHoldem (AAAI 2022) ·
Heinrich & Silver, [NFSP](https://arxiv.org/abs/1603.01121) (2016) ·
Brown, Bakhtin, Lerer & Gong, [ReBeL](https://arxiv.org/abs/2007.13544) (NeurIPS 2020)
