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
| tabular CFR, CFR+, DCFR, PCFR+, MCCFR | `headsup/algos/tabular.py` | reference solvers for the small games |
| Pluribus blueprint (Linear MCCFR + pruning) | `headsup/blueprint.py` | `python -m headsup.blueprint --game nlhe --iterations 20000000 --out runs/bp.pt` |
| real-time search (depth-limited, Pluribus mode) | `headsup/search.py` | player spec `search:<blueprint>[@pluribus]` |

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

ESCHER's paper has no deep Leduc results. Its Leduc experiment is tabular with oracle history
values (`python -m headsup.algos.oracle`, 500 trajectories per iteration). At 1000 iterations we
measure NashConv 0.44 for OS-MCCFR (paper ≈0.44), 0.15 for DREAM (≈0.37) and 0.10 for ESCHER
(≈0.54). The regret-estimator variance is 3.5 for ESCHER vs 132 for DREAM (paper: 5.3 vs 280).
Tabular references: CFR+ 0.24, DCFR 0.15 mA/g at 1000 iterations.

### NL abstraction (chips/hand, 1 chip = 500 mbb; ± standard errors)

| strategy | LBR | head-to-head |
|---|---|---|
| DeepCFR, paper features + net, 300 it. (`models/deepcfr_policy.pth`) | 1.33 ± 0.09 | ties its SD-CFR average (−0.04 ± 0.05) |
| tabular blueprint, 20 M it., 40 min (`models/blueprint_nlhe.pt`) | **0.60 ± 0.09** | **+0.87 ± 0.04** vs the DeepCFR net |
| Pluribus-mode search on the blueprint | −0.78 ± 0.39 | within noise of the blueprint |

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

(total exploitability, mbb/g). The Deep CFR numbers measured on the earlier game (80 mbb/g total after 450
iterations) are not comparable with the paper and are being re-run with the paper's network and loss weights.
Best-response exploitability of the NL strategies is being re-measured (the earlier estimates missed the board
chance factor).

## Layout

```
headsup/engine.py game.py cards.py     rules, action trees, observation encoding
headsup/env.py players.py              vectorised envs, batched players, player specs
headsup/model.py numpy_model.py        networks (torch, numpy mirror)
headsup/cpp/headsup_cpp.cpp            C++ kernels: engine, evaluator, MLP, traversals, solvers, blueprint
headsup/deepcfr/ sdcfr.py              hold'em trainer, reservoir memories, SD-CFR player
headsup/blueprint.py search.py         tabular blueprint, real-time search
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
Farina, Kroer & Sandholm, [PCFR+](https://arxiv.org/abs/2007.14358) (AAAI 2021)
