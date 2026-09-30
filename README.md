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
| FHP | `--game fhp` | limit, blinds 50/100, bets 100, two rounds, showdown after the flop (DeepCFR paper) |
| HULH | `--game hulh` | limit, bets 100 / 100 / 200 / 200, four rounds |
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
paper's hyperparameters; `python -m headsup.deepcfr.train -h` lists the network / budget options.

## Evaluation

| tool | measures |
|---|---|
| `python -m headsup.algos.holdem_br --policy <spec> --cards all` | best-response exploitability: exact over hands, boards enumerated (FHP) or sampled per street; SD-CFR averages exactly |
| `python -m headsup.lbr --policy <spec>` | Local Best Response, a lower bound |
| `python -m headsup.compare <spec> <spec> --bots` | head-to-head chips/hand ± SE |
| `python -m headsup.exploit --policy <spec>` | PPO exploiters, a weak lower bound |

## Results

### Leduc: reproductions

Exploitability of the average strategy in milli-antes per game (mean over seats, the papers' unit),
mean ± sd over 3 seeds (`python -m headsup.algos.leduc_report runs/leduc_v4`):

| SD-CFR paper setup (by iteration) | 30 | 120 | 300 | 1000 |
|---|---|---|---|---|
| SD-CFR, ours | 282 ± 24 | 100 ± 9 | 89 ± 12 | **87 ± 20** |
| SD-CFR, Steinberger (2019) Fig. 1a | 381 | 154 | 112 | 89 |
| Deep CFR, ours | 297 ± 48 | 136 ± 26 | 132 ± 23 | **127 ± 4** |
| Deep CFR, Steinberger (2019) Fig. 1a | 401 | 170 | 143 | 110 |

| DREAM paper setup (by nodes touched) | 3e6 | 1e7 | 1.4e7 | paper (plot read-off) |
|---|---|---|---|---|
| ES-SD-CFR, 346 traversals | 90 ± 12 | 58 ± 2 | **53 ± 8** | ≈40 at 1.5e7 |
| DREAM, 900 traversals | 117 ± 16 | 85 ± 6 | **70 ± 4** | ≈56 at 1.2e7 |

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

Best-response exploitability of the NL strategies and the FHP reproduction (DeepCFR paper: 37 mbb/g
total exploitability) are being re-measured: the earlier estimates missed the board chance factor
and the FHP runs used the uniform regret-matching fallback and unscaled chip targets.

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
