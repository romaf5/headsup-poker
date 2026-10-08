"""Leduc reproduction tables: our deep CFR curves next to the papers' published ones.

    python -m headsup.algos.leduc_report                     # the runs of runs/scripts/leduc_v*.sh (README "Results")
    python -m headsup.algos.leduc_report --row "dreamp,dream,DREAM (mine),runs/x/dream_s*.json"

A row is ``setup,algo,label,glob`` over ``--json`` curves (or logs) of ``headsup.algos.deep``: ``sdcfrp`` = the SD-CFR
paper's Leduc setup (x-axis: iterations), ``dreamp`` = the DREAM paper's (x-axis: nodes touched, as the DREAM code
counts them), ``pdcfrp`` / ``pdcfrk`` = the Deep PDCFR+ paper's Leduc / Kuhn setup (x-axis: episodes).  Values are
the exploitability of the average strategy in milli-antes per game (mean over the two seats, the papers' unit), mean
± sd over seeds of each run's mean over the evaluations within ±10 % of x (the deep curves fluctuate by ±10-20 %
between evaluations).

The last section is ReBeL (``headsup.algos.rebel``): reference numbers - none of them an official result, the paper has
no Leduc experiment - and our runs ``runs/leduc_rebel/leduc_s*.json`` by epoch (``--rebel`` / ``--rebel-oracle``).
"""

import argparse
import glob
import json
import re

import numpy as np

# Steinberger (2019), Fig. 1a (read off the vector plot; 5 runs): iteration -> (SD-CFR, Deep CFR)
SDCFR_PAPER = {30: (381, 401), 60: (227, 230), 90: (186, 199), 120: (154, 170), 210: (139, 155), 300: (112, 143),
               510: (95, 115), 1020: (89, 110), 2010: (69, 93), 3000: (67, 86), 4980: (59, 80)}
# Steinberger, Lerer & Brown (2020), Fig. 2 top-left: 3-seed mean curves read off the raster (log axes), up to where
# all seeds' bands end (single-seed tails beyond are excluded): nodes touched -> mA/g
DREAM_PAPER = {"sdcfr": {1e6: 123, 1.84e6: 85, 2.55e6: 67, 3.5e6: 57, 4.86e6: 53, 6.7e6: 49, 1e7: 46},
               "dream": {1.84e6: 88, 2.55e6: 78, 3.5e6: 67, 4.86e6: 60, 6.7e6: 62, 1e7: 57, 1.28e7: 56}}
# our nodes_touched / the DREAM code's "States Seen": it counts decision and terminal nodes (deals happen inside
# env.step) and its learned-baseline sampler skips terminals reached by a traverser action; measured on trained
# Leduc policies with each sampler's own rules
NODE_RATIO = {"sdcfr": 1.23, "deepcfr": 1.23, "dream": 1.64}
# Xu et al. (2025), Fig. 1 (4-seed means read off the PDF's vector curves; x 1000 = mA/g): episodes -> value; the
# last point is the mean over 9-10 M episodes
PDCFR_PAPER = {("pdcfrp", "dcfr+"): {1e6: 152, 2e6: 121, 4e6: 114, 8e6: 85, 9.5e6: 89},
               ("pdcfrp", "pdcfr+"): {1e6: 158, 2e6: 121, 4e6: 115, 8e6: 88, 9.5e6: 90},
               ("pdcfrk", "dcfr+"): {1e6: 8.7, 2e6: 7.0, 4e6: 5.2, 8e6: 5.6, 9.5e6: 5.3},
               ("pdcfrk", "pdcfr+"): {1e6: 4.1, 2e6: 4.3, 4e6: 4.1, 8e6: 4.3, 9.5e6: 3.3}}
# ReBeL on Leduc, mA/g.  Search with exact leaf values = the rest of the game solved at every leaf query, what a perfect
# value network would give: {search steps T (T / 2 updates per player): (random-iterate mixture, "unsafe" average
# policy)}, exact exploitability; from the research brief's scratch prototype (unreviewed; its ranges without mass stay
# zero where the official code and ours make them uniform, and the search amplifies rounding differences: +-5 % at small T)
REBEL_EXACT_LEAVES = {64: (134.9, 165.9), 128: (80.1, 92.6), 256: (49.9, 68.4), 512: (32.7, 58.0), 1024: (21.8, 53.5), 2048: (14.0, 45.4)}
# full-game tabular Linear CFR (headsup.algos.tabular) with the same number of updates per player
REBEL_FULL_LCFR = {64: 139.5, 128: 55.6, 256: 41.0, 512: 16.2, 1024: 10.6, 2048: 5.2}
# pinouche/poker_self_play, docs/paradigm_b.md (third party, single seed, not verified: DCFR, leaves AFTER the board
# card, gadget re-solving at test time): {test-time search iterations: (its trained net, its exact leaf solving)}
REBEL_PINOUCHE = {300: (58.5, 57.1), 1000: (46.6, 49.7), 3000: (26.9, 39.2), 10000: (16.4, 29.3)}
# Student of Games (Schmid et al., arXiv 2112.03178, Fig. 3A): plateau of SoG(s, 1) on Leduc read off the plot by a
# literature survey (not re-digitised): {simulations per move s: mA/g}
REBEL_SOG = {33: 200, 100: 105, 300: 55, 1000: 22}
_REBEL_ROWS = (("exploitability", "policy played in expectation (exact mixture over the stopping steps)", 1000.0),
               ("exploitability_sampled", "average of K sampled playthrough policies (the paper's protocol)", 1000.0),
               ("exploitability_unsafe", "unsafe search (average strategy and its ranges)", 1000.0),
               ("value_error", "value net RMS error on the probe PBSs, milli-antes", 1000.0),
               ("value_error_search", "value net RMS error at the search's own leaf PBSs, milli-antes", 1000.0),
               ("root_value", "player 0's game value by the root solve, milli-antes (exact: -85.6)", 1000.0),
               ("examples", "training examples generated, thousands", 0.001))

_NAMES = {"sdcfr": "SD-CFR", "deepcfr": "Deep CFR", "dream": "DREAM", "dcfr+": "VR-DeepDCFR+", "pdcfr+": "VR-DeepPDCFR+"}

DEFAULT_ROWS = [
    ("sdcfrp", "sdcfr", "SD-CFR, authors' net (`--arch pokerrl`)", "runs/leduc_v6/reffull_sdcfr_s*.json"),
    ("sdcfrp", "sdcfr", "SD-CFR, MLP + `--masked-loss`", "runs/leduc_v6/mlpmasked_sdcfr_s*.json"),
    ("sdcfrp", "sdcfr", "SD-CFR, plain MLP", "runs/leduc_v5/sdcfrp_sdcfr_s*.json"),
    ("sdcfrp", "deepcfr", "Deep CFR, authors' net", "runs/leduc_v6/reffull_deepcfr_s*.json"),
    ("sdcfrp", "deepcfr", "Deep CFR, plain MLP", "runs/leduc_v5/sdcfrp_deepcfr_s*.json"),
    ("dreamp", "sdcfr", "ES-SD-CFR, DREAM-code settings", "runs/leduc_v10/exact_sdcfr_s*.json"),
    ("dreamp", "sdcfr", "ES-SD-CFR, plain MLP", "runs/leduc_v4/dreamp_sdcfr_s*.json"),
    ("dreamp", "dream", "DREAM, DREAM-code settings + shared baseline", "runs/leduc_v11/repobase_dream_s*.json"),
    ("dreamp", "dream", "DREAM, DREAM-code settings, per-player baselines", "runs/leduc_v10/exact_dream_s*.json"),
    ("dreamp", "dream", "DREAM, plain MLP", "runs/leduc_v4/dreamp_dream_s*.json"),
    ("pdcfrp", "dcfr+", "VR-DeepDCFR+, ours", "runs/leduc_pdcfr/leduc_dcfr+_s*.json"),
    ("pdcfrp", "pdcfr+", "VR-DeepPDCFR+, ours", "runs/leduc_pdcfr/leduc_pdcfr+_s*.json"),
    ("pdcfrk", "dcfr+", "VR-DeepDCFR+, ours", "runs/leduc_pdcfr/kuhn_dcfr+_s*.json"),
    ("pdcfrk", "pdcfr+", "VR-DeepPDCFR+, ours", "runs/leduc_pdcfr/kuhn_pdcfr+_s*.json"),
]

_LINE = re.compile(r"it (\d+): exploitability current ([\d.]+) average ([\d.]+)\s+nodes ([\d.e+]+)")


def load_run(path):
    """[(iteration, average mA/g, nodes touched, episodes or None)] of one run (a --json curve or a log)."""
    if path.endswith(".json"):
        return [(c["iteration"], 1000 * c["average"], c["nodes_touched"], c.get("episodes")) for c in json.load(open(path))["curve"]]
    rows = {int(i): (int(i), 1000 * float(a), float(n), None) for i, _, a, n in _LINE.findall(open(path).read())}
    return [rows[k] for k in sorted(rows)]


def _interp(points, x):
    """Log-x interpolation of a paper curve (None outside it; the last point covers x up to 2 % beyond)."""
    xs = sorted(points)
    if xs[-1] < x <= 1.02 * xs[-1]:
        x = xs[-1]
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(np.log(x), np.log(xs), [points[k] for k in xs]))


def _at(rows, key, x, scale=1.0, window=0.1):
    """Mean of a run's evaluations whose x (iteration, or nodes / scale) lies within ±window of x."""
    vals = [r[1] for r in rows if r[key] is not None and (1 - window) * x <= r[key] / scale <= (1 + window) * x]
    return float(np.mean(vals)) if vals else None


def _fmt(vals, digits=0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    return f"{np.mean(vals):.{digits}f}" + (f" ± {np.std(vals, ddof=1):.{digits}f}" if len(vals) > 1 else "")


def table(rows, setup, xs):
    """Markdown table: one line per row of ``setup`` and per algorithm the paper's line."""
    episodes = setup in ("pdcfrp", "pdcfrk")
    head = "iteration" if setup == "sdcfrp" else "episodes" if episodes else "nodes touched (DREAM code's count)"
    out = [f"| {head} | " + " | ".join(f"{x:g}" if setup == "sdcfrp" else f"{x:.2g}".replace("e+0", "e") for x in xs) + " |",
           "|---|" + "---|" * len(xs)]
    digits = 1 if setup == "pdcfrk" else 0
    done = set()
    for st, algo, label, pattern in rows:
        if st != setup:
            continue
        runs = [load_run(p) for p in sorted(glob.glob(pattern))]
        runs = [r for r in runs if r]
        if setup == "sdcfrp":
            cells = [_fmt([_at(r, 0, x) for r in runs]) for x in xs]
        elif episodes:  # the final column (9.5e6) averages the evaluations between 9 M and 10 M episodes
            cells = [_fmt([_at(r, 3, x, window=0.0527 if x == 9.5e6 else 0.1) for r in runs], digits) for x in xs]
        else:
            if algo not in NODE_RATIO:
                raise ValueError(f"no DREAM-code node count ratio for {algo!r} (known: {sorted(NODE_RATIO)})")
            cells = [_fmt([_at(r, 2, x, NODE_RATIO[algo]) for r in runs]) for x in xs]
        out.append(f"| {label} ({len(runs)}) | " + " | ".join(cells) + " |")
        if algo not in done:
            done.add(algo)
            if setup == "sdcfrp":
                col = 0 if algo == "sdcfr" else 1
                paper = [_interp({k: v[col] for k, v in SDCFR_PAPER.items()}, x) for x in xs]
            elif episodes:
                paper = [PDCFR_PAPER.get((setup, algo), {}).get(x) for x in xs]
            else:
                paper = [_interp(DREAM_PAPER[algo], x) if algo in DREAM_PAPER else None for x in xs]
            name = "ES-SD-CFR" if (algo, setup) == ("sdcfr", "dreamp") else _NAMES.get(algo, algo)
            out.insert(len(out) - 1, f"| **{name}, paper** | " + " | ".join("–" if v is None else f"**{v:.{digits}f}**" for v in paper) + " |")
    return "\n".join(out)


def rebel_references(oracle_pattern="runs/leduc_rebel/oracle_T*.json"):
    """Markdown tables of the ReBeL reference numbers on Leduc, each labelled with its source and what it measures."""
    xs = sorted(REBEL_EXACT_LEAVES)
    ours = {}
    for path in sorted(glob.glob(oracle_pattern)):
        for c in json.load(open(path))["curve"]:
            ours.setdefault(c["iters"], []).append(c)

    def cells(f):
        return " | ".join("–" if x not in ours else _fmt([f(c) for c in ours[x]], 1) for x in xs)

    out = ["| search steps T (T / 2 updates per player) | " + " | ".join(str(x) for x in xs) + " |", "|---|" + "---|" * len(xs),
           "| full-game tabular Linear CFR, average strategy (ours) | " + " | ".join(f"{REBEL_FULL_LCFR[x]:.1f}" for x in xs) + " |",
           "| search with exact leaf values, random-iterate mixture (the brief's prototype) | "
           + " | ".join(f"{REBEL_EXACT_LEAVES[x][0]:.1f}" for x in xs) + " |",
           "| search with exact leaf values, unsafe average policy (the brief's prototype) | "
           + " | ".join(f"{REBEL_EXACT_LEAVES[x][1]:.1f}" for x in xs) + " |",
           f"| search with exact leaf values, random-iterate mixture (ours, `--oracle`) | {cells(lambda c: 1000 * c['exploitability'])} |",
           "| search with exact leaf values, unsafe average policy (ours, `--oracle`) | "
           f"{cells(lambda c: 1000 * c['exploitability_unsafe'])} |",
           ""]
    xs = sorted(REBEL_PINOUCHE)
    out += ["| test-time search iterations | " + " | ".join(str(x) for x in xs) + " |", "|---|" + "---|" * len(xs),
            "| pinouche/poker_self_play, trained net (third party, single seed, unverified; DCFR, leaves after the board card, gadget "
            "re-solving) | " + " | ".join(f"{REBEL_PINOUCHE[x][0]:.1f}" for x in xs) + " |",
            "| pinouche/poker_self_play, exact leaf solving (same source) | "
            + " | ".join(f"{REBEL_PINOUCHE[x][1]:.1f}" for x in xs) + " |",
            ""]
    xs = sorted(REBEL_SOG)
    out += ["| simulations per move | " + " | ".join(str(x) for x in xs) + " |", "|---|" + "---|" * len(xs),
            "| Student of Games, Fig. 3A: plateau of SoG(s, 1) (read off the plot by a survey; another algorithm - GT-CFR with a "
            "learned value and policy net) | " + " | ".join(f"~{REBEL_SOG[x]}" for x in xs) + " |"]
    return "\n".join(out)


def rebel_table(pattern="runs/leduc_rebel/leduc_s*.json", epochs=None, columns=8, last=3):
    """Markdown table of our ReBeL runs (``--json`` curves of headsup.algos.rebel): mean ± sd over the seeds at the
    evaluated epochs every run has (thinned to ``columns``, the first and the last kept), one line per quantity.
    The final column is each run's mean over the ``last`` evaluations that every run has (its header names their
    epochs, so runs of different lengths are compared at the same ones) - the paper reports the mean of three
    checkpoints for Liar's Dice: single evaluations of a search with a learned value function scatter by 20-30 %."""
    runs = [json.load(open(p)) for p in sorted(glob.glob(pattern))]
    runs = [{c["epoch"]: c for c in r["curve"]} for r in runs if r.get("curve")]
    if not runs:
        return f"(no runs match {pattern})"
    common = sorted(set.intersection(*(set(r) for r in runs)))
    if epochs is None:
        keep = np.unique(np.round(np.linspace(0, len(common) - 1, min(columns, len(common)))).astype(int))
        epochs = [common[i] for i in keep]
    tail = common[-last:]
    samples = sorted({c["samples"] for r in runs for c in r.values() if c.get("samples")})
    head = f"| epoch ({len(runs)} run{'s' if len(runs) > 1 else ''}) | "
    out = [head + " | ".join(str(e) for e in epochs) + f" | last {len(tail)} evaluations (epochs {' / '.join(map(str, tail))}) |",
           "|---|" + "---|" * (len(epochs) + 1)]
    for key, label, scale in _REBEL_ROWS:
        if key == "exploitability_sampled" and samples:
            label = label.replace("K", "K = " + " / ".join(str(k) for k in samples))
        cells = [_fmt([scale * r[e][key] for r in runs if e in r and r[e].get(key) is not None], 1) for e in epochs]
        tails = [[r[e][key] for e in tail if r[e].get(key) is not None] for r in runs]
        cells.append(_fmt([scale * float(np.mean(t)) for t in tails if t], 1))
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _parse_row(text):
    """"setup,algo,label,glob" - the label may contain commas (the glob is everything after the last one)."""
    setup, algo, rest = text.split(",", 2)
    label, pattern = rest.rsplit(",", 1)
    return setup, algo, label, pattern


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--row", action="append", default=None, help="setup,algo,label,glob (repeatable; default: DEFAULT_ROWS)")
    p.add_argument("--rebel", default="runs/leduc_rebel/leduc_s*.json",
                   help="glob of the ReBeL runs (--json curves of headsup.algos.rebel)")
    p.add_argument("--rebel-oracle", default="runs/leduc_rebel/oracle_T*.json", help="glob of the exact-leaf calibration runs (--oracle)")
    args = p.parse_args(argv)
    rows = [_parse_row(r) for r in args.row] if args.row else DEFAULT_ROWS
    print("SD-CFR paper setup (1500 traversals, 750 warm-started updates, 1M buffers), mA/g by iteration:\n")
    print(table(rows, "sdcfrp", (500, 1000, 2000, 3000, 5000)))
    print("\nDREAM paper setup (346 ES / 900 OS traversals, 3000 x 2048 fits from scratch, 2M buffers), mA/g by nodes touched:\n")
    print(table(rows, "dreamp", (2.55e6, 3.5e6, 6.7e6, 1e7, 1.28e7)))
    xs = (1e6, 2e6, 4e6, 8e6, 9.5e6)
    print("\nDeep PDCFR+ paper setup (10 000 episodes per player and iteration), mA/g by episodes (last column: mean over 9-10 M):\n")
    print("Leduc\n")
    print(table(rows, "pdcfrp", xs))
    print("\nKuhn\n")
    print(table(rows, "pdcfrk", xs))
    print("\nReBeL on Leduc, exact exploitability in mA/g. References (no official Leduc result exists):\n")
    print(rebel_references(args.rebel_oracle))
    print(f"\nReBeL, our runs ({args.rebel}), mean ± sd over seeds:\n")
    print(rebel_table(args.rebel))


if __name__ == "__main__":
    main()
