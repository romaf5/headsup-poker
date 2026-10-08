"""Leduc reproduction tables: our deep CFR curves next to the papers' published ones.

    python -m headsup.algos.leduc_report                     # the runs of runs/scripts/leduc_v*.sh (README "Results")
    python -m headsup.algos.leduc_report --row "dreamp,dream,DREAM (mine),runs/x/dream_s*.json"

A row is ``setup,algo,label,glob`` over ``--json`` curves (or logs) of ``headsup.algos.deep``: ``sdcfrp`` = the SD-CFR
paper's Leduc setup (x-axis: iterations), ``dreamp`` = the DREAM paper's (x-axis: nodes touched, as the DREAM code
counts them), ``pdcfrp`` / ``pdcfrk`` = the Deep PDCFR+ paper's Leduc / Kuhn setup (x-axis: episodes), ``nfspp`` /
``nfspk`` = NFSP (``headsup.algos.nfsp``) on Leduc / Kuhn by iterations of 128 steps (Leduc: against the NFSP paper),
``nfspd`` = NFSP on Leduc by nodes touched against the DREAM paper's NFSP curve.  Values are the exploitability of the
average strategy in milli-antes per game (mean over the two seats, the papers' unit), mean ± sd over seeds of each
run's mean over the evaluations within ±10 % of x (the deep curves fluctuate by ±10-20 % between evaluations).
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
# Heinrich & Silver (2016), Fig. 1a, the calibrated setup with 64 hidden units (read off the vector plot; the paper's
# exploitability - inferred: antes, mean over the seats - x 1000): iteration -> mA/g
NFSP_PAPER = {1e3: 1880, 1e4: 1040, 1e5: 430, 2e5: 257, 5e5: 158, 1e6: 128, 2e6: 77, 3e6: 75}
# Steinberger, Lerer & Brown (2020), Fig. 2 top-right, the NFSP curve (read off the raster, about +-3 %): nodes touched
# (decisions only: 128 per iteration) -> mA/g
NFSP_DREAM = {2e7: 147, 3e7: 116, 5e7: 92, 1e8: 71, 2e8: 61, 3.2e8: 58}
_NAMES = {"sdcfr": "SD-CFR", "deepcfr": "Deep CFR", "dream": "DREAM", "dcfr+": "VR-DeepDCFR+", "pdcfr+": "VR-DeepPDCFR+",
          "nfsp": "NFSP (64 units)"}

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
    ("nfspp", "nfsp", "NFSP, `--preset paper`", "runs/leduc_nfsp/leduc_paper_s*.json"),
    ("nfspp", "nfsp", "NFSP, `--preset dream`", "runs/leduc_nfsp/leduc_dream_s*.json"),
    ("nfspd", "nfsp", "NFSP, `--preset dream`", "runs/leduc_nfsp/leduc_dream_s*.json"),
    ("nfspd", "nfsp", "NFSP, `--preset paper`", "runs/leduc_nfsp/leduc_paper_s*.json"),
    ("nfspk", "nfsp", "NFSP, `--preset paper`", "runs/leduc_nfsp/kuhn_paper_s*.json"),
    ("nfspk", "nfsp", "NFSP, `--preset dream`", "runs/leduc_nfsp/kuhn_dream_s*.json"),
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
    episodes, iterations = setup in ("pdcfrp", "pdcfrk"), setup in ("sdcfrp", "nfspp", "nfspk")
    head = ("iteration" if iterations else "episodes" if episodes else "nodes touched (128 per iteration)" if setup == "nfspd"
            else "nodes touched (DREAM code's count)")
    out = [f"| {head} | " + " | ".join(f"{x:g}" if setup == "sdcfrp" else f"{x:.2g}".replace("e+0", "e") for x in xs) + " |",
           "|---|" + "---|" * len(xs)]
    digits = 1 if setup in ("pdcfrk", "nfspk") else 0
    done = set()
    for st, algo, label, pattern in rows:
        if st != setup:
            continue
        runs = [load_run(p) for p in sorted(glob.glob(pattern))]
        runs = [r for r in runs if r]
        if iterations:
            cells = [_fmt([_at(r, 0, x) for r in runs], digits) for x in xs]
        elif setup == "nfspd":  # NFSP counts decisions, as the DREAM code's NFSP does
            cells = [_fmt([_at(r, 2, x) for r in runs]) for x in xs]
        elif episodes:  # the final column (9.5e6) averages the evaluations between 9 M and 10 M episodes
            cells = [_fmt([_at(r, 3, x, window=0.0527 if x == 9.5e6 else 0.1) for r in runs], digits) for x in xs]
        else:
            if algo not in NODE_RATIO:
                raise ValueError(f"no DREAM-code node count ratio for {algo!r} (known: {sorted(NODE_RATIO)})")
            cells = [_fmt([_at(r, 2, x, NODE_RATIO[algo]) for r in runs]) for x in xs]
        out.append(f"| {label} ({len(runs)}) | " + " | ".join(cells) + " |")
        if algo not in done and setup != "nfspk":  # (no published NFSP curve for Kuhn)
            done.add(algo)
            if setup in ("nfspp", "nfspd"):
                paper = [_interp(NFSP_PAPER if setup == "nfspp" else NFSP_DREAM, x) for x in xs]
            elif setup == "sdcfrp":
                col = 0 if algo == "sdcfr" else 1
                paper = [_interp({k: v[col] for k, v in SDCFR_PAPER.items()}, x) for x in xs]
            elif episodes:
                paper = [PDCFR_PAPER.get((setup, algo), {}).get(x) for x in xs]
            else:
                paper = [_interp(DREAM_PAPER[algo], x) if algo in DREAM_PAPER else None for x in xs]
            name = "ES-SD-CFR" if (algo, setup) == ("sdcfr", "dreamp") else _NAMES.get(algo, algo)
            name = "NFSP, DREAM paper" if setup == "nfspd" else f"{name}, paper"
            out.insert(len(out) - 1, f"| **{name}** | " + " | ".join("–" if v is None else f"**{v:.{digits}f}**" for v in paper) + " |")
    return "\n".join(out)


def _parse_row(text):
    """"setup,algo,label,glob" - the label may contain commas (the glob is everything after the last one)."""
    setup, algo, rest = text.split(",", 2)
    label, pattern = rest.rsplit(",", 1)
    return setup, algo, label, pattern


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--row", action="append", default=None, help="setup,algo,label,glob (repeatable; default: DEFAULT_ROWS)")
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
    print("\nNFSP (128 steps and 2 SGD steps per network per iteration), mA/g by iteration:\n")
    print("Leduc, against Heinrich & Silver's Fig. 1a\n")
    print(table(rows, "nfspp", (1e3, 1e4, 1e5, 2e5, 5e5, 1e6, 2e6, 3e6)))
    print("\nKuhn\n")
    print(table(rows, "nfspk", (1e3, 1e4, 1e5, 1e6, 3e6)))
    print("\nNFSP on Leduc against the DREAM paper's NFSP curve, mA/g by nodes touched:\n")
    print(table(rows, "nfspd", (2e7, 3e7, 5e7, 1e8, 2e8, 3.2e8)))


if __name__ == "__main__":
    main()
