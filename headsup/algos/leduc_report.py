"""Leduc reproduction tables: our deep CFR curves next to the papers' published ones.

Reads the logs (or ``--json`` curves) written by ``headsup.algos.deep`` into one directory, named
``<protocol>_<algo>_s<seed>.log``: ``sdcfrp`` = the SD-CFR paper's Leduc setup (x-axis: iterations),
``dreamp`` = the DREAM paper's (x-axis: nodes touched).  Values are the exploitability of the
average strategy in milli-antes per game (mean over the two seats, the unit of both papers' Leduc
plots), mean ± sd over seeds.

    python -m headsup.algos.leduc_report runs/leduc_v4
"""

import argparse
import glob
import os
import re

import numpy as np

# Steinberger (2019), Fig. 1a (read off the vector plot; 5 runs): iteration -> (SD-CFR, Deep CFR)
SDCFR_PAPER = {30: (381, 401), 60: (227, 230), 90: (186, 199), 120: (154, 170), 210: (139, 155), 300: (112, 143),
               510: (95, 115), 1020: (89, 110), 2010: (69, 93), 3000: (67, 86), 4980: (59, 80)}
# Steinberger, Lerer & Brown (2020), Leduc figure (raster read-off, approximate): nodes touched -> value
DREAM_PAPER = {"sdcfr": {1.5e7: 40}, "dream": {1.15e7: 56}}

_LINE = re.compile(r"it (\d+): exploitability current ([\d.]+) average ([\d.]+)\s+nodes ([\d.e+]+)")


def load_curves(directory):
    """{(protocol, algo): {seed: [(iteration, average mA/g, current mA/g, nodes)]}} from the logs."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*_s*.log"))):
        m = re.match(r"(\w+?)_(\w+)_s(\d+)\.log$", os.path.basename(path))
        if not m:
            continue
        rows = [(int(a), 1000 * float(c), 1000 * float(b), float(n)) for a, b, c, n in _LINE.findall(open(path).read())]
        if rows:
            out.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = rows
    return out


def _interp(points, x, log=True):
    xs = sorted(points)
    if x < xs[0] or x > xs[-1]:
        return None
    ys = [points[k] for k in xs]
    return float(np.interp(np.log(x) if log else x, np.log(xs) if log else xs, ys))


def _at(rows, key, x):
    """Linear interpolation of a run's average exploitability at iteration / nodes ``x`` (None if outside)."""
    xs = np.array([r[key] for r in rows], dtype=float)
    ys = np.array([r[1] for r in rows])
    if len(xs) == 0 or x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def _fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    return f"{np.mean(vals):.0f}" + (f" ± {np.std(vals, ddof=1):.0f}" if len(vals) > 1 else "")


def sdcfr_table(curves, iterations=(30, 60, 120, 300, 510, 1000)):
    lines = ["| iteration | " + " | ".join(str(i) for i in iterations) + " |", "|---|" + "---|" * len(iterations)]
    for algo, col in (("sdcfr", 0), ("deepcfr", 1)):
        runs = curves.get(("sdcfrp", algo), {})
        ours = [_fmt([_at(r, 0, i) for r in runs.values()]) for i in iterations]
        paper = [f"{_interp({k: v[col] for k, v in SDCFR_PAPER.items()}, i):.0f}" for i in iterations]
        name = "SD-CFR" if algo == "sdcfr" else "Deep CFR"
        lines.append(f"| {name}, ours ({len(runs)} seeds) | " + " | ".join(ours) + " |")
        lines.append(f"| {name}, paper | " + " | ".join(paper) + " |")
    return "\n".join(lines)


def dream_table(curves, nodes=(1e6, 3e6, 1e7, 1.5e7)):
    lines = ["| nodes touched | " + " | ".join(f"{n:.1e}" for n in nodes) + " | paper |", "|---|" + "---|" * (len(nodes) + 1)]
    for algo, name in (("sdcfr", "ES-SD-CFR (346 trav.)"), ("deepcfr", "Deep CFR (346 trav.)"), ("dream", "DREAM (900 trav.)"), ("escher", "ESCHER")):
        runs = curves.get(("dreamp", algo), {})
        if not runs:
            continue
        ours = [_fmt([_at(r, 3, n) for r in runs.values()]) for n in nodes]
        paper = DREAM_PAPER.get(algo, {})
        ref = ", ".join(f"≈{v} @ {k:.2g}" for k, v in paper.items()) or "–"
        lines.append(f"| {name}, ours ({len(runs)} seeds) | " + " | ".join(ours) + f" | {ref} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory")
    args = p.parse_args(argv)
    curves = load_curves(args.directory)
    print("SD-CFR paper protocol (1500 traversals, 750 warm-started updates, 1M buffers), mA/g:\n")
    print(sdcfr_table(curves))
    print("\nDREAM paper protocol (advantage nets 3000 x 2048 from scratch, 2M buffers), mA/g by nodes touched:\n")
    print(dream_table(curves))


if __name__ == "__main__":
    main()
