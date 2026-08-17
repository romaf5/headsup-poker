"""Collect the evaluation artefacts of training runs into one Markdown table.

    python -m headsup.summarize runs/base runs/abl_* [--json out.json]

Reads, per run directory: ``eval.json`` (trainer: bots, head-to-head, LBR final), ``compare.json``
(headsup.compare), ``lbr_policy.json`` / ``lbr_sdcfr.json`` / ``lbr_last_iterate.json`` and the LBR
curve files ``lbr_*_t<N>.json`` (headsup.lbr), ``exploit_policy/results.json`` /
``exploit_sdcfr/results.json`` (headsup.exploit).  Missing files are simply skipped.
"""

import argparse
import glob
import json
import os
import re


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def summarize_run(run):
    out = {"run": os.path.basename(os.path.normpath(run))}
    ev = _load(os.path.join(run, "eval.json"))
    if ev:
        out["iteration"] = ev.get("iteration")
        out["model_config"] = ev.get("model_config")
        for key in ("deepcfr_chips_per_hand", "sdcfr_chips_per_hand"):
            if key in ev:
                out[key] = ev[key]
        if "deepcfr_vs_sdcfr_chips_per_hand" in ev:
            out["deepcfr_vs_sdcfr"] = (ev["deepcfr_vs_sdcfr_chips_per_hand"], ev.get("deepcfr_vs_sdcfr_se"))
        if "lbr_chips_per_hand" in ev:
            out["lbr_final"] = {k: (v, ev.get("lbr_se", {}).get(k)) for k, v in ev["lbr_chips_per_hand"].items()}
    cmp_ = _load(os.path.join(run, "compare.json"))
    if cmp_:
        out["compare"] = {k: (v["mean"], v["se"]) for k, v in cmp_.items()}
    for name in ("policy", "sdcfr", "last_iterate"):
        d = _load(os.path.join(run, f"lbr_{name}.json"))
        if d:
            out[f"lbr_{name}"] = (d["lbr_chips_per_hand"], d["se"], d["hands"])
    curve = {}
    for path in glob.glob(os.path.join(run, "lbr_*_t*.json")):
        m = re.search(r"lbr_(\w+?)_t(\d+)\.json$", path)
        d = _load(path)
        if m and d:
            curve.setdefault(m.group(1), {})[int(m.group(2))] = (d["lbr_chips_per_hand"], d["se"])
    if curve:
        out["lbr_curve"] = curve
    for name in ("policy", "sdcfr"):
        d = _load(os.path.join(run, f"exploit_{name}", "results.json"))
        if d:
            lb = d.get("lower_bound", {})
            out[f"exploit_{name}"] = (lb.get("chips_per_hand"), lb.get("se"), lb.get("exploiter"))
        for path in (f"br_{name}.json", f"br_{name}_12.json"):  # best-response exploitability (headsup.algos.holdem_br)
            d = _load(os.path.join(run, path))
            if d:
                out[f"br_{name}"] = (d["exploitability_chips"], None, d["boards"])
                break
    return out


def _fmt(pair, digits=2):
    if pair is None or pair[0] is None:
        return "–"
    m, se = pair[0], pair[1] if len(pair) > 1 else None
    return f"{m:+.{digits}f}" + (f" ± {se:.{digits}f}" if se is not None else "")


def markdown(rows):
    cfg = lambda r: (r.get("model_config") or {})
    head = ["run", "features", "net", "cards", "rm", "policy vs random/call/allin", "SD-CFR vs random/call/allin",
            "policy vs SD-CFR", "LBR policy", "LBR SD-CFR", "LBR last iterate", "PPO policy", "PPO SD-CFR", "BR policy", "BR SD-CFR"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        c = cfg(r)
        bots = lambda d: "/".join(f"{d[k]:+.2f}" for k in ("random", "call", "allin")) if d else "–"
        cells = [
            r["run"], c.get("features", "–"), c.get("arch", "–"), c.get("cards", "–"), c.get("rm_fallback", "–"),
            bots(r.get("deepcfr_chips_per_hand")), bots(r.get("sdcfr_chips_per_hand")),
            _fmt(r.get("deepcfr_vs_sdcfr")),
            _fmt(r.get("lbr_policy")), _fmt(r.get("lbr_sdcfr")), _fmt(r.get("lbr_last_iterate")),
            _fmt(r.get("exploit_policy")), _fmt(r.get("exploit_sdcfr")),
            _fmt(r.get("br_policy")), _fmt(r.get("br_sdcfr")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def small_game_table(directory, iterations=(10, 20, 50, 100, 200), unit=1000.0):
    """Markdown table of the small-game reproduction curves written by ``headsup.algos.deep --json``
    (``<algo>.json`` with a ``curve`` of exploitabilities per iteration) and the tabular baselines
    (``tabular.json``: {name: [{iteration, exploitability}]}); values x ``unit`` (chips -> milli-antes)."""
    lines = []
    deep = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        data = _load(path)
        if not data or name.startswith("tabular"):
            continue
        curve = {c["iteration"]: c for c in data.get("curve", [])}
        deep[name] = (data.get("args", {}), curve)
    if deep:
        cols = " | ".join(f"it. {i}" for i in iterations)
        lines.append(f"| algorithm (traversals / it.) | {cols} | current @ last |")
        lines.append("|---|" + "---|" * (len(iterations) + 1))
        for name, (args, curve) in deep.items():
            vals = []
            for i in iterations:
                c = curve.get(i)
                vals.append(f"{unit * c['average']:.0f}" if c else "–")
            last = curve[max(curve)] if curve else None
            cur = f"{unit * last['current']:.0f} (it. {max(curve)})" if last else "–"
            lines.append(f"| {name} ({args.get('traversals', '?')}) | " + " | ".join(vals) + f" | {cur} |")
    tab = _load(os.path.join(directory, "tabular.json"))
    if tab:
        lines.append("")
        lines.append("| tabular | exploitability of the average strategy by iteration |")
        lines.append("|---|---|")
        for name, curve in tab.items():
            lines.append(f"| {name} | " + ", ".join(f"it. {c['iteration']}: {unit * c['exploitability']:.0f}" for c in curve) + " |")
    return "\n".join(lines)


def holdem_br_table(runs):
    """Markdown table of the hold'em best-response exploitability files (headsup.algos.holdem_br
    --json): per run the policy net (``br_policy*.json``) and the SD-CFR average after t
    iterations (``br_sdcfr_t<N>.json``), in mbb/g."""
    lines = ["| run | policy net | SD-CFR average by iteration (mbb/g) |", "|---|---|---|"]
    for run in runs:
        pol = None
        for name in ("br_policy.json", "br_policy_12.json"):
            d = _load(os.path.join(run, name))
            if d:
                pol = f"{d['exploitability_mbb']:.0f} ({d['boards']} boards)"
                break
        curve = []
        for path in glob.glob(os.path.join(run, "br_sdcfr_t*.json")):
            m = re.search(r"_t(\d+)\.json$", path)
            d = _load(path)
            if m and d:
                curve.append((int(m.group(1)), d["exploitability_mbb"]))
        curve.sort()
        lines.append(f"| {os.path.basename(os.path.normpath(run))} | {pol or '–'} | "
                     + (", ".join(f"t={t}: {v:.0f}" for t, v in curve) if curve else "–") + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run directories")
    p.add_argument("--json", default=None)
    p.add_argument("--small-game", default=None, help="directory of small-game reproduction curves (headsup.algos.deep --json)")
    p.add_argument("--br", action="store_true", help="print the hold'em best-response exploitability table of the runs instead")
    args = p.parse_args(argv)
    if args.small_game:
        print(small_game_table(args.small_game))
        if not args.runs:
            return
    if args.br:
        print(holdem_br_table(args.runs))
        return
    rows = [summarize_run(r) for r in args.runs]
    print(markdown(rows))
    for r in rows:
        if "lbr_curve" in r:
            print(f"\nLBR curve {r['run']} (chips/hand, lower is better):")
            for name, pts in r["lbr_curve"].items():
                print(f"  {name:12s} " + "  ".join(f"@{t}: {v[0]:+.2f}" for t, v in sorted(pts.items())))
        if "compare" in r:
            print(f"\nhead-to-head {r['run']} (row vs column, chips/hand):")
            for k, v in r["compare"].items():
                print(f"  {k}: {v[0]:+.3f} ± {v[1]:.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
