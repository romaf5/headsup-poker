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
    return out


def _fmt(pair, digits=2):
    if pair is None or pair[0] is None:
        return "–"
    m, se = pair[0], pair[1] if len(pair) > 1 else None
    return f"{m:+.{digits}f}" + (f" ± {se:.{digits}f}" if se is not None else "")


def markdown(rows):
    cfg = lambda r: (r.get("model_config") or {})
    head = ["run", "features", "net", "cards", "rm", "policy vs random/call/allin", "SD-CFR vs random/call/allin",
            "policy vs SD-CFR", "LBR policy", "LBR SD-CFR", "LBR last iterate", "PPO policy", "PPO SD-CFR"]
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
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
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
