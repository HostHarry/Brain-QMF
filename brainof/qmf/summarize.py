"""Collect every results.json below an output folder and print one Markdown table (mean +- std over seeds per tag).

  python -m qmf.summarize --out_root $OUT
"""
import os
import json
import argparse

import numpy as np


def fmt(values, scale=100.0, digits=2):
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not values:
        return "-"
    return f"{np.mean(values) * scale:.{digits}f} +- {np.std(values) * scale:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()
    runs = {}
    for folder, _, files in os.walk(args.out_root):
        if "results.json" in files:
            with open(os.path.join(folder, "results.json")) as f:
                result = json.load(f)
            runs.setdefault(result["args"]["tag"], []).append(result)
    if not runs:
        raise SystemExit("no results.json found")
    sigmas = sorted({s for rs in runs.values() for r in rs for s in r.get("noise", {})}, key=float)
    header = ["tag", "streams", "fusion", "lambda", "seeds", "test BAC", "test AUROC", "BAC m1", "BAC m2",
              "BAC fused-stream", "r(w,loss) m1", "r(w,loss) m2"] + [f"BAC noise {s}" for s in sigmas]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for tag in sorted(runs):
        rs = runs[tag]
        a = rs[0]["args"]
        test = [r["test"] for r in rs]
        row = [tag, a["streams"], a["fusion"], str(a["lamb"]), str(len(rs)),
               fmt([t.get("bac") for t in test]), fmt([t.get("auroc") for t in test]),
               fmt([t.get("bac_m1") for t in test]), fmt([t.get("bac_m2") for t in test]),
               fmt([t.get("bac_fused") for t in test]),
               fmt([t.get("pearson_w_loss_m1") for t in test], scale=1.0, digits=3),
               fmt([t.get("pearson_w_loss_m2") for t in test], scale=1.0, digits=3)]
        row += [fmt([r["noise"].get(s, {}).get("bac") for r in rs]) for s in sigmas]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
