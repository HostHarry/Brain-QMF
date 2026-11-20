"""Export aggregate results from a BrainOF-QMF out directory into a fresh results repo."""
import argparse
import csv
import gzip
import json
import math
import re
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


ARGS = ("cfg", "batch_size", "accumulation_steps", "disable_amp", "eval_mode", "streams", "fusion",
        "lamb", "alpha", "beta", "temperature", "rank_mode", "reg_start_epoch", "lr_scale", "seed",
        "tag", "noise_sigmas", "noise_seed", "no_save", "batch_trace")
BATCH = ("split", "epoch", "batch", "n", "sigma", "loss", "rank_raw", "rank_weighted", "fusion_ce_term",
         "accuracy", "rank_active", "lr", "optimizer_update", "accumulation_size")
HEAD = ("ce", "hard_ce", "rank_raw", "w_mean", "w_std", "weighted_true_margin", "accuracy",
        "without_correct", "without_flips")
ROUTER = ("count", "sigmoid_mean", "gate_mass_per_token", "bias_used")
TRAIN = re.compile(r"Train: \[(\d+)/\d+\]\[(\d+)/(\d+)\] lr ([\de.+-]+) loss ([\de.+-]+)"
                   r" reg ([\de.+-]+) mem ([\d.]+)MB")


def pick(d, keys):
    return {k: d[k] for k in keys if k in d}


def metrics(d):
    if not all(isinstance(v, (int, float)) for v in d.values()):
        raise ValueError("Expected only aggregate scalar metrics")
    return {k: v if math.isfinite(v) else None for k, v in d.items()}


def write(path, data):
    with path.open("x") as out:
        json.dump(data, out, allow_nan=False, indent=2)
        out.write("\n")


def export_batches(src, dest):
    directory = dest / "batches"
    directory.mkdir()
    counts = Counter()
    buf = []
    part = 0

    def flush():
        nonlocal part
        part += 1
        path = directory / f"part-{part:03d}.jsonl.gz"
        with path.open("xb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=4) as z:
            z.writelines(buf)
        buf.clear()

    with src.open("rb") as f:
        for line in f:
            raw = json.loads(line)
            row = pick(raw, BATCH)
            row["streams"] = {s: pick(head, HEAD) for s, head in raw["streams"].items()}
            row["moe"] = {k: {field: m[field] for field in ROUTER} for k, m in raw["moe"].items()}
            counts[row["split"]] += 1
            buf.append((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
            if len(buf) == 1000:
                flush()
    if buf:
        flush()
    assert counts["train"] == 9000 and counts["val"] == 1950 and counts["noise_test"] == 160
    write(dest / "batch_counts.json", dict(counts))
    return sum(counts.values())


def export_run(job):
    args_file, out = job
    src = args_file.parent
    a = json.loads(args_file.read_text())
    traced = (src / "batches.jsonl").is_file()
    stamp = src.name.rsplit("_", 2)[-2:]
    group = "traced" if traced else "original"
    complete = (src / "results.json").is_file()
    tag = a["tag"]
    dst = out / "runs" / group / (tag if complete else f"{tag}-partial-{stamp[1]}")
    dst.mkdir(parents=True, exist_ok=False)
    config = pick(a, ARGS)
    config["pretrained_model"] = Path(a["pretrained_model"]).name
    write(dst / "config.json", config)
    epochs = [{"epoch": e["epoch"], **{k: metrics(e[k]) for k in ("train", "val", "test") if k in e}}
              for e in json.loads((src / "epochs.json").read_text())]
    write(dst / "epochs.json", epochs)
    row = dict(series=group, run=tag, timestamp="_".join(stamp), status="complete" if complete else "superseded",
               seed=a["seed"], streams=a["streams"], fusion=a["fusion"], epochs=len(epochs),
               best_epoch_1based=None, test_bac=None, test_auroc=None, traced_batches=0,
               path=dst.relative_to(out).as_posix())
    if complete:
        r = json.loads((src / "results.json").read_text())
        result = pick(r, ("streams", "seed", "best_epoch", "train_time"))
        result.update(val=metrics(r["val"]), test=metrics(r["test"]), noise={k: metrics(v) for k, v in r["noise"].items()})
        write(dst / "results.json", result)
        row.update(best_epoch_1based=r["best_epoch"] + 1, test_bac=r["test"]["bac"], test_auroc=r["test"]["auroc"])
    else:
        s = json.loads((src / "superseded.json").read_text())
        write(dst / "superseded.json", pick(s, ("status", "reason", "old_batch_size", "old_accumulation_steps", "stopped_utc")))
    if traced:
        row["traced_batches"] = export_batches(src / "batches.jsonl", dst)
    else:
        with (src / "log_rank0.log").open() as log, (dst / "sampled_train.csv").open("x", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "batch", "batches_in_epoch", "lr", "loss", "reg", "memory_mb"])
            for line in log:
                m = TRAIN.search(line)
                if m:
                    w.writerow(m.groups())
    print(group, dst.name, row["status"], row["traced_batches"], flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    p, out = args.source.resolve(), args.dest.resolve()
    if (out / "runs").exists():
        parser.error("use a destination without a runs directory")
    traced = p / "traced_seed3407_20261120"
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(export_run, [(f, out) for f in sorted(p.rglob("qmf_args.json"))]))
    assert Counter((r["series"], r["status"]) for r in rows) == {
        ("original", "complete"): 8, ("original", "superseded"): 3, ("traced", "complete"): 8}
    with (out / "experiments.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    analysis = out / "analysis"
    analysis.mkdir()
    for name in ("coverage.csv", "qmf_heads.csv", "qmf_epochs.csv", "moe_layers.csv", "qmf_moe.png",
                 "findings.md", "analyze.py"):
        shutil.copyfile(traced / "analysis" / name, analysis / name)
    shutil.copyfile(p / "diagnostics_20261120/historical_stream_contribution.csv",
                    analysis / "historical_stream_contribution.csv")


if __name__ == "__main__":
    main()
