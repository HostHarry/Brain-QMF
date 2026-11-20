"""CPU-only, frozen-checkpoint analysis of the traced LEMON runs."""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREFIX = re.compile(rb'^\{"split":"([^"]+)","epoch":(\d+),"batch":(\d+),')


def save(name, rows):
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def bac(y, pred):
    return np.mean([(pred[y == k] == k).mean() for k in (0, 1)])


def effective(p):
    return np.exp(-(p * np.log(p.clip(1e-30))).sum(axis=-1))


def main():
    moe, heads, timeline, coverage = [], [], [], []
    y_common = None
    for f in sorted((ROOT / "parallel").glob("*/results/LEMON/*/results.json")):
        d = json.loads(f.read_text())
        args, streams = d["args"], d["streams"]
        tag, best = args["tag"], d["best_epoch"] + 1
        epochs = json.loads(f.with_name("epochs.json").read_text())
        assert len(epochs) == 50 and best == max(epochs, key=lambda e: (e["val"]["bac"], e["epoch"]))["epoch"] + 1
        for e in epochs:
            for s in streams:
                timeline.append(dict(run=tag, epoch=e["epoch"]+1, stream=s, train_loss=e["train"]["loss"],
                                     rank_raw=e["train"]["reg"], val_bac=e["val"]["bac"],
                                     val_stream_bac=e["val"][f"bac_{s}"], val_w=e["val"][f"w_mean_{s}"]))
        batches, layers, accum, eval_rows = Counter(), {}, {}, defaultdict(list)
        with f.with_name("batches.jsonl").open("rb") as h:
            for line in h:
                split, epoch, batch = PREFIX.match(line).groups()
                split, epoch, batch = split.decode(), int(epoch), int(batch)
                if split != "noise_test":
                    batches[split, epoch] += 1
                    assert batches[split, epoch] == batch
                use_eval = epoch == best and split in ("test_at_best", "val")
                use_train = tag in ("R7", "R8") and split == "train" and epoch in (1, 5, 10, best, 50)
                if not (use_train or use_eval):
                    continue
                r = json.loads(line)
                if use_eval:
                    eval_rows[split].append((r["idx"], r["y"], r["streams"]))
                key = split, epoch
                names = layers.setdefault(key, tuple(r["moe"]))
                assert tuple(r["moe"]) == names
                a = np.array([[m["count"], m["gate_mass_per_token"], m["bias_used"]]
                              for m in r["moe"].values()])
                counts = a[:, 0]
                assert counts.shape == (len(names), 8) and np.all(counts.sum(1) > 0)
                tokens = counts.sum(1, keepdims=True) / 2
                load = counts / (2 * tokens)
                eff_batch = effective(load)[:, None] * tokens
                stats = np.concatenate((counts, a[:, 1] * tokens, np.abs(a[:, 2]).max(1, keepdims=True) * tokens,
                                        eff_batch, tokens), axis=1)
                if key not in accum:
                    accum[key] = np.zeros_like(stats)
                accum[key] += stats
        assert all(batches["train", e] == 180 and batches["val", e] == 39 for e in range(1, 51))
        assert batches["test_at_best", best] == 40
        coverage.append(dict(run=tag, epochs=50, train_batches=9000, val_batches=1950, best_epoch=best,
                             clean_best_test_batches=40, clean_best_test_n=d["test"]["n"]))
        for (split, ep), stats in accum.items():
            for name, v in zip(layers[split, ep], stats):
                load = v[:8] / v[:8].sum()
                mass = v[8:16] / v[8:16].sum()
                moe.append(dict(run=tag, split=split, epoch=ep, stream=name.split(".")[1], layer=name,
                                tokens=int(v[-1]), active=int((load > 0).sum()), effective=effective(load),
                                mean_batch_effective=v[-2]/v[-1], top_share=load.max(), min_share=load.min(),
                                top_mass_share=mass.max(), mean_max_abs_bias=v[-3]/v[-1],
                                **{f"load_{i}": x for i, x in enumerate(load)},
                                **{f"mass_{i}": x for i, x in enumerate(mass)}))
        for split, rows in eval_rows.items():
            metrics = d["test" if split == "test_at_best" else "val"]
            y = np.concatenate([r[1] for r in rows])
            idx = np.concatenate([r[0] for r in rows])
            np.testing.assert_array_equal(idx, np.arange(metrics["n"]))
            lg = np.stack([np.concatenate([r[2][s]["logits"] for r in rows]) for s in streams]).astype("f")
            w = np.stack([np.concatenate([r[2][s]["weight"] for r in rows]) for s in streams]).astype("f")
            parts = lg * w[:, :, None]
            pred = parts.sum(0).argmax(1)
            np.testing.assert_allclose(bac(y, pred), metrics["bac"])
            if split == "test_at_best":
                with np.load(f.with_name("test_outputs.npz")) as saved:
                    np.testing.assert_array_equal(y, saved["y"])
                    np.testing.assert_allclose(parts.sum(0), saved["fused"], atol=1e-6)
                if y_common is not None:
                    np.testing.assert_array_equal(y, y_common)
                y_common = y
            center = lg.mean(-1)
            centered = lg - center[:, :, None]
            t = args["temperature"]
            cw = args["beta"] - args["alpha"] * t * np.logaddexp.reduce(centered / t, axis=-1)
            if len(streams) > 1 and args["fusion"] == "qmf":
                np.testing.assert_allclose(w, cw - args["alpha"] * center, atol=1e-6)
            eq_bac = bac(y, lg.sum(0).argmax(1))
            centered_bac = bac(y, (lg * cw[:, :, None]).sum(0).argmax(1))
            for i, s in enumerate(streams):
                remaining = np.delete(parts, i, axis=0).sum(0).argmax(1) if len(streams) > 1 else None
                if remaining is not None:
                    np.testing.assert_allclose(bac(y, remaining), metrics[f"bac_without_{s}"])
                heads.append(dict(run=tag, split=split, epoch=best, stream=s, n=len(y),
                                  full_bac=bac(y, pred), stream_bac=bac(y, lg[i].argmax(1)),
                                  w_mean=w[i].mean(), logit_mean=center[i].mean(), centered_qmf_w=cw[i].mean(),
                                  abs_weighted_margin=np.abs(parts[i, :, 1]-parts[i, :, 0]).mean(),
                                  without_bac=bac(y, remaining) if remaining is not None else None,
                                  flips=int((pred != remaining).sum()) if remaining is not None else None,
                                  equal_weight_bac=eq_bac, centered_gate_bac=centered_bac))
        print(tag, "verified", flush=True)
    assert len(coverage) == 8
    save("coverage.csv", coverage)
    save("qmf_heads.csv", heads)
    save("qmf_epochs.csv", timeline)
    save("moe_layers.csv", moe)


if __name__ == "__main__":
    main()
