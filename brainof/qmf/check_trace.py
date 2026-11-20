"""Verify recorded forward passes against independent NumPy reconstructions (LEMON Base)."""
import argparse
import json
from pathlib import Path

import numpy as np


def check(row, args):
    streams = row["streams"]
    near = np.testing.assert_allclose
    y = np.array(row["y"])
    n = len(y)
    assert n == row["n"] == len(row["idx"])
    near(sum(v["ce"] for v in streams.values()) + row["fusion_ce_term"] + row["rank_weighted"], row["loss"],
         atol=2e-6)
    near(sum(v["rank_raw"] for v in streams.values()), row["rank_raw"], atol=2e-6)
    near(row["rank_raw"] * args["lamb"], row["rank_weighted"], atol=2e-6)
    parts = []
    for v in streams.values():
        lg = np.asarray(v["logits"], dtype=np.float32)
        w = np.asarray(v["weight"], dtype=np.float32)
        assert lg.shape == (n, 2) and w.shape == (n,)
        if len(streams) == 1 or args["fusion"] == "static":
            expected = 1 / len(streams)
        else:
            t = args["temperature"]
            expected = args["beta"] - args["alpha"] * t * np.logaddexp.reduce(lg / t, axis=1)
        near(w, expected, atol=2e-6)
        parts.append(lg * w[:, None])
    pred = np.sum(parts, axis=0).argmax(1)
    near((pred == y).mean(), row["accuracy"])
    for i, (s, v) in enumerate(streams.items()):
        if len(parts) > 1:
            without = sum(p for j, p in enumerate(parts) if i != j).argmax(1)
            assert (without != pred).sum() == v["without_flips"]
            assert (without == y).sum() == v["without_correct"]
        assert any(k.startswith(f"nets.{s}.") for k in row["moe"])
    for layer in row["moe"].values():
        count, gate = np.array(layer["count"]), np.array(layer["gate_mass_per_token"])
        prob = np.array(layer["sigmoid_mean"])
        assert len(count) == len(gate) == len(prob) == len(layer["bias_used"]) == 8
        assert np.all(count >= 0) and 0 < count.sum() and count.sum() % 2 == 0
        assert np.all((prob >= 0) & (prob <= 1)) and np.all(gate >= 0)
        assert np.all(gate[count == 0] == 0)
        near(gate.sum(), 1., atol=4e-3)  # top-2 weights may be rounded to bf16 before application


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--rows", type=int, default=1)
    opt = parser.parse_args()
    args = json.loads(opt.file.with_name("qmf_args.json").read_text())
    with opt.file.open() as f:
        for _ in range(opt.rows):
            row = json.loads(next(f))
            check(row, args)
    print(f"{args['tag']}: checked {opt.rows} rows; last {row['split']} epoch={row['epoch']} "
          f"batch={row['batch']} loss={row['loss']:.6f} routers={len(row['moe'])}")


if __name__ == "__main__":
    main()
