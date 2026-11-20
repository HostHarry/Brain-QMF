# Brain-QMF: LEMON experiment results

Two local runs of the R1-R8 Brain-OF Base + QMF ablation matrix. Both use seed
3407; these are repeated runs, **not** two independent seeds. The archive
contains aggregate experiment outputs, not the LEMON participant dataset.

## Clean test balanced accuracy (%)

| Run | Branches and fusion | Nov 13 | Nov 20 (traced) |
| --- | --- | ---: | ---: |
| R1 | fMRI alone | 55.13 | 58.29 |
| R2 | EEG alone | 86.65 | 83.06 |
| R3 | Original joint backbone alone | 70.32 | 68.26 |
| R4 | fMRI + EEG, equal weights | 61.42 | 58.70 |
| R5 | fMRI + EEG, QMF, lambda=0 | 55.81 | 43.71 |
| R6 | fMRI + EEG, QMF, lambda=0.1 | 42.27 | 49.97 |
| R7 | fMRI + EEG + joint, QMF, lambda=0.1 | 47.97 | 63.30 |
| R8 | fMRI + EEG + joint, equal weights | 81.98 | 69.36 |

Each completed run used 50 epochs, batch size 64 and 8 accumulation steps.
The table reports BAC on all 2,556 clean paired test windows at the best
validation epoch, **not** the batch average. Windows come from 30 participants,
only four in the minority age group. The official paper reports 76.77 +/- 0.60%
for the Base joint backbone on its own five-seed evaluation; it is **not** a
third run in this archive. See [Brain-OF](https://arxiv.org/abs/2602.23410).

## Data map

- `experiments.csv`: all 19 attempts, labeled complete or superseded, with epoch
  counts and clean test BAC/AUROC. `original` means the Nov 13 **series**, not a
  different architecture. Partial runs have no final test result.
- `runs/{original,traced}/R*/`: sanitized training arguments, epoch metrics and,
  for finished runs, selected validation/test and noise-sweep results. Undefined
  correlations are JSON `null`. `epochs.json.epoch` and `results.json.best_epoch`
  are zero-based; the index CSV and batch traces use one-based epochs.
- `runs/original/`: includes three superseded batch-16 attempts (R3: 26 recorded
  epochs, R7: 18, R8: 12). `sampled_train.csv` extracts the periodic training log
  lines; full per-batch traces did not exist for this series.
- `runs/traced/R*/batches/part-*.jsonl.gz`: 91,440 batch-level rows in 96 ordered
  shards, including all 72,000 training batches, validation, test-at-best and
  noise tests. `batch_counts.json` gives exact split counts by run.
- `analysis/`: validation trajectories, clean-test branch ablations, MoE routing
  distributions, a plot and [interpretation](analysis/findings.md). The original
  analysis script operates on the **private** raw run tree, not on this export.
- `configs/`: traced task configuration. Checkpoint SHA-256 (reference only):
  `2d04b79581b8285f22813ea85389af72cee4085dbd42734f684152245a5f067a`.
  Key traced runtime versions: torch 2.10.0+cu128, torchvision 0.25.0+cu128,
  numpy 2.4.3, timm 1.0.12, scikit-learn 1.8.0 and lmdb 2.2.0.

## Reading a batch

```python
import gzip
import json
from pathlib import Path

shards = Path("runs/traced/R7/batches").glob("part-*.jsonl.gz")
with gzip.open(min(shards), "rt") as f:
    row = json.loads(next(f))
print(row["epoch"], row["batch"], row["loss"], row["streams"]["m2"]["w_mean"])
```

`m1` = fMRI, `m2` = EEG, `fused` = joint backbone. `loss` is before gradient
accumulation scaling. `ce` includes label smoothing; `hard_ce` does not.
`rank_raw` and `rank_weighted` distinguish the rank penalty and its contribution
to the objective. `w_mean`/`w_std` summarize the raw weights (not normalized
across branches). `without_flips` removes only the given output term with the
remaining weights fixed. Each named router has eight routed experts:
`count` is top-2 assignment count; `sigmoid_mean` is the pre-bias gate score;
`gate_mass_per_token` is applied gate mass per token; `bias_used` is the bias
used for selection. The always-active shared expert is excluded.

## Scope and integrity

This is a results-only export. No raw EEG/fMRI, participant identifiers,
sample indices or labels, per-window logits/weights, model weights, raw machine
logs, account credentials, or unrelated LLaDA/MMaDA results are included.
`scripts/export_results.py` builds the aggregate tables/traces from the private
run tree. Check the exported files with `sha256sum -c SHA256SUMS`.
