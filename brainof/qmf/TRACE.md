# LEMON batch traces

Each completed train microbatch and each evaluation batch writes one flushed JSON line to
`batches.jsonl` in its run directory. `epoch` and `batch` are one-based; noise evaluations use the
selected model's epoch. The older `results.json.best_epoch` is zero-based. `idx` indexes the split.
Stream names: `m1` = fMRI, `m2` = EEG, `fused` = serial fMRI-then-EEG backbone, not the final decision.

`loss` is the objective before dividing by `accumulation_size` for backpropagation:
sum of streams' label-smoothed `ce`, plus `fusion_ce_term` (zero in a single-stream run), plus
`rank_weighted`. `hard_ce` uses hard labels and supplies the loss history. `rank_active` indicates
whether the history-based ranking penalty was computed. With the original settings this starts in
epoch 2 only in dynamic multistream runs with nonzero lambda. `lr` is the rate used before stepping
the scheduler (layer decay is 1 here).

`streams.*.weight` and `logits` are paired, sample by sample, with `idx` and `y`.
The dynamic coefficient is `alpha * (-T * logsumexp(logits / T)) + beta`, not a probability
distribution over streams. Static weights are equal; single-stream weights are 1. `weighted_true_margin`
averages the contribution to the true class logit minus the other class logit. `without_flips` and
`without_correct` remove one decision stream without retraining or changing other weights. Removing a
unimodal stream is not the same as removing that modality from the joint backbone.

`moe` is keyed by the full stream and router path. Arrays use routed expert index order:
`count` is top-k assignments; divide by its sum for the load distribution. `sigmoid_mean` is the
mean raw gate score per expert before the correction bias; `bias_used` is the bias used in this
forward pass. `gate_mass_per_token` is the sum of applied top-k gate weights per expert divided by
the number of routed tokens (including zero mass on tokens not assigned to that expert). It sums
to approximately one for this top-2 model. The shared expert always runs; it is not one of those
eight routed experts. No token activations or extra model forwards are needed for these measures.

Check records: `python -m qmf.check_trace /path/to/batches.jsonl --rows 8`.
