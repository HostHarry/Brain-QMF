# BrainOF-QMF / LEMON, seed 3407

All eight runs completed. Each has 50 epochs, 9,000 training batches, 1,950 validation
batches, and 40 clean test batches at the validation-selected epoch. The test metrics
below are balanced accuracy (BAC) over all 2,556 paired windows, not batch-averaged BAC.
The windows come from 30 participants (26/4 by age group), 60 distinct fMRI sample keys
and 2,556 distinct EEG sample keys. The windows are not independent participants.

| Run | Model | Selected epoch | Clean test BAC (%) |
| --- | --- | ---: | ---: |
| R1 | fMRI | 38 | 58.29 |
| R2 | EEG | 20 | 83.06 |
| R3 | Joint backbone alone | 6 | 68.26 |
| R4 | fMRI + EEG, static | 7 | 58.70 |
| R5 | fMRI + EEG, QMF, lambda=0 | 33 | 43.71 |
| R6 | fMRI + EEG, QMF, lambda=0.1 | 27 | 49.97 |
| R7 | fMRI + EEG + joint, QMF, lambda=0.1 | 32 | 63.30 |
| R8 | fMRI + EEG + joint, static | 23 | 69.36 |

## QMF

In R7 the EEG head scores 85.77% on its own, yet removing its logit term changes zero
test decisions. Removing fMRI changes 542 and raises BAC to 78.87%; removing the joint
head changes 97 and lowers BAC to 61.14%. These operations do not remove EEG or fMRI
from the joint backbone. Mean weights (fMRI/EEG/joint) are 0.523/0.096/0.252.
The fMRI weight exceeds EEG on every clean test window in R5, R6 and R7.
R6 improves over R5 but the EEG head in R6 still has 85.35% BAC vs 49.97% combined.

At the run settings, w = 0.1 logsumexp(logits). Adding a common offset to both class
logits leaves branch softmax and branch CE unchanged but changes w by 0.1 times the
offset. In R7 the mean logit offsets are +3.708 for fMRI and -0.017 for EEG; they
explain about 87% of the difference in mean weight. The ranking term compares samples
within a stream and cannot directly calibrate constant offsets between streams.
Freezing R7 logits and assigning equal weights gives 77.00% vs 58.91% BAC on
validation and 79.40% vs 63.30% on test. Centering logits for the gate alone gives
74.89%/79.53%. These are post-hoc diagnostics, not new trained models. R8 is a
separately trained static control; it is not the equal-weight replay of R7.

## MoE

Define effective experts per layer as exp(entropy of that layer's top-2 assignment
shares). Eight indicates uniform use; this measure is distinct from gate mass.
In R7 epoch 32 the median over 52 layers is 7.93 on aggregated training traffic vs
5.41 on clean test traffic. In the fMRI/EEG/joint streams, the clean test medians
are 4.54/6.02/5.44. Different layers and branches have independent experts.

Most concentrated: fMRI ARNESS layer 2 (zero-based index 1). On test, expert indices
1 and 5 account for 49.80% and 49.01% of all top-2 assignments. Expert 1 receives
91.10% of the applied routed gate mass on test and 91.38% on validation. Training
for that epoch has 7.73 aggregate effective experts in this layer, but its token-weighted
mean within-batch effective count is only 3.93. Bias correction balances counts over
training batches; it is frozen at evaluation and does not equalize gate mass. R8 also
shows concentration (maximum single-expert routed mass: 87.23% across fMRI layers).
The shared expert is always active and is not counted. None of this establishes
that an expert causes the classification errors.

## Implications

Prioritize per-stream logit-scale/offset calibration on validation before changing
expert capacity. Compare a fixed equal gate and a centered/calibrated gate using
the same frozen backbone; include EEG alone and separate learning-rate controls
(current peak lr fMRI/EEG ratio: 15). Verify across seeds and at participant level;
do not tune on these test results. Separate training with and without MoE bias
correction would be needed to attribute model quality to load balancing.

`qmf_heads.csv` contains validation/test replay and ablations. `qmf_epochs.csv`
contains validation trajectory and training regularization. `moe_layers.csv` contains
per-layer assignment and gate-mass distributions; `coverage.csv` contains coverage.
`analyze.py` rebuilds the CSVs on CPU and verifies the clean predictions and BAC.
