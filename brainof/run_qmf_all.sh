#!/bin/bash
# Runs the experiment matrix. Usage (from the BrainOF repository root, conda env "brainof" active):
#   LMDB=... OUT=... CKPT=... bash run_qmf_all.sh
# Optional: SEEDS="3407" RUNS="R3 R6 R7" to run a subset. Finished runs are skipped on restart.
set -euo pipefail
: "${LMDB:?set LMDB}" "${OUT:?set OUT}" "${CKPT:?set CKPT}"
SEEDS=${SEEDS:-"3407 3408 3409 3410 3411"}
RUNS=${RUNS:-"R1 R2 R3 R4 R5 R6 R7 R8"}
BATCH_SIZE=${BATCH_SIZE:-64}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-8}
trace_flags=()
if [[ ${BATCH_TRACE:-0} == 1 ]]; then trace_flags+=(--batch_trace); fi
# peak learning rates of the official LEMON configs: fMRI 7.5e-5, EEG 5e-6, multimodal 4e-5 (= base of the YAML)
LR_SCALE="m1=1.875,m2=0.125,fused=1.0"

declare -A SPEC=(
  [R1]="--streams m1"
  [R2]="--streams m2"
  [R3]="--streams fused"
  [R4]="--streams m1,m2 --fusion static"
  [R5]="--streams m1,m2 --fusion qmf --lamb 0"
  [R6]="--streams m1,m2 --fusion qmf --lamb 0.1"
  [R7]="--streams m1,m2,fused --fusion qmf --lamb 0.1"
  [R8]="--streams m1,m2,fused --fusion static"
)

mkdir -p "$OUT/done"
for seed in $SEEDS; do
  for run in $RUNS; do
    marker="$OUT/done/${run}_seed${seed}"
    if [ -f "$marker" ]; then echo "skip $run seed $seed"; continue; fi
    save_flag="--no_save"
    if [ "$run" = "R7" ]; then save_flag=""; fi
    # shellcheck disable=SC2086
    torchrun --standalone --nnodes=1 --nproc_per_node=1 main_finetuning_qmf.py \
      --cfg configs/BrainOF_Base_LEMON_qmf.yaml --root "$LMDB" --out_root "$OUT/parallel/${run}_seed${seed}" \
      --batch_size "$BATCH_SIZE" --accumulation_steps "$ACCUMULATION_STEPS" \
      --pretrained_model "$CKPT" --lr_scale "$LR_SCALE" --seed "$seed" --tag "$run" \
      "${trace_flags[@]}" $save_flag ${SPEC[$run]}
    touch "$marker"
  done
done
if [[ ${SKIP_SUMMARY:-0} != 1 ]]; then python -m qmf.summarize --out_root "$OUT"; fi
