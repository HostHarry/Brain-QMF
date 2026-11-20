#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")" && pwd)
cd "$repo"
export PATH="$repo/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BATCH_TRACE=1 SKIP_SUMMARY=1 SEEDS=3407 BATCH_SIZE=64 ACCUMULATION_STEPS=8
: "${OUT:?provide a new output directory}" "${CKPT:?}" "${LMDB:?}"
mkdir "$OUT"
exec 9>"$OUT/queue.lock"
flock -n 9
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
    echo "GPU already in use; not starting this queue" >&2; exit 1
fi
python -m pip freeze > "$OUT/environment.txt"
tar -czf "$OUT/training_source.tgz" --exclude=.venv --exclude=__pycache__ --exclude=install.log \
    configs models datasets utils qmf main_finetuning_qmf.py run_qmf_all.sh run_traced_pair.sh
sha256sum "$CKPT" "$OUT/training_source.tgz" > "$OUT/sha256.txt"
printf 'driver_pid=%s\nGPU 0: R7 R6 R3 R1\nGPU 1: R8 R5 R4 R2\n' "$$" > "$OUT/queue.txt"
CUDA_VISIBLE_DEVICES=0 RUNS="R7 R6 R3 R1" bash run_qmf_all.sh > "$OUT/gpu0.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 RUNS="R8 R5 R4 R2" bash run_qmf_all.sh > "$OUT/gpu1.log" 2>&1 & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
if ((rc == 0)); then
    python -m qmf.summarize --out_root "$OUT" > "$OUT/summary.md"
fi
echo "$rc" > "$OUT/queue.exit"
exit "$rc"
