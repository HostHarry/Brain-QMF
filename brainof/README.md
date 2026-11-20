# BrainOF-QMF source

This directory contains the code for the R1-R8 LEMON experiments, including
the original Brain-OF model and training entry points, the QMF extension,
LEMON data preparation, and per-batch MoE/QMF telemetry. Run commands **from
this directory**. No atlas, dataset, model weights or virtual environment is
included.

## Reproduce the traced matrix

1. Create a Python environment and install a CUDA build of PyTorch compatible
   with your GPU. The traced run used Python 3.12, torch 2.10.0+cu128 and
   torchvision 0.25.0+cu128. `requirements.txt` lists the upstream project's
   dependencies, including optional preprocessing and pretraining packages.
2. Supply your own pretrained [Brain-OF Base checkpoint](https://huggingface.co/han-ning/Brain-OF-Base)
   and paired LEMON LMDB root containing `LEMON/{train,val,test}/BrainSignal.lmdb`.
   If building these yourself, see `python -m qmf.prep_lemon --help` and the
   commands in `qmf/prep_lemon.py`. For the `fmri` and `build` stages, point
   `BRAINOF_ATLAS` at your local `BN_Atlas_246_2mm.nii.gz`.
3. First check the QMF formulas without data or GPU, then start the matrix:

```bash
python -m qmf.smoke_test
export LMDB=/absolute/private/path/to/lmdb
export CKPT=/absolute/private/path/to/brainof_base_50M_A22M.pth
export OUT=/absolute/private/path/to/new/run
SEEDS=3407 BATCH_TRACE=1 bash run_qmf_all.sh
```

`RUNS="R3 R7"` selects a subset. To run the two-GPU schedule used in this
study, set the same variables and use `bash run_traced_pair.sh` instead. It
requires two idle GPUs and an output directory that does not yet exist.
Do not commit the output directory: unlike the published aggregate tables,
raw traces contain per-sample fields and R7 saves a trained checkpoint.

| Component | Location |
| --- | --- |
| Fine-tuning, evaluation and checkpoint selection | `main_finetuning_qmf.py` |
| Modality branches and fusion objective | `qmf/model.py`, `qmf/core.py` |
| Backbone and MoE routers | `models/transformer*.py` |
| Trace writer, field definitions and checks | `qmf/telemetry.py`, `qmf/TRACE.md`, `qmf/check_trace.py` |
| R1-R8 recipes and hyperparameters | `run_qmf_all.sh`, `configs/BrainOF_Base_LEMON_qmf.yaml` |

## Provenance

Backbone and upstream training code: [HanningGuo/BrainOF](https://github.com/HanningGuo/BrainOF).
See the original [Apache-2.0 license](LICENSE) and [citation](CITATION.cff).
The experiment fork adds QMF and batch-level router instrumentation. The QMF
training entry point, model, loss, telemetry, launch scripts and traced config
are the versions used for the archived run. In this public copy the optional
timing scripts write to `--out_root`, and the LEMON preparation helper loads
an external atlas and omits the upstream participant lists. For preprocessing
scripts for unrelated datasets, see the upstream repository.
