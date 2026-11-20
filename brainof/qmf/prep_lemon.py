"""LEMON (EEG + fMRI) preparation for Brain-OF: download helper, official preprocessing, LMDB, pairing.

Run from the BrainOF repository root:
  python -m qmf.prep_lemon fmri  --raw $RAW --work $WORK
  python -m qmf.prep_lemon build --raw $RAW --work $WORK --lmdb $LMDB --seed 3407
  python -m qmf.prep_lemon pair  --lmdb $LMDB --seed 3407
  python -m qmf.prep_lemon check --lmdb $LMDB

The signal processing itself is the official dataset_maker/make_LEMON.py (functions `preprocessing` and
`preparing`). Set BRAINOF_ATLAS to a separately obtained 2 mm Brainnetome atlas for fmri/build.
"""
import os
import sys
import json
import glob
import pickle
import shutil
import argparse
import subprocess
import importlib
from urllib.request import Request
from urllib.error import HTTPError
from qmf.http_retry import urlopen_retry, is_retryable_network_error

import numpy as np

GWDG = "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON"
FMRI_NAME = "{sub}_ses-01_task-rest_acq-AP_run-01_MNI2mm.nii.gz"
FMRI_URL = GWDG + "/MRI_MPILMBB_LEMON/MRI_Preprocessed_Derivetives/{sub}/func/" + FMRI_NAME
SPLITS = ("train", "val", "test")


def load_official():
    """Import preprocessing only for stages that need the atlas and raw-signal dependencies."""
    return importlib.import_module("dataset_maker.make_LEMON")


def read_labels(csv_path):
    """{subject: (age_bin, gender)} with the same column positions as make_LEMON.py (0: ID, 1: gender, 2: age)."""
    import pandas as pd
    demographics = pd.read_csv(csv_path)
    return {str(row.iloc[0]): (row.iloc[2], row.iloc[1]) for _, row in demographics.iterrows()}


def h5_subject(name):
    return os.path.basename(name).split("_")[0]


# ---------------------------------------------------------------- stage 1: fMRI
def stage_fmri(args):
    official = load_official()
    labels = read_labels(os.path.join(args.raw, "Participants_LEMON.csv"))
    manifest = {}
    manifest_path = os.path.join(args.raw, "fmri_size_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            rows = json.load(f)
        manifest = {r["subject"]: r for r in rows}
        assert len(rows) == len(manifest) and set(manifest) == set(labels), "fMRI manifest cohort mismatch"
        assert all(r["exit"] == 0 and r["status"] in (200, 404) and
                   (r["status"] == 404 or isinstance(r["bytes"], int) and r["bytes"] > 0)
                   for r in rows), "Invalid fMRI size manifest"
        print("Using verified fMRI size manifest; downloaded sizes will still be checked", flush=True)
    out_root = os.path.join(args.work, "h5", "LEMON_fMRI")
    pool = os.path.join(out_root, "_pool")
    tmp = os.path.join(args.work, "nifti_tmp")
    os.makedirs(pool, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    status_path = os.path.join(args.work, "fmri_status.json")
    status = json.load(open(status_path)) if os.path.isfile(status_path) else {}
    subjects = sorted(labels) if not args.subjects else args.subjects.split(",")
    for n, sub in enumerate(subjects):
        done = glob.glob(os.path.join(pool, f"{sub}_*_sample*.h5"))
        if len(done) == 2 or status.get(sub) in ("no_file", "failed", "excluded_official_eeg_only"):
            continue
        local = os.path.join(args.fmri_local, sub, "func", FMRI_NAME.format(sub=sub)) if args.fmri_local else None
        if local and os.path.isfile(local):
            nifti, downloaded = local, False
        else:
            nifti, downloaded = os.path.join(tmp, FMRI_NAME.format(sub=sub)), True
            url = args.fmri_url.format(sub=sub)
            print(f"[{n + 1}/{len(subjects)}] downloading {url}", flush=True)
            if shutil.disk_usage(tmp).free < 10 * 2**30:
                raise SystemExit("Less than 10 GiB free; pausing before the next fMRI download.")
            entry = manifest.get(sub)
            if entry and entry["url"] == url:
                if entry["status"] == 404:
                    status[sub] = "no_file"
                    json.dump(status, open(status_path, "w"), indent=1)
                    continue
                expected_size = entry["bytes"]
            else:
                try:
                    with urlopen_retry(Request(url, method="HEAD"), timeout=60) as response:
                        expected_size = int(response.headers["Content-Length"])
                except HTTPError as exc:
                    if exc.code == 404:
                        status[sub] = "no_file"
                        json.dump(status, open(status_path, "w"), indent=1)
                        continue
                    raise
            if not os.path.isfile(nifti) or os.path.getsize(nifti) != expected_size:
                partial = nifti + ".part"
                subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--continue-at", "-",
                                "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1024",
                                "--retry", "5", "--retry-delay", "30", "--retry-all-errors", "--output", partial, url], check=True)
                if os.path.getsize(partial) != expected_size:
                    raise RuntimeError(f"Incomplete fMRI download: {sub}")
                os.replace(partial, nifti)
        official.preparing((nifti, labels[sub], "_pool"), out_root)
        ok = len(glob.glob(os.path.join(pool, f"{sub}_*_sample*.h5"))) == 2
        status[sub] = "ok" if ok else "failed"
        json.dump(status, open(status_path, "w"), indent=1)
        if ok and downloaded and not args.keep_nifti and os.path.isfile(nifti):
            os.remove(nifti)
        print(f"[{n + 1}/{len(subjects)}] {sub}: {status[sub]}", flush=True)
        if not ok:
            print(f"Preprocessing failed for {sub}; raw file retained for inspection. "
                  "Continuing other participants; the build gate will reject unresolved failures.", flush=True)
    counts = {v: sum(1 for s in status.values() if s == v)
              for v in ("ok", "no_file", "failed", "excluded_official_eeg_only")}
    print("fMRI stage finished:", counts, "- failures are listed in", os.path.join(out_root, "lemon_error.log"))


# ---------------------------------------------------------------- stage 2: split, EEG, LMDB
def write_lmdb(h5_dir, lmdb_dir, dataset):
    """Same record layout as dataset_maker/transform2lmdb.py (keys sorted for determinism)."""
    import h5py
    import lmdb
    os.makedirs(lmdb_dir, exist_ok=True)
    path = os.path.join(lmdb_dir, "BrainSignal.lmdb")
    if os.path.exists(path):
        raise SystemExit(f"{path} already exists - delete it first.")
    env = lmdb.open(path, map_size=1 << 40)
    names = sorted(n for n in os.listdir(h5_dir) if n.endswith(".h5"))
    key_list = []
    for start in range(0, len(names), 500):
        with env.begin(write=True) as txn:
            for name in names[start:start + 500]:
                with h5py.File(os.path.join(h5_dir, name), "r") as f:
                    sample = {"signal": f["signal"][()], "y": f["y"][()]}
                txn.put(name.encode("ascii"), pickle.dumps(sample))
                key_list.append({"dataset": dataset, "key": name})
    with env.begin(write=True) as txn:
        txn.put("__keys__".encode("ascii"), pickle.dumps(key_list))
    env.close()
    return len(key_list)


def stage_build(args):
    from sklearn.model_selection import train_test_split
    official = load_official()
    labels = read_labels(os.path.join(args.raw, "Participants_LEMON.csv"))
    eeg_dir = os.path.join(args.raw, "EEG")
    h5_eeg = os.path.join(args.work, "h5", "LEMON_EEG")
    h5_fmri = os.path.join(args.work, "h5", "LEMON_fMRI")
    pool = os.path.join(h5_fmri, "_pool")
    split_path = os.path.join(args.work, "split.json")

    eeg_subs = sorted({h5_subject(p) for p in glob.glob(os.path.join(eeg_dir, "*_EC.set"))} & set(labels))
    fmri_subs = sorted({h5_subject(p) for p in glob.glob(os.path.join(pool, "*.h5"))} & set(labels))
    if not eeg_subs or not fmri_subs:
        raise SystemExit("No usable EEG .set files or fMRI samples: check --raw, run the fmri stage first, and make "
                         "sure the IDs in Participants_LEMON.csv use the same scheme as the file names.")
    common = sorted(set(eeg_subs) & set(fmri_subs))
    eeg_only = sorted(set(eeg_subs) - set(fmri_subs))
    fmri_only = sorted(set(fmri_subs) - set(eeg_subs))

    # subject-level 70/15/15 split of the participants that have both modalities, stratified by gender
    # (same procedure as make_LEMON.py, plus a fixed random_state)
    genders = [labels[s][1] for s in common]
    train_subs, tmp_subs, _, tmp_genders = train_test_split(common, genders, test_size=0.3, stratify=genders,
                                                            random_state=args.seed)
    val_subs, test_subs = train_test_split(tmp_subs, test_size=0.5, stratify=tmp_genders, random_state=args.seed)
    split = {"seed": args.seed, "train": sorted(train_subs), "val": sorted(val_subs), "test": sorted(test_subs),
             "eeg_only_train": eeg_only, "fmri_only_train": fmri_only}
    if os.path.isfile(split_path) and json.load(open(split_path)) != split:
        raise SystemExit(f"{split_path} was made with another seed or other data: use a new --work directory.")
    json.dump(split, open(split_path, "w"), indent=1)
    of = {s: "train" for s in train_subs + eeg_only + fmri_only}
    of.update({s: "val" for s in val_subs})
    of.update({s: "test" for s in test_subs})
    print({k: len(v) for k, v in split.items() if k != "seed"})

    # EEG: official preprocessing (61-channel order, 10 s segments, 200 Hz, average reference, micro-volt)
    for n, sub in enumerate(eeg_subs):
        for cond in ("EC", "EO"):
            fname = os.path.join(eeg_dir, f"{sub}_{cond}.set")
            if not os.path.isfile(fname) and cond == "EO":
                fname = os.path.join(eeg_dir, f"{sub}_E0.set")  # one participant is stored with a zero on the server
            stem = os.path.basename(fname).split(".")[0]
            if not os.path.isfile(fname) or glob.glob(os.path.join(h5_eeg, of[sub], f"{stem}_sample*.h5")):
                continue
            official.preprocessing((fname, labels[sub], of[sub]), h5_eeg, n_jobs=args.n_jobs)
        print(f"EEG [{n + 1}/{len(eeg_subs)}] {sub} -> {of[sub]}", flush=True)

    # fMRI: copy the pooled samples into their split folder
    for name in sorted(os.listdir(pool)):
        sub = h5_subject(name)
        if name.endswith(".h5") and sub in of:
            os.makedirs(os.path.join(h5_fmri, of[sub]), exist_ok=True)
            shutil.copy2(os.path.join(pool, name), os.path.join(h5_fmri, of[sub], name))

    for dataset, h5_root in (("LEMON_EEG", h5_eeg), ("LEMON_fMRI", h5_fmri)):
        for s in SPLITS:
            n = write_lmdb(os.path.join(h5_root, s), os.path.join(args.lmdb, dataset, s), dataset)
            print(f"{dataset}/{s}: {n} samples")


# ---------------------------------------------------------------- stage 3: pairing
def stage_pair(args):
    """LEMON = (fMRI, EEG) pairs of the same participant; the dominant modality (EEG) is the last element.

    Every EEG segment is paired with one fMRI sample of the same participant, drawn with a seeded generator.
    """
    import lmdb
    for s in SPLITS:
        src = {}
        for dataset in ("LEMON_fMRI", "LEMON_EEG"):
            env = lmdb.open(os.path.join(args.lmdb, dataset, s, "BrainSignal.lmdb"), readonly=True, lock=False,
                            readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                keys = sorted(k["key"] for k in pickle.loads(txn.get("__keys__".encode("ascii"))))
            src[dataset] = (env, keys)
        fmri_env, fmri_keys = src["LEMON_fMRI"]
        eeg_env, eeg_keys = src["LEMON_EEG"]
        by_sub = {}
        for k in fmri_keys:
            by_sub.setdefault(h5_subject(k), []).append(k)

        out_dir = os.path.join(args.lmdb, "LEMON", s)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "BrainSignal.lmdb")
        if os.path.exists(out_path):
            raise SystemExit(f"{out_path} already exists - delete it first.")
        out_env = lmdb.open(out_path, map_size=1 << 40)
        rng = np.random.default_rng(args.seed + SPLITS.index(s))
        key_list = []
        for start in range(0, len(eeg_keys), 500):
            with out_env.begin(write=True) as out_txn, fmri_env.begin(write=False) as ft, \
                    eeg_env.begin(write=False) as et:
                for eeg_key in eeg_keys[start:start + 500]:
                    candidates = by_sub.get(h5_subject(eeg_key))
                    if not candidates:
                        continue  # EEG-only participant
                    fmri_key = candidates[int(rng.integers(len(candidates)))]
                    fmri_sample = pickle.loads(ft.get(fmri_key.encode("ascii")))
                    eeg_sample = pickle.loads(et.get(eeg_key.encode("ascii")))
                    name = fmri_key.split(".")[0] + " & " + eeg_key.split(".")[0]
                    paired = {"signal": (fmri_sample["signal"], eeg_sample["signal"]), "y": eeg_sample["y"]}
                    out_txn.put(name.encode("ascii"), pickle.dumps(paired))
                    key_list.append({"dataset": "LEMON", "key": name})
        with out_env.begin(write=True) as out_txn:
            out_txn.put("__keys__".encode("ascii"), pickle.dumps(key_list))
        out_env.close()
        print(f"LEMON/{s}: {len(key_list)} pairs")


# ---------------------------------------------------------------- stage 4: report
def stage_check(args):
    import lmdb
    subjects = {}
    for dataset in ("LEMON_fMRI", "LEMON_EEG", "LEMON"):
        for s in SPLITS:
            env = lmdb.open(os.path.join(args.lmdb, dataset, s, "BrainSignal.lmdb"), readonly=True, lock=False)
            with env.begin(write=False) as txn:
                keys = pickle.loads(txn.get("__keys__".encode("ascii")))
                first = pickle.loads(txn.get(keys[0]["key"].encode("ascii")))
            subs = {h5_subject(k["key"]) for k in keys}
            subjects[(dataset, s)] = subs
            signal = first["signal"]
            shape = [np.asarray(x).shape for x in signal] if isinstance(signal, tuple) else np.asarray(signal).shape
            print(f"{dataset:11s} {s:5s} samples {len(keys):6d} participants {len(subs):4d} "
                  f"first signal {shape} first y {np.asarray(first['y']).tolist()}")
    for dataset in ("LEMON_fMRI", "LEMON_EEG", "LEMON"):
        tr, va, te = (subjects[(dataset, s)] for s in SPLITS)
        assert not (tr & va) and not (tr & te) and not (va & te), f"{dataset}: participant leakage between splits"
    print("no participant appears in two splits: OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["fmri", "build", "pair", "check"])
    parser.add_argument("--raw", help="folder with Participants_LEMON.csv and EEG/*.set, *.fdt")
    parser.add_argument("--work", help="working folder for HDF5 files and bookkeeping")
    parser.add_argument("--lmdb", help="LMDB root (the --root of the training scripts)")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--n_jobs", type=int, default=8, help="threads for EEG resampling")
    parser.add_argument("--subjects", default="", help="fmri stage: comma separated subset (for a quick test)")
    parser.add_argument("--fmri_url", default=FMRI_URL, help="URL template with {sub}")
    parser.add_argument("--fmri_local", default="", help="use <dir>/<sub>/func/<file> instead of downloading")
    parser.add_argument("--keep_nifti", action="store_true", help="keep the 2.3 GB NIfTI files after parcellation")
    args = parser.parse_args()
    needed = {"fmri": ("raw", "work"), "build": ("raw", "work", "lmdb"), "pair": ("lmdb",), "check": ("lmdb",)}
    for name in needed[args.stage]:
        if not getattr(args, name):
            parser.error(f"--{name} is required for stage {args.stage}")
    {"fmri": stage_fmri, "build": stage_build, "pair": stage_pair, "check": stage_check}[args.stage](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        if not is_retryable_network_error(exc):
            raise
        import traceback
        traceback.print_exc()
        sys.exit(75)
