import os
import mne
import h5py
import numpy as np
import nibabel as nib
from typing import Union
from pathlib import Path


ch_names = ['F1', 'F2', 'FC1', 'FC2', 'C1', 'C2', 'CP1', 'CP2', 'P1', 'P2',
            'Fp1', 'Fp2', 'AF3', 'AF4', 'F3', 'F4', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4', 'P3', 'P4', 'PO3', 'PO4', 'O1', 'O2',
            'F5', 'F6', 'FC5',  'FC6', 'C5',  'C6', 'CP5',  'CP6', 'P5',  'P6', 'PO7', 'PO8',
            'AF7', 'AF8', 'F7', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'P7', 'P8', 'PO9', 'PO10',
            'AFz', 'Fz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']

# Public source release: the atlas and participant lists are not distributed.
atlas_path = os.environ.get("BRAINOF_ATLAS")
if not atlas_path or not Path(atlas_path).is_file():
    raise FileNotFoundError("Set BRAINOF_ATLAS to your local BN_Atlas_246_2mm.nii.gz")
BN_atlas = nib.load(atlas_path).get_fdata()


def preprocessing(subject: tuple, save_path: Union[str, Path], n_jobs: int = 8, preload: bool = True):
    fname, label, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the raw eeg data
        eeg_raw = mne.io.read_raw_eeglab(fname, preload=preload)
        # pick common channels between whole channel and current subject, and reorder those channel with specific order
        eeg_raw.reorder_channels(list(filter(lambda x: x not in (set(ch_names) - set(eeg_raw.ch_names)), ch_names)))

        # find the boundary and the start of brain signal
        boundary = [index for index, value in enumerate(eeg_raw.annotations.description) if value in ['boundary', '32766']]
        boundary.append(eeg_raw.annotations.description.size - 1)
        sample_event = []
        for i in range(len(boundary) - 1):
            if eeg_raw.annotations.onset[boundary[i]+1] + 10 <= eeg_raw.annotations.onset[boundary[i+1]]:
                sample_event.append(mne.make_fixed_length_events(eeg_raw, start=eeg_raw.annotations.onset[boundary[i]+1], stop=eeg_raw.annotations.onset[boundary[i+1]], id=10, duration=10, overlap=0))
        # set the 10-second length event
        eeg_raw.sample_event = np.concatenate(sample_event)
        eeg_raw.sample_event_id = {'10 seconds': 10}
        # segment the whole signals into 10 seconds sample
        eeg_epochs = mne.Epochs(eeg_raw, eeg_raw.sample_event, eeg_raw.sample_event_id, baseline=None, tmin=0, tmax=10,
                                preload=True, reject=None, proj=False)

        # resample brain signal to 200 HZ
        eeg_epochs.resample(200, n_jobs=n_jobs)
        # set average reference
        eeg_epochs.set_eeg_reference(ref_channels='average', ch_type='eeg')

        signals = eeg_epochs.get_data(units='uV')
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            age, gender = label
            if age in ['20-25', '25-30', '30-35', '35-40']:
                target_age = np.array([1, 0])
            elif age in ['55-60', '60-65', '65-70', '70-75', '75-80']:
                target_age = np.array([0, 1])

            # Gender: 1=female 2=male
            if gender == 1:
                target_gender = np.array([1, 0])
            elif gender == 2:
                target_gender = np.array([0, 1])

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=(target_age, target_gender))
    except:
        with open(os.path.join(save_path, "lemon_error.log"), "a") as f:
            f.write(fname + "\n")


def preparing(subject: tuple, save_path: Union[str, Path], ts_length: int = 320):
    fname, label, split = subject
    if not os.path.exists(os.path.join(save_path, split)):
        os.makedirs(os.path.join(save_path, split))
    try:
        # read the preprocessed fMRI data
        fMRI_voxel = nib.load(fname).get_fdata()
        # parcel voxel signal to brain region
        fMRI_roi = np.zeros((int(BN_atlas.max()), fMRI_voxel.shape[-1]))
        roi = np.unique(BN_atlas)
        # exclude roi 0
        roi = roi[roi != 0]
        for i, value in enumerate(roi):
            fMRI_roi[i] = fMRI_voxel[BN_atlas == value].mean(axis=0)

        signals = np.stack((fMRI_roi[:, :ts_length], fMRI_roi[:, ts_length:2*ts_length]))
        for idx, signal in enumerate(signals):
            save_fname = fname.split(os.sep)[-1].split('.')[0] + f'_sample{idx:04d}.h5'
            age, gender = label
            if age in ['20-25', '25-30', '30-35', '35-40']:
                target_age = np.array([1, 0])
            elif age in ['55-60', '60-65', '65-70', '70-75', '75-80']:
                target_age = np.array([0, 1])

            # Gender: 1=female 2=male
            if gender == 1:
                target_gender = np.array([1, 0])
            elif gender == 2:
                target_gender = np.array([0, 1])

            with h5py.File(os.path.join(save_path, split, save_fname), "w") as hf:
                hf.create_dataset("signal", data=signal)
                hf.create_dataset("y", data=(target_age, target_gender))
    except:
        with open(os.path.join(save_path, "lemon_error.log"), "a") as f:
            f.write(fname + "\n")
