"""
Dataset loaders for BCI Motor Imagery datasets.

Supports:
- BCI Competition IV Dataset 2a (4-class, 9 subjects, 22 EEG + 3 EOG channels)
- BCI Competition IV Dataset 2b (2-class, 9 subjects, 3 EEG + 3 EOG channels)
- PhysioNet MI (4-class, 109 subjects, 64 EEG channels)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from preprocessing import EEGPreprocessor, euclidean_alignment


# ──────────────────────────────────────────────────────────────────────
# BCI Competition IV Dataset 2a Loader
# ──────────────────────────────────────────────────────────────────────

def load_bci_iv_2a_gdf(filepath: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load BCI Competition IV 2a data from GDF files using MNE.

    Args:
        filepath: Path to .gdf file
    Returns:
        data: (n_channels, n_samples) EEG data
        events: (n_events, 2) [sample_idx, class_label]
        srate: Sampling rate
    """
    import mne
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    srate = raw.info["sfreq"]

    # Get events - BCI IV 2a uses event codes 769-772 for classes 1-4
    events_mne, event_id = mne.events_from_annotations(raw, verbose=False)

    # Map event codes to class labels (0-indexed)
    # 769=left hand(0), 770=right hand(1), 771=both feet(2), 772=tongue(3)
    class_mapping = {}
    for name, code in event_id.items():
        if "769" in name or "left" in name.lower():
            class_mapping[code] = 0
        elif "770" in name or "right" in name.lower():
            class_mapping[code] = 1
        elif "771" in name or "feet" in name.lower() or "foot" in name.lower():
            class_mapping[code] = 2
        elif "772" in name or "tongue" in name.lower():
            class_mapping[code] = 3

    # Extract EEG channels (first 22) and build event array
    eeg_ch_names = raw.ch_names[:22]
    raw_eeg = raw.copy().pick_channels(eeg_ch_names)
    data = raw_eeg.get_data().astype(np.float32)

    events_list = []
    for ev in events_mne:
        if ev[2] in class_mapping:
            events_list.append([ev[0], class_mapping[ev[2]]])
    events = np.array(events_list) if events_list else np.empty((0, 2), dtype=np.int64)

    return data, events, srate


def load_bci_iv_2a_mat(filepath: str, labels_filepath: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load BCI Competition IV 2a data from .mat files.

    Args:
        filepath: Path to data .mat file
        labels_filepath: Path to labels .mat file (for evaluation data)
    Returns:
        data: (n_channels, n_samples)
        events: (n_events, 2) [sample_idx, class_label]
        srate: 250 Hz
    """
    from scipy.io import loadmat

    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    srate = 250.0

    # Handle different .mat structures
    if "data" in mat:
        raw_struct = mat["data"]
        if hasattr(raw_struct, "__iter__") and not isinstance(raw_struct, str):
            # Multiple runs within file
            all_data = []
            all_events = []
            cumulative_samples = 0
            for run in raw_struct:
                x = run.X.T  # (n_channels, n_samples)
                all_data.append(x)
                if hasattr(run, "y") and run.y is not None:
                    trial_starts = run.trial
                    labels = run.y
                    for start, label in zip(trial_starts, labels):
                        all_events.append([start + cumulative_samples, int(label) - 1])
                cumulative_samples += x.shape[1]
            data = np.hstack(all_data).astype(np.float32)
            events = np.array(all_events, dtype=np.int64) if all_events else np.empty((0, 2), dtype=np.int64)
        else:
            data = raw_struct.X.T.astype(np.float32)
            events_list = []
            if hasattr(raw_struct, "y") and raw_struct.y is not None:
                for start, label in zip(raw_struct.trial, raw_struct.y):
                    events_list.append([int(start), int(label) - 1])
            events = np.array(events_list, dtype=np.int64) if events_list else np.empty((0, 2), dtype=np.int64)
    else:
        raise ValueError(f"Unrecognized .mat format in {filepath}")

    # Load separate labels file if provided
    if labels_filepath and os.path.exists(labels_filepath):
        labels_mat = loadmat(labels_filepath, squeeze_me=True)
        if "classlabel" in labels_mat:
            external_labels = labels_mat["classlabel"].astype(np.int64) - 1
            if len(external_labels) == len(events):
                events[:, 1] = external_labels

    return data[:22], events, srate  # First 22 channels are EEG


def load_bci_iv_2a_subject(
    data_dir: str, subject_id: int, session: str = "T"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load a single subject from BCI IV 2a.

    Args:
        data_dir: Directory containing dataset files
        subject_id: Subject number (1-9)
        session: 'T' for training, 'E' for evaluation
    """
    # Try GDF format first
    gdf_path = os.path.join(data_dir, f"A{subject_id:02d}{session}.gdf")
    if os.path.exists(gdf_path):
        data, events, srate = load_bci_iv_2a_gdf(gdf_path)
        # For evaluation session, load true labels
        if session == "E":
            labels_path = os.path.join(data_dir, f"A{subject_id:02d}E.mat")
            if os.path.exists(labels_path):
                from scipy.io import loadmat
                mat_labels = loadmat(labels_path, squeeze_me=True)
                if "classlabel" in mat_labels:
                    true_labels = mat_labels["classlabel"].astype(np.int64) - 1
                    if len(true_labels) == len(events):
                        events[:, 1] = true_labels
        return data, events, srate

    # Try .mat format
    mat_path = os.path.join(data_dir, f"A{subject_id:02d}{session}.mat")
    labels_path = os.path.join(data_dir, f"A{subject_id:02d}{session}_labels.mat")
    if os.path.exists(mat_path):
        return load_bci_iv_2a_mat(mat_path, labels_path if os.path.exists(labels_path) else None)

    raise FileNotFoundError(
        f"No data file found for subject {subject_id}, session {session} in {data_dir}. "
        f"Expected {gdf_path} or {mat_path}"
    )


# ──────────────────────────────────────────────────────────────────────
# BCI Competition IV Dataset 2b Loader
# ──────────────────────────────────────────────────────────────────────

def load_bci_iv_2b_subject(
    data_dir: str, subject_id: int, session: int = 1
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load a single subject/session from BCI IV 2b (2-class: left/right hand).
    Uses 3 bipolar EEG channels (C3, Cz, C4).

    Args:
        data_dir: Directory containing dataset files
        subject_id: Subject number (1-9)
        session: Session number (1-5; sessions 1-3 without feedback, 4-5 with)
    """
    import mne

    gdf_path = os.path.join(data_dir, f"B{subject_id:02d}{session:02d}T.gdf")
    if not os.path.exists(gdf_path):
        gdf_path = os.path.join(data_dir, f"B{subject_id:02d}{session:02d}E.gdf")

    if os.path.exists(gdf_path):
        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
        srate = raw.info["sfreq"]
        events_mne, event_id = mne.events_from_annotations(raw, verbose=False)

        class_mapping = {}
        for name, code in event_id.items():
            if "769" in name:
                class_mapping[code] = 0  # left hand
            elif "770" in name:
                class_mapping[code] = 1  # right hand

        # Use first 3 EEG channels (C3, Cz, C4)
        eeg_names = raw.ch_names[:3]
        data = raw.copy().pick_channels(eeg_names).get_data().astype(np.float32)

        events_list = []
        for ev in events_mne:
            if ev[2] in class_mapping:
                events_list.append([ev[0], class_mapping[ev[2]]])
        events = np.array(events_list, dtype=np.int64) if events_list else np.empty((0, 2), dtype=np.int64)

        return data, events, srate

    raise FileNotFoundError(f"No data file found for subject {subject_id}, session {session} in {data_dir}")


# ──────────────────────────────────────────────────────────────────────
# PhysioNet MI Dataset Loader
# ──────────────────────────────────────────────────────────────────────

def load_physionet_subject(
    data_dir: str, subject_id: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load PhysioNet Motor Movement/Imagery dataset for one subject.

    Motor imagery runs: 4, 8, 12 (open/close left or right fist)
                        6, 10, 14 (open/close both fists or both feet)

    Classes: 0=left fist, 1=right fist, 2=both fists, 3=both feet

    Args:
        data_dir: Root directory (or None to auto-download)
        subject_id: Subject number (1-109)
    """
    import mne

    # PhysioNet MI run numbers for imagery tasks
    imagery_runs_lr = [4, 8, 12]  # Left/Right fist imagery
    imagery_runs_bf = [6, 10, 14]  # Both fists / Both feet imagery

    all_data = []
    all_events = []
    cumulative_samples = 0
    srate = 160.0

    for run_type, runs in [("lr", imagery_runs_lr), ("bf", imagery_runs_bf)]:
        for run in runs:
            # Try local files first
            edf_name = f"S{subject_id:03d}R{run:02d}.edf"
            edf_path = os.path.join(data_dir, f"S{subject_id:03d}", edf_name)

            if os.path.exists(edf_path):
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            else:
                # Auto-download from PhysioNet using MNE
                try:
                    from mne.datasets import eegbci
                    raw_fnames = eegbci.load_data(subject_id, [run], path=data_dir)
                    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
                except Exception:
                    continue

            srate = raw.info["sfreq"]
            events_mne, event_id = mne.events_from_annotations(raw, verbose=False)
            data = raw.get_data().astype(np.float32)
            all_data.append(data)

            # Map annotations: T0=rest, T1=left/both_fists, T2=right/both_feet
            for ev in events_mne:
                sample_idx = ev[0] + cumulative_samples
                ann_code = ev[2]
                ann_name = ""
                for name, code in event_id.items():
                    if code == ann_code:
                        ann_name = name
                        break

                if "T1" in ann_name:
                    label = 0 if run_type == "lr" else 2  # left fist / both fists
                elif "T2" in ann_name:
                    label = 1 if run_type == "lr" else 3  # right fist / both feet
                else:
                    continue  # skip rest
                all_events.append([sample_idx, label])

            cumulative_samples += data.shape[1]

    if not all_data:
        raise FileNotFoundError(f"No PhysioNet data found for subject {subject_id} in {data_dir}")

    data = np.hstack(all_data).astype(np.float32)
    events = np.array(all_events, dtype=np.int64)

    return data, events, srate


# ──────────────────────────────────────────────────────────────────────
# PyTorch Dataset Classes
# ──────────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    """
    PyTorch Dataset for preprocessed EEG trials.
    """

    def __init__(self, trials: np.ndarray, labels: np.ndarray, subject_id: int = -1):
        """
        Args:
            trials: (n_trials, n_channels, n_samples)
            labels: (n_trials,)
            subject_id: Subject identifier (for domain adaptation)
        """
        self.trials = torch.FloatTensor(trials)
        self.labels = torch.LongTensor(labels)
        self.subject_id = subject_id

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return (1, n_channels, n_samples) for Conv2d compatibility
        return self.trials[idx].unsqueeze(0), self.labels[idx]


class MultiSubjectDataset(Dataset):
    """
    Combines data from multiple subjects with domain labels.
    Used for multi-source transfer learning and domain adaptation.
    """

    def __init__(self, subject_datasets: List[EEGDataset]):
        self.datasets = subject_datasets
        self._cumulative_sizes = []
        cumsum = 0
        for ds in subject_datasets:
            cumsum += len(ds)
            self._cumulative_sizes.append(cumsum)
        self.total_len = cumsum

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        ds_idx = 0
        local_idx = idx
        for i, cs in enumerate(self._cumulative_sizes):
            if idx < cs:
                ds_idx = i
                local_idx = idx - (self._cumulative_sizes[i - 1] if i > 0 else 0)
                break
        trial, label = self.datasets[ds_idx][local_idx]
        domain = torch.tensor(ds_idx, dtype=torch.long)
        return trial, label, domain


# ──────────────────────────────────────────────────────────────────────
# Data Loading Utilities
# ──────────────────────────────────────────────────────────────────────

def load_and_preprocess_subject(
    dataset_name: str,
    data_dir: str,
    subject_id: int,
    preproc_config,
    session: str = "T",
    apply_ea: bool = False,
) -> EEGDataset:
    """
    Load, preprocess, and wrap a single subject's data as an EEGDataset.

    Args:
        dataset_name: 'bci_iv_2a', 'bci_iv_2b', or 'physionet'
        data_dir: Dataset root directory
        subject_id: Subject number
        preproc_config: PreprocessingConfig
        session: Session identifier
        apply_ea: Apply Euclidean Alignment
    """
    print(f"  Loading {dataset_name} subject {subject_id} (session={session})...")

    eog_channels = None
    if dataset_name == "bci_iv_2a":
        data, events, srate = load_bci_iv_2a_subject(data_dir, subject_id, session)
        eog_channels = [22, 23, 24] if data.shape[0] > 22 else None
    elif dataset_name == "bci_iv_2b":
        sess_num = 4 if session == "T" else 5
        data, events, srate = load_bci_iv_2b_subject(data_dir, subject_id, sess_num)
        eog_channels = [3, 4, 5] if data.shape[0] > 3 else None
    elif dataset_name == "physionet":
        data, events, srate = load_physionet_subject(data_dir, subject_id)
        eog_channels = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if len(events) == 0:
        raise ValueError(f"No events found for subject {subject_id}")

    preprocessor = EEGPreprocessor(preproc_config)
    trials, labels = preprocessor.process_trials(
        data, events, srate, eog_channels, apply_ea
    )
    print(f"    → {len(labels)} trials, shape={trials.shape}, classes={np.unique(labels)}")

    return EEGDataset(trials, labels, subject_id)


def create_source_target_loaders(
    dataset_name: str,
    data_dir: str,
    source_subjects: List[int],
    target_subject: int,
    preproc_config,
    train_config,
    n_target_trials_per_class: int = 10,
    apply_ea: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for source pretraining, target fine-tuning, and target evaluation.

    Args:
        dataset_name: Dataset identifier
        data_dir: Dataset root directory
        source_subjects: List of source subject IDs
        target_subject: Target subject ID
        preproc_config: PreprocessingConfig
        train_config: TrainConfig
        n_target_trials_per_class: Calibration trials per class for target
        apply_ea: Apply Euclidean Alignment per-subject
    Returns:
        source_loader: DataLoader for source subject data
        target_train_loader: DataLoader for target calibration data
        target_test_loader: DataLoader for target test data
    """
    print("Loading source subjects...")
    source_datasets = []
    for sid in source_subjects:
        try:
            ds = load_and_preprocess_subject(
                dataset_name, data_dir, sid, preproc_config, "T", apply_ea
            )
            source_datasets.append(ds)
        except (FileNotFoundError, ValueError) as e:
            print(f"  Warning: Skipping subject {sid}: {e}")

    if not source_datasets:
        raise RuntimeError("No source subjects loaded successfully.")

    source_combined = ConcatDataset(source_datasets)
    source_loader = DataLoader(
        source_combined,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Load target subject
    print(f"Loading target subject {target_subject}...")
    target_ds = load_and_preprocess_subject(
        dataset_name, data_dir, target_subject, preproc_config, "T", apply_ea
    )

    # Split target into calibration (few-shot) and evaluation
    n_classes = len(torch.unique(target_ds.labels))
    cal_indices = []
    test_indices = []
    for c in range(n_classes):
        class_indices = (target_ds.labels == c).nonzero(as_tuple=True)[0].tolist()
        np.random.shuffle(class_indices)
        n_cal = min(n_target_trials_per_class, len(class_indices) // 2)
        cal_indices.extend(class_indices[:n_cal])
        test_indices.extend(class_indices[n_cal:])

    target_train = Subset(target_ds, cal_indices)
    target_test = Subset(target_ds, test_indices)

    target_train_loader = DataLoader(
        target_train,
        batch_size=min(train_config.batch_size, len(cal_indices)),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    target_test_loader = DataLoader(
        target_test,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Source: {len(source_combined)} trials | "
          f"Target cal: {len(cal_indices)} | Target test: {len(test_indices)}")

    return source_loader, target_train_loader, target_test_loader
