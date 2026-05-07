"""
EEG Preprocessing Pipeline for BCI Motor Imagery.

Includes:
- Bandpass filtering (4-38 Hz)
- Notch filtering (50/60 Hz line noise)
- Common Average Reference (CAR)
- ICA-based artifact removal (EOG, EMG)
- Exponential Moving Standardization
- Euclidean Alignment for transfer learning
- Trial extraction and windowing
"""
import numpy as np
from scipy import signal
from scipy.linalg import sqrtm, inv
from sklearn.decomposition import FastICA
from typing import Optional, Tuple, List


def bandpass_filter(
    data: np.ndarray, low: float, high: float, srate: float, order: int = 5
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter.

    Args:
        data: EEG data (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        low: Lower cutoff frequency (Hz)
        high: Upper cutoff frequency (Hz)
        srate: Sampling rate (Hz)
        order: Filter order
    Returns:
        Filtered data with same shape as input
    """
    nyq = srate / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    if data.ndim == 2:
        return signal.filtfilt(b, a, data, axis=-1).astype(np.float32)
    elif data.ndim == 3:
        return np.stack(
            [signal.filtfilt(b, a, data[i], axis=-1) for i in range(data.shape[0])]
        ).astype(np.float32)
    raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def notch_filter(
    data: np.ndarray, freq: float, srate: float, quality: float = 30.0
) -> np.ndarray:
    """Remove line noise using notch filter."""
    b, a = signal.iirnotch(freq, quality, srate)
    if data.ndim == 2:
        return signal.filtfilt(b, a, data, axis=-1).astype(np.float32)
    elif data.ndim == 3:
        return np.stack(
            [signal.filtfilt(b, a, data[i], axis=-1) for i in range(data.shape[0])]
        ).astype(np.float32)
    raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR).
    Subtracts mean across channels at each time point.

    Args:
        data: (n_channels, n_samples) or (n_trials, n_channels, n_samples)
    """
    if data.ndim == 2:
        return (data - data.mean(axis=0, keepdims=True)).astype(np.float32)
    elif data.ndim == 3:
        return (data - data.mean(axis=1, keepdims=True)).astype(np.float32)
    raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def ica_artifact_removal(
    data: np.ndarray,
    n_components: Optional[int] = None,
    eog_channels: Optional[List[int]] = None,
    correlation_threshold: float = 0.3,
) -> np.ndarray:
    """
    ICA-based artifact removal (EOG, muscle artifacts).

    Identifies artifactual components by correlation with EOG channels
    or by statistical properties (kurtosis, variance).

    Args:
        data: Continuous EEG (n_channels, n_samples)
        n_components: Number of ICA components (default: n_channels)
        eog_channels: Indices of EOG channels for correlation-based rejection
        correlation_threshold: Threshold for rejecting components correlated with EOG
    Returns:
        Cleaned EEG data
    """
    n_ch, n_samp = data.shape
    if n_components is None:
        n_components = n_ch

    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    sources = ica.fit_transform(data.T)  # (n_samples, n_components)
    mixing_matrix = ica.mixing_  # (n_channels, n_components)

    # Identify artifact components
    reject_idx = set()

    if eog_channels is not None and len(eog_channels) > 0:
        # Correlation-based rejection using EOG channels
        for eog_ch in eog_channels:
            eog_signal = data[eog_ch]
            for comp_idx in range(n_components):
                corr = np.abs(np.corrcoef(eog_signal, sources[:, comp_idx])[0, 1])
                if corr > correlation_threshold:
                    reject_idx.add(comp_idx)

    # Statistical rejection: high kurtosis (blinks) or extreme variance
    from scipy.stats import kurtosis as compute_kurtosis

    kurt = compute_kurtosis(sources, axis=0)
    kurt_threshold = np.mean(kurt) + 3 * np.std(kurt)
    for comp_idx in range(n_components):
        if kurt[comp_idx] > kurt_threshold:
            reject_idx.add(comp_idx)

    # Zero out artifact components
    sources_clean = sources.copy()
    for idx in reject_idx:
        sources_clean[:, idx] = 0.0

    # Reconstruct cleaned signal
    cleaned = (sources_clean @ mixing_matrix.T).T  # (n_channels, n_samples)
    return cleaned.astype(np.float32)


def exponential_moving_standardize(
    data: np.ndarray, init_samples: int = 250, eps: float = 1e-4
) -> np.ndarray:
    """
    Exponential Moving Standardization (EMS).
    Adaptive standardization that handles non-stationarity.

    Args:
        data: (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        init_samples: Number of initial samples to compute statistics
        eps: Small constant for numerical stability
    """
    def _ems_single(x: np.ndarray) -> np.ndarray:
        """Apply EMS to a single trial (n_channels, n_samples)."""
        n_ch, n_samp = x.shape
        factor = 1.0 / init_samples
        out = np.zeros_like(x)

        # Initialize with first `init_samples`
        init_block = x[:, :min(init_samples, n_samp)]
        running_mean = init_block.mean(axis=1)
        running_var = init_block.var(axis=1) + eps

        for t in range(n_samp):
            running_mean = (1 - factor) * running_mean + factor * x[:, t]
            running_var = (1 - factor) * running_var + factor * (
                x[:, t] - running_mean
            ) ** 2
            out[:, t] = (x[:, t] - running_mean) / np.sqrt(running_var + eps)
        return out

    if data.ndim == 2:
        return _ems_single(data).astype(np.float32)
    elif data.ndim == 3:
        return np.stack([_ems_single(data[i]) for i in range(data.shape[0])]).astype(
            np.float32
        )
    raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def euclidean_alignment(trials: np.ndarray) -> np.ndarray:
    """
    Euclidean Alignment (EA) for transfer learning.
    Aligns the covariance matrices of EEG data from different subjects/sessions
    to a common reference, reducing distribution shift.

    Reference: He & Wu, "Transfer Learning for Brain-Computer Interfaces", 2020

    Args:
        trials: (n_trials, n_channels, n_samples)
    Returns:
        Aligned trials with same shape
    """
    n_trials, n_ch, n_samp = trials.shape

    # Compute mean covariance matrix (reference matrix)
    cov_matrices = []
    for i in range(n_trials):
        x = trials[i]
        cov = (x @ x.T) / n_samp
        cov_matrices.append(cov)
    mean_cov = np.mean(cov_matrices, axis=0)

    # Compute R^{-1/2} (whitening transform)
    R_inv_sqrt = inv(sqrtm(mean_cov)).real.astype(np.float32)

    # Apply alignment: x_aligned = R^{-1/2} @ x
    aligned = np.stack([R_inv_sqrt @ trials[i] for i in range(n_trials)])
    return aligned.astype(np.float32)


def resample(data: np.ndarray, orig_srate: float, target_srate: float) -> np.ndarray:
    """Resample EEG data to target sampling rate."""
    if abs(orig_srate - target_srate) < 1e-3:
        return data

    n_samples_orig = data.shape[-1]
    n_samples_new = int(n_samples_orig * target_srate / orig_srate)

    if data.ndim == 2:
        return signal.resample(data, n_samples_new, axis=-1).astype(np.float32)
    elif data.ndim == 3:
        return signal.resample(data, n_samples_new, axis=-1).astype(np.float32)
    raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def segment_trials(
    continuous: np.ndarray,
    events: np.ndarray,
    srate: float,
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fixed-length trials from continuous EEG.

    Args:
        continuous: (n_channels, n_samples) continuous recording
        events: (n_events, 2) array with columns [sample_idx, class_label]
        srate: Sampling rate
        tmin: Start time relative to event (seconds)
        tmax: End time relative to event (seconds)
    Returns:
        trials: (n_trials, n_channels, n_samples)
        labels: (n_trials,)
    """
    start_offset = int(tmin * srate)
    end_offset = int(tmax * srate)
    trial_len = end_offset - start_offset
    n_channels = continuous.shape[0]

    trials = []
    labels = []
    for event_sample, label in events:
        start = int(event_sample) + start_offset
        end = start + trial_len
        if start >= 0 and end <= continuous.shape[1]:
            trials.append(continuous[:, start:end])
            labels.append(label)

    return np.array(trials, dtype=np.float32), np.array(labels, dtype=np.int64)


class EEGPreprocessor:
    """
    Complete EEG preprocessing pipeline.

    Usage:
        preprocessor = EEGPreprocessor(config)
        trials, labels = preprocessor.process(raw_data, events, orig_srate)
    """

    def __init__(self, config):
        self.config = config

    def process_continuous(
        self,
        data: np.ndarray,
        orig_srate: float,
        eog_channels: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Process continuous EEG recording.

        Args:
            data: (n_channels, n_samples)
            orig_srate: Original sampling rate
            eog_channels: EOG channel indices for ICA artifact removal
        Returns:
            processed: Cleaned continuous data
            new_srate: Sampling rate after resampling
        """
        cfg = self.config

        # 1. Resample
        data = resample(data, orig_srate, cfg.target_srate)
        current_srate = cfg.target_srate

        # 2. Notch filter (line noise)
        data = notch_filter(data, cfg.notch_freq, current_srate, cfg.notch_quality)

        # 3. Bandpass filter
        data = bandpass_filter(
            data, cfg.low_freq, cfg.high_freq, current_srate, cfg.filter_order
        )

        # 4. Common Average Reference
        if cfg.use_car:
            data = common_average_reference(data)

        # 5. ICA artifact removal
        if cfg.use_ica:
            data = ica_artifact_removal(
                data, n_components=cfg.n_ica_components, eog_channels=eog_channels
            )

        return data, current_srate

    def process_trials(
        self,
        data: np.ndarray,
        events: np.ndarray,
        orig_srate: float,
        eog_channels: Optional[List[int]] = None,
        apply_ea: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full pipeline: continuous processing → trial extraction → trial-level processing.

        Args:
            data: (n_channels, n_samples) continuous
            events: (n_events, 2) [sample_idx, label]
            orig_srate: Original sampling rate
            eog_channels: EOG channel indices
            apply_ea: Whether to apply Euclidean Alignment
        Returns:
            trials: (n_trials, n_channels, n_samples) preprocessed
            labels: (n_trials,) class labels
        """
        cfg = self.config

        # Process continuous data
        data, srate = self.process_continuous(data, orig_srate, eog_channels)

        # Adjust event sample indices for new sampling rate
        scale_factor = cfg.target_srate / orig_srate
        events_resampled = events.copy()
        events_resampled[:, 0] = (events[:, 0] * scale_factor).astype(int)

        # Extract trials
        trials, labels = segment_trials(
            data, events_resampled, srate, cfg.tmin, cfg.tmax
        )

        # Exponential moving standardization per trial
        if cfg.use_ems:
            trials = exponential_moving_standardize(trials, cfg.ems_init_samples)

        # Euclidean Alignment
        if apply_ea:
            trials = euclidean_alignment(trials)

        return trials, labels


def preprocess_subject(
    raw_data: np.ndarray,
    events: np.ndarray,
    orig_srate: float,
    config,
    eog_channels: Optional[List[int]] = None,
    apply_ea: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to preprocess a single subject's data.
    """
    preprocessor = EEGPreprocessor(config)
    return preprocessor.process_trials(
        raw_data, events, orig_srate, eog_channels, apply_ea
    )
