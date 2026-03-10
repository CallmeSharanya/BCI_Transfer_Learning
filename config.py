"""
Configuration for BCI Transfer Learning project.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PreprocessingConfig:
    """EEG preprocessing parameters."""
    # Sampling rate after resampling
    target_srate: float = 250.0
    # Bandpass filter (Hz)
    low_freq: float = 4.0
    high_freq: float = 38.0
    filter_order: int = 5
    # Notch filter for line noise
    notch_freq: float = 50.0  # 50Hz (EU) or 60Hz (US)
    notch_quality: float = 30.0
    # EOG artifact removal via ICA
    use_ica: bool = True
    n_ica_components: Optional[int] = None  # None = n_channels
    # Common Average Reference
    use_car: bool = True
    # Trial window (seconds relative to cue onset)
    tmin: float = 0.5
    tmax: float = 4.0
    # Exponential moving standardization
    use_ems: bool = True
    ems_init_samples: int = 250


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Root directories for datasets
    bci_iv_2a_path: str = "./data/bci_iv_2a"
    bci_iv_2b_path: str = "./data/bci_iv_2b"
    physionet_path: str = "./data/physionet_mi"
    # Which datasets to use
    datasets: List[str] = field(default_factory=lambda: ["bci_iv_2a"])
    # BCI IV 2a: subjects 1-9, classes: left hand, right hand, both feet, tongue
    # BCI IV 2b: subjects 1-9, classes: left hand, right hand
    # PhysioNet MI: subjects 1-109, classes: left fist, right fist, both fists, both feet
    n_classes_2a: int = 4
    n_classes_2b: int = 2
    n_classes_physionet: int = 4
    # Validation split ratio
    val_ratio: float = 0.2


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Model type: 'eegnet', 'shallowconvnet', 'deepconvnet'
    model_type: str = "eegnet"
    # EEGNet hyperparameters
    F1: int = 8       # Number of temporal filters
    F2: int = 16      # Number of pointwise filters
    D: int = 2        # Depth multiplier for depthwise convolution
    kernel_length: int = 64  # Length of temporal convolution kernel
    dropout_rate: float = 0.5
    # Input shape (set dynamically)
    n_channels: int = 22   # BCI IV 2a default
    n_samples: int = 875   # 3.5s * 250Hz


@dataclass
class TransferConfig:
    """Transfer learning configuration."""
    # TL strategy: 'fine_tune', 'feature_extract', 'domain_adapt', 'progressive', 'multi_source'
    strategy: str = "fine_tune"
    # Source subjects (list of subject IDs)
    source_subjects: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    # Target subject
    target_subject: int = 9
    # Number of target calibration trials per class (few-shot)
    n_target_trials_per_class: int = 10
    # Fine-tuning: which layers to freeze
    freeze_feature_extractor: bool = True
    # Domain adaptation: MMD/CORAL weight
    domain_loss_weight: float = 0.5
    # Euclidean Alignment (EA)
    use_euclidean_alignment: bool = True


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 64
    n_epochs_pretrain: int = 200
    n_epochs_finetune: int = 100
    lr_pretrain: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 20  # Early stopping patience
    seed: int = 42
    device: str = "cuda"  # 'cuda' or 'cpu'
    num_workers: int = 4
    # Mixed precision training
    use_amp: bool = True
    # Gradient clipping
    max_grad_norm: float = 1.0
