"""
Deep Learning Models for EEG Motor Imagery Classification.

Implements:
- EEGNet: Compact CNN with depthwise/separable convolutions (Lawhern et al., 2018)
- ShallowConvNet: Shallow architecture (Schirrmeister et al., 2017)
- DeepConvNet: Deep architecture (Schirrmeister et al., 2017)

All models have a clear feature_extractor / classifier split for transfer learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────
# EEGNet
# ──────────────────────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs.
    Reference: Lawhern et al., J. Neural Eng., 2018.

    Architecture:
        1. Temporal convolution (band-pass filtering)
        2. Depthwise convolution (spatial filtering per temporal feature)
        3. Separable convolution (temporal pattern mixing)
        4. Classifier head

    Args:
        n_channels: Number of EEG channels
        n_samples: Number of time samples per trial
        n_classes: Number of output classes
        F1: Number of temporal filters
        F2: Number of pointwise filters
        D: Depth multiplier for depthwise convolution
        kernel_length: Length of temporal convolution kernel
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 875,
        n_classes: int = 4,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        # Block 1: Temporal + Spatial filtering
        self.block1 = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial convolution
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            # Depthwise temporal convolution
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),  # Pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Feature extractor = block1 + block2
        self.feature_extractor = nn.Sequential(self.block1, self.block2)

        # Compute flattened feature size
        self._feature_size = self._get_feature_size()

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, n_classes),
        )

    def _get_feature_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.feature_extractor(x)
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head (for TL)."""
        features = self.feature_extractor(x)
        return features.view(features.size(0), -1)

    @property
    def feature_dim(self) -> int:
        return self._feature_size


# ──────────────────────────────────────────────────────────────────────
# ShallowConvNet
# ──────────────────────────────────────────────────────────────────────

class ShallowConvNet(nn.Module):
    """
    Shallow ConvNet for EEG decoding.
    Reference: Schirrmeister et al., Human Brain Mapping, 2017.

    Mimics FBCSP: temporal convolution → spatial convolution → squaring → log → pooling.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 875,
        n_classes: int = 4,
        n_temporal_filters: int = 40,
        temporal_kernel_size: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.feature_extractor = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, n_temporal_filters, (1, temporal_kernel_size), bias=False),
            # Spatial convolution
            nn.Conv2d(n_temporal_filters, n_temporal_filters, (n_channels, 1), bias=False),
            nn.BatchNorm2d(n_temporal_filters),
        )

        # Non-linear activation: square + log (applied in forward)
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
        self.drop = nn.Dropout(dropout_rate)

        self._feature_size = self._get_feature_size()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, n_classes),
        )

    def _get_feature_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.feature_extractor(x)
            x = x ** 2
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-6))
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x ** 2  # Square nonlinearity
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # Log nonlinearity
        x = self.drop(x)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        return x.view(x.size(0), -1)

    @property
    def feature_dim(self) -> int:
        return self._feature_size


# ──────────────────────────────────────────────────────────────────────
# DeepConvNet
# ──────────────────────────────────────────────────────────────────────

class DeepConvNet(nn.Module):
    """
    Deep ConvNet for EEG decoding.
    Reference: Schirrmeister et al., Human Brain Mapping, 2017.

    Four convolutional blocks with increasing filter depth.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 875,
        n_classes: int = 4,
        n_filters_1: int = 25,
        temporal_kernel_size: int = 10,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Block 1: Temporal + Spatial
        self.block1 = nn.Sequential(
            nn.Conv2d(1, n_filters_1, (1, temporal_kernel_size), bias=False),
            nn.Conv2d(n_filters_1, n_filters_1, (n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters_1),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout_rate),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(n_filters_1, 50, (1, temporal_kernel_size), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout_rate),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, temporal_kernel_size), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout_rate),
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, temporal_kernel_size), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(dropout_rate),
        )

        self.feature_extractor = nn.Sequential(
            self.block1, self.block2, self.block3, self.block4
        )

        self._feature_size = self._get_feature_size()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, n_classes),
        )

    def _get_feature_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.feature_extractor(x)
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return features.view(features.size(0), -1)

    @property
    def feature_dim(self) -> int:
        return self._feature_size


# ──────────────────────────────────────────────────────────────────────
# EEG-Inception (Inception-EEGNet)
# ──────────────────────────────────────────────────────────────────────

class InceptionTemporalBlock(nn.Module):
    """
    Inception-style multi-scale temporal convolution block.
    Applies parallel temporal convolutions with different kernel sizes
    to capture EEG features at multiple time scales (different frequency bands).
    """

    def __init__(self, in_channels: int, out_channels_per_branch: int, n_eeg_channels: int):
        super().__init__()
        # Branch 1: Short kernel (high frequency ~beta)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_per_branch, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(out_channels_per_branch),
        )
        # Branch 2: Medium kernel (mid frequency ~alpha/mu)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_per_branch, (1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(out_channels_per_branch),
        )
        # Branch 3: Long kernel (low frequency ~theta)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_per_branch, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(out_channels_per_branch),
        )
        self.out_channels = out_channels_per_branch * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        # Trim to same temporal length (padding can differ by 1)
        min_t = min(b1.size(-1), b2.size(-1), b3.size(-1))
        return torch.cat([b1[..., :min_t], b2[..., :min_t], b3[..., :min_t]], dim=1)


class _InceptionFeatureExtractor(nn.Module):
    """Wraps inception blocks into a single nn.Module for TL compatibility."""

    def __init__(self, inception1, spatial1, inception2, temporal2):
        super().__init__()
        self.inception1 = inception1
        self.spatial1 = spatial1
        self.inception2 = inception2
        self.temporal2 = temporal2

    def forward(self, x):
        x = self.inception1(x)
        x = self.spatial1(x)
        x = self.inception2(x)
        x = self.temporal2(x)
        return x


class EEGInception(nn.Module):
    """
    EEG-Inception: Inception-style architecture for EEG classification.

    Combines multi-scale temporal feature extraction (Inception blocks) with
    depthwise spatial convolutions, similar to EEGNet but capturing multiple
    frequency bands simultaneously.

    Reference: Santamaria-Vazquez et al., "EEG-Inception", 2020
               Zhang et al., "A multi-scale convolutional neural network for
               motor imagery classification", 2021

    Advantages over standard EEGNet:
    - Multi-scale temporal filters capture theta, alpha/mu, and beta simultaneously
    - Better suited for motor imagery where discriminative bands vary across subjects
    - Reduces the need to manually pick the optimal filter kernel size

    Args:
        n_channels: Number of EEG channels
        n_samples: Number of time samples per trial
        n_classes: Number of output classes
        F_per_branch: Filters per inception branch (total = 3 * F_per_branch)
        D: Depth multiplier for depthwise spatial convolution
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_samples: int = 875,
        n_classes: int = 4,
        F_per_branch: int = 8,
        D: int = 2,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        F_total = F_per_branch * 3  # Combined channels from 3 inception branches

        # Block 1: Multi-scale temporal inception + depthwise spatial
        self.inception1 = InceptionTemporalBlock(1, F_per_branch, n_channels)
        self.spatial1 = nn.Sequential(
            nn.Conv2d(F_total, F_total * D, (n_channels, 1), groups=F_total, bias=False),
            nn.BatchNorm2d(F_total * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: Second inception layer at reduced time resolution
        self.inception2 = InceptionTemporalBlock(F_total * D, F_per_branch, 1)
        self.temporal2 = nn.Sequential(
            nn.BatchNorm2d(F_total),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Feature extractor as nn.Module for TL framework compatibility
        self.feature_extractor = _InceptionFeatureExtractor(
            self.inception1, self.spatial1, self.inception2, self.temporal2
        )
        self._feature_size = self._get_feature_size()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, n_classes),
        )

    def _get_feature_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.feature_extractor(x)
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return features.view(features.size(0), -1)

    @property
    def feature_dim(self) -> int:
        return self._feature_size


# ──────────────────────────────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────────────────────────────

def build_model(
    model_type: str,
    n_channels: int,
    n_samples: int,
    n_classes: int,
    model_config=None,
) -> nn.Module:
    """
    Build a model by name.

    Args:
        model_type: 'eegnet', 'shallowconvnet', 'deepconvnet'
        n_channels: Number of EEG channels
        n_samples: Number of time samples
        n_classes: Number of classes
        model_config: Optional ModelConfig for hyperparameters
    """
    if model_type == "eegnet":
        kwargs = dict(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
        )
        if model_config:
            kwargs.update(
                F1=model_config.F1,
                F2=model_config.F2,
                D=model_config.D,
                kernel_length=model_config.kernel_length,
                dropout_rate=model_config.dropout_rate,
            )
        return EEGNet(**kwargs)
    elif model_type == "shallowconvnet":
        return ShallowConvNet(n_channels, n_samples, n_classes)
    elif model_type == "deepconvnet":
        return DeepConvNet(n_channels, n_samples, n_classes)
    elif model_type == "eeginception":
        return EEGInception(n_channels, n_samples, n_classes)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Use 'eegnet', 'shallowconvnet', 'deepconvnet', or 'eeginception'."
        )
