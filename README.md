# BCI Transfer Learning for Motor Imagery

Transfer Learning framework for EEG-based Brain-Computer Interface (BCI) Motor Imagery classification using PyTorch.

## Overview

EEG-based BCIs suffer from **inter-subject** and **intra-subject non-stationarity**, making it impractical to build a universal model. This project applies Transfer Learning (TL) to reduce calibration effort by leveraging data from existing subjects/sessions to bootstrap a new subject's decoder.

## Supported Datasets

| Dataset | Subjects | Classes | EEG Channels | Sampling Rate |
|---------|----------|---------|-------------|---------------|
| **BCI Competition IV 2a** | 9 | 4 (L-hand, R-hand, Feet, Tongue) | 22 | 250 Hz |
| **BCI Competition IV 2b** | 9 | 2 (L-hand, R-hand) | 3 (C3, Cz, C4) | 250 Hz |
| **PhysioNet MI** | 109 | 4 (L-fist, R-fist, Both fists, Both feet) | 64 | 160 Hz |

## Preprocessing Pipeline

1. **Resampling** → 250 Hz (configurable)
2. **Notch Filter** → 50/60 Hz line noise removal
3. **Bandpass Filter** → 4–38 Hz (mu + beta rhythms)
4. **Common Average Reference (CAR)** → spatial noise reduction
5. **ICA Artifact Removal** → EOG/EMG artifact rejection (FastICA + kurtosis-based component rejection)
6. **Exponential Moving Standardization** → adaptive normalization for non-stationarity
7. **Euclidean Alignment (EA)** → covariance matrix alignment across subjects for TL

## Models

| Model | Description | Params (22ch, 4cls) |
|-------|-------------|---------------------|
| **EEGNet** | Compact CNN with depthwise/separable convolutions | ~2.6K |
| **ShallowConvNet** | Mimics FBCSP with square/log activations | ~35K |
| **DeepConvNet** | 4-block deep CNN with increasing depth | ~70K |

All models expose a `feature_extractor` / `classifier` split for TL.

## Transfer Learning Strategies

| Strategy | Description |
|----------|-------------|
| **Fine-Tuning** | Pretrain on source subjects → fine-tune all (or classifier) layers on target |
| **Feature Extraction** | Freeze pretrained feature extractor → train only classifier on target |
| **Domain Adaptation (MMD/CORAL)** | Minimize distribution shift between source/target features |
| **DANN** | Adversarial domain adaptation with gradient reversal layer |
| **Progressive Networks** | Frozen source column + new target column with lateral connections |
| **Multi-Source Transfer** | Attention-weighted combination of per-source models |

## Installation

```bash
pip install -r requirements.txt
```

## Data Setup

### BCI Competition IV 2a/2b
Download from [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets):
```
data/
  bci_iv_2a/
    A01T.gdf, A01E.gdf, ..., A09T.gdf, A09E.gdf
  bci_iv_2b/
    B0101T.gdf, ..., B0905E.gdf
```

### PhysioNet MI
Auto-downloaded via MNE, or place manually:
```
data/
  physionet_mi/
    S001/S001R04.edf, ...
```

## Usage

### Single Experiment
```bash
# Fine-tune EEGNet on BCI IV 2a, target subject 9
python main.py --dataset bci_iv_2a --model eegnet --strategy fine_tune --target 9

# DANN with DeepConvNet on PhysioNet
python main.py --dataset physionet --model deepconvnet --strategy dann --target 5

# Feature extraction with 5 calibration trials per class
python main.py --strategy feature_extract --n-cal 5 --target 3
```

### Compare All Strategies
```bash
python main.py --compare-all --dataset bci_iv_2a --model eegnet --target 9
```

### Leave-One-Subject-Out Cross-Validation
```bash
python main.py --cross-subject --dataset bci_iv_2a --model eegnet --strategy fine_tune
```

### Command-Line Options
```
--dataset         bci_iv_2a | bci_iv_2b | physionet
--model           eegnet | shallowconvnet | deepconvnet
--strategy        fine_tune | feature_extract | domain_adapt | dann | progressive | multi_source
--target          Target subject ID
--source          Source subject IDs (default: all except target)
--n-cal           Calibration trials per class (default: 10)
--epochs-pretrain Pretraining epochs (default: 200)
--epochs-finetune Fine-tuning epochs (default: 100)
--batch-size      Batch size (default: 64)
--no-ea           Disable Euclidean Alignment
--no-ica          Disable ICA artifact removal
--device          cuda | cpu
--data-dir        Root data directory (default: ./data)
```

## Project Structure

```
BCI_Transfer_Learning/
├── config.py              # All configuration dataclasses
├── preprocessing.py       # EEG preprocessing pipeline
├── datasets.py            # Dataset loaders (2a, 2b, PhysioNet) + PyTorch Datasets
├── models.py              # EEGNet, ShallowConvNet, DeepConvNet
├── transfer_learning.py   # TL strategies (Fine-tune, DA, DANN, Progressive, Multi-source)
├── trainer.py             # Training loops, evaluation, metrics
├── main.py                # CLI entry point
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## References

1. Lawhern et al., "EEGNet: A Compact CNN for EEG-based BCIs", *J. Neural Eng.*, 2018
2. Schirrmeister et al., "Deep Learning with CNNs for EEG Decoding", *Human Brain Mapping*, 2017
3. He & Wu, "Transfer Learning for Brain-Computer Interfaces", *IEEE Trans. NSRE*, 2020
4. Ganin et al., "Domain-Adversarial Training of Neural Networks", *JMLR*, 2016
5. Tangermann et al., "Review of the BCI Competition IV", *Frontiers in Neuroscience*, 2012
