# pet-doi-ml

[![CI](https://github.com/aupbrown/ml-doi-calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/aupbrown/ml-doi-calibration/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Machine learning methods for Depth-of-Interaction (DOI) calibration in Positron Emission
Tomography (PET) imaging. Accurate DOI estimation reduces parallax error in ring-based PET
scanners, improving spatial resolution near the edges of the field of view. This repository
provides data pipelines, model training, and evaluation tools for comparing classical and deep
learning approaches to DOI calibration.

---

## Directory Structure

```
pet-doi-ml/
├── configs/
│   └── default.yaml        # Default experiment configuration
├── data/                   # Raw and processed data (not tracked in git)
│   └── .gitkeep
├── notebooks/              # Exploratory analysis notebooks
├── results/                # Model outputs, metrics, plots (not tracked in git)
│   └── .gitkeep
├── src/
│   └── pet_doi_ml/         # Main Python package
│       └── __init__.py
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   └── test_package.py
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI pipeline
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/aupbrown/ml-doi-calibration.git
cd ml-doi-calibration

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Usage

```bash
# Train a model using the default configuration
python -m pet_doi_ml.train --config configs/default.yaml

# Override specific config values
python -m pet_doi_ml.train --config configs/default.yaml model.type=mlp training.epochs=100
```

---

## Dataset

Data were acquired at the **University of Arizona Radiation Imaging Laboratory** using a
collimator-scan technique for ground-truth DOI labeling.

| Parameter | Value |
|---|---|
| Digitizer | CAEN DT5730 |
| Sample rate | 62.5 MHz (16 ns/sample) |
| Channels | 16 (7 anodes + 8 cathodes + 1 steering electrode) |
| Samples/waveform | 2001 (32 µs window) |
| Source | Cs-137 (662 keV photopeak) |
| Motor steps | 41 × 8 mm = 320 mm total scan range |
| Raw data size | ~127 GB (41 binary files, int16 Fortran-column-major) |

Raw waveform files (`Decoded_XXXX.bin`) are not redistributed here. Contact the authors
for data access. Processed feature arrays (`X_normalized.npy`, `y_ground_truth.npy`) will
be made available via a Zenodo data release (DOI forthcoming).

---

## Methods

Features are extracted by **emulating the RENA-3 ASIC signal processing chain**, ensuring
that models trained on Arizona data can be deployed on Yi Gu's Stanford CZT PET system
(which uses identical RENA-3 electronics) without retraining.

The emulation pipeline (implemented in `src/pet_doi_ml/features/`):

1. **Baseline subtraction** — remove per-channel DC offset from pre-trigger region
2. **RC-CR pulse shaping** — convolve with 2.8 µs peaking-time kernel (matches RENA-3 shaper)
3. **Leading-edge trigger** — detect threshold crossing (emulates RENA-3 discriminator)
4. **Peak amplitude extraction** — sample-and-hold in a window after trigger
5. **12-bit ADC quantization** — scale to 0–4095 counts

From these emulated ASIC outputs, 15 physics-motivated features are derived per event
(cathode-to-anode ratio, charge-sharing fraction, strip multiplicities, etc.). See
`data/RENA3_EMULATION_PIPELINE.md` for the complete specification.

---

## Data Notes

Data files are **not tracked in git**. The `data/` directory is present as a folder skeleton
only. Expected layout:

```
data/
├── CZT_.../          # Raw CAEN binary files (Decoded_XXXX.bin)
├── processed/        # Feature arrays output by run_ingestion()
└── .gitkeep
```

See `configs/default.yaml` for path configuration.

---

## Reproducibility

All random seeds are set via `configs/default.yaml` (`data.seed`). Feature extraction is
fully deterministic given identical raw binary files. To reproduce results exactly:

```bash
pip install -e ".[dev]"
python -m pet_doi_ml.pipeline --config configs/default.yaml
```

Processed feature arrays (`X_normalized.npy`, `y_ground_truth.npy`) will be archived with
a permanent DOI on Zenodo to support result reproducibility without re-running the full
127 GB ingestion pipeline.

---

## Results

| Model | MAE (mm) | RMSE (mm) | Notes |
|---|---|---|---|
| — | — | — | Results forthcoming |

---

## Configuration

All experiment parameters are controlled via YAML config files:

```bash
# Copy the default config and modify as needed
cp configs/default.yaml configs/my_experiment.yaml

# Pass to training script
python -m pet_doi_ml.train --config configs/my_experiment.yaml
```

Key configuration sections: `data`, `model`, `training`, `logging`, `output`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{brown2026petdoi,
  author       = {Brown, Austin},
  title        = {pet-doi-ml: Machine Learning for Depth-of-Interaction Calibration in PET Imaging},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/aupbrown/ml-doi-calibration}},
}
```

---

## Acknowledgments

- **Max Teicheira** (University of California, Santa Cruz
  ) — experimental data acquisition, detector
  characterization scripts, and CZT methods reference documentation
- **Yi Gu** (Stanford University) — motivation for RENA-3 ASIC emulation approach
- University of Arizona Radiation Imaging Laboratory

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
