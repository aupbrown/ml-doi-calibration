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

## Data Notes

Data files are **not tracked in git**. The `data/` directory is present as a folder skeleton
only. Expected layout:

```
data/
├── raw/          # Original detector output files
└── processed/    # Feature-engineered, split datasets
```

See `configs/default.yaml` for path configuration.

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

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
