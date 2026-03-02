# Contributing

Thank you for your interest in contributing to pet-doi-ml.

## Development Setup

```bash
git clone https://github.com/aupbrown/ml-doi-calibration.git
cd ml-doi-calibration
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- **Formatter/linter**: ruff (`line-length = 88`, rules E, W, F, I, B)
- **Type checking**: mypy with `ignore_missing_imports = true`
- **No magic numbers**: all physical constants belong in `src/pet_doi_ml/constants.py`
- Run checks before submitting: `make lint && make type-check && make test`

## Branch Conventions

| Prefix | Use |
|---|---|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `experiment/` | Model experiments or ablations |
| `docs/` | Documentation only |

## Pull Requests

- All CI checks (lint, type-check, tests) must pass
- New modules must include tests in `tests/`
- Config parameters must be added to `configs/default.yaml` and documented
- Include reproducibility notes: random seeds, data file versions used

## Data Provenance

Raw waveform data originates from a collimator-scan experiment at the University of
Arizona Radiation Imaging Laboratory. Do not commit raw or processed data files to the
repository. The `data/` and `results/` directories are gitignored by design.

## Reporting Issues

Use the GitHub issue templates:
- **Bug report**: unexpected behavior with steps to reproduce
- **Feature request**: new functionality with motivation and proposed approach
