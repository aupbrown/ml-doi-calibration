## Description

<!-- What does this PR do and why? -->

## Type of Change

- [ ] Bug fix
- [ ] New feature / module
- [ ] Refactor
- [ ] Experiment (model, config, ablation)
- [ ] Documentation

## Checklist

- [ ] `make lint` passes (ruff check + format)
- [ ] `make type-check` passes (mypy)
- [ ] `make test` passes (pytest)
- [ ] New modules have corresponding tests in `tests/`
- [ ] New config parameters added to `configs/default.yaml` with comments
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Random seeds set in config (not hardcoded) if results depend on randomness
- [ ] No data files, model weights, or `.npy` arrays committed
