# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `src/pet_doi_ml/constants.py` — centralized physical constants and channel maps
- `src/pet_doi_ml/config.py` — Pydantic v2 config models for ingestion and emulation
- `src/pet_doi_ml/data/loader.py` — chunked binary loader with Fortran-order reshape
- `src/pet_doi_ml/features/emulation.py` — RENA-3 ASIC emulation (RC-CR shaping, trigger, ADC)
- `src/pet_doi_ml/features/extraction.py` — 15 physics-motivated RENA-3 compatible features
- `src/pet_doi_ml/pipeline.py` — end-to-end orchestration with validation plots
- `CITATION.cff`, `CONTRIBUTING.md`, issue templates, PR template

## [0.1.0] — 2026-02-XX

### Added
- Initial project scaffold: `src/pet_doi_ml/` package, `pyproject.toml`, `configs/default.yaml`
- GitHub Actions CI: lint (ruff), type-check (mypy), test (pytest + codecov)
- Pre-commit hooks: ruff formatting + pre-commit-hooks
- Makefile targets: `install`, `lint`, `format`, `type-check`, `test`, `clean`
- `README.md` with project description, installation, usage, and citation
- MIT License
