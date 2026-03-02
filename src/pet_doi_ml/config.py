"""Pydantic v2 configuration models for the pet-doi-ml pipeline.

Usage:
    config = load_config(Path("configs/default.yaml"))
    config.ingest.data_dir   # resolved absolute Path
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from pet_doi_ml.constants import (
    MOTOR_STEP_MM,
    NUM_CHANNELS,
    NUM_MOTOR_STEPS,
    RENA3_ADC_MAX,
    RENA3_SHAPING_TIME_S,
    SAMPLES_PER_WAVEFORM,
)

# Project root is two levels above this file (src/pet_doi_ml/config.py → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class IngestConfig(BaseModel):
    data_dir: Path
    chunk_size: int = 500
    step_indices: list[int] | None = None  # None → use all NUM_MOTOR_STEPS files
    num_channels: int = NUM_CHANNELS
    samples_per_waveform: int = SAMPLES_PER_WAVEFORM
    motor_step_mm: float = MOTOR_STEP_MM

    @field_validator("data_dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, v: object) -> Path:
        p = Path(str(v))
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p.resolve()

    def effective_step_indices(self) -> list[int]:
        if self.step_indices is not None:
            return self.step_indices
        return list(range(NUM_MOTOR_STEPS))


class EmulationConfig(BaseModel):
    shaping_time_s: float = RENA3_SHAPING_TIME_S
    trigger_threshold_fraction: float = 0.10
    trigger_skip_samples: int = 10
    peak_search_window_samples: int = 50
    # None → auto-fit from 99th percentile of first chunk
    adc_scale: float | None = None
    adc_max: int = RENA3_ADC_MAX


class PipelineConfig(BaseModel):
    ingest: IngestConfig
    emulation: EmulationConfig
    output_dir: Path

    @field_validator("output_dir", mode="before")
    @classmethod
    def resolve_output_dir(cls, v: object) -> Path:
        p = Path(str(v))
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p.resolve()


def load_config(path: Path) -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file."""
    raw = yaml.safe_load(path.read_text())

    ingest_raw = raw.get("ingest", {})
    emulation_raw = raw.get("emulation", {})
    output_dir = raw.get("output", {}).get("processed_dir", "data/processed")

    return PipelineConfig(
        ingest=IngestConfig(**ingest_raw),
        emulation=EmulationConfig(**emulation_raw),
        output_dir=output_dir,
    )
