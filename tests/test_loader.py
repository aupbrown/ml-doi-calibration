"""Tests for src/pet_doi_ml/data/loader.py.

All tests use synthetic binary data written to tmp_path — no real data files needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pet_doi_ml.config import IngestConfig
from pet_doi_ml.constants import NUM_CHANNELS, SAMPLES_PER_WAVEFORM
from pet_doi_ml.data.loader import BinaryLoader


def _write_synthetic_file(
    path,
    n_events: int,
    n_channels: int = NUM_CHANNELS,
    n_samples: int = SAMPLES_PER_WAVEFORM,
    fill_value: int = 0,
) -> None:
    """Write a synthetic CAEN-format binary file (Fortran column-major int16)."""
    arr = np.full((n_samples, n_events, n_channels), fill_value, dtype=np.int16)
    arr.ravel(order="F").tofile(path)


def _make_config(tmp_path, chunk_size: int = 10) -> IngestConfig:
    return IngestConfig(data_dir=tmp_path, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Core correctness test: Fortran reshape must not scramble axes
# ---------------------------------------------------------------------------


def test_fortran_reshape_correctness(tmp_path):
    """Known values placed at specific (event, channel, sample) positions must
    survive the Fortran-order binary write and read-back without scrambling."""
    n_events, n_ch, n_s = 4, NUM_CHANNELS, SAMPLES_PER_WAVEFORM

    # Build a reference array in (events, channels, samples) order
    ref = np.zeros((n_s, n_events, n_ch), dtype=np.int16)
    # Plant sentinel values
    ref[100, 0, 0] = 111   # event 0, ch 0, sample 100
    ref[500, 1, 3] = 222   # event 1, ch 3, sample 500
    ref[2000, 3, 7] = 333  # event 3, ch 7, sample 2000

    # Write in Fortran order (as the CAEN digitizer does)
    ref.ravel(order="F").tofile(tmp_path / "Decoded_0000.bin")

    config = _make_config(tmp_path, chunk_size=n_events)
    loader = BinaryLoader(config)
    chunks = list(loader.iter_file_chunks(0))
    assert len(chunks) == 1
    data, doi = chunks[0]

    # Shape must be (n_events, n_channels, n_samples)
    assert data.shape == (n_events, n_ch, n_s)

    # Sentinels must be at the correct (event, channel, sample) positions
    assert data[0, 0, 100] == 111
    assert data[1, 3, 500] == 222
    assert data[3, 7, 2000] == 333


# ---------------------------------------------------------------------------
# Chunk iteration
# ---------------------------------------------------------------------------


def test_iter_file_chunks_shapes(tmp_path):
    """Chunks should cover all events with correct shapes and no overlap."""
    n_events = 25
    chunk_size = 10
    _write_synthetic_file(tmp_path / "Decoded_0000.bin", n_events)

    config = _make_config(tmp_path, chunk_size=chunk_size)
    loader = BinaryLoader(config)
    chunks = list(loader.iter_file_chunks(0))

    # 25 events / 10 per chunk → 3 chunks (10, 10, 5)
    assert len(chunks) == 3
    assert chunks[0][0].shape == (10, NUM_CHANNELS, SAMPLES_PER_WAVEFORM)
    assert chunks[1][0].shape == (10, NUM_CHANNELS, SAMPLES_PER_WAVEFORM)
    assert chunks[2][0].shape == (5, NUM_CHANNELS, SAMPLES_PER_WAVEFORM)


def test_doi_label_assignment(tmp_path):
    """DOI label for step N must be N * MOTOR_STEP_MM."""
    _write_synthetic_file(tmp_path / "Decoded_0000.bin", 5)
    _write_synthetic_file(tmp_path / "Decoded_0020.bin", 5)
    _write_synthetic_file(tmp_path / "Decoded_0040.bin", 5)

    config = _make_config(tmp_path, chunk_size=100)
    loader = BinaryLoader(config)

    for step, expected_doi in [(0, 0.0), (20, 160.0), (40, 320.0)]:
        chunks = list(loader.iter_file_chunks(step))
        assert chunks[0][1] == pytest.approx(expected_doi)


# ---------------------------------------------------------------------------
# count_events
# ---------------------------------------------------------------------------


def test_count_events(tmp_path):
    """count_events() must derive event count purely from file size."""
    n_events = 17
    _write_synthetic_file(tmp_path / "Decoded_0000.bin", n_events)
    config = _make_config(tmp_path)
    loader = BinaryLoader(config)
    assert loader.count_events(0) == n_events


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_data_dir_raises():
    with pytest.raises(FileNotFoundError):
        BinaryLoader(IngestConfig(data_dir="/nonexistent/path"))
