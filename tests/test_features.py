"""Tests for RENA-3 emulation and feature extraction.

All tests use small synthetic arrays — no file I/O, no real data required.
"""

from __future__ import annotations

import numpy as np
import pytest

from pet_doi_ml.config import EmulationConfig
from pet_doi_ml.constants import (
    ANODE_INDICES,
    DT_S,
    PRE_TRIGGER_SAMPLES,
    RENA3_SHAPING_TIME_S,
    SAMPLES_PER_WAVEFORM,
    STEERING_CHANNEL,
)
from pet_doi_ml.features.emulation import Rena3Emulator
from pet_doi_ml.features.extraction import FEATURE_NAMES, extract_features


def _default_emulator() -> Rena3Emulator:
    return Rena3Emulator(EmulationConfig())


# ---------------------------------------------------------------------------
# RC-CR kernel
# ---------------------------------------------------------------------------


def test_rccr_kernel_peak_time():
    """Kernel peak should occur at tau = shaping_time_s / 2 (≈ sample 87)."""
    emulator = _default_emulator()
    kernel = emulator._kernel
    tau = RENA3_SHAPING_TIME_S / 2.0
    expected_peak_sample = int(round(tau / DT_S))
    actual_peak_sample = int(np.argmax(kernel))
    assert abs(actual_peak_sample - expected_peak_sample) <= 2


def test_rccr_kernel_normalized():
    """Kernel sum should be 1.0 (normalized by sum in _build_rccr_kernel)."""
    emulator = _default_emulator()
    assert emulator._kernel.sum() == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Baseline subtraction
# ---------------------------------------------------------------------------


def test_baseline_subtraction():
    """Pre-trigger region mean should be ~0 after baseline subtraction."""
    wf = np.random.randn(5, 16, SAMPLES_PER_WAVEFORM).astype(np.float64)
    # Add a large per-channel offset in the pre-trigger region
    offset = np.random.uniform(100, 500, (5, 16, 1))
    wf[:, :, :PRE_TRIGGER_SAMPLES] += offset

    emulator = _default_emulator()
    corrected = emulator._subtract_baseline(wf)

    pre_trigger_mean = corrected[:, :, :PRE_TRIGGER_SAMPLES].mean(axis=2)
    assert np.allclose(pre_trigger_mean, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Full emulation chain on a step-function input
# ---------------------------------------------------------------------------


def test_emulation_on_step_function():
    """A step-function waveform should produce a peaked shaped output and
    yield positive, non-zero ADC counts."""
    n_events = 3
    wf = np.zeros((n_events, 16, SAMPLES_PER_WAVEFORM), dtype=np.int16)
    # Step: 0 for pre-trigger, 1000 after
    wf[:, :, PRE_TRIGGER_SAMPLES:] = 1000

    emulator = _default_emulator()
    adc = emulator.process_chunk(wf)

    assert adc.shape == (n_events, 16)
    assert adc.dtype == np.uint16
    # All channels should have triggered and produced non-zero ADC values
    assert np.all(adc > 0)
    # No saturation (should be well below 4095 for a step of 1000 ADC counts)
    assert np.all(adc < 4096)


# ---------------------------------------------------------------------------
# ADC scale locking
# ---------------------------------------------------------------------------


def test_adc_scale_locked_after_first_chunk():
    """adc_scale must be fit from the first chunk and unchanged on the second."""
    wf = np.zeros((10, 16, SAMPLES_PER_WAVEFORM), dtype=np.int16)
    wf[:, :, PRE_TRIGGER_SAMPLES:] = 500

    emulator = _default_emulator()
    assert emulator._adc_scale is None

    emulator.process_chunk(wf)
    scale_after_first = emulator._adc_scale
    assert scale_after_first is not None and scale_after_first > 0

    emulator.process_chunk(wf)
    assert emulator._adc_scale == scale_after_first  # unchanged


# ---------------------------------------------------------------------------
# Feature extraction shapes and dtypes
# ---------------------------------------------------------------------------


def test_feature_extraction_shapes():
    """extract_features must return (N, 15) float32."""
    adc = np.ones((100, 16), dtype=np.uint16) * 100
    feats = extract_features(adc)
    assert feats.shape == (100, 15)
    assert feats.dtype == np.float32


def test_feature_names_length():
    assert len(FEATURE_NAMES) == 15


# ---------------------------------------------------------------------------
# Physics correctness
# ---------------------------------------------------------------------------


def test_car_physics():
    """CAR = cathode / anode should match expected ratio."""
    adc = np.zeros((10, 16), dtype=np.uint16)
    # Set anode channels (indices from ANODE_INDICES) to 1000
    for idx in ANODE_INDICES:
        adc[:, idx] = 1000
    # Set cathode channels (8-15) to 500
    adc[:, 8:] = 500

    feats = extract_features(adc)
    car_idx = list(FEATURE_NAMES).index("CAR")
    expected_car = 500.0 / (1000.0 + 1e-6)
    assert np.allclose(feats[:, car_idx], expected_car, rtol=1e-4)


def test_steering_channel_excluded():
    """Setting the steering channel (ch 3) high must not affect E_anode_primary."""
    adc = np.zeros((5, 16), dtype=np.uint16)
    # All real anode channels to 100
    for idx in ANODE_INDICES:
        adc[:, idx] = 100
    # Steering channel to a very large value
    adc[:, STEERING_CHANNEL] = 10_000

    feats = extract_features(adc)
    e_anode_idx = list(FEATURE_NAMES).index("E_anode_primary")
    assert np.all(feats[:, e_anode_idx] == pytest.approx(100.0))


# ---------------------------------------------------------------------------
# Vectorized shaping numerical equivalence
# ---------------------------------------------------------------------------


def test_vectorized_shaping_matches_scipy():
    """Vectorized FFT shaping must match scipy fftconvolve row-by-row to 1e-10."""
    from scipy import signal as sp_signal

    emulator = _default_emulator()
    rng = np.random.default_rng(42)
    wf = rng.standard_normal((3, 16, SAMPLES_PER_WAVEFORM))

    shaped_vec = emulator._shape_waveforms(wf)

    n_events = wf.shape[0]
    flat = wf.reshape(n_events * 16, SAMPLES_PER_WAVEFORM)
    ref_flat = np.apply_along_axis(
        lambda row: sp_signal.fftconvolve(row, emulator._kernel, mode="same"),
        axis=1,
        arr=flat,
    )
    ref = ref_flat.reshape(n_events, 16, SAMPLES_PER_WAVEFORM)

    assert np.allclose(shaped_vec, ref, atol=1e-10)


def test_quantize_scale_uses_per_event_max():
    """Per-event max 99th pct gives signal-level scale for mixed noise+signal chunk."""
    emulator = _default_emulator()
    rng = np.random.default_rng(0)
    n_noise = 170
    n_signal = 30  # 15% signal events — mirrors first-chunk observation
    noise = rng.uniform(0.0, 5.0, (n_noise, 16))
    signal = rng.uniform(100.0, 300.0, (n_signal, 16))
    peaks = np.vstack([noise, signal])
    rng.shuffle(peaks)  # interleave
    emulator._quantize(peaks)
    # With old code: scale ≈ 5  (noise-level flat 99th pct over 200×16 values)
    # With new code: scale > 50 (per-event max 99th pct reaches signal region)
    assert emulator._adc_scale is not None
    assert emulator._adc_scale > 50


def test_car_high_depth():
    """Higher cathode relative to anode → higher CAR (deeper interaction)."""
    adc_shallow = np.zeros((1, 16), dtype=np.uint16)
    adc_deep = np.zeros((1, 16), dtype=np.uint16)

    for idx in ANODE_INDICES:
        adc_shallow[:, idx] = 1000
        adc_deep[:, idx] = 500
    adc_shallow[:, 8:] = 200
    adc_deep[:, 8:] = 800

    feats_shallow = extract_features(adc_shallow)
    feats_deep = extract_features(adc_deep)
    car_idx = list(FEATURE_NAMES).index("CAR")
    assert feats_deep[0, car_idx] > feats_shallow[0, car_idx]
