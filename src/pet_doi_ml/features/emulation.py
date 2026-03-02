"""RENA-3 ASIC signal processing emulation from raw CZT waveforms.

Emulates the hardware operations performed by the RENA-3 readout ASIC:
  1A. Baseline subtraction (pre-trigger mean removal)
  1B. RC-CR pulse shaping (2.8 µs peaking time)
  1C. Leading-edge trigger detection (threshold crossing)
  1D. Peak amplitude extraction (sample-and-hold window after trigger)
  1E. 12-bit ADC quantization (0–4095 counts)

This ensures features extracted from Arizona CAEN data are directly comparable
to RENA-3 outputs from the Stanford CZT PET system.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

from pet_doi_ml.config import EmulationConfig
from pet_doi_ml.constants import (
    DT_S,
    NUM_CHANNELS,
    PRE_TRIGGER_SAMPLES,
    RENA3_ADC_MAX,
    SAMPLES_PER_WAVEFORM,
)


class Rena3Emulator:
    """Emulate RENA-3 ASIC readout from raw int16 waveform chunks.

    The emulator is stateful: ADC scale is auto-fit from the 99th percentile of
    the first chunk and then locked for all subsequent chunks to ensure
    consistent quantization across files.
    """

    def __init__(self, config: EmulationConfig) -> None:
        self._config = config
        self._adc_scale: float | None = config.adc_scale
        self._kernel = self._build_rccr_kernel(config.shaping_time_s)

    @staticmethod
    def _build_rccr_kernel(shaping_time_s: float) -> NDArray[np.float64]:
        """Build RC-CR impulse response kernel.

        The RC-CR differentiator-integrator has peaking time = 2*tau.
        We use tau = shaping_time_s / 2 so that the peak occurs at shaping_time_s.
        The kernel runs for 10*tau to reach negligible amplitude before truncation.
        """
        tau = shaping_time_s / 2.0
        t = np.arange(0.0, 10.0 * tau, DT_S)
        h = (t / tau**2) * np.exp(-t / tau)
        h /= h.sum()  # normalize so shaped amplitudes are in the same units as input
        return h

    def _subtract_baseline(
        self, wf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        baseline = wf[:, :, :PRE_TRIGGER_SAMPLES].mean(axis=2, keepdims=True)
        return wf - baseline

    def _shape_waveforms(
        self, wf: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply RC-CR pulse shaping to all channels of all events.

        Reshapes (N, 16, 2001) → (N*16, 2001), convolves each row with the
        RC-CR kernel, then reshapes back.
        """
        n_events = wf.shape[0]
        flat = wf.reshape(n_events * NUM_CHANNELS, SAMPLES_PER_WAVEFORM)

        shaped_flat = np.apply_along_axis(
            lambda row: sp_signal.fftconvolve(row, self._kernel, mode="same"),
            axis=1,
            arr=flat,
        )
        return shaped_flat.reshape(n_events, NUM_CHANNELS, SAMPLES_PER_WAVEFORM)

    def _extract_peaks(
        self, shaped: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Detect trigger and extract peak amplitude per channel per event.

        Uses absolute value to handle both anode (+) and cathode (-) polarity
        without branching. Builds a boolean window mask for variable trigger
        positions using broadcasting — no Python loops over events.
        """
        cfg = self._config
        abs_s = np.abs(shaped)  # (N, 16, 2001)

        # Compute per-channel peak in the post-pre-trigger region for thresholding
        post_trigger_max = abs_s[:, :, PRE_TRIGGER_SAMPLES:].max(axis=2)  # (N, 16)
        thresh = cfg.trigger_threshold_fraction * post_trigger_max[:, :, np.newaxis]

        # Leading-edge trigger: first sample at or above threshold, skipping head
        above = abs_s[:, :, cfg.trigger_skip_samples:] >= thresh  # (N, 16, T')
        # argmax on bool returns 0 when no True; we detect that separately
        trigger_rel = np.argmax(above, axis=2)  # (N, 16) — relative to skip offset
        triggered_any = above.any(axis=2)       # (N, 16)

        trigger_abs = np.where(
            triggered_any,
            trigger_rel + cfg.trigger_skip_samples,
            SAMPLES_PER_WAVEFORM,  # sentinel: untriggered channels yield 0 peak
        )  # (N, 16)

        # Peak search window: [trigger_abs, trigger_abs + window)
        end_idx = np.minimum(
            trigger_abs + cfg.peak_search_window_samples, SAMPLES_PER_WAVEFORM
        )  # (N, 16)

        # Vectorised window mask via broadcasting
        sample_idx = np.arange(SAMPLES_PER_WAVEFORM)  # (2001,)
        s = sample_idx[np.newaxis, np.newaxis, :]
        in_window = (s >= trigger_abs[:, :, np.newaxis]) & (
            s < end_idx[:, :, np.newaxis]
        )  # (N, 16, 2001)

        peaks = np.where(in_window, abs_s, 0.0).max(axis=2)  # (N, 16)
        return peaks

    def _quantize(self, peaks: NDArray[np.float64]) -> NDArray[np.uint16]:
        """Scale float peaks to 12-bit ADC counts.

        On the first call, auto-fits the scale from the 99th percentile of all
        channel peaks in the current chunk and locks it for all future calls.
        """
        if self._adc_scale is None:
            self._adc_scale = float(np.percentile(peaks, 99))
            if self._adc_scale == 0.0:
                self._adc_scale = 1.0  # guard against all-zero chunk

        scaled = peaks / self._adc_scale * RENA3_ADC_MAX
        return np.clip(np.round(scaled), 0, RENA3_ADC_MAX).astype(np.uint16)

    def debug_process_chunk(
        self,
        waveforms_int16: NDArray[np.int16],
        n_events: int = 4,
    ) -> dict[str, NDArray]:
        """Return intermediate emulation stages for visualization.

        Does NOT call _quantize, so ADC scale is never fitted or modified.
        Safe to call at any point without side effects.

        Args:
            waveforms_int16: Shape (N, 16, 2001), raw int16 ADC samples.
            n_events: Number of events to process (takes the first min(n, N)).

        Returns:
            Dict with keys:
                ``kernel``      — 1-D RC-CR kernel array (float64)
                ``raw_f64``     — (n, 16, 2001) baseline-subtracted float64
                ``shaped``      — (n, 16, 2001) RC-CR shaped float64
                ``peaks``       — (n, 16) float64 peak amplitudes (un-quantized)
                ``trigger_idx`` — (n, 16) int sample index of leading-edge trigger
        """
        n = min(n_events, waveforms_int16.shape[0])
        wf = waveforms_int16[:n].astype(np.float64)
        raw_f64 = self._subtract_baseline(wf)
        shaped = self._shape_waveforms(raw_f64)

        # Recover trigger indices (mirrors _extract_peaks logic, no quantization)
        cfg = self._config
        abs_s = np.abs(shaped)
        post_trigger_max = abs_s[:, :, PRE_TRIGGER_SAMPLES:].max(axis=2)
        thresh = cfg.trigger_threshold_fraction * post_trigger_max[:, :, np.newaxis]
        above = abs_s[:, :, cfg.trigger_skip_samples:] >= thresh
        trigger_rel = np.argmax(above, axis=2)
        triggered_any = above.any(axis=2)
        trigger_idx = np.where(
            triggered_any,
            trigger_rel + cfg.trigger_skip_samples,
            SAMPLES_PER_WAVEFORM,
        ).astype(np.int64)

        peaks = self._extract_peaks(shaped)

        return {
            "kernel": self._kernel,
            "raw_f64": raw_f64,
            "shaped": shaped,
            "peaks": peaks,
            "trigger_idx": trigger_idx,
        }

    def process_chunk(
        self, waveforms_int16: NDArray[np.int16]
    ) -> NDArray[np.uint16]:
        """Run the full RENA-3 emulation chain on one chunk of waveforms.

        Args:
            waveforms_int16: Shape (N, 16, 2001), raw int16 ADC samples.

        Returns:
            Shape (N, 16), uint16 emulated RENA-3 ADC counts.
        """
        wf = waveforms_int16.astype(np.float64)
        wf = self._subtract_baseline(wf)
        wf = self._shape_waveforms(wf)
        peaks = self._extract_peaks(wf)
        return self._quantize(peaks)
