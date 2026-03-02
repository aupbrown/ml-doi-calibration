"""Binary waveform loader for CAEN DT5730 digitizer output files.

Files are headerless int16 little-endian binaries written in Fortran column-major
order with logical shape (time, events, channels). Correct loading requires:

    raw.reshape(SAMPLES_PER_WAVEFORM, n_events, NUM_CHANNELS, order="F")
        .transpose(1, 2, 0)
    # → (n_events, NUM_CHANNELS, SAMPLES_PER_WAVEFORM)

Using C-order or wrong axis order silently scrambles all channels and time samples.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pet_doi_ml.config import IngestConfig


class BinaryLoader:
    """Chunked loader for CAEN binary waveform files.

    Iterates over events in memory-efficient chunks using np.memmap so that
    a full multi-GB file is never materialized in RAM.
    """

    def __init__(self, config: IngestConfig) -> None:
        if not config.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir not found: {config.data_dir}")
        self._config = config
        self._samples_per_event = config.num_channels * config.samples_per_waveform

    def _file_path(self, step_index: int) -> Path:
        return self._config.data_dir / f"Decoded_{step_index:04d}.bin"

    def count_events(self, step_index: int) -> int:
        """Return the number of events in a file without loading it."""
        path = self._file_path(step_index)
        n_int16 = path.stat().st_size // 2  # each int16 = 2 bytes
        return n_int16 // self._samples_per_event

    def iter_file_chunks(
        self,
        step_index: int,
    ) -> Generator[tuple[NDArray[np.int16], float], None, None]:
        """Yield (waveforms, doi_mm) chunks from one motor-position file.

        Args:
            step_index: Motor step index (0 to NUM_MOTOR_STEPS-1). The DOI ground
                truth label is ``step_index * motor_step_mm`` millimetres.

        Yields:
            waveforms: int16 array of shape
                (chunk_size, num_channels, samples_per_waveform).
                Conversion to float is deferred to the emulation step.
            doi_mm: Ground-truth depth-of-interaction in millimetres for every event
                in this chunk.
        """
        path = self._file_path(step_index)
        doi_mm = step_index * self._config.motor_step_mm
        chunk_size = self._config.chunk_size
        n_channels = self._config.num_channels
        n_samples = self._config.samples_per_waveform

        raw: NDArray[np.int16] = np.memmap(path, dtype="<i2", mode="r")
        n_events = len(raw) // self._samples_per_event

        for start in range(0, n_events, chunk_size):
            end = min(start + chunk_size, n_events)
            n_chunk = end - start

            flat = raw[start * self._samples_per_event : end * self._samples_per_event]

            # Fortran column-major reshape: disk layout is (time, events, channels)
            chunk = (
                flat.reshape(n_samples, n_chunk, n_channels, order="F")
                .transpose(1, 2, 0)
                .copy()  # copy to detach from memmap before yielding
            )
            yield chunk, doi_mm


def load_motor_positions(data_dir: Path) -> NDArray[np.float64]:
    """Read motor encoder positions from MotorPositions.bin.

    The file stores 41 float64 little-endian values starting at byte offset 341.
    Values are in inches (raw encoder readback). This function is informational;
    the ground-truth DOI label used in training is step_index * MOTOR_STEP_MM.

    Returns:
        Array of shape (NUM_CHANNELS,) with encoder positions in inches.
    """
    motor_file = data_dir / "MotorPositions.bin"
    if not motor_file.exists():
        return np.array([], dtype=np.float64)

    data = motor_file.read_bytes()
    offset = 341
    n_positions = (len(data) - offset) // 8
    return np.frombuffer(data[offset : offset + n_positions * 8], dtype="<f8")
