"""End-to-end ingestion pipeline: raw waveforms → RENA-3 features → saved arrays.

Entry points:
    run_ingestion(config)          — load, emulate, extract, normalize, save
    run_validation_plots(X, y, dir) — physics sanity-check plots
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from pet_doi_ml.config import PipelineConfig
from pet_doi_ml.data.loader import BinaryLoader
from pet_doi_ml.features.emulation import Rena3Emulator
from pet_doi_ml.features.extraction import FEATURE_NAMES, extract_features


def run_ingestion(config: PipelineConfig) -> dict[str, Path]:
    """Load all waveform files, emulate RENA-3, extract features, and save.

    Args:
        config: Validated PipelineConfig.

    Returns:
        Dict mapping output names to saved file paths:
        ``{"X_raw", "X_normalized", "y_ground_truth", "feature_names", "scaler"}``.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    loader = BinaryLoader(config.ingest)
    emulator = Rena3Emulator(config.emulation)

    features_list: list[NDArray[np.float32]] = []
    labels_list: list[NDArray[np.float32]] = []

    step_indices = config.ingest.effective_step_indices()
    t0 = time.monotonic()

    for step_index in step_indices:
        for chunk_i16, doi_mm in loader.iter_file_chunks(step_index):
            adc_counts = emulator.process_chunk(chunk_i16)
            features = extract_features(
                adc_counts, config.emulation.trigger_threshold_fraction
            )
            n = features.shape[0]
            features_list.append(features)
            labels_list.append(np.full(n, doi_mm, dtype=np.float32))

        elapsed = time.monotonic() - t0
        total_events = sum(f.shape[0] for f in features_list)
        print(
            f"  step {step_index:02d}/{step_indices[-1]:02d} done — "
            f"{total_events:,} events, {elapsed:.1f}s elapsed"
        )

    X_raw: NDArray[np.float32] = np.concatenate(features_list, axis=0)
    y: NDArray[np.float32] = np.concatenate(labels_list, axis=0)

    scaler = StandardScaler()
    X_norm: NDArray[np.float32] = scaler.fit_transform(X_raw).astype(np.float32)

    out = config.output_dir
    paths: dict[str, Path] = {
        "X_raw": out / "X_raw.npy",
        "X_normalized": out / "X_normalized.npy",
        "y_ground_truth": out / "y_ground_truth.npy",
        "feature_names": out / "feature_names.txt",
        "scaler": out / "scaler.pkl",
    }

    np.save(paths["X_raw"], X_raw)
    np.save(paths["X_normalized"], X_norm)
    np.save(paths["y_ground_truth"], y)
    paths["feature_names"].write_text("\n".join(FEATURE_NAMES) + "\n")
    joblib.dump(scaler, paths["scaler"])

    total_elapsed = time.monotonic() - t0
    print(
        f"\nIngestion complete: {X_raw.shape[0]:,} events × {X_raw.shape[1]} features "
        f"in {total_elapsed:.1f}s"
    )
    print(f"Outputs saved to: {out}")
    return paths


def run_validation_plots(
    X_raw: NDArray[np.float32],
    y: NDArray[np.float32],
    output_dir: Path,
) -> list[Path]:
    """Generate physics validation plots from raw (un-normalized) features.

    Plots:
        1. Feature distributions: 5×3 histogram grid for all 15 features
        2. CAR vs depth: monotonic increase expected (primary DOI indicator)
        3. Anode amplitude vs depth: decrease near cathode (charge trapping)

    Args:
        X_raw: Shape (N, 15) un-normalized feature matrix.
        y: Shape (N,) ground-truth DOI values in mm.
        output_dir: Directory where PNG files are saved.

    Returns:
        List of saved plot paths.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # --- Plot 1: Feature distributions ---
    fig, axes = plt.subplots(5, 3, figsize=(14, 16))
    for i, (ax, name) in enumerate(zip(axes.flat, FEATURE_NAMES, strict=False)):
        ax.hist(X_raw[:, i], bins=100, edgecolor="none")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    for ax in axes.flat[len(FEATURE_NAMES) :]:
        ax.set_visible(False)
    fig.suptitle("RENA-3 Feature Distributions", fontsize=12)
    fig.tight_layout()
    p = output_dir / "validation_feature_distributions.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # --- Plot 2: CAR vs depth ---
    car_idx = list(FEATURE_NAMES).index("CAR")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y, X_raw[:, car_idx], alpha=0.03, s=1, rasterized=True)
    ax.set_xlabel("DOI ground truth (mm)")
    ax.set_ylabel("CAR (cathode-to-anode ratio)")
    ax.set_title("CAR vs Depth — physics validation (expect monotonic increase)")
    fig.tight_layout()
    p = output_dir / "validation_car_vs_depth.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # --- Plot 3: Anode amplitude vs depth ---
    ea_idx = list(FEATURE_NAMES).index("E_anode_primary")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y, X_raw[:, ea_idx], alpha=0.03, s=1, rasterized=True)
    ax.set_xlabel("DOI ground truth (mm)")
    ax.set_ylabel("E_anode_primary (ADC counts)")
    ax.set_title(
        "Anode amplitude vs depth — expect decrease near cathode (charge trapping)"
    )
    fig.tight_layout()
    p = output_dir / "validation_anode_amplitude_vs_depth.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    return saved
