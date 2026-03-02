"""End-to-end ingestion pipeline: raw waveforms → RENA-3 features → saved arrays.

Entry points:
    run_ingestion(config)          — load, emulate, extract, normalize, save
    run_validation_plots(X, y, dir) — physics sanity-check plots
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler

from pet_doi_ml.config import PipelineConfig
from pet_doi_ml.constants import (
    DT_S,
    RENA3_ADC_MAX,
    RENA3_SHAPING_TIME_S,
)
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


def run_scientific_validation(
    X_raw: NDArray[np.float32],
    y: NDArray[np.float32],
    debug_data: dict[str, NDArray],
    emulator: Rena3Emulator,
    output_dir: Path,
) -> dict[str, object]:
    """Generate 6 scientific validation plots and return computed metrics.

    Plots are routed to subdirectories of ``output_dir``:
        emulation/   — V1 kernel, V2 waveform transform
        physics/     — V4 CAR scatter, V5 CAR boxplot, V6 charge trapping
        data_quality/ — V3 energy spectrum

    A ``reports/validation_report.json`` is also written.

    Args:
        X_raw: Shape (N, 15) un-normalized feature matrix from run_ingestion().
        y: Shape (N,) ground-truth DOI values in mm.
        debug_data: Dict returned by Rena3Emulator.debug_process_chunk().
        emulator: Fitted Rena3Emulator (used only to access the kernel).
        output_dir: Base validation directory (e.g. results/validation/).

    Returns:
        Dict of computed metrics and pass/fail booleans.
    """
    import matplotlib.pyplot as plt

    emulation_dir = output_dir / "emulation"
    physics_dir = output_dir / "physics"
    dq_dir = output_dir / "data_quality"
    reports_dir = output_dir / "reports"
    for d in (emulation_dir, physics_dir, dq_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {}
    car_idx = list(FEATURE_NAMES).index("CAR")
    ea_idx = list(FEATURE_NAMES).index("E_anode_primary")

    # ── V1: RC-CR kernel shape ──────────────────────────────────────────────
    kernel = debug_data["kernel"]
    t_kernel = np.arange(len(kernel)) * DT_S * 1e6  # µs
    peak_sample = int(np.argmax(kernel))
    tau_s = RENA3_SHAPING_TIME_S / 2.0
    peak_error_s = abs(peak_sample * DT_S - tau_s)
    kernel_pass = bool(peak_error_s < DT_S * 2)

    # Theoretical kernel for overlay
    tau_us = tau_s * 1e6
    t_th = t_kernel
    h_th = (t_th / tau_us**2) * np.exp(-t_th / tau_us)
    if h_th.sum() > 0:
        h_th /= h_th.sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_kernel, kernel, label="emulated kernel", lw=2)
    ax.plot(t_kernel, h_th, "--", label=f"theory τ={tau_us:.1f} µs", lw=1.5)
    ax.axvline(tau_us, color="gray", linestyle=":", label=f"τ={tau_us:.1f} µs")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Normalized amplitude")
    err_ns = peak_error_s * 1e9
    ax.set_title(
        f"V1 — RC-CR Kernel  |  peak error = {err_ns:.0f} ns  |  pass={kernel_pass}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(emulation_dir / "rccr_kernel.png", dpi=150)
    plt.close(fig)

    metrics["kernel_peak_error_s"] = float(peak_error_s)
    metrics["kernel_pass"] = kernel_pass

    # ── V2: Waveform transformation ─────────────────────────────────────────
    raw_f64 = debug_data["raw_f64"]    # (n, 16, 2001)
    shaped = debug_data["shaped"]      # (n, 16, 2001)
    n_show = raw_f64.shape[0]
    t_wf = np.arange(raw_f64.shape[2]) * DT_S * 1e6  # µs

    # Use primary anode channel (channel 0 by default)
    anode_ch = 0
    shaped_ch = shaped[:, anode_ch, :]
    peak_vals = np.abs(shaped_ch).max(axis=1)
    mean_vals = np.abs(shaped_ch).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(mean_vals > 0, peak_vals / mean_vals, 0.0)
    wf_peak_to_mean = float(ratios.mean())
    wf_pass = bool(wf_peak_to_mean > 2.0)

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 3 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax2 = ax.twinx()
        ax.plot(
            t_wf, raw_f64[i, anode_ch],
            color="steelblue", lw=1, label="raw (baseline-sub)",
        )
        ax2.plot(
            t_wf, shaped[i, anode_ch],
            color="darkorange", lw=1, label="RC-CR shaped",
        )
        ax.set_ylabel("Raw (ADC units)", color="steelblue", fontsize=8)
        ax2.set_ylabel("Shaped (a.u.)", color="darkorange", fontsize=8)
        ax.set_title(f"Event {i}  |  ch{anode_ch} anode", fontsize=9)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time (µs)")
    fig.suptitle(
        f"V2 — Waveform Transform  |  peak/mean = {wf_peak_to_mean:.1f}"
        f"  |  pass={wf_pass}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(emulation_dir / "waveform_transform.png", dpi=150)
    plt.close(fig)

    metrics["waveform_peak_to_mean"] = wf_peak_to_mean
    metrics["waveform_pass"] = wf_pass

    # ── V3: Energy spectrum ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(X_raw[:, ea_idx], bins=200, range=(0, RENA3_ADC_MAX), edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("E_anode_primary (ADC counts)")
    ax.set_ylabel("Counts (log scale)")
    ax.set_title("V3 — Energy Spectrum  |  expect photopeak cluster (Cs-137, 662 keV)")
    fig.tight_layout()
    fig.savefig(dq_dir / "energy_spectrum.png", dpi=150)
    plt.close(fig)

    # ── V4: CAR vs DOI scatter ───────────────────────────────────────────────
    rho_car, _ = sp_stats.spearmanr(X_raw[:, car_idx], y)
    car_pass = bool(rho_car > 0.3)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y, X_raw[:, car_idx], alpha=0.03, s=1, rasterized=True)
    ax.set_xlabel("DOI ground truth (mm)")
    ax.set_ylabel("CAR (E_cathode / E_anode)")
    ax.set_title(f"V4 — CAR vs DOI  |  Spearman ρ = {rho_car:.3f}  |  pass={car_pass}")
    fig.tight_layout()
    fig.savefig(physics_dir / "car_vs_doi.png", dpi=150)
    plt.close(fig)

    metrics["spearman_car_doi"] = float(rho_car)
    metrics["car_doi_pass"] = car_pass

    # ── V5: CAR box plot per step ────────────────────────────────────────────
    unique_doi = np.unique(y)
    step_labels = [f"{d:.0f}" for d in unique_doi]
    car_by_step = [X_raw[y == d, car_idx] for d in unique_doi]
    medians = [np.median(g) for g in car_by_step]
    monotone = bool(len(medians) >= 2 and medians[-1] > medians[0])

    fig, ax = plt.subplots(figsize=(max(6, len(unique_doi)), 5))
    ax.boxplot(car_by_step, labels=step_labels, sym="", whis=(5, 95))  # type: ignore[call-arg]
    ax.set_xlabel("DOI ground truth (mm)")
    ax.set_ylabel("CAR")
    ax.set_title(f"V5 — CAR per Depth Step  |  monotone={monotone}")
    fig.tight_layout()
    fig.savefig(physics_dir / "car_boxplot.png", dpi=150)
    plt.close(fig)

    metrics["car_median_monotone"] = monotone
    metrics["car_boxplot_pass"] = monotone

    # ── V6: Charge trapping signature ────────────────────────────────────────
    rho_ea, _ = sp_stats.spearmanr(X_raw[:, ea_idx], y)
    ct_pass = bool(rho_ea < -0.1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y, X_raw[:, ea_idx], alpha=0.03, s=1, rasterized=True)
    ax.set_xlabel("DOI ground truth (mm)")
    ax.set_ylabel("E_anode_primary (ADC counts)")
    ax.set_title(
        f"V6 — Charge Trapping  |  Spearman ρ = {rho_ea:.3f}  |  pass={ct_pass}"
    )
    fig.tight_layout()
    fig.savefig(physics_dir / "charge_trapping.png", dpi=150)
    plt.close(fig)

    metrics["spearman_eap_doi"] = float(rho_ea)
    metrics["charge_trapping_pass"] = ct_pass

    # ── Write JSON report ────────────────────────────────────────────────────
    report_path = reports_dir / "validation_report.json"
    report_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nValidation report saved: {report_path}")

    return metrics


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
