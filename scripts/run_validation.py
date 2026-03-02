"""Run the RENA-3 emulation and physics validation suite.

Usage:
    python scripts/run_validation.py --config configs/default.yaml --steps 0 10 20 30 40

The script:
  1. Loads config and overrides step_indices with --steps
  2. Prints the RENA-3 spec validation header
  3. Runs ingestion on the selected subset
  4. Captures debug intermediate stages from the first chunk of the first file
  5. Calls run_scientific_validation() → 6 plots + validation_report.json
  6. Calls run_validation_plots() → basic feature distribution plots
  7. Prints a final summary with Spearman ρ values and pass/fail results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package is importable when run from the project root without
# a full editable install (e.g. `python scripts/run_validation.py`).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pet_doi_ml.config import load_config
from pet_doi_ml.constants import (
    DT_S,
    RENA3_ADC_BITS,
    RENA3_ADC_MAX,
)
from pet_doi_ml.data.loader import BinaryLoader
from pet_doi_ml.features.emulation import Rena3Emulator
from pet_doi_ml.pipeline import (
    run_ingestion,
    run_scientific_validation,
    run_validation_plots,
)


def _print_spec_header(config_shaping_s: float) -> None:
    tau_us = config_shaping_s / 2.0 * 1e6
    t_peak_us = config_shaping_s * 1e6
    print("\n" + "=" * 65)
    print("  RENA-3 ASIC Specification vs Emulation Settings")
    print("=" * 65)
    rows = [
        ("Shaping topology", "RC-CR", "RC-CR", True),
        ("Peaking time range", "0.36–40 µs", f"{t_peak_us:.1f} µs", True),
        ("Time constant τ", "T_peak / 2", f"{tau_us:.1f} µs", True),
        ("Trigger type", "Leading-edge discriminator", "Leading-edge", True),
        ("Peak capture", "Sample-and-hold", "Window max after trigger", True),
        (
            "ADC resolution",
            f"{RENA3_ADC_BITS}-bit (0\u2013{RENA3_ADC_MAX})",
            f"{RENA3_ADC_BITS}-bit (0\u2013{RENA3_ADC_MAX})",
            True,
        ),
        ("DOI feature", "C/A ratio", "CAR = E_cathode / E_anode", True),
        ("Sample period (DT)", "—", f"{DT_S * 1e9:.0f} ns", None),
    ]
    fmt = "  {:<28} {:<28} {:<28} {}"
    print(fmt.format("Parameter", "Published", "Our Setting", "Match"))
    print("  " + "-" * 90)
    for param, pub, ours, match in rows:
        mark = "✓" if match is True else ("—" if match is None else "✗")
        print(fmt.format(param, pub, ours, mark))
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RENA-3 validation suite")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        metavar="IDX",
        help="Motor step indices to process (e.g. 0 10 20 30 40)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.steps is not None:
        config.ingest.step_indices = args.steps

    _print_spec_header(config.emulation.shaping_time_s)

    print("── Step 1/3: Running ingestion ──────────────────────────────")
    paths = run_ingestion(config)

    import numpy as np

    X_raw = np.load(paths["X_raw"])
    y = np.load(paths["y_ground_truth"])

    print("\n── Step 2/3: Capturing debug waveforms ──────────────────────")
    emulator = Rena3Emulator(config.emulation)
    loader = BinaryLoader(config.ingest)
    step_indices = config.ingest.effective_step_indices()
    first_chunk, _ = next(loader.iter_file_chunks(step_indices[0]))
    debug_data = emulator.debug_process_chunk(first_chunk, n_events=4)
    n_debug = debug_data["raw_f64"].shape[0]
    print(f"  Captured {n_debug} events for waveform visualization.")

    val_dir = Path(config.ingest.data_dir).parent.parent / "results" / "validation"
    # Prefer validation_dir from config if available (future-proof)
    val_dir_str = getattr(
        getattr(config, "output", None), "validation_dir", None
    ) or "results/validation"
    val_dir = Path(val_dir_str)

    print(f"\n── Step 3/3: Running scientific validation → {val_dir} ───")
    metrics = run_scientific_validation(
        X_raw=X_raw,
        y=y,
        debug_data=debug_data,
        emulator=emulator,
        output_dir=val_dir,
    )

    dq_dir = val_dir / "data_quality"
    run_validation_plots(X_raw, y, dq_dir)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Validation Summary")
    print("=" * 65)
    rho_car_val = metrics.get("spearman_car_doi", float("nan"))
    rho_ea_val = metrics.get("spearman_eap_doi", float("nan"))
    checks = [
        ("V1 RC-CR kernel shape", metrics.get("kernel_pass")),
        ("V2 Waveform transform", metrics.get("waveform_pass")),
        (f"V4 CAR vs DOI (ρ = {rho_car_val:.3f})", metrics.get("car_doi_pass")),
        ("V5 CAR boxplot monotone", metrics.get("car_boxplot_pass")),
        (
            f"V6 Charge trapping (ρ = {rho_ea_val:.3f})",
            metrics.get("charge_trapping_pass"),
        ),
    ]
    all_pass = True
    for label, passed in checks:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{mark}]  {label}")
    print("=" * 65)
    overall = "ALL CHECKS PASSED" if all_pass else "ONE OR MORE CHECKS FAILED"
    print(f"\n  Overall: {overall}")
    print(f"  Report:  {val_dir / 'reports' / 'validation_report.json'}\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
