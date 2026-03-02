"""Extract 15 RENA-3 compatible physics features from emulated ADC outputs.

All operations are fully vectorized (no Python loops over events). Input is the
uint16 ADC count array output by Rena3Emulator.process_chunk().
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pet_doi_ml.constants import ANODE_INDICES, CATHODE_INDICES

FEATURE_NAMES: tuple[str, ...] = (
    "E_anode_primary",
    "E_cathode_primary",
    "CAR",
    "E_anode_secondary",
    "charge_share_ratio",
    "E_anode_sum",
    "E_cathode_sum",
    "log_CAR",
    "cathode_fraction",
    "anode_fraction",
    "N_triggered_anodes",
    "N_triggered_cathodes",
    "anode_signal_spread",
    "cathode_signal_spread",
    "CAR_squared",
)

_EPS = 1e-6
_ANODE_IDX = list(ANODE_INDICES)
_CATHODE_IDX = list(CATHODE_INDICES)


def extract_features(
    adc_counts: NDArray[np.uint16],
    trigger_threshold_fraction: float = 0.10,
) -> NDArray[np.float32]:
    """Compute 15 RENA-3 compatible features from emulated ADC counts.

    Args:
        adc_counts: Shape (N, 16), uint16 ADC values from Rena3Emulator.
            Channel 3 (steering electrode) is automatically excluded from
            anode features via ANODE_INDICES.
        trigger_threshold_fraction: Fraction of primary amplitude used to
            classify a channel as "triggered" for multiplicity features.

    Returns:
        Float32 array of shape (N, 15) with features in FEATURE_NAMES order.
    """
    anode = adc_counts[:, _ANODE_IDX].astype(np.float32)   # (N, 7)
    cathode = adc_counts[:, _CATHODE_IDX].astype(np.float32)  # (N, 8)

    # Primary amplitudes
    e_anode_primary = anode.max(axis=1)        # (N,)
    e_cathode_primary = cathode.max(axis=1)    # (N,)

    # Cathode-to-anode ratio (primary DOI feature)
    car = e_cathode_primary / (e_anode_primary + _EPS)  # (N,)

    # Secondary anode (charge sharing)
    anode_sorted = np.sort(anode, axis=1)[:, ::-1]  # descending per event
    e_anode_secondary = anode_sorted[:, 1]           # (N,)
    charge_share_ratio = (e_anode_secondary + _EPS) / (e_anode_primary + _EPS)

    # Energy sums
    e_anode_sum = anode.sum(axis=1)    # (N,)
    e_cathode_sum = cathode.sum(axis=1)  # (N,)

    # Derived ratio features
    log_car = np.log(car + _EPS)
    total = e_anode_sum + e_cathode_sum + _EPS
    cathode_fraction = e_cathode_sum / total
    anode_fraction = e_anode_sum / total

    # Multiplicity: channels above a threshold relative to the primary
    thresh_a = trigger_threshold_fraction * e_anode_primary[:, np.newaxis]
    n_triggered_anodes = (anode > thresh_a).sum(axis=1).astype(np.float32)

    thresh_c = trigger_threshold_fraction * e_cathode_primary[:, np.newaxis]
    n_triggered_cathodes = (cathode > thresh_c).sum(axis=1).astype(np.float32)

    # Signal spread
    anode_spread = anode.std(axis=1)
    cathode_spread = cathode.std(axis=1)

    # Nonlinear CAR term
    car_squared = car**2

    return np.column_stack(
        [
            e_anode_primary,
            e_cathode_primary,
            car,
            e_anode_secondary,
            charge_share_ratio,
            e_anode_sum,
            e_cathode_sum,
            log_car,
            cathode_fraction,
            anode_fraction,
            n_triggered_anodes,
            n_triggered_cathodes,
            anode_spread,
            cathode_spread,
            car_squared,
        ]
    ).astype(np.float32)
