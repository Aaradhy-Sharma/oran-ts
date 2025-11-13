"""PHY mapping utilities for sandbox Sionna integration.

This module provides a sinr->rate mapping. If Sionna is available, a more
accurate mapping could be used; otherwise, we fall back to a realistic MCS
spectral-efficiency table (approximate) which is more practical than pure
Shannon for radio link modeling in this simulator.
"""
import math
from typing import Union


def sinr_to_rate_bps(sinr_linear: Union[float, int], bandwidth_hz: float) -> float:
    """Map linear SINR to achievable bits/s for a given bandwidth.

    Uses an approximate MCS table (spectral efficiencies in bits/s/Hz).
    If sinr_linear is invalid or <=0, returns 0.
    """
    try:
        sinr_db = 10 * math.log10(sinr_linear) if sinr_linear > 0 else -999.0
    except Exception:
        return 0.0

    # Approximate MCS thresholds and spectral efficiencies (LTE-like)
    # thresholds in dB and corresponding bits/s/Hz
    mcs_table = [
        (-999.0, 0.1523),
        (-6.0, 0.2344),
        (-4.0, 0.3770),
        (-2.0, 0.6016),
        (0.0, 0.8770),
        (2.0, 1.1758),
        (4.0, 1.4766),
        (6.0, 1.9141),
        (8.0, 2.4063),
        (10.0, 2.7305),
        (12.0, 3.3223),
        (14.0, 3.9023),
        (16.0, 4.5234),
        (18.0, 5.1152),
        (20.0, 5.5547),
        (22.0, 6.0),
    ]

    # Find the highest spectral efficiency whose threshold <= sinr_db
    eff = 0.0
    for thr_db, se in mcs_table:
        if sinr_db >= thr_db:
            eff = se
        else:
            break

    # Efficiency multiplier to account for coding/overheads
    efficiency_factor = 0.9

    return bandwidth_hz * eff * efficiency_factor
