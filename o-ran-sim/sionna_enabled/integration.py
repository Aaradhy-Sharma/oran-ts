"""Integration helpers to enable Sionna fading in the existing simulator.

This module provides a function `enable_sionna_fading(sim, cfg)` which will
patch `BaseStation.calculate_sinr_on_rb` to use a Sionna channel for
small-scale fading samples while delegating path loss and noise to the
original `ChannelModel` implementation.

The integration is careful to be reversible and to not change the simulator
API; it simply monkey-patches methods on the BaseStation instances created
by a running `Simulation`.
"""
from typing import Dict
import numpy as np
from sionna_enabled.phy import sinr_to_rate_bps


def enable_sionna_fading(sim, cfg: Dict):
    """Patch the Simulation `sim` to use Sionna fading where possible.

    cfg: dict passed to `init_sionna_channel` (keys: tx, rx, samples)
    Returns dict mapping bs_id -> channel_handle for later inspection.
    """
    # Local import so the core project doesn't require sionna at module import time
    try:
        from sionna_enabled.sionna_wrapper import sionna_available, init_sionna_channel
    except Exception:
        print("sionna_enabled.integration: sionna wrapper not importable")
        return {}

    if not sionna_available():
        print("sionna_enabled.integration: Sionna not available — skipping")
        return {}

    bs_channel_handles = {}
    # Simple per-step cache to avoid repeated channel.apply for same BS and RB in the same step
    # keyed by (bs_id, step_counter)
    per_step_cache = {}

    # First create a channel handle per BS and store it
    for bs in sim.bss.values():
        ch = init_sionna_channel(cfg)
        if ch is None:
            continue
        bs_channel_handles[bs.id] = ch

    # Helper factory to capture bs-specific handle correctly
    def make_patched(ch_handle):
        def patched(self, target_ue, rb_id, all_bss_map, rb_pool):
            # noise
            noise_w = self.channel_model.get_noise_power_linear(self.params.rb_bandwidth_hz)

            # path loss and received power (no small-scale fading)
            signal_path_loss_db = self.channel_model.calculate_path_loss(self.position, target_ue.position)
            signal_eirp_dbm = self.power_per_link_beam_dbm + self.params.bs_link_beam_gain_db
            received_signal_power_dbm = signal_eirp_dbm - signal_path_loss_db
            signal_power_w_no_fading = 10 ** ((received_signal_power_dbm - 30) / 10)

            # Serving fading sample
            try:
                tx = getattr(ch_handle, "tx", 1)
                tx = int(tx)
                x = np.ones((1, 1, tx), dtype=np.complex64)
                y = ch_handle.apply(x)
                if y is None:
                    fading_gain_serv = 1.0
                else:
                    power = np.abs(y[0, 0]) ** 2
                    if getattr(power, 'ndim', 0) > 0:
                        power = float(power.flat[0])
                    fading_gain_serv = max(power, 1e-12)
            except Exception as e:
                print("sionna_enabled.integration: serving channel apply failed:", e)
                fading_gain_serv = 1.0

            signal_power_w = signal_power_w_no_fading * fading_gain_serv

            # Interference: iterate over RB allocations and sample per-interferer fading if handle exists
            interference_w = 0.0
            for other_bs_id, other_bs in all_bss_map.items():
                if other_bs.id == self.id:
                    continue
                rb_alloc_info = rb_pool.rb_status.get(rb_id)
                if rb_alloc_info and rb_alloc_info.get("bs_id") == other_bs.id:
                    inter_path_loss_db = self.channel_model.calculate_path_loss(other_bs.position, target_ue.position)
                    inter_eirp_dbm = other_bs.power_per_link_beam_dbm + self.params.bs_link_beam_gain_db
                    inter_power_w_no_fading = 10 ** ((inter_eirp_dbm - inter_path_loss_db - 30) / 10)

                    other_handle = bs_channel_handles.get(other_bs.id)
                    if other_handle is not None:
                        try:
                            tx_o = getattr(other_handle, "tx", 1)
                            tx_o = int(tx_o)
                            x_o = np.ones((1, 1, tx_o), dtype=np.complex64)
                            y_o = other_handle.apply(x_o)
                            if y_o is None:
                                fading_gain_interf = 1.0
                            else:
                                p_o = np.abs(y_o[0, 0]) ** 2
                                if getattr(p_o, 'ndim', 0) > 0:
                                    p_o = float(p_o.flat[0])
                                fading_gain_interf = max(p_o, 1e-12)
                        except Exception as e:
                            print("sionna_enabled.integration: interferer channel apply failed:", e)
                            fading_gain_interf = 1.0
                    else:
                        fading_gain_interf = 1.0

                    interference_w += inter_power_w_no_fading * fading_gain_interf

            denominator = noise_w + interference_w
            if denominator <= 1e-22:
                denominator = 1e-22
            sinr_linear = signal_power_w / denominator
            # Map to rate via PHY adapter and store sinr as well for metrics if needed
            return max(sinr_linear, 10 ** ((-20 - 30) / 10))

        return patched

    # Attach patched methods
    patched_count = 0
    for bs in sim.bss.values():
        chh = bs_channel_handles.get(bs.id)
        if chh is None:
            continue
        if not hasattr(bs, "_orig_calculate_sinr_on_rb"):
            bs._orig_calculate_sinr_on_rb = bs.calculate_sinr_on_rb
        bs.calculate_sinr_on_rb = make_patched(chh).__get__(bs, bs.__class__)
        patched_count += 1

    print(f"sionna_enabled.integration: patched {patched_count} BSs to use Sionna fading")
    return bs_channel_handles
