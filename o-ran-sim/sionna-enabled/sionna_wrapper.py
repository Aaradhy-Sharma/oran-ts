"""Adapter to use Sionna channel models from a NumPy-based simulator.

The wrapper guards imports so the rest of the codebase doesn't need TF/Sionna
installed. It exposes a minimal API:
 - sionna_available() -> bool
 - init_sionna_channel(config: dict) -> ChannelHandle | None

ChannelHandle.apply(np_complex_samples) -> np_complex_samples (rx samples)
"""
from typing import Optional
import numpy as np


def sionna_available() -> bool:
    try:
        import tensorflow as tf  # noqa: F401
        import sionna  # noqa: F401
        return True
    except Exception:
        return False


class ChannelHandle:
    """Thin wrapper around a Sionna channel object.

    The wrapper converts NumPy complex arrays to TF tensors, runs the
    Sionna channel, and converts back to NumPy arrays. It keeps the API
    minimal so it can be swapped into `sim_core` code paths where needed.
    """

    def __init__(self, channel_obj, rx: int, tx: int):
        self.channel_obj = channel_obj
        self.rx = rx
        self.tx = tx

    def apply(self, x_np: np.ndarray) -> np.ndarray:
        """Apply channel on input samples.

        x_np: np.complex64 or complex128 numpy array shaped (batch, samples, tx)
        returns: np array shaped (batch, samples, rx)
        """
        import tensorflow as tf

        # Convert to TF complex tensor
        x_tf = tf.convert_to_tensor(x_np)
        # Ensure complex dtype
        if not x_tf.dtype.is_complex:
            real = tf.cast(x_tf, tf.float32)
            imag = tf.zeros_like(real)
            x_tf = tf.complex(real, imag)

        # Sionna channels typically expect shape (batch, symbols, tx)
        y_tf = self.channel_obj(x_tf)

        # Convert back to numpy
        y_np = y_tf.numpy()
        return y_np


def init_sionna_channel(config: dict) -> Optional[ChannelHandle]:
    """Initialize and return a ChannelHandle or None if unavailable.

    Minimal supported config keys:
    - "type": "rayleigh" (current minimal implementation)
    - "tx": int
    - "rx": int
    - "samples": int (symbols per call)
    """
    if not sionna_available():
        return None

    try:
        import tensorflow as tf
        from sionna.channel import RayleighChannel

        tx = int(config.get("tx", 1))
        rx = int(config.get("rx", 1))
        # Create a simple Rayleigh channel object
        ch = RayleighChannel(tx=tx, rx=rx, normalize=True)

        return ChannelHandle(ch, rx=rx, tx=tx)
    except Exception as e:
        # Return None on any failure to keep integration optional
        print("sionna_wrapper: failed to init Sionna channel:", e)
        return None
