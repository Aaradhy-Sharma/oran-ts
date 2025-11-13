"""Sionna-enabled integration package (sandbox).

This package contains an adapter that lets you experiment with Sionna channel
models without changing the original simulator code. Use the runner script to
launch a short simulation that monkey-patches BS channel handling to use the
Sionna-backed fading samples where available.
"""

__all__ = ["sionna_wrapper", "runner"]
