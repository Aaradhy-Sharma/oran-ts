Sionna-enabled sandbox
======================

This folder contains a lightweight integration of the Sionna library into the
simulator without changing the original code. The idea is to provide a safe
place to experiment with Sionna channel models and to keep the base project
untouched.

Files
- `sionna_wrapper.py`: guarded imports and adapter to create Sionna channel
  objects and apply them to NumPy arrays.
- `runner.py`: a smoke runner that creates a `Simulation`, and when Sionna is
  available, attaches Sionna channel handles to the created BaseStation
  objects. The current patch is non-invasive: it monkey-patches methods for
  testing and doesn't replace core logic.

How to run
----------
1. Create a virtualenv and install TensorFlow + Sionna per your platform.
2. From the repository root run:

```bash
python sionna-enabled/runner.py
```

Notes
-----
- This is intentionally conservative: it does not fully replace the simulator's
  fading calculation yet. It provides a place to iterate quickly. If you want
  a deeper integration (direct Sionna fading inserted into `calculate_sinr_on_rb`),
  we can implement that next by refactoring `sim_core/entities.py` to accept an
  optional fading generator object.
