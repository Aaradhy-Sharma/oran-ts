Sionna-enabled sandbox (package)
================================

This package provides a guarded Sionna adapter (`sionna_wrapper.py`) and a
`runner.py` that demonstrates how to patch the existing simulator to use
Sionna channel models. Run the runner as a quick smoke test:

```bash
python sionna_enabled/runner.py
```

The runner adds the project root to `sys.path` so imports of `sim_core` work
when running the script directly.
