# Sionna integration summary

This document lists all places where the Sionna integration sandbox (`sionna_enabled/`)
interacts with the original simulator and explains the rationale for each change.

IMPORTANT: All changes live inside `sionna_enabled/` and the original project files
are not modified. The integration uses runtime monkey-patching for experimentation.

Files added (sionna_enabled/)

- `sionna_wrapper.py`
  - Guards imports of TensorFlow and Sionna, exposing `sionna_available()` and
    `init_sionna_channel(config)` to create Sionna channel handles.
  - Purpose: allow optional Sionna usage without requiring TF at import time.

- `integration.py`
  - Provides `enable_sionna_fading(sim, cfg)` which creates a per-BS Sionna
    channel handle and monkey-patches `BaseStation.calculate_sinr_on_rb` for the
    lifetime of the `Simulation` instance.
  - Applies Sionna small-scale fading to the serving link and to each
    interfering BS (per-interferer fading). Path-loss, shadowing and noise are
    delegated to the original `ChannelModel` to preserve existing behavior.
  - Rationale: Non-invasive way to test Sionna-driven fading without changing
    the core simulator. It allows comparing original vs Sionna-influenced
    results side-by-side.

- `phy.py`
  - Provides `sinr_to_rate_bps()` which maps linear SINR to Mbps using an
    MCS-like spectral efficiency table. This is used by the sandbox for more
    realistic rate mapping when needed (can be swapped for Sionna PHY blocks
    later).

- `sionna_runner.py`
  - A sandbox-local replacement for per-agent simulation runs that constructs
    a `Simulation`, optionally calls `enable_sionna_fading`, runs steps, and
    returns `(metrics_history, params_dict)` where `params_dict` includes
    `use_sionna` metadata.

- `run_all_experiments_sionna.py`
  - Sandbox batch runner that mirrors the original `run_all_experiments.py` but
    supports an explicit `--sionna` flag and uses `sionna_runner` for per-agent
    runs, honoring the experiment spec in `experiment_spec.py`.

- `experiment_spec.py`
  - Encodes default placements, step counts, and repetitions. Used by the
    sandbox runner to run the experiments from the paper reproducibly.

- `README.md` and helper runner scripts
  - Documentation and convenience scripts for quick smoke tests.

Design decisions and rationale

- Sandbox-first approach: keeps the original simulator untouched and allows
  rapid experimentation. Once the Sionna wiring is validated, a gradual
  refactor (dependency injection) can be applied to the core code for
  production-quality integration.

- Per-BS Sionna channels: Each BaseStation receives its own Sionna channel
  handle. This lets us get per-interferer fading samples and improves
  realism compared to a single fading generator.

- Path-loss/shadowing retained in core: We keep deterministic large-scale
  quantities in the original `ChannelModel` and only replace small-scale
  fading with Sionna. This minimizes behavioral drift from original codepaths
  and isolates the effect of small-scale fading.

- Safeguards: All Sionna-invoking code uses guarded imports and falls back to
  unity fading if Sionna is not available, ensuring reproducibility and
  no-hard-failure behavior.

Next steps (recommended)

1. Replace the Shannon capacity mapping in `sim_core` with Sionna PHY blocks
   in the sandbox, evaluate differences, then optionally refactor core code
   to accept a pluggable PHY mapping provider.
2. Implement batched Sionna sampling for performance (important for large
   experiments and RL training).
3. Refactor `sim_core` to accept a fading-provider interface (clean DI) and
   remove monkey-patching for better maintainability.
4. Consider a TF-native training pipeline for the DQN if end-to-end
   differentiability is desired.

If you want, I can start implementing any of the recommended next steps in the
sandbox so we can evaluate impact before modifying the core code.
