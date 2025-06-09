# Reinforcement Learning for Traffic Steering in Open Radio Access Networks (O-RAN)

## Project Overview

This repository hosts the simulation codebase accompanying the research paper titled **"Reinforcement Learning based traffic steering in Open Radio Access Network (ORAN)"** by **Arnab Gain, Iti Saha Misra, Aaradhy Sharma and Dr.Shankar K. Ghosh**.

The project implements a comprehensive simulation environment for a cellular network, specifically focusing on **traffic steering** and **resource allocation** within an O-RAN framework. It evaluates the performance of various Reinforcement Learning (RL) agents against a traditional baseline algorithm under diverse and challenging network conditions.

The core objectives addressed by the RL agents are:
*   Maximizing the number of User Equipments (UEs) that receive high-speed internet.
*   Optimizing the Base Station (BS) load distribution across the network.
*   Minimizing unnecessary handovers due to dynamic network conditions.

The simulation incorporates realistic channel models (path loss, shadowing, small-scale fading), UE mobility, and dual connectivity capabilities, providing a robust platform for comparative analysis of traffic steering algorithms.

## Features

*   **Modular Design:** Code is logically separated into `sim_core`, `rl_agents`, `utils`, and `gui` modules for clarity and maintainability. </br>

*   **Diverse RL Agents:** Includes implementations of:
    *   **Baseline Agent:** A traditional, hysteresis-based handover mechanism with greedy resource allocation.
    *   **Tabular Q-Learning:** A fundamental value-based RL algorithm.
    *   **SARSA:** An on-policy temporal-difference control algorithm.
    *   **Expected SARSA:** An off-policy variant of SARSA, using expected Q-values.
    *   **N-Step SARSA:** Generalizes SARSA to use n-step returns for potentially faster learning.
    *   **Deep Q-Network (DQN):** A sophisticated deep reinforcement learning agent using neural networks with residual connections, double Q-learning, and prioritized experience replay (requires TensorFlow).
    *   **NO_DECAY_DQN:** A variant of DQN with stable exploration rate.
*   **Realistic Network Modeling:**
    *   UE mobility with random walk model.
    *   Base Station (BS) and UE placement (Uniform Random or Poisson Point Process).
    *   Detailed channel model (Path Loss, Log-Normal Shadowing, Rayleigh Small-Scale Fading).
    *   SINR calculation with inter-cell interference and thermal noise.
    *   Shannon capacity for data rate calculation.
    *   Resource Block (RB) allocation and management.
    *   Support for UE **dual connectivity**.
*   **Interactive GUI:** A Tkinter-based graphical user interface (`gui.py`) allows for:
    *   Configuration of all simulation parameters.
    *   Real-time visualization of the network state (BSs, UEs, connections, RBs).
    *   Step-by-step or full simulation execution.
    *   Logging of simulation events and metrics.
*   **Comprehensive Runner:** The `final_runner.py` script provides:
    *   Support for both uniform and PPP placement methods.
    *   Progress tracking with tqdm.
    *   Comprehensive logging and visualization.
    *   Research-quality plots for each metric.
    *   Results saved in `FINAL/results/` directory.

## Project Structure

```
o-ran-sim/
├── main.py                   # Entry point for the GUI application
├── final_runner.py           # Main script for running comprehensive simulations
├── sim_core/                 # Core simulation logic and entities
│   ├── __init__.py           # Package initializer
│   ├── constants.py          # Global constants (Boltzmann, Kelvin, TF_AVAILABLE)
│   ├── helpers.py            # Math utility functions (dB conversions, etc.)
│   ├── params.py             # SimParams class for all configuration parameters
│   ├── channel.py            # ChannelModel class (path loss, noise, Shannon capacity)
│   ├── resource.py           # ResourceBlockPool class (RB allocation)
│   ├── entities.py           # BaseStation and UserEquipment classes
│   └── simulation.py         # Main Simulation logic (runs steps, calculates rewards)
├── rl_agents/                # Implementations of various Reinforcement Learning agents
│   ├── __init__.py           # Package initializer
│   ├── base.py               # RLAgentBase abstract class
│   ├── baseline.py           # Baseline (classical) agent
│   ├── tabular_q.py          # Tabular Q-Learning agent
│   ├── sarsa.py              # SARSA agent
│   ├── expected_sarsa.py     # Expected SARSA agent
│   ├── nstep_sarsa.py        # N-Step SARSA agent
│   └── dqn.py                # Deep Q-Network agent (requires TensorFlow)
├── gui.py                    # Tkinter-based Graphical User Interface
├── utils/                    # Utility functions
│   ├── __init__.py           # Package initializer
│   ├── logger.py             # LogHandler for console, GUI, and file logging
│   └── saver.py              # SaveHandler for JSON, CSV, and plot saving
└── requirements.txt          # Python dependencies
LICENSE.md                    # License for the project
README.md                     # This file
```

## Getting Started

### Prerequisites

Before running the simulation, ensure you have Python 3.8+ installed. You can install the necessary libraries using `pip`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
numpy
matplotlib
scipy
pandas
tensorflow # Only required if you wish to run the DQN agent
tqdm       # For progress tracking
```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Aaradhy-Sharma/oran-ts.git 
    cd o-ran-sim
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ensure correct directory structure:**
    Verify that the file structure matches the "Project Structure" section above, with empty `__init__.py` files in `sim_core`, `rl_agents`, and `utils` directories.

## How to Run

There are two primary ways to run the simulation:

### 1. Interactive GUI Mode

For interactive exploration and visualization:

```bash
# Using python
python main.py

# Or using uv (faster alternative)
uv run main.py
```

This will launch the Tkinter GUI.
*   Adjust parameters on the left pane.
*   Click "Setup Sim" to initialize the network.
*   Use "Run Step" to advance the simulation one time step at a time.
*   Use "Run All Steps" to run the simulation to completion.
*   The visualization panel will update in real-time.
*   The log panel provides detailed information.

### 2. Comprehensive Simulation Mode

For running comprehensive simulations with all agents:

```bash
# Using python
python final_runner.py
# Or with PPP placement
python final_runner.py --placement PPP --lambda-bs 0.5 --lambda-ue 2.0

# Or using uv (faster alternative)
uv run final_runner.py
# Or with PPP placement
uv run final_runner.py --placement PPP --lambda-bs 0.5 --lambda-ue 2.0
```

The `final_runner.py` script provides:
*   Command-line interface for parameter configuration
*   Support for both uniform and PPP placement methods
*   Progress tracking with tqdm
*   Comprehensive logging
*   Research-quality plots for each metric
*   Results saved in `FINAL/results/` directory

## Key Performance Indicators (KPIs)

The simulation tracks several KPIs to evaluate agent performance:

*   **Average UE Throughput (Mbps):** Mean data rate achieved by connected UEs.
*   **% Satisfied UEs:** Percentage of connected UEs meeting their target throughput.
*   **Average BS Load Factor:** Metric indicating how balanced the load is across Base Stations (closer to 1.0 is better).
*   **Step Reward:** The reward received by the RL agent at each time step, reflecting overall network performance based on satisfaction, load, and handover penalties.
*   **Handovers per Step / Cumulative Handovers:** Measures the frequency of handovers, a critical factor for network stability and signaling overhead.
*   **Average SINR (dB):** Mean Signal-to-Interference-plus-Noise Ratio for connected UEs.
*   **Average RBs per UE:** Resources allocated per UE.
*   **Epsilon Decay:** (For RL agents) Visualizes the exploration-exploitation balance over time.

## Output Structure

The simulation generates output in the following directory structure:

```
FINAL/
├── logs/                    # Log files directory
│   └── final_simulation_YYYYMMDD_HHMMSS.log
└── results/                 # Results directory
    └── final_comparison_[placement]_YYYYMMDD_HHMMSS/
        ├── metrics/         # CSV files with detailed metrics
        └── plots/           # Research-quality plots for each metric
```

## Contact

*   Aaradhy Sharma - as783@snu.edu.in // tc.shadical@gmail.com

