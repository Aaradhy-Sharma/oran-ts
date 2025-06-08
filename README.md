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
*   **Automated Experiment Runner:** A powerful `runner.py` script for:
    *   Defining multiple experiment scenarios with varying network parameters (e.g., density, mobility, resource scarcity).
    *   Running all specified RL agents across these scenarios for multiple random seeds.
    *   Aggregating performance metrics over multiple runs for statistical significance.
    *   Generating comparative plots for key performance indicators (KPIs).
    *   Saving raw metrics (JSON) and aggregated data (CSV) for detailed offline analysis.
    *   Saving generated plots to disk.
*   **Command-Line Interface:** A new `run_comparison.py` script for:
    *   Quick comparison of all agents with configurable parameters.
    *   Support for both uniform and PPP placement methods.
    *   Progress tracking with tqdm.
    *   Comprehensive logging and visualization.
    *   Research-quality plots for each metric.

## Project Structure

```
o-ran-sim/
├── main.py                   # Entry point for the GUI application
├── runner.py                 # Automated experiment execution and analysis
├── run_comparison.py         # Quick comparison script with CLI
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

There are three primary ways to run the simulation:

### 1. Interactive GUI Mode

For interactive exploration and visualization:

```bash
python main.py
```

This will launch the Tkinter GUI.
*   Adjust parameters on the left pane.
*   Click "Setup Sim" to initialize the network.
*   Use "Run Step" to advance the simulation one time step at a time.
*   Use "Run All Steps" to run the simulation to completion.
*   The visualization panel will update in real-time.
*   The log panel provides detailed information.

### 2. Automated Experiment Runner Mode

For systematic evaluation and generating comparative results:

```bash
python runner.py
```

This script will:
*   Execute predefined experiment scenarios (e.g., varying UE/BS density, mobility).
*   Run each specified RL agent multiple times per scenario with different random seeds for statistical robustness.
*   Log the progress and any errors to a file in the `automated_sim_results` directory.
*   Aggregate the metrics for each agent across multiple runs.
*   Generate comparative plots (displayed and optionally saved).
*   Save raw per-run metrics (JSON) and aggregated per-scenario metrics (CSV) to the `automated_sim_results` directory.

### 3. Quick Comparison Mode

For quick comparison of all agents with configurable parameters:

```bash
# Run with uniform placement (default)
python run_comparison.py

# Run with PPP placement
python run_comparison.py --placement PPP --lambda-bs 0.5 --lambda-ue 2.0
```

This script provides:
*   Command-line interface for parameter configuration
*   Support for both uniform and PPP placement methods
*   Progress tracking with tqdm
*   Comprehensive logging
*   Research-quality plots for each metric
*   Results saved in `simulation_results/` directory

## Understanding the Experiment Scenarios in `runner.py`

The `runner.py` script includes a `EXPERIMENT_SCENARIOS` dictionary that defines various test conditions. Each scenario is a dictionary of `SimParams` attributes that override the `base_sim_params`. This allows for flexible testing of how different agents perform under specific stresses.

Examples of included scenarios:

*   **`Default_Config`**: Baseline parameters for comparison.
*   **`High_Density_Dynamic`**: Tests performance under high UE and BS density with increased mobility and channel variability. Designed to challenge tabular methods due to large state space and highlight generalization of DRL.
*   **`BS_Hotspot_Uneven_Load`**: Simulates a network where UEs cluster, potentially leading to uneven load distribution across BSs, testing load balancing strategies.
*   **`Sparse_Network_Challenging_Coverage`**: Evaluates performance in networks with fewer BSs and more severe channel conditions, stressing coverage maintenance.
*   **`High_Traffic_Low_Resources`**: Tests resource allocation efficiency when demand is high but available resources are limited.
*   **`Extreme_Mobility`**: Assesses adaptability and handover efficiency in very rapidly changing environments.

Feel free to modify the `EXPERIMENT_SCENARIOS` dictionary in `runner.py` to define your own test cases.

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
*   **DQN Loss:** (For DQN agent) Tracks the training loss of the neural network.

## Recent Improvements

### DQN Agent Enhancements
*   Sophisticated network architecture with residual connections
*   Improved loss tracking and visualization
*   Double Q-learning implementation
*   Learning rate scheduling with exponential decay
*   Enhanced state representation with more granular levels
*   Proper gradient clipping and L2 regularization
*   Prioritized experience replay

### Simulation Framework Updates
*   Support for both uniform and PPP placement methods
*   Improved progress tracking with tqdm
*   Enhanced logging system
*   Research-quality plots with improved styling
*   Better error handling and reporting
*   Command-line interface for quick comparisons

## Output Structure

The simulation generates output in two main directories:

### 1. `automated_sim_results/` (for `runner.py`)
[Previous detailed structure remains unchanged]

### 2. `simulation_results/` (for `run_comparison.py`)
```
simulation_results/
└── comparison_run_[placement]_YYYYMMDD_HHMMSS/
    ├── simulation_logs/                    # Log files for each run
    ├── metrics/                           # CSV files with detailed metrics
    └── plots/                             # Research-quality plots for each metric
```

## Contact

*   Aaradhy Sharma - as783@snu.edu.in // tc.shadical@gmail.com

