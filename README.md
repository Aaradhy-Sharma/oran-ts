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
    *   **Deep Q-Network (DQN):** A deep reinforcement learning agent using neural networks for Q-value approximation (requires TensorFlow).
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

## Project Structure

```
o-ran-sim/
├── main.py                   # Entry point for the GUI application
├── runner.py                 # Automated experiment execution and analysis
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

You will be prompted at the end to decide whether to save the generated plots.

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

## Output Structure (`automated_sim_results` directory)

When `runner.py` is executed, it creates a dedicated directory named `automated_sim_results/` in the project root. Inside this, a unique timestamped subdirectory is created for each complete experiment run (e.g., `automated_sim_results/Experiment_Run_YYYYMMDD_HHMMSS/`). This ensures that results from different full experiment sessions do not overwrite each other.

Within each timestamped experiment run directory, you will find the following:

```
automated_sim_results/
└── Experiment_Run_YYYYMMDD_HHMMSS/
    ├── runner_log_YYYYMMDD_HHMMSS.log     # Comprehensive log of the entire experiment run
    ├── raw_metrics_per_run/               # Contains raw JSON data for each individual simulation run
    │   └── [scenario_name]_run[#]_seed[#]/ # Subdirectory for each specific run (scenario-agent-seed combo)
    │       └── [AgentType]_metrics_YYYYMMDD_HHMMSS.json  # Raw metrics and params for one simulation run
    ├── aggregated_csv_data/               # Contains aggregated CSV data per scenario
    │   └── [ScenarioName]_aggregated_metrics.csv # Aggregated performance data for all agents in a scenario
    └── comparison_plots/                  # Contains saved comparison plots (PNG format)
        └── Experiment_Run_YYYYMMDD_HHMMSS_[ScenarioName]_performance_comparison.png # Plot for each scenario
```

### Accessing the Data

1.  **Full Experiment Log (`runner_log_*.log`):**
    *   This file provides a detailed chronological account of the entire automated experiment process. It includes start/end times for each run, parameter overrides, and any warnings or errors encountered. Useful for debugging or reviewing the exact sequence of events.

2.  **Raw Metrics per Run (`raw_metrics_per_run/`):**
    *   Each JSON file in this subdirectory represents the complete metrics history (`metrics_history`) and the exact simulation parameters (`simulation_parameters`) for a *single* agent running under a *specific* scenario with a *single* random seed.
    *   These JSON files are ideal if you need to re-process the data, perform custom analysis on individual runs, or inspect the exact configuration that led to a particular result. They can be opened with any text editor or JSON viewer, or loaded into Python using `json.load()`.

3.  **Aggregated CSV Data (`aggregated_csv_data/`):**
    *   For each experiment scenario, a `.csv` file is generated (e.g., `Default_Config_aggregated_metrics.csv`).
    *   These CSV files contain the **mean performance metrics over time (averaged across all random seeds)** for all agents within that specific scenario.
    *   Each row typically represents a time step, and columns include `time_step`, various KPIs (e.g., `avg_ue_throughput_mbps`, `reward`), and an `Agent` column indicating which agent the data belongs to.
    *   CSV files are highly versatile and can be directly opened in spreadsheet software (Microsoft Excel, Google Sheets, LibreOffice Calc) for quick visualization, filtering, and statistical analysis. They can also be easily loaded into Python using `pandas.read_csv()` for advanced data manipulation.

4.  **Comparison Plots (`comparison_plots/`):**
    *   These are high-resolution PNG image files, with one file for each experiment scenario.
    *   Each plot visually compares the performance trends of all tested RL agents (and the baseline) across multiple KPIs over the simulation duration for that specific scenario.
    *   These are excellent for quick visual comparison and for including directly in reports or presentations.

This structured output facilitates comprehensive analysis of the RL agents' performance, allowing researchers to draw robust conclusions from the simulation results.

## Contact

*   Aaradhy Sharma - as783@snu.edu.in // tc.shadical@gmail.com 

