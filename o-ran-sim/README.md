# O-RAN RL Traffic Steering Simulator

A comprehensive simulation framework for evaluating Reinforcement Learning (RL) based traffic steering algorithms in Open Radio Access Network (O-RAN) environments.

## Overview

This project implements and compares various RL algorithms for optimizing traffic steering decisions in cellular networks. The simulator supports both uniform and Poisson Point Process (PPP) placement of base stations and user equipment, with optional integration of advanced channel models using Sionna.

## Features

- **Multiple RL Algorithms:**
  - Baseline (Greedy)
  - Tabular Q-Learning
  - SARSA
  - Expected SARSA
  - N-Step SARSA
  - DQN (Deep Q-Network) - requires TensorFlow
  - **DQNFixed** (Improved DQN with better stability) - requires TensorFlow

- **Flexible Network Topology:**
  - Uniform placement
  - Poisson Point Process (PPP) placement
  - Configurable number of base stations and user equipment

- **Advanced Channel Modeling:**
  - Basic path loss with shadowing
  - Optional Sionna integration for realistic channel models

- **Comprehensive Evaluation:**
  - Throughput analysis
  - Handover metrics
  - Fairness evaluation (Jain's fairness index)
  - Resource utilization tracking

## Directory Structure

```
o-ran-sim/
├── rl_agents/          # RL algorithm implementations
│   ├── base.py         # Base RL agent interface
│   ├── baseline.py     # Greedy baseline agent
│   ├── tabular_q.py    # Tabular Q-Learning
│   ├── sarsa.py        # SARSA algorithm
│   ├── expected_sarsa.py  # Expected SARSA
│   ├── nstep_sarsa.py  # N-Step SARSA
│   ├── dqn.py          # Deep Q-Network (requires TensorFlow)
│   └── dqn_fixed.py    # Improved DQN with better stability
├── sim_core/           # Core simulation components
│   ├── channel.py      # Channel modeling
│   ├── entities.py     # Base Station and UE entities
│   ├── simulation.py   # Main simulation engine
│   ├── resource.py     # Resource block management
│   ├── params.py       # Simulation parameters
│   └── constants.py    # Global constants
├── utils/              # Utility modules
│   ├── logger.py       # Logging utilities
│   └── saver.py        # Results saving utilities
├── sionna_enabled/     # Sionna integration (optional)
│   ├── sionna_wrapper.py  # Sionna adapter
│   ├── phy.py          # PHY layer utilities
│   └── runner.py       # Sionna experiment runner
├── tests/              # Test suite
│   └── test_baseline_smoke.py
├── report/             # Mid-term report LaTeX files
├── main.py             # GUI application entry point
├── gui.py              # Tkinter-based GUI
├── final_runner.py     # Batch simulation runner
├── dqnFixedRunner.py   # DQN comparison experiment runner
├── run_all_experiments.py  # Comprehensive experiment suite
└── requirements.txt    # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd o-ran-sim
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Optional: Install TensorFlow for DQN

The DQN agent requires TensorFlow. If not installed, the simulator will automatically skip DQN experiments:

```bash
pip install tensorflow>=2.12.0
```

### Optional: Install Sionna for Advanced Channel Modeling

For realistic channel models using Sionna:

```bash
pip install sionna>=0.14.0
```

## Usage

### GUI Mode

Launch the interactive GUI for single simulations with real-time visualization:

```bash
python main.py
```

### Batch Experiments

Run comprehensive experiments comparing all algorithms:

```bash
# Run for uniform placement
python run_all_experiments.py --placements uniform --steps 200

# Run for PPP placement
python run_all_experiments.py --placements PPP --steps 200

# Run for both placements
python run_all_experiments.py --placements uniform PPP --steps 200
```

### DQN Performance Comparison

Compare original DQN vs improved DQN (DQNFixed) vs Baseline:

```bash
# Compare all DQN variants with uniform placement
python dqnFixedRunner.py --placement uniform --steps 200 --runs 3

# Compare with PPP placement
python dqnFixedRunner.py --placement PPP --steps 300 --runs 5

# Compare specific agents
python dqnFixedRunner.py --agents Baseline DQNFixed --steps 200
```

This will generate:
- Time-series plots (throughput, satisfaction, handovers, fairness, loss)
- Final metrics comparison bar charts  
- Summary CSV with statistics
- Individual run metrics in JSON format

### Single Algorithm Test

Run a specific RL algorithm:

```bash
python final_runner.py
```

### Run Tests

Execute the test suite:

```bash
python -m pytest tests/
# or
python tests/test_baseline_smoke.py
```

## Configuration

Simulation parameters can be configured in `sim_core/params.py`:

- **Network Parameters:**
  - `num_bss`: Number of base stations
  - `num_ues`: Number of user equipment
  - `placement_method`: "uniform" or "PPP"
  - `area_size`: Simulation area dimensions

- **Channel Parameters:**
  - `carrier_freq_ghz`: Carrier frequency
  - `path_loss_exponent`: Path loss exponent
  - `shadowing_std_dev_db`: Shadowing standard deviation

- **RL Parameters:**
  - `learning_rate`: Learning rate for RL algorithms
  - `discount_factor`: Discount factor (gamma)
  - `epsilon_start`: Initial exploration rate
  - `epsilon_decay`: Exploration decay rate

## Results

Simulation results are saved in the following directories:

- `res/`: Experiment results with metrics and plots
- `FINAL/`: Final simulation logs and outputs
- `res_test/`: Test run results

Each experiment generates:
- JSON files with per-agent metrics
- Aggregated CSV with comparative metrics
- Visualization plots (throughput, handovers, fairness)

## Key Metrics

- **System Throughput**: Total network throughput over time
- **Handover Count**: Number of handovers per UE
- **Fairness Index**: Jain's fairness index for resource allocation
- **Resource Utilization**: Average RB utilization per BS
- **Learning Curves**: Cumulative rewards for RL agents

## DQN Improvements (DQNFixed)

The improved DQN agent (`DQNFixed`) addresses several performance issues in the original implementation:

### Key Improvements:

1. **Running State Normalization**: Uses exponential moving average of mean/variance instead of per-sample normalization for consistent feature scaling
2. **Simplified Action Space**: Reduced from 72 to 27 actions for faster learning
3. **Better Reward Scaling**: Scales rewards by 0.1 without aggressive clipping
4. **Optimized Learning**: Constant learning rate (0.0003) instead of fast exponential decay
5. **Larger Batch Size**: 128 instead of 64 for more stable gradient updates
6. **Improved Target Updates**: Soft updates every step (tau=0.01) instead of hard updates every 1000 steps
7. **Prioritized Experience Replay**: With proper importance sampling weights
8. **Double DQN**: Reduces overestimation bias in Q-value estimates

### Performance Comparison:

Run the comparison script to see the improvements:

```bash
python dqnFixedRunner.py --placement uniform --steps 200 --runs 3
```

Expected improvements:
- **Higher throughput** due to better exploration-exploitation balance
- **Better satisfaction rates** from improved learning stability
- **Lower training loss** from optimized hyperparameters
- **Faster convergence** from simplified action space

## Contributing

Contributions are welcome! Please ensure:
1. Code follows PEP 8 style guidelines
2. All tests pass before submitting
3. New features include appropriate tests
4. Documentation is updated

## License

[Specify your license here]

## Citation

If you use this simulator in your research, please cite:

```
[Add citation details here]
```

## Acknowledgments

This project is part of research on RL-based traffic steering optimization in O-RAN environments.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## References

1. O-RAN Alliance. "O-RAN Architecture Description," 2020.
2. R. S. Sutton and A. G. Barto, "Reinforcement Learning: An Introduction," 2nd ed., MIT Press, 2018.
3. C. J. C. H. Watkins and P. Dayan, "Q-learning," Machine Learning, vol. 8, pp. 279–292, 1992.
4. G. A. Rummery and M. Niranjan, "On-line Q-learning using connectionist systems," Technical Report CUED/F-INFENG/TR 166, Cambridge University, 1994.
