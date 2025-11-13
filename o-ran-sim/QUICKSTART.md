# Quick Start Guide

## Installation and First Run

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd o-ran-sim

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Your First Simulation

#### Option A: GUI Mode (Interactive)

```bash
python main.py
```

This launches an interactive GUI where you can:
- Configure simulation parameters
- Select RL algorithm
- Visualize network topology
- See real-time metrics
- Plot results

**Basic Steps in GUI:**
1. Set number of Base Stations (e.g., 3)
2. Set number of User Equipment (e.g., 10)
3. Set simulation steps (e.g., 100)
4. Select RL agent (e.g., "Baseline" or "TabularQLearning")
5. Click "Run Simulation"
6. View results in the plots panel

#### Option B: Command Line (Batch)

```bash
# Run quick experiment with default settings
python final_runner.py
```

This will run a single simulation and save results to the `res/` directory.

### 3. Run Comprehensive Experiments

```bash
# Compare all algorithms on uniform placement
python run_all_experiments.py --placements uniform --steps 200

# Results will be saved in res/ directory with:
# - Individual agent metrics (JSON)
# - Aggregated comparison (CSV)
# - Visualization plots (PNG)
```

### 4. Run Tests

```bash
# Quick smoke test
python tests/test_baseline_smoke.py

# Or with pytest
python -m pytest tests/
```

## Understanding the Results

After running experiments, check the `res/` directory:

```
res/final_comparison_uniform_YYYYMMDD_HHMMSS/
├── aggregated_metrics.csv          # Summary of all agents
├── Baseline_metrics_*.json         # Detailed Baseline metrics
├── TabularQLearning_metrics_*.json # Detailed Q-Learning metrics
├── SARSA_metrics_*.json           # Detailed SARSA metrics
└── plots/
    ├── throughput_comparison.png   # Throughput over time
    ├── handover_comparison.png     # Handover counts
    └── fairness_comparison.png     # Fairness metrics
```

## Common Configurations

### Fast Test Run (Quick Validation)

```python
# In sim_core/params.py, set:
num_bss = 2
num_ues = 5
total_sim_steps = 50
```

### Standard Evaluation

```python
num_bss = 3
num_ues = 10
total_sim_steps = 200
```

### Large-Scale Simulation

```python
num_bss = 7
num_ues = 30
total_sim_steps = 500
```

## Customizing Parameters

Edit `sim_core/params.py` to customize:

```python
class SimParams:
    # Network topology
    num_bss = 3                    # Number of base stations
    num_ues = 10                   # Number of users
    placement_method = "uniform"   # "uniform" or "PPP"
    
    # Simulation
    total_sim_steps = 200          # Simulation duration
    
    # RL parameters
    learning_rate = 0.1            # Alpha
    discount_factor = 0.9          # Gamma
    epsilon_start = 0.3            # Initial exploration
    epsilon_decay = 0.995          # Exploration decay
    
    # Channel parameters
    carrier_freq_ghz = 2.0         # Carrier frequency
    path_loss_exponent = 3.5       # Path loss exponent
```

## Comparing RL Algorithms

The simulator includes:

1. **Baseline**: Greedy algorithm (highest SINR)
2. **Tabular Q-Learning**: Off-policy TD learning
3. **SARSA**: On-policy TD learning
4. **Expected SARSA**: On-policy with expectation
5. **N-Step SARSA**: N-step lookahead
6. **DQN**: Deep Q-Network (requires TensorFlow)

To compare all algorithms:

```bash
python run_all_experiments.py --placements uniform PPP --steps 200
```

## Troubleshooting

### Issue: TensorFlow not found

**Solution**: DQN is optional. To install:
```bash
pip install tensorflow>=2.12.0
```

### Issue: GUI doesn't launch

**Solution**: Ensure tkinter is installed:
```bash
# On Ubuntu/Debian
sudo apt-get install python3-tk

# On macOS (usually pre-installed)
# On Windows (comes with Python)
```

### Issue: Sionna channel models not working

**Solution**: Sionna is optional. To install:
```bash
pip install sionna>=0.14.0
```

## Next Steps

1. **Modify Parameters**: Edit `sim_core/params.py` for different scenarios
2. **Add New Agents**: See CONTRIBUTING.md for guidelines
3. **Analyze Results**: Use the generated plots and CSV files
4. **Run Experiments**: Try different network configurations

## Getting Help

- Check README.md for detailed documentation
- Read CONTRIBUTING.md for development guidelines
- Open an issue on GitHub for bugs or questions
- Review the code documentation and comments

## Example Workflow

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Quick test
python tests/test_baseline_smoke.py

# 3. Run GUI to understand the system
python main.py

# 4. Run batch experiments
python run_all_experiments.py --placements uniform --steps 200

# 5. Analyze results
cd res/final_comparison_uniform_*/
cat aggregated_metrics.csv
# Open plots in plots/ directory

# 6. Customize and re-run
# Edit sim_core/params.py
python run_all_experiments.py --placements PPP --steps 300
```

