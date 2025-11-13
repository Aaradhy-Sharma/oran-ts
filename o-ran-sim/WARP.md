# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an O-RAN (Open Radio Access Network) simulation framework that implements reinforcement learning agents for intelligent traffic steering and handover optimization in cellular networks. The simulator models base stations (BSs), user equipment (UEs), radio resource blocks (RBs), and channel conditions to evaluate different RL algorithms for network optimization.

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Simulations
```bash
# Run GUI application (interactive simulation)
python main.py

# Run comprehensive batch comparison (all agents)
python final_runner.py

# Run with specific placement method
python final_runner.py --placement PPP --lambda-bs 0.5 --lambda-ue 2.0
python final_runner.py --placement uniform
```

### Testing Individual Components
```bash
# Test specific RL agent (example commands)
python -c "from rl_agents.dqn import DQNAgent; print('DQN agent can be imported')"
python -c "from sim_core.simulation import Simulation; from sim_core.params import SimParams; s = Simulation(SimParams(), print); print('Simulation initialized')"

# Quick parameter validation
python -c "from sim_core.params import SimParams; p = SimParams(); print(p.to_dict())"
```

### Development Workflow
```bash
# Check TensorFlow availability for DQN agents
python -c "from rl_agents.dqn import TF_AVAILABLE; print(f'TensorFlow available: {TF_AVAILABLE}')"

# Run single agent simulation programmatically
python -c "
from sim_core.params import SimParams
from sim_core.simulation import Simulation
params = SimParams()
params.total_sim_steps = 10
params.rl_agent_type = 'Baseline'
sim = Simulation(params, print)
for i in range(5): sim.run_step()
"
```

## Code Architecture

### Core Components

**sim_core/** - Main simulation engine
- `simulation.py`: Central orchestrator managing BSs, UEs, RL agents, and time steps
- `entities.py`: BaseStation and UserEquipment classes with mobility, measurements, and connectivity
- `params.py`: SimParams class containing all configurable parameters
- `channel.py`: Radio channel modeling (path loss, shadowing, SINR calculations)  
- `resource.py`: Resource Block pool management and allocation tracking

**rl_agents/** - Reinforcement learning implementations
- `base.py`: RLAgentBase abstract class defining agent interface
- `baseline.py`: Traditional handover algorithm (A3 condition + TTT)
- `dqn.py`: Deep Q-Network with prioritized experience replay and target networks
- `tabular_q.py`, `sarsa.py`, `expected_sarsa.py`, `nstep_sarsa.py`: Tabular RL methods

**utils/** - Supporting utilities
- `logger.py`: Logging infrastructure for GUI and batch runs
- `saver.py`: Results saving and data persistence

### Key Architectural Patterns

**Agent-Environment Interface**: RL agents receive network state (UE positions, SINR, load factors) and output handover/resource allocation decisions. The simulation applies these actions and provides rewards based on throughput, satisfaction, and handover costs.

**Dual Connectivity Support**: UEs can connect to two BSs simultaneously (serving_bs_1, serving_bs_2) with independent RB allocations, enabling load balancing and improved rates.

**Placement Flexibility**: Supports both uniform random placement and Poisson Point Process (PPP) spatial distributions for realistic network topologies.

**Metrics Pipeline**: Comprehensive metrics collection (throughput, SINR, handovers, satisfaction rates) with time-series storage for analysis and plotting.

## Agent-Specific Implementation Notes

### DQN Agent (`rl_agents/dqn.py`)
- Requires TensorFlow installation
- Uses prioritized experience replay and target networks
- Deep neural architecture with batch normalization and residual connections
- Handles TensorFlow import gracefully with `TF_AVAILABLE` flag

### Tabular Agents
- State discretization based on SINR bins and load factor categories
- Epsilon-greedy exploration with configurable decay schedules
- Q-table or policy table storage depending on algorithm (Q-learning vs SARSA variants)

### Baseline Agent
- Implements 3GPP-standard A3 handover condition
- Time-to-trigger (TTT) mechanism to prevent ping-pong handovers
- Greedy resource allocation prioritizing UEs below target throughput

## Configuration Management

Parameters are centralized in `SimParams` class with sensible defaults. Key configuration areas:

- **Network Topology**: `num_bss`, `num_ues`, `sim_area_x/y`, placement method
- **Radio Parameters**: TX power, path loss exponent, shadowing, RB bandwidth
- **RL Training**: Learning rates, epsilon decay, network architecture, replay buffer size
- **Performance Targets**: `target_ue_throughput_mbps`, satisfaction thresholds

## GUI vs Batch Execution

**GUI Mode** (`gui.py` → `main.py`): Interactive simulation with real-time visualization, parameter tuning, and single-agent runs with step-by-step execution.

**Batch Mode** (`final_runner.py`): Automated comparison of multiple agents with statistical analysis, research-quality plots, and results persistence to timestamped directories.

## Results and Analysis

Batch runs generate structured output in `FINAL/results/` with:
- Time-series metrics in CSV format per agent
- Comparative plots (throughput, satisfaction, handovers, rewards)
- Execution logs with detailed agent decisions

The simulation tracks key performance indicators:
- Average UE throughput and percentage of satisfied UEs
- Handover frequency (balancing performance vs stability)
- Base station load distribution and resource utilization
- SINR distributions and channel quality metrics

## Dependencies and Compatibility

- **Required**: numpy, matplotlib, scipy, pandas (scientific computing stack)
- **Optional**: tensorflow (enables DQN agents)
- **Python**: Tested on 3.12.7, compatible with 3.8+
- **Platform**: Cross-platform (Windows/macOS/Linux)

TensorFlow dependency is handled gracefully - DQN agents are disabled if unavailable, other agents remain functional.