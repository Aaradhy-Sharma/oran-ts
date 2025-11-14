# DQN Fixed Runner - Quick Reference

## Quick Test (Recommended First Run)

Test that everything works with a quick 50-step run:

```bash
python dqnFixedRunner.py --placement uniform --steps 50 --runs 1 --agents Baseline DQNFixed
```

## Standard Comparison

Compare all three agents (Baseline, DQN, DQNFixed) with 200 steps:

```bash
python dqnFixedRunner.py --placement uniform --steps 200 --runs 3
```

## Extended Evaluation

For more thorough results with PPP placement:

```bash
python dqnFixedRunner.py --placement PPP --steps 300 --runs 5
```

## Command Options

```
--placement [uniform|PPP]  # Network topology
--steps N                  # Simulation steps (default: 200)
--runs N                   # Runs per agent (default: 3)
--agents AGENT1 AGENT2     # Specific agents to test
```

## Available Agents

- `Baseline` - Greedy baseline (no learning)
- `DQN` - Original DQN implementation
- `DQNFixed` - Improved DQN with better stability

## Output Location

Results are saved to: `FINAL/dqn_comparison/dqn_comparison_YYYYMMDD_HHMMSS/`

Contains:
- `plots/comprehensive_comparison.png` - 6-panel time-series comparison
- `plots/final_metrics_comparison.png` - Bar charts of final metrics
- `summary_metrics.csv` - Statistical summary
- `metrics/*.json` - Individual run data

## Expected Runtime

- Quick test (50 steps, 1 run): ~30 seconds
- Standard (200 steps, 3 runs): ~5-10 minutes
- Extended (300 steps, 5 runs): ~15-25 minutes

## Troubleshooting

### TensorFlow not found
```bash
pip install tensorflow>=2.12.0
```

### Out of memory
Reduce steps or runs:
```bash
python dqnFixedRunner.py --steps 100 --runs 1
```

### Want to test only DQNFixed
```bash
python dqnFixedRunner.py --agents Baseline DQNFixed --steps 200 --runs 3
```

## Example Output

```
SUMMARY STATISTICS
================================================================================

Baseline:
  Avg Throughput: 45.23 ± 2.1 Mbps
  Satisfaction Rate: 78.5% ± 3.2%
  Fairness Index: 0.842 ± 0.023
  Total Handovers: 12.3 ± 2.1

DQN:
  Avg Throughput: 42.15 ± 5.3 Mbps
  Satisfaction Rate: 72.1% ± 8.4%
  Fairness Index: 0.801 ± 0.042
  Total Handovers: 18.7 ± 4.2

DQNFixed:
  Avg Throughput: 48.92 ± 1.8 Mbps
  Satisfaction Rate: 82.3% ± 2.1%
  Fairness Index: 0.867 ± 0.015
  Total Handovers: 11.2 ± 1.5
```

## Understanding the Improvements

DQNFixed should show:
- **Higher throughput** - Better resource allocation
- **Higher satisfaction** - More UEs meeting QoS targets
- **Better fairness** - More equitable resource distribution
- **Lower variance** - More stable learning
- **Lower loss** - Better training convergence
