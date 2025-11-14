#!/usr/bin/env python3
"""
DQN Fixed Comparison Runner

This script compares the performance of:
1. Baseline (Greedy) Agent
2. Original DQN Agent
3. Fixed DQN Agent

It runs multiple experiments and generates comprehensive comparison plots.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import json
import pandas as pd
from tqdm import tqdm
import argparse

# Import core simulation components
from sim_core.params import SimParams
from sim_core.simulation import Simulation
from sim_core.constants import TF_AVAILABLE as TF_AVAILABLE_GLOBAL

# Import utilities
from utils.saver import SaveHandler


def setup_output_dir(base_dir="FINAL/dqn_comparison"):
    """Setup output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"dqn_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    return output_dir


def create_params(placement_method="uniform", total_steps=200):
    """Create simulation parameters."""
    params = SimParams()
    params.placement_method = placement_method
    params.total_sim_steps = total_steps
    params.num_bss = 3
    params.num_ues = 10
    
    # Set specific values for PPP
    if placement_method == "PPP":
        params.lambda_bs = 3.0  # Expected 3 BSs per km^2
        params.lambda_ue = 10.0  # Expected 10 UEs per km^2
    
    # RL parameters
    params.rl_learning_rate = 0.0003
    params.rl_gamma = 0.99
    params.rl_epsilon_start = 1.0
    params.rl_epsilon_end = 0.05
    params.rl_epsilon_decay_steps = total_steps * 5  # Decay over 5 episodes
    params.rl_batch_size = 128
    params.rl_replay_buffer_size = 50000
    params.rl_target_update_freq = 500
    params.rl_use_soft_updates = True
    params.rl_tau = 0.01
    
    return params


def calculate_fairness_index(throughputs):
    """Calculate Jain's fairness index."""
    if not throughputs or len(throughputs) == 0:
        return 0.0
    throughputs = np.array([t for t in throughputs if t > 0])
    if len(throughputs) == 0:
        return 0.0
    n = len(throughputs)
    sum_throughputs = np.sum(throughputs)
    sum_squares = np.sum(throughputs ** 2)
    if sum_squares == 0:
        return 0.0
    return (sum_throughputs ** 2) / (n * sum_squares)


def run_single_experiment(agent_type: str, params: SimParams, run_seed: int, output_dir: str):
    """Run a single experiment with a specific agent."""
    # Set random seeds
    random.seed(run_seed)
    np.random.seed(run_seed)
    if TF_AVAILABLE_GLOBAL:
        import tensorflow as tf
        tf.random.set_seed(run_seed)
    
    # Create log function
    log_messages = []
    def log_func(msg):
        log_messages.append(msg)
    
    # Set agent type
    params.rl_agent_type = agent_type
    
    print(f"\n{'='*60}")
    print(f"Running: {agent_type} (Seed: {run_seed})")
    print(f"Placement: {params.placement_method}")
    print(f"{'='*60}")
    
    # Run simulation
    sim = Simulation(params, log_func)
    
    # Track metrics over time
    episode_metrics = {
        'timestep': [],
        'avg_throughput': [],
        'satisfaction_rate': [],
        'num_handovers': [],
        'fairness_index': [],
        'avg_reward': [],
        'epsilon': []
    }
    
    pbar = tqdm(total=params.total_sim_steps, desc=f"{agent_type}")
    for step in range(params.total_sim_steps):
        if not sim.run_step():
            break
        pbar.update(1)
    pbar.close()
    
    # Extract metrics from simulation history
    for metrics in sim.metrics_history:
        # Calculate fairness index from UE throughputs
        ue_throughputs = []
        for ue in sim.ues.values():
            if ue.serving_bs_1 or ue.serving_bs_2:
                ue_throughputs.append(ue.current_total_rate_mbps)
        fairness = calculate_fairness_index(ue_throughputs)
        
        episode_metrics['timestep'].append(metrics['time_step'])
        episode_metrics['avg_throughput'].append(metrics.get('avg_ue_throughput_mbps', 0))
        episode_metrics['satisfaction_rate'].append(
            metrics.get('percentage_satisfied_ues', 0) / 100.0
        )
        episode_metrics['num_handovers'].append(metrics.get('total_handovers_cumulative', 0))
        episode_metrics['fairness_index'].append(fairness)
        episode_metrics['avg_reward'].append(metrics.get('reward', 0))
        
        # Get epsilon if available
        if hasattr(sim.rl_agent, 'current_epsilon'):
            episode_metrics['epsilon'].append(sim.rl_agent.current_epsilon)
        else:
            episode_metrics['epsilon'].append(0.0)
    
    # Calculate final metrics from last entries in history
    if sim.metrics_history:
        last_metrics = sim.metrics_history[-1]
        
        # Calculate final fairness
        ue_throughputs = []
        for ue in sim.ues.values():
            if ue.serving_bs_1 or ue.serving_bs_2:
                ue_throughputs.append(ue.current_total_rate_mbps)
        final_fairness = calculate_fairness_index(ue_throughputs)
        
        final_metrics = {
            'avg_throughput_mbps': last_metrics.get('avg_ue_throughput_mbps', 0),
            'avg_satisfaction': last_metrics.get('percentage_satisfied_ues', 0) / 100.0,
            'fairness_index': final_fairness,
            'total_handovers': sim.total_handovers_cumulative
        }
    else:
        final_metrics = {
            'avg_throughput_mbps': 0,
            'avg_satisfaction': 0,
            'fairness_index': 0,
            'total_handovers': 0
        }
    
    # Add agent-specific metrics
    if hasattr(sim.rl_agent, 'episode_losses') and sim.rl_agent.episode_losses:
        final_metrics['avg_loss'] = np.mean(sim.rl_agent.episode_losses[-100:])
        final_metrics['final_loss'] = sim.rl_agent.episode_losses[-1] if sim.rl_agent.episode_losses else 0.0
        episode_metrics['losses'] = sim.rl_agent.episode_losses
    
    # Save metrics
    results = {
        'agent_type': agent_type,
        'run_seed': run_seed,
        'placement_method': params.placement_method,
        'final_metrics': final_metrics,
        'time_series': episode_metrics,
        'log_messages': log_messages[-100:]  # Last 100 messages
    }
    
    # Save to file
    metrics_file = os.path.join(output_dir, "metrics", f"{agent_type}_{run_seed}.json")
    with open(metrics_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\n{agent_type} Results:")
    print(f"  Avg Throughput: {final_metrics.get('avg_throughput_mbps', 0):.2f} Mbps")
    print(f"  Satisfaction Rate: {final_metrics.get('avg_satisfaction', 0):.2%}")
    print(f"  Total Handovers: {sim.total_handovers_cumulative}")
    print(f"  Fairness Index: {final_metrics.get('fairness_index', 0):.3f}")
    if 'avg_loss' in final_metrics:
        print(f"  Avg Loss (last 100): {final_metrics['avg_loss']:.4f}")
    
    return results


def plot_comparison(all_results, output_dir):
    """Generate comprehensive comparison plots."""
    print("\nGenerating comparison plots...")
    
    # Organize results by agent type
    agent_results = {}
    for result in all_results:
        agent_type = result['agent_type']
        if agent_type not in agent_results:
            agent_results[agent_type] = []
        agent_results[agent_type].append(result)
    
    # Define colors for each agent
    colors = {
        'Baseline': '#2E86AB',
        'DQN': '#A23B72',
        'DQNFixed': '#F18F01',
        'DQN_Fixed': '#F18F01'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DQN Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Throughput over time
    ax = axes[0, 0]
    for agent_type, results in agent_results.items():
        # Average across runs
        all_timesteps = []
        all_throughputs = []
        for result in results:
            ts = result['time_series']
            all_timesteps.append(ts['timestep'])
            all_throughputs.append(ts['avg_throughput'])
        
        # Average and plot
        if all_timesteps:
            min_len = min(len(t) for t in all_timesteps)
            timesteps = all_timesteps[0][:min_len]
            throughputs = np.mean([t[:min_len] for t in all_throughputs], axis=0)
            ax.plot(timesteps, throughputs, label=agent_type, color=colors.get(agent_type, 'gray'), linewidth=2)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Avg Throughput (Mbps)')
    ax.set_title('Average Throughput Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Satisfaction rate over time
    ax = axes[0, 1]
    for agent_type, results in agent_results.items():
        all_timesteps = []
        all_satisfaction = []
        for result in results:
            ts = result['time_series']
            all_timesteps.append(ts['timestep'])
            all_satisfaction.append(ts['satisfaction_rate'])
        
        if all_timesteps:
            min_len = min(len(t) for t in all_timesteps)
            timesteps = all_timesteps[0][:min_len]
            satisfaction = np.mean([s[:min_len] for s in all_satisfaction], axis=0)
            ax.plot(timesteps, satisfaction, label=agent_type, color=colors.get(agent_type, 'gray'), linewidth=2)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Satisfaction Rate')
    ax.set_title('Satisfaction Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative handovers
    ax = axes[0, 2]
    for agent_type, results in agent_results.items():
        all_timesteps = []
        all_handovers = []
        for result in results:
            ts = result['time_series']
            all_timesteps.append(ts['timestep'])
            all_handovers.append(ts['num_handovers'])
        
        if all_timesteps:
            min_len = min(len(t) for t in all_timesteps)
            timesteps = all_timesteps[0][:min_len]
            handovers = np.mean([h[:min_len] for h in all_handovers], axis=0)
            ax.plot(timesteps, handovers, label=agent_type, color=colors.get(agent_type, 'gray'), linewidth=2)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Handovers')
    ax.set_title('Cumulative Handovers Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Fairness index over time
    ax = axes[1, 0]
    for agent_type, results in agent_results.items():
        all_timesteps = []
        all_fairness = []
        for result in results:
            ts = result['time_series']
            all_timesteps.append(ts['timestep'])
            all_fairness.append(ts['fairness_index'])
        
        if all_timesteps:
            min_len = min(len(t) for t in all_timesteps)
            timesteps = all_timesteps[0][:min_len]
            fairness = np.mean([f[:min_len] for f in all_fairness], axis=0)
            ax.plot(timesteps, fairness, label=agent_type, color=colors.get(agent_type, 'gray'), linewidth=2)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Fairness Index')
    ax.set_title('Fairness Index Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Average reward over time (for RL agents)
    ax = axes[1, 1]
    for agent_type, results in agent_results.items():
        if agent_type == 'Baseline':
            continue
        all_timesteps = []
        all_rewards = []
        for result in results:
            ts = result['time_series']
            all_timesteps.append(ts['timestep'])
            all_rewards.append(ts['avg_reward'])
        
        if all_timesteps:
            min_len = min(len(t) for t in all_timesteps)
            timesteps = all_timesteps[0][:min_len]
            rewards = np.mean([r[:min_len] for r in all_rewards], axis=0)
            ax.plot(timesteps, rewards, label=agent_type, color=colors.get(agent_type, 'gray'), linewidth=2)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Loss comparison (for DQN agents)
    ax = axes[1, 2]
    for agent_type, results in agent_results.items():
        if agent_type == 'Baseline':
            continue
        all_losses = []
        for result in results:
            ts = result['time_series']
            if 'losses' in ts and ts['losses']:
                all_losses.append(ts['losses'])
        
        if all_losses:
            min_len = min(len(l) for l in all_losses)
            losses = np.mean([l[:min_len] for l in all_losses], axis=0)
            # Plot smoothed losses
            window = 50
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=agent_type, color=colors.get(agent_type, 'gray'), linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (smoothed)')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "plots", "comprehensive_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {plot_file}")
    plt.close()
    
    # Create bar chart for final metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Final Metrics Comparison', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('avg_throughput_mbps', 'Average Throughput (Mbps)'),
        ('avg_satisfaction', 'Satisfaction Rate'),
        ('fairness_index', 'Fairness Index'),
        ('total_handovers', 'Total Handovers')
    ]
    
    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        agent_names = []
        metric_values = []
        metric_stds = []
        
        for agent_type, results in agent_results.items():
            values = []
            for result in results:
                fm = result['final_metrics']
                if metric_key == 'total_handovers':
                    # Special handling for handovers
                    if 'num_handovers' in result['time_series']:
                        values.append(result['time_series']['num_handovers'][-1])
                else:
                    values.append(fm.get(metric_key, 0))
            
            if values:
                agent_names.append(agent_type)
                metric_values.append(np.mean(values))
                metric_stds.append(np.std(values))
        
        x_pos = np.arange(len(agent_names))
        bars = ax.bar(x_pos, metric_values, yerr=metric_stds, capsize=5,
                     color=[colors.get(name, 'gray') for name in agent_names],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_names, rotation=45, ha='right')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "plots", "final_metrics_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved final metrics plot: {plot_file}")
    plt.close()


def create_summary_report(all_results, output_dir):
    """Create summary CSV report."""
    print("\nCreating summary report...")
    
    summary_data = []
    for result in all_results:
        agent_type = result['agent_type']
        run_seed = result['run_seed']
        fm = result['final_metrics']
        ts = result['time_series']
        
        row = {
            'Agent': agent_type,
            'Seed': run_seed,
            'Placement': result['placement_method'],
            'Avg Throughput (Mbps)': fm.get('avg_throughput_mbps', 0),
            'Satisfaction Rate': fm.get('avg_satisfaction', 0),
            'Fairness Index': fm.get('fairness_index', 0),
            'Total Handovers': ts['num_handovers'][-1] if ts['num_handovers'] else 0,
            'Final Epsilon': ts['epsilon'][-1] if ts['epsilon'] else 0,
        }
        
        if 'avg_loss' in fm:
            row['Avg Loss (last 100)'] = fm['avg_loss']
            row['Final Loss'] = fm['final_loss']
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_file = os.path.join(output_dir, "summary_metrics.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved summary report: {csv_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    for agent_type in df['Agent'].unique():
        agent_df = df[df['Agent'] == agent_type]
        print(f"\n{agent_type}:")
        print(f"  Avg Throughput: {agent_df['Avg Throughput (Mbps)'].mean():.2f} ± {agent_df['Avg Throughput (Mbps)'].std():.2f} Mbps")
        print(f"  Satisfaction Rate: {agent_df['Satisfaction Rate'].mean():.2%} ± {agent_df['Satisfaction Rate'].std():.2%}")
        print(f"  Fairness Index: {agent_df['Fairness Index'].mean():.3f} ± {agent_df['Fairness Index'].std():.3f}")
        print(f"  Total Handovers: {agent_df['Total Handovers'].mean():.1f} ± {agent_df['Total Handovers'].std():.1f}")
        if 'Avg Loss (last 100)' in agent_df.columns and not agent_df['Avg Loss (last 100)'].isna().all():
            print(f"  Avg Loss: {agent_df['Avg Loss (last 100)'].mean():.4f} ± {agent_df['Avg Loss (last 100)'].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Compare DQN agent variants')
    parser.add_argument('--placement', type=str, default='uniform', 
                       choices=['uniform', 'PPP'],
                       help='Placement method (uniform or PPP)')
    parser.add_argument('--steps', type=int, default=200,
                       help='Number of simulation steps')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per agent')
    parser.add_argument('--agents', nargs='+', 
                       default=['Baseline', 'DQN', 'DQNFixed'],
                       help='Agents to test')
    args = parser.parse_args()
    
    print("="*80)
    print("DQN FIXED COMPARISON RUNNER")
    print("="*80)
    print(f"Placement: {args.placement}")
    print(f"Steps per run: {args.steps}")
    print(f"Runs per agent: {args.runs}")
    print(f"Agents: {', '.join(args.agents)}")
    print("="*80)
    
    # Check TensorFlow availability
    if not TF_AVAILABLE_GLOBAL and ('DQN' in args.agents or 'DQNFixed' in args.agents or 'DQN_Fixed' in args.agents):
        print("\nWARNING: TensorFlow not available. DQN agents will be skipped.")
        args.agents = [a for a in args.agents if a not in ['DQN', 'DQNFixed', 'DQN_Fixed']]
    
    # Setup output directory
    output_dir = setup_output_dir()
    print(f"\nOutput directory: {output_dir}")
    
    # Create parameters
    params = create_params(args.placement, args.steps)
    
    # Run experiments
    all_results = []
    start_time = time.time()
    
    for agent_type in args.agents:
        for run_idx in range(args.runs):
            seed = 42 + run_idx
            result = run_single_experiment(agent_type, params, seed, output_dir)
            all_results.append(result)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"All experiments completed in {elapsed_time/60:.1f} minutes")
    print(f"{'='*80}")
    
    # Generate plots
    plot_comparison(all_results, output_dir)
    
    # Create summary report
    create_summary_report(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
