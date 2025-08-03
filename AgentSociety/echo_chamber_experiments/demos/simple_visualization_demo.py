"""
Simple Working Visualization Demo for Dynamic Belief Evolution

This script demonstrates the core visualization capabilities that are working
and provides a reliable way to see your experimental results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Import experiment components
from experiments.dynamic_evolution.experiment import DynamicEvolutionExperiment, DynamicEvolutionConfig
from core.dynamic_parameters import CrisisType
from core.agent import TopicType


def create_and_run_demo_experiment():
    """Create and run a sample experiment"""
    
    print("🧪 Running Demo Experiment...")
    
    config = DynamicEvolutionConfig(
        name="Simple Demo",
        description="Basic visualization demo",
        num_agents=30,
        topic=TopicType.HEALTHCARE,
        num_rounds=12,
        crisis_scenario=CrisisType.PANDEMIC,
        crisis_severity=0.8,
        interactions_per_round=60,
        belief_history_tracking=True,
        random_seed=42
    )
    
    experiment = DynamicEvolutionExperiment(config)
    results = experiment.run_full_experiment()
    
    print("✅ Experiment completed successfully!")
    return results


def create_basic_visualizations(results):
    """Create basic but effective visualizations"""
    
    print("\n📊 Creating Visualizations...")
    
    # Create output directory
    save_path = Path("demo_outputs")
    save_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: Core Evolution Plot
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Polarization over time
    rounds = range(len(results.polarization_over_time))
    axes[0,0].plot(rounds, results.polarization_over_time, 
                   'o-', linewidth=3, markersize=6, color='darkred')
    axes[0,0].set_title('Population Polarization Evolution', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Round')
    axes[0,0].set_ylabel('Polarization')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Parameter evolution (if available)
    if results.parameter_evolution:
        for param, values in list(results.parameter_evolution.items())[:3]:  # Top 3 parameters
            axes[0,1].plot(rounds[:len(values)], values, 
                          'o-', linewidth=2, label=param.replace('_', ' ').title())
        axes[0,1].set_title('Parameter Evolution', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Round')
        axes[0,1].set_ylabel('Parameter Value')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Sample belief trajectories
    if results.belief_trajectories:
        sample_agents = list(results.belief_trajectories.keys())[:8]  # Sample 8 agents
        for agent_id in sample_agents:
            trajectory = results.belief_trajectories[agent_id]
            axes[1,0].plot(range(len(trajectory)), trajectory, 
                          alpha=0.7, linewidth=1.5)
        axes[1,0].set_title('Sample Agent Belief Trajectories', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Round')
        axes[1,0].set_ylabel('Belief Strength')
        axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Crisis impact summary
    if results.crisis_impact_metrics:
        metrics = list(results.crisis_impact_metrics.items())[:5]  # Top 5 metrics
        metric_names, metric_values = zip(*metrics)
        
        bars = axes[1,1].bar(range(len(metric_names)), metric_values, 
                            color='steelblue', alpha=0.7)
        axes[1,1].set_title('Crisis Impact Metrics', fontsize=14, fontweight='bold')
        axes[1,1].set_xticks(range(len(metric_names)))
        axes[1,1].set_xticklabels([name.replace('_', ' ').title() for name in metric_names], 
                                 rotation=45, ha='right')
        axes[1,1].set_ylabel('Impact Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                          f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig1.savefig(save_path / "core_evolution_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path / 'core_evolution_analysis.png'}")
    
    # Figure 2: Belief Distribution Heatmap
    if results.belief_trajectories:
        fig2, ax = plt.subplots(figsize=(12, 8))
        
        # Create belief matrix
        agent_ids = sorted(list(results.belief_trajectories.keys()))[:25]  # Top 25 agents
        num_rounds = len(results.polarization_over_time)
        
        belief_matrix = np.array([
            results.belief_trajectories[agent_id][:num_rounds] 
            for agent_id in agent_ids
        ])
        
        # Create heatmap
        im = ax.imshow(belief_matrix, cmap='RdBu_r', aspect='auto', 
                      vmin=-1, vmax=1, interpolation='bilinear')
        
        ax.set_title('Agent Belief Evolution Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('Simulation Round', fontsize=14)
        ax.set_ylabel('Agent ID', fontsize=14)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Belief Strength', fontsize=12)
        
        plt.tight_layout()
        fig2.savefig(save_path / "belief_evolution_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path / 'belief_evolution_heatmap.png'}")
    
    # Figure 3: Crisis Timeline
    fig3, ax = plt.subplots(figsize=(14, 6))
    
    # Main polarization line
    ax.plot(rounds, results.polarization_over_time, 
           'o-', linewidth=4, markersize=8, color='darkred', label='Polarization')
    
    # Add crisis phases
    total_rounds = len(rounds)
    pre_crisis = int(0.2 * total_rounds)
    crisis_peak = int(0.6 * total_rounds)
    
    ax.axvspan(0, pre_crisis, alpha=0.2, color='green', label='Pre-Crisis')
    ax.axvspan(pre_crisis, crisis_peak, alpha=0.2, color='red', label='Crisis Period')
    ax.axvspan(crisis_peak, total_rounds, alpha=0.2, color='blue', label='Recovery')
    
    # Annotations
    peak_pol = max(results.polarization_over_time)
    peak_round = results.polarization_over_time.index(peak_pol)
    ax.annotate(f'Peak: {peak_pol:.3f}', 
               xy=(peak_round, peak_pol), xytext=(peak_round+2, peak_pol+0.02),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=12, fontweight='bold')
    
    ax.set_title('Crisis-Driven Polarization Timeline', fontsize=16, fontweight='bold')
    ax.set_xlabel('Simulation Round', fontsize=14)
    ax.set_ylabel('Population Polarization', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3.savefig(save_path / "crisis_timeline.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path / 'crisis_timeline.png'}")
    
    return [fig1, fig2, fig3]


def create_summary_report(results):
    """Create a summary report"""
    
    print("\n📋 Creating Summary Report...")
    
    save_path = Path("demo_outputs")
    report_path = save_path / "visualization_summary.md"
    
    with open(report_path, 'w') as f:
        f.write("# Dynamic Belief Evolution - Visualization Summary\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Experiment Results\n\n")
        f.write(f"- **Agents:** {results.config.num_agents}\n")
        f.write(f"- **Rounds:** {results.config.num_rounds}\n")
        f.write(f"- **Crisis Type:** {results.config.crisis_scenario.value if results.config.crisis_scenario else 'None'}\n")
        f.write(f"- **Final Polarization:** {results.polarization_over_time[-1]:.3f}\n")
        f.write(f"- **Polarization Change:** {results.polarization_over_time[-1] - results.polarization_over_time[0]:+.3f}\n")
        f.write(f"- **Peak Polarization:** {max(results.polarization_over_time):.3f}\n\n")
        
        f.write("## Generated Visualizations\n\n")
        f.write("1. **core_evolution_analysis.png** - Multi-panel analysis overview\n")
        f.write("2. **belief_evolution_heatmap.png** - Agent belief evolution heatmap\n") 
        f.write("3. **crisis_timeline.png** - Crisis-driven polarization timeline\n\n")
        
        if results.crisis_impact_metrics:
            f.write("## Crisis Impact Metrics\n\n")
            for metric, value in results.crisis_impact_metrics.items():
                f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")
    
    print(f"✅ Saved: {report_path}")


def main():
    """Main demonstration function"""
    
    print("🎨 Simple Dynamic Belief Evolution Visualization Demo")
    print("=" * 60)
    
    # 1. Run experiment
    results = create_and_run_demo_experiment()
    
    # 2. Create visualizations
    figures = create_basic_visualizations(results)
    
    # 3. Create summary
    create_summary_report(results)
    
    print("\n" + "=" * 60)
    print("🎉 VISUALIZATION DEMO COMPLETE!")
    print("=" * 60)
    print("\n📁 Outputs saved to: demo_outputs/")
    print("\n🎯 Created Visualizations:")
    print("   📊 core_evolution_analysis.png - Multi-panel overview")
    print("   🔥 belief_evolution_heatmap.png - Agent belief heatmap")
    print("   📈 crisis_timeline.png - Crisis-driven timeline")
    print("   📋 visualization_summary.md - Summary report")
    
    print("\n✨ Your visualization system is working perfectly!")
    print("🔬 Use these as templates for your own experiments!")


if __name__ == "__main__":
    main()