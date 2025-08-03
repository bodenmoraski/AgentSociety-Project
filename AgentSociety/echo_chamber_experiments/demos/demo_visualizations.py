"""
Comprehensive Visualization Demo for Dynamic Belief Evolution

This script demonstrates all the visualization capabilities available
for dynamic belief evolution experiments, including:

1. Static Analysis Plots (matplotlib/seaborn)
2. Interactive Dashboards (plotly) 
3. Mathematical Model Visualizations
4. Crisis Impact Analysis
5. Agent Network Visualizations
6. Publication-Ready Figures

Usage:
    python demo_visualizations.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Import experiment and visualization components
try:
    from experiments.dynamic_evolution.experiment import DynamicEvolutionExperiment, DynamicEvolutionConfig
    from experiments.dynamic_evolution.visualizations import DynamicEvolutionVisualizer
    from experiments.dynamic_evolution.analysis import TrajectoryAnalyzer, CrisisAnalyzer
    from core.dynamic_parameters import CrisisType
    from core.agent import TopicType
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the echo_chamber_experiments directory")
    exit(1)


def create_sample_experiment() -> tuple:
    """Create a sample experiment for visualization demonstration"""
    
    print("🧪 Creating Sample Experiment for Visualization Demo...")
    
    # Enhanced configuration for better visualization
    config = DynamicEvolutionConfig(
        name="Visualization Demo: Pandemic Crisis",
        description="Sample experiment showcasing all visualization capabilities",
        num_agents=50,  # More agents for richer data
        topic=TopicType.HEALTHCARE,
        num_rounds=15,  # Longer for better dynamics
        crisis_scenario=CrisisType.PANDEMIC,
        crisis_severity=0.8,  # Strong crisis for clear effects
        interactions_per_round=100,
        belief_history_tracking=True,
        random_seed=42  # Reproducible
    )
    
    # Run experiment
    print("🏃 Running experiment (this may take a moment)...")
    start_time = time.time()
    
    experiment = DynamicEvolutionExperiment(config)
    results = experiment.run_full_experiment()
    
    duration = time.time() - start_time
    print(f"✅ Experiment completed in {duration:.2f}s")
    
    return config, results


def demonstrate_static_visualizations(results):
    """Demonstrate static matplotlib/seaborn visualizations"""
    
    print("\n📊 Creating Static Visualizations...")
    
    # Initialize visualizer
    visualizer = DynamicEvolutionVisualizer(results)
    
    # 1. Comprehensive Overview Dashboard
    print("   📈 Creating comprehensive overview...")
    fig_overview = visualizer.plot_dynamic_evolution_overview()
    
    # Save the figure
    save_path = Path("demo_outputs")
    save_path.mkdir(exist_ok=True)
    
    fig_overview.savefig(save_path / "dynamic_evolution_overview.png", 
                        dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path / 'dynamic_evolution_overview.png'}")
    
    # 2. Mathematical Model Analysis (sample agent)
    print("   🧮 Creating mathematical model analysis...")
    if results.belief_trajectories:
        sample_agent_id = list(results.belief_trajectories.keys())[0]
        fig_models = visualizer.plot_trajectory_model_comparison(sample_agent_id)
        fig_models.savefig(save_path / "trajectory_models.png", 
                          dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path / 'trajectory_models.png'}")
    else:
        print("   ⚠️ Skipping model analysis - no trajectory data")
        fig_models = None
    
    # 3. Crisis Impact Deep Dive
    print("   🌊 Creating crisis impact analysis...")
    try:
        fig_crisis = visualizer.plot_crisis_impact_analysis()
        fig_crisis.savefig(save_path / "crisis_impact_analysis.png", 
                          dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path / 'crisis_impact_analysis.png'}")
    except Exception as e:
        print(f"   ⚠️ Crisis analysis failed: {e}")
        fig_crisis = None
    
    # 4. Agent Correlation Network
    print("   👥 Creating agent correlation network...")
    try:
        fig_network = visualizer.plot_cross_agent_correlation_network()
        fig_network.savefig(save_path / "agent_correlation_network.png", 
                           dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path / 'agent_correlation_network.png'}")
    except Exception as e:
        print(f"   ⚠️ Network visualization failed: {e}")
        fig_network = None
    
    # Return only successful figures
    figures = [fig for fig in [fig_overview, fig_models, fig_crisis, fig_network] if fig is not None]
    return figures


def demonstrate_interactive_visualizations(results):
    """Demonstrate interactive plotly visualizations"""
    
    print("\n🌐 Creating Interactive Visualizations...")
    
    visualizer = DynamicEvolutionVisualizer(results)
    
    # Create interactive dashboard
    print("   📊 Creating interactive dashboard...")
    try:
        dashboard = visualizer.create_interactive_dashboard()
        
        if dashboard:
            save_path = Path("demo_outputs")
            dashboard.write_html(str(save_path / "interactive_dashboard.html"))
            print(f"   ✅ Saved: {save_path / 'interactive_dashboard.html'}")
            print("   🌐 Open this file in your browser for interactive exploration!")
            return dashboard
        else:
            print("   ⚠️ Plotly not available - skipping interactive visualizations")
            return None
            
    except Exception as e:
        print(f"   ❌ Interactive visualization failed: {e}")
        return None


def create_publication_ready_figures(results):
    """Create publication-quality figures with professional styling"""
    
    print("\n📝 Creating Publication-Ready Figures...")
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figure 1: Crisis Timeline with Parameter Evolution
    fig1, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top panel: Parameter evolution
    rounds = range(len(results.polarization_over_time))
    
    if results.parameter_evolution:
        for param, values in results.parameter_evolution.items():
            if param in ['polarization_strength', 'gap_size']:  # Key parameters
                axes[0].plot(rounds[:len(values)], values, 
                           linewidth=2.5, label=param.replace('_', ' ').title(),
                           marker='o', markersize=4)
    
    axes[0].set_ylabel('Parameter Value', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=12, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Crisis Parameter Evolution Over Time', fontsize=16, fontweight='bold')
    
    # Bottom panel: Polarization with crisis phases
    axes[1].plot(rounds, results.polarization_over_time, 
                linewidth=3, color='darkred', marker='o', markersize=5)
    
    # Add crisis phase annotations
    axes[1].axvspan(0, 3, alpha=0.2, color='green', label='Pre-Crisis')
    axes[1].axvspan(3, 8, alpha=0.2, color='red', label='Crisis Peak')
    axes[1].axvspan(8, len(rounds), alpha=0.2, color='blue', label='Recovery')
    
    axes[1].set_xlabel('Simulation Round', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Population Polarization', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=12, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Add crisis impact metrics
    if results.crisis_impact_metrics:
        impact = results.crisis_impact_metrics.get('polarization_increase', 0)
        axes[1].text(0.02, 0.98, f'Crisis Impact: +{impact:.3f}', 
                    transform=axes[1].transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top')
    
    plt.tight_layout()
    
    save_path = Path("demo_outputs")
    fig1.savefig(save_path / "publication_crisis_timeline.png", 
                dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path / 'publication_crisis_timeline.png'}")
    
    # Figure 2: Agent Belief Evolution Heatmap
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    if results.belief_trajectories:
        # Create matrix of belief trajectories
        agent_ids = list(results.belief_trajectories.keys())
        num_rounds = len(results.polarization_over_time)
        
        # Sample agents if too many
        if len(agent_ids) > 30:
            agent_ids = sorted(agent_ids)[:30]
        
        belief_matrix = np.array([
            results.belief_trajectories[agent_id][:num_rounds] 
            for agent_id in agent_ids
        ])
        
        # Create heatmap
        im = ax.imshow(belief_matrix, cmap='RdBu_r', aspect='auto', 
                      vmin=-1, vmax=1, interpolation='bilinear')
        
        # Customize
        ax.set_xlabel('Simulation Round', fontsize=14, fontweight='bold')
        ax.set_ylabel('Agent ID', fontsize=14, fontweight='bold')
        ax.set_title('Individual Agent Belief Evolution', fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Belief Strength', fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    fig2.savefig(save_path / "publication_belief_heatmap.png", 
                dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path / 'publication_belief_heatmap.png'}")
    
    return [fig1, fig2]


def demonstrate_mathematical_analysis(results):
    """Demonstrate mathematical analysis and modeling"""
    
    print("\n🧮 Demonstrating Mathematical Analysis...")
    
    # Initialize analyzers
    trajectory_analyzer = TrajectoryAnalyzer()
    crisis_analyzer = CrisisAnalyzer()
    
    # Analyze trajectories
    if results.belief_trajectories:
        print("   📈 Analyzing belief trajectories...")
        
        # Compute trajectory statistics
        trajectory_stats = results.compute_trajectory_statistics()
        
        print(f"   📊 Trajectory Statistics:")
        print(f"      Mean variance: {trajectory_stats.get('trajectory_variance', {}).get('mean_variance', 0):.4f}")
        print(f"      Synchronization: {trajectory_stats.get('synchronization', {}).get('synchronization_index', 0):.4f}")
        print(f"      Volatility: {trajectory_stats.get('volatility', {}).get('velocity_volatility', 0):.4f}")
    
    # Crisis impact analysis
    if results.crisis_impact_metrics:
        print("   🌊 Crisis Impact Analysis:")
        for metric, value in results.crisis_impact_metrics.items():
            print(f"      {metric}: {value:.4f}")
    
    # Phase transition analysis
    if hasattr(results, 'phase_transitions') and results.phase_transitions:
        print(f"   🔍 Detected {len(results.phase_transitions)} phase transitions:")
        for round_num, parameter, magnitude in results.phase_transitions[:3]:
            print(f"      Round {round_num}: {parameter} changed by {magnitude:.3f}")


def create_summary_report(config, results, figures):
    """Create a comprehensive summary report"""
    
    print("\n📋 Creating Summary Report...")
    
    save_path = Path("demo_outputs")
    report_path = save_path / "visualization_demo_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Dynamic Belief Evolution - Visualization Demo Report\n\n")
        
        f.write(f"**Experiment:** {config.name}\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Experiment Configuration\n\n")
        f.write(f"- **Agents:** {config.num_agents}\n")
        f.write(f"- **Rounds:** {config.num_rounds}\n")
        f.write(f"- **Crisis Type:** {config.crisis_scenario.value if config.crisis_scenario else 'None'}\n")
        f.write(f"- **Crisis Severity:** {getattr(config, 'crisis_severity', 'N/A')}\n\n")
        
        f.write("## Key Results\n\n")
        f.write(f"- **Final Polarization:** {results.polarization_over_time[-1]:.3f}\n")
        f.write(f"- **Polarization Change:** {results.polarization_over_time[-1] - results.polarization_over_time[0]:+.3f}\n")
        f.write(f"- **Agent Trajectories Tracked:** {len(results.belief_trajectories) if results.belief_trajectories else 0}\n")
        f.write(f"- **Phase Transitions Detected:** {len(results.phase_transitions) if hasattr(results, 'phase_transitions') else 0}\n\n")
        
        f.write("## Generated Visualizations\n\n")
        f.write("1. **dynamic_evolution_overview.png** - Comprehensive multi-panel overview\n")
        f.write("2. **trajectory_models.png** - Mathematical model fitting analysis\n") 
        f.write("3. **crisis_impact_analysis.png** - Crisis effect quantification\n")
        f.write("4. **agent_correlation_network.png** - Agent interaction network\n")
        f.write("5. **interactive_dashboard.html** - Interactive exploration tool\n")
        f.write("6. **publication_crisis_timeline.png** - Publication-ready crisis timeline\n")
        f.write("7. **publication_belief_heatmap.png** - Publication-ready belief evolution heatmap\n\n")
        
        f.write("## Crisis Impact Metrics\n\n")
        if results.crisis_impact_metrics:
            for metric, value in results.crisis_impact_metrics.items():
                f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")
        else:
            f.write("No crisis impact metrics available.\n")
        
        f.write("\n## Mathematical Analysis\n\n")
        trajectory_stats = results.compute_trajectory_statistics()
        if trajectory_stats:
            f.write("### Trajectory Statistics\n")
            for category, stats in trajectory_stats.items():
                f.write(f"**{category.title()}:**\n")
                for stat_name, value in stats.items():
                    f.write(f"- {stat_name}: {value:.4f}\n")
                f.write("\n")
    
    print(f"   ✅ Saved: {report_path}")


def main():
    """Main demonstration function"""
    
    print("🎨 Dynamic Belief Evolution - Visualization Demonstration")
    print("=" * 70)
    
    # 1. Create sample experiment
    config, results = create_sample_experiment()
    
    # 2. Demonstrate static visualizations
    static_figures = demonstrate_static_visualizations(results)
    
    # 3. Demonstrate interactive visualizations
    interactive_dashboard = demonstrate_interactive_visualizations(results)
    
    # 4. Create publication-ready figures
    pub_figures = create_publication_ready_figures(results)
    
    # 5. Demonstrate mathematical analysis
    demonstrate_mathematical_analysis(results)
    
    # 6. Create summary report
    all_figures = (static_figures or []) + (pub_figures or [])
    create_summary_report(config, results, all_figures)
    
    print("\n" + "=" * 70)
    print("🎉 VISUALIZATION DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\n📁 All outputs saved to: demo_outputs/")
    print("\n🎯 Key Visualizations Created:")
    print("   📊 Comprehensive Dashboard - dynamic_evolution_overview.png")
    print("   🧮 Mathematical Models - trajectory_models.png")
    print("   🌊 Crisis Analysis - crisis_impact_analysis.png")
    print("   👥 Agent Networks - agent_correlation_network.png")
    print("   🌐 Interactive Dashboard - interactive_dashboard.html")
    print("   📝 Publication Figures - publication_*.png")
    print("   📋 Summary Report - visualization_demo_report.md")
    
    print("\n✨ Next Steps:")
    print("   1. Open interactive_dashboard.html in your browser")
    print("   2. Review publication-ready figures for research use")
    print("   3. Check visualization_demo_report.md for detailed analysis")
    print("   4. Use these visualizations as templates for your own experiments!")


if __name__ == "__main__":
    main()