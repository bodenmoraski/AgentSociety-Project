"""
Command Line Interface for Dynamic Belief Evolution Experiments

Extends the main CLI with specialized commands for running dynamic evolution
experiments with crisis scenarios and time-varying parameters.

Usage:
    python run_dynamic_evolution.py --scenario pandemic --severity 0.8
    python run_dynamic_evolution.py --config configs/dynamic_pandemic.json
    python run_dynamic_evolution.py --custom --agents 100 --rounds 25
"""

import click
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import experiment components
from experiments.dynamic_evolution.experiment import (
    DynamicEvolutionExperiment, 
    DynamicEvolutionConfig,
    run_pandemic_experiment,
    run_election_experiment, 
    run_economic_shock_experiment,
    compare_crisis_scenarios
)
from experiments.dynamic_evolution.visualizations import DynamicEvolutionVisualizer
from core.dynamic_parameters import CrisisType, CrisisScenarioGenerator
from core.agent import TopicType
from core.network import NetworkConfig


def load_dynamic_config_from_file(config_file: str) -> DynamicEvolutionConfig:
    """Load dynamic evolution configuration from JSON file"""
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Convert string enums to proper types
    if 'topic' in config_data:
        topic_map = {
            'gun_control': TopicType.GUN_CONTROL,
            'climate_change': TopicType.CLIMATE_CHANGE,
            'healthcare': TopicType.HEALTHCARE,
            'taxation': TopicType.TAXATION,
            'immigration': TopicType.IMMIGRATION,
            'tech_regulation': TopicType.TECHNOLOGY_REGULATION
        }
        config_data['topic'] = topic_map.get(config_data['topic'], TopicType.GUN_CONTROL)
    
    if 'crisis_scenario' in config_data:
        crisis_map = {
            'pandemic': CrisisType.PANDEMIC,
            'election': CrisisType.ELECTION,
            'economic_shock': CrisisType.ECONOMIC_SHOCK,
            'natural_disaster': CrisisType.NATURAL_DISASTER,
            'social_unrest': CrisisType.SOCIAL_UNREST,
            'technological_disruption': CrisisType.TECHNOLOGICAL_DISRUPTION
        }
        config_data['crisis_scenario'] = crisis_map.get(config_data['crisis_scenario'])
    
    if 'network_config' in config_data:
        config_data['network_config'] = NetworkConfig(**config_data['network_config'])
    
    return DynamicEvolutionConfig(**config_data)


def create_custom_dynamic_config(agents: int, rounds: int, scenario: str, severity: float,
                               topic: str, intervention: Optional[str], 
                               intervention_round: Optional[int], seed: Optional[int],
                               timestamp: str) -> DynamicEvolutionConfig:
    """Create custom dynamic evolution configuration"""
    
    # Map scenario string to crisis type
    crisis_map = {
        'pandemic': CrisisType.PANDEMIC,
        'election': CrisisType.ELECTION,
        'economic': CrisisType.ECONOMIC_SHOCK,
        'disaster': CrisisType.NATURAL_DISASTER,
        'unrest': CrisisType.SOCIAL_UNREST,
        'tech': CrisisType.TECHNOLOGICAL_DISRUPTION
    }
    crisis_type = crisis_map.get(scenario, CrisisType.PANDEMIC)
    
    # Map topic string to enum
    topic_map = {
        'gun_control': TopicType.GUN_CONTROL,
        'climate_change': TopicType.CLIMATE_CHANGE,
        'healthcare': TopicType.HEALTHCARE,
        'taxation': TopicType.TAXATION,
        'immigration': TopicType.IMMIGRATION,
        'tech_regulation': TopicType.TECHNOLOGY_REGULATION
    }
    topic_enum = topic_map.get(topic, TopicType.GUN_CONTROL)
    
    return DynamicEvolutionConfig(
        name=f"Custom Dynamic Evolution - {scenario.title()} ({timestamp})",
        description=f"Custom {scenario} crisis scenario with severity {severity}",
        num_agents=agents,
        topic=topic_enum,
        num_rounds=rounds,
        crisis_scenario=crisis_type,
        crisis_severity=severity,
        intervention_type=intervention,
        intervention_round=intervention_round,
        random_seed=seed,
        network_config=NetworkConfig(
            network_type="preferential_attachment",
            homophily_strength=0.7,
            average_connections=6
        )
    )


@click.command()
@click.option('--scenario', '-s', type=click.Choice([
    'pandemic', 'election', 'economic', 'disaster', 'unrest', 'tech'
]), help='Crisis scenario type')
@click.option('--config', '-c', type=str, help='Load configuration from JSON file')
@click.option('--custom', is_flag=True, help='Create custom experiment with CLI parameters')
@click.option('--agents', '-n', type=int, default=100, help='Number of agents (custom mode)')
@click.option('--rounds', '-r', type=int, default=25, help='Number of rounds (custom mode)')
@click.option('--severity', type=float, default=0.7, help='Crisis severity 0.0-1.0')
@click.option('--topic', '-t', type=click.Choice([
    'gun_control', 'climate_change', 'healthcare', 'taxation', 'immigration', 'tech_regulation'
]), default='healthcare', help='Topic for discussion')
@click.option('--intervention', type=click.Choice([
    'fact_check', 'diverse_exposure', 'bridge_building'
]), help='Intervention type')
@click.option('--intervention-round', type=int, help='Round to apply intervention')
@click.option('--optimize-timing', is_flag=True, help='Optimize intervention timing')
@click.option('--output', '-o', type=str, default='results', help='Output directory')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--no-viz', is_flag=True, help='Skip visualization generation')
@click.option('--interactive', is_flag=True, help='Generate interactive dashboard')
@click.option('--report', is_flag=True, help='Generate comprehensive report')
def run(scenario, config, custom, agents, rounds, severity, topic, intervention, 
        intervention_round, optimize_timing, output, seed, no_viz, interactive, report):
    """
    Run dynamic belief evolution experiments with crisis scenarios.
    
    This command runs experiments with time-varying belief parameters to study
    how crisis events affect opinion dynamics and polarization over time.
    """
    
    print("🌀 DYNAMIC BELIEF EVOLUTION EXPERIMENTS")
    print("=" * 60)
    
    try:
        # Create output directory
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load or create experiment configuration
        if config:
            print(f"📄 Loading configuration from: {config}")
            experiment_config = load_dynamic_config_from_file(config)
            
        elif scenario:
            print(f"🎭 Running predefined {scenario} scenario")
            
            # Use convenience functions for predefined scenarios
            if scenario == 'pandemic':
                results = run_pandemic_experiment(
                    num_agents=agents, 
                    duration=rounds, 
                    severity=severity,
                    random_seed=seed,
                    intervention_type=intervention,
                    intervention_round=intervention_round,
                    optimize_intervention_timing=optimize_timing
                )
                experiment_config = results.config
                
            elif scenario == 'election':
                peak_polarization = 0.5 + 0.4 * severity  # Map severity to peak polarization
                results = run_election_experiment(
                    num_agents=agents,
                    duration=rounds,
                    peak_polarization=peak_polarization,
                    random_seed=seed,
                    intervention_type=intervention,
                    intervention_round=intervention_round
                )
                experiment_config = results.config
                
            elif scenario == 'economic':
                results = run_economic_shock_experiment(
                    num_agents=agents,
                    duration=rounds,
                    shock_severity=severity,
                    random_seed=seed,
                    intervention_type=intervention,
                    intervention_round=intervention_round
                )
                experiment_config = results.config
                
            else:
                # Create custom scenario
                experiment_config = create_custom_dynamic_config(
                    agents, rounds, scenario, severity, topic, intervention,
                    intervention_round, seed, timestamp
                )
                experiment = DynamicEvolutionExperiment(experiment_config)
                results = experiment.run_full_experiment()
                
        elif custom:
            print(f"🔧 Running custom dynamic evolution experiment")
            experiment_config = create_custom_dynamic_config(
                agents, rounds, 'pandemic', severity, topic, intervention,
                intervention_round, seed, timestamp
            )
            
            # Add optimization if requested
            if optimize_timing:
                experiment_config.optimize_intervention_timing = True
            
            experiment = DynamicEvolutionExperiment(experiment_config)
            results = experiment.run_full_experiment()
            
        else:
            print("❌ Error: Must specify --scenario, --config, or --custom")
            print("   Use --help for usage information")
            return
        
        # Run experiment if not already run
        if 'results' not in locals():
            experiment = DynamicEvolutionExperiment(experiment_config)
            results = experiment.run_full_experiment()
        
        # Save results
        results_file = output_dir / f"dynamic_evolution_{timestamp}.json"
        results.save_to_file(str(results_file))
        print(f"💾 Results saved to: {results_file}")
        
        # Print summary
        print(f"\n📊 Experiment Summary:")
        print(f"   Final polarization: {results.polarization_over_time[-1]:.3f}")
        print(f"   Phase transitions: {len(results.phase_transitions)}")
        
        if results.crisis_impact_metrics:
            metrics = results.crisis_impact_metrics
            print(f"   Crisis impact:")
            print(f"     Polarization increase: {metrics.get('polarization_increase', 0):.3f}")
            print(f"     Recovery ratio: {metrics.get('recovery_ratio', 0):.3f}")
        
        if results.optimal_intervention_round is not None:
            print(f"   Optimal intervention round: {results.optimal_intervention_round}")
        
        # Generate visualizations
        if not no_viz:
            print(f"\n📊 Generating visualizations...")
            
            viz_dir = output_dir / f"visualizations_{timestamp}"
            viz_dir.mkdir(exist_ok=True)
            
            visualizer = DynamicEvolutionVisualizer(results)
            
            try:
                # Overview dashboard
                overview_fig = visualizer.plot_dynamic_evolution_overview()
                overview_path = viz_dir / "dynamic_evolution_overview.png"
                overview_fig.savefig(overview_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"   📈 Overview saved to: {overview_path}")
                
                # Correlation network
                if len(results.belief_trajectories) > 1:
                    network_fig = visualizer.plot_cross_agent_correlation_network()
                    network_path = viz_dir / "correlation_network.png"
                    network_fig.savefig(network_path, dpi=300, bbox_inches='tight')
                    print(f"   🕸️ Network analysis saved to: {network_path}")
                
                # Interactive dashboard
                if interactive:
                    dashboard = visualizer.create_interactive_dashboard()
                    if dashboard:
                        dashboard_path = viz_dir / "interactive_dashboard.html"
                        dashboard.write_html(str(dashboard_path))
                        print(f"   🎛️ Interactive dashboard saved to: {dashboard_path}")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not generate some visualizations: {e}")
        
        # Generate comprehensive report
        if report:
            print(f"\n📋 Generating comprehensive report...")
            try:
                visualizer = DynamicEvolutionVisualizer(results)
                report_files = visualizer.generate_comprehensive_report(str(output_dir / f"report_{timestamp}"))
                
                print(f"   📄 Report generated with {len(report_files)} files:")
                for component, file_path in report_files.items():
                    print(f"     {component}: {file_path}")
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not generate comprehensive report: {e}")
        
        print(f"\n✅ Dynamic evolution experiment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


@click.command()
@click.option('--agents', '-n', type=int, default=100, help='Number of agents for comparison')
@click.option('--duration', '-d', type=int, default=25, help='Duration in rounds')
@click.option('--output', '-o', type=str, default='results', help='Output directory')
@click.option('--seed', type=int, help='Random seed for reproducibility')
def compare(agents, duration, output, seed):
    """
    Compare different crisis scenarios under identical conditions.
    
    Runs pandemic, election, and economic shock scenarios with the same
    population and duration to analyze comparative crisis dynamics.
    """
    
    print("🔄 CRISIS SCENARIO COMPARISON")
    print("=" * 50)
    
    try:
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = Path(output) / f"comparison_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎭 Comparing crisis scenarios...")
        print(f"   Agents: {agents}, Duration: {duration} rounds")
        if seed:
            print(f"   Random seed: {seed}")
        
        # Run comparison
        start_time = time.time()
        scenario_results = compare_crisis_scenarios(agents, duration)
        duration_seconds = time.time() - start_time
        
        print(f"\n📊 Comparison completed in {duration_seconds:.1f} seconds")
        
        # Save individual results
        for scenario_name, results in scenario_results.items():
            result_file = output_dir / f"{scenario_name}_results.json"
            results.save_to_file(str(result_file))
            print(f"   💾 {scenario_name.title()} results: {result_file}")
        
        # Generate comparative analysis
        comparison_summary = {}
        
        for scenario_name, results in scenario_results.items():
            summary = {
                'final_polarization': results.polarization_over_time[-1] if results.polarization_over_time else 0,
                'peak_polarization': max(results.polarization_over_time) if results.polarization_over_time else 0,
                'num_phase_transitions': len(results.phase_transitions),
                'crisis_impact': results.crisis_impact_metrics
            }
            comparison_summary[scenario_name] = summary
            
            print(f"\n📈 {scenario_name.title()} Summary:")
            print(f"   Final polarization: {summary['final_polarization']:.3f}")
            print(f"   Peak polarization: {summary['peak_polarization']:.3f}")
            print(f"   Phase transitions: {summary['num_phase_transitions']}")
        
        # Save comparison summary
        summary_file = output_dir / "comparison_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(comparison_summary, f, indent=2, default=str)
        print(f"\n📋 Comparison summary: {summary_file}")
        
        # Generate comparative visualizations
        print(f"\n📊 Generating comparative visualizations...")
        
        try:
            import matplotlib.pyplot as plt
            
            # Comparative polarization evolution
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = {'pandemic': '#e74c3c', 'election': '#3498db', 'economic_shock': '#2ecc71'}
            
            for scenario_name, results in scenario_results.items():
                if results.polarization_over_time:
                    rounds = range(len(results.polarization_over_time))
                    ax.plot(rounds, results.polarization_over_time, 
                           label=scenario_name.replace('_', ' ').title(),
                           color=colors.get(scenario_name, 'gray'),
                           linewidth=3, alpha=0.8)
            
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Polarization Index', fontsize=12)
            ax.set_title('Crisis Scenario Comparison: Polarization Evolution', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            comparison_plot_path = output_dir / "scenario_comparison.png"
            fig.savefig(comparison_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"   📈 Comparison plot: {comparison_plot_path}")
            
        except Exception as e:
            print(f"⚠️  Could not generate comparative visualizations: {e}")
        
        print(f"\n✅ Crisis scenario comparison completed!")
        
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


@click.command()
@click.option('--name', '-n', required=True, help='Configuration name')
@click.option('--scenario', '-s', type=click.Choice([
    'pandemic', 'election', 'economic', 'disaster', 'unrest', 'tech'
]), default='pandemic', help='Crisis scenario type')
@click.option('--agents', type=int, default=100, help='Number of agents')
@click.option('--rounds', type=int, default=25, help='Number of rounds')
@click.option('--severity', type=float, default=0.7, help='Crisis severity 0.0-1.0')
@click.option('--output', '-o', type=str, default='configs', help='Output directory for config file')
def create_config(name, scenario, agents, rounds, severity, output):
    """Create a configuration file template for dynamic evolution experiments"""
    
    config_template = {
        "name": f"Dynamic Evolution: {name}",
        "description": f"Dynamic belief evolution experiment with {scenario} crisis scenario",
        "experiment_type": "dynamic_evolution",
        
        "num_agents": agents,
        "topic": "healthcare",
        "num_rounds": rounds,
        "interactions_per_round": agents * 3,
        
        "crisis_scenario": scenario,
        "crisis_severity": severity,
        
        "belief_history_tracking": True,
        "sample_interval": 1,
        "phase_detection_threshold": 0.1,
        
        "optimize_intervention_timing": False,
        "intervention_type": None,
        "intervention_round": None,
        
        "network_config": {
            "network_type": "preferential_attachment",
            "homophily_strength": 0.7,
            "average_connections": 6,
            "dynamic_rewiring": True,
            "bridge_probability": 0.04
        },
        
        "derivative_epsilon": 0.1,
        "trajectory_smoothing": 0.05,
        
        "save_detailed_history": True,
        "save_network_snapshots": False,
        "random_seed": None
    }
    
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = output_dir / f"dynamic_{name.lower().replace(' ', '_')}.json"
    with open(config_file, 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print(f"📝 Dynamic evolution configuration created: {config_file}")


@click.group()
def cli():
    """Dynamic Belief Evolution Experiments
    
    Advanced experiments studying belief dynamics during crisis scenarios
    with time-varying parameters and mathematical trajectory analysis.
    """
    pass


cli.add_command(run)
cli.add_command(compare)
cli.add_command(create_config, name='create-config')


if __name__ == '__main__':
    cli()