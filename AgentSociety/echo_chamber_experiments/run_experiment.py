#!/usr/bin/env python3
"""
Echo Chamber Experiments - Main Runner Script

This script provides a command-line interface to run various echo chamber experiments
and generate visualizations of social dynamics and belief propagation.

Usage:
    python run_experiment.py --experiment basic_polarization
    python run_experiment.py --experiment intervention_study --output results/
    python run_experiment.py --custom --agents 100 --rounds 20 --topic climate_change
"""

import click
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import traceback

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

try:
    # Try importing as installed package first
    from echo_chamber_experiments.core.experiment import EchoChamberExperiment, ExperimentConfig, run_predefined_experiment
    from echo_chamber_experiments.core.network import NetworkConfig
    from echo_chamber_experiments.core.agent import TopicType, PersonalityType
    from echo_chamber_experiments.visualizations.plots import EchoChamberVisualizer
except ImportError:
    # Fallback to local development imports
    from core.experiment import EchoChamberExperiment, ExperimentConfig, run_predefined_experiment
    from core.network import NetworkConfig
    from core.agent import TopicType, PersonalityType
    from visualizations.plots import EchoChamberVisualizer


@click.command()
@click.option('--experiment', '-e', type=click.Choice([
    'basic_polarization', 'intervention_study', 'bridge_building', 'large_scale'
]), help='Run a predefined experiment')
@click.option('--custom', is_flag=True, help='Run custom experiment with specified parameters')
@click.option('--agents', '-n', type=int, default=50, help='Number of agents (custom mode)')
@click.option('--rounds', '-r', type=int, default=15, help='Number of rounds (custom mode)')
@click.option('--topic', '-t', type=click.Choice([
    'gun_control', 'climate_change', 'healthcare', 'taxation', 'immigration', 'tech_regulation'
]), default='gun_control', help='Topic for discussion (custom mode)')
@click.option('--network', type=click.Choice([
    'random', 'small_world', 'scale_free', 'preferential_attachment'
]), default='preferential_attachment', help='Network type (custom mode)')
@click.option('--belief-dist', type=click.Choice([
    'polarized', 'normal', 'uniform'
]), default='polarized', help='Initial belief distribution (custom mode)')
@click.option('--homophily', type=float, default=0.7, help='Homophily strength 0-1 (custom mode)')
@click.option('--intervention', type=click.Choice([
    'fact_check', 'diverse_exposure', 'bridge_building'
]), help='Intervention type (custom mode)')
@click.option('--intervention-round', type=int, help='Round to apply intervention (custom mode)')
@click.option('--output', '-o', type=str, default='results', help='Output directory')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--no-viz', is_flag=True, help='Skip visualization generation')
@click.option('--interactive', is_flag=True, help='Generate interactive dashboard')
@click.option('--config-file', type=str, help='Load experiment config from JSON file')
def main(experiment, custom, agents, rounds, topic, network, belief_dist, homophily,
         intervention, intervention_round, output, seed, no_viz, interactive, config_file):
    """
    Run echo chamber experiments to study social dynamics and belief propagation.
    
    This tool simulates AI agents with different beliefs, personalities, and social networks
    to study how echo chambers form and how interventions might reduce polarization.
    """
    
    print("🏛️  ECHO CHAMBER SOCIAL DYNAMICS EXPERIMENTS")
    print("=" * 60)
    
    try:
        # Create output directory
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load or create experiment configuration
        if config_file:
            config = load_config_from_file(config_file)
        elif experiment:
            # Run predefined experiment
            print(f"🚀 Running predefined experiment: {experiment}")
            
            # Override seed if provided
            kwargs = {}
            if seed is not None:
                kwargs['random_seed'] = seed
            
            results = run_predefined_experiment(experiment, **kwargs)
            config = results.config
            
        elif custom:
            # Create custom experiment
            print(f"🔧 Running custom experiment")
            config = create_custom_config(
                agents, rounds, topic, network, belief_dist, homophily,
                intervention, intervention_round, seed, timestamp
            )
            
            # Run experiment
            experiment_runner = EchoChamberExperiment(config)
            results = experiment_runner.run_full_experiment()
        else:
            print("❌ Error: Must specify either --experiment or --custom")
            print("   Use --help for usage information")
            return
        
        if not 'results' in locals():
            # If we loaded config from file, run the experiment
            experiment_runner = EchoChamberExperiment(config)
            results = experiment_runner.run_full_experiment()
        
        # Save results
        results_file = output_dir / f"{config.name.lower().replace(' ', '_')}_{timestamp}.json"
        results.save_to_file(str(results_file))
        print(f"💾 Results saved to: {results_file}")
        
        # Generate summary report
        visualizer = EchoChamberVisualizer(results)
        summary = visualizer.generate_summary_report()
        
        summary_file = output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        print(f"📋 Summary report saved to: {summary_file}")
        
        # Print key findings
        print("\n" + "=" * 60)
        print("KEY FINDINGS:")
        print(f"🎯 Final polarization: {results.polarization_over_time[-1]:.3f}")
        print(f"🏠 Echo chambers formed: {len(results.final_echo_chambers)}")
        print(f"🌉 Bridge agents: {len(results.bridge_agents)}")
        
        if config.intervention_type and config.intervention_round:
            pre = results.polarization_over_time[config.intervention_round - 1] if config.intervention_round > 0 else 0
            post = results.polarization_over_time[-1]
            effect = post - pre
            print(f"🛠️  Intervention effect: {effect:+.3f} ({'Helpful' if effect < -0.05 else 'Harmful' if effect > 0.05 else 'Neutral'})")
        
        # Generate visualizations
        if not no_viz:
            print(f"\n📊 Generating visualizations...")
            
            viz_dir = output_dir / f"visualizations_{timestamp}"
            viz_dir.mkdir(exist_ok=True)
            
            # Static plots
            try:
                import matplotlib.pyplot as plt
                
                # Belief evolution
                fig1 = visualizer.plot_belief_evolution()
                if fig1:
                    fig1.savefig(viz_dir / "belief_evolution.png", dpi=300, bbox_inches='tight')
                    plt.close(fig1)
                
                # Network analysis
                fig2 = visualizer.plot_network_analysis()
                if fig2:
                    fig2.savefig(viz_dir / "network_analysis.png", dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                
                # Agent analysis
                fig3 = visualizer.plot_agent_analysis()
                if fig3:
                    fig3.savefig(viz_dir / "agent_analysis.png", dpi=300, bbox_inches='tight')
                    plt.close(fig3)
                
                # Network graph
                fig4 = visualizer.create_network_graph()
                if fig4:
                    fig4.savefig(viz_dir / "network_graph.png", dpi=300, bbox_inches='tight')
                    plt.close(fig4)
                
                print(f"📈 Static plots saved to: {viz_dir}")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not generate static plots: {e}")
            
            # Interactive dashboard
            if interactive:
                try:
                    dashboard = visualizer.create_comprehensive_dashboard()
                    dashboard_file = viz_dir / "interactive_dashboard.html"
                    dashboard.write_html(str(dashboard_file))
                    print(f"🌐 Interactive dashboard saved to: {dashboard_file}")
                    print(f"   Open in your browser to explore results!")
                    
                except Exception as e:
                    print(f"⚠️  Warning: Could not generate interactive dashboard: {e}")
        
        # Export data for further analysis
        if results.agents_history:
            df = results.to_dataframe()
            csv_file = output_dir / f"agent_data_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"📊 Agent data exported to: {csv_file}")
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📁 All outputs saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error:")
        print(f"   {str(e)}")
        print(f"\n🔍 Full traceback:")
        traceback.print_exc()
        sys.exit(1)


def create_custom_config(agents, rounds, topic, network, belief_dist, homophily,
                        intervention, intervention_round, seed, timestamp):
    """Create custom experiment configuration"""
    
    # Convert string topic to enum
    topic_map = {
        'gun_control': TopicType.GUN_CONTROL,
        'climate_change': TopicType.CLIMATE_CHANGE,
        'healthcare': TopicType.HEALTHCARE,
        'taxation': TopicType.TAXATION,
        'immigration': TopicType.IMMIGRATION,
        'tech_regulation': TopicType.TECHNOLOGY_REGULATION
    }
    
    network_config = NetworkConfig(
        network_type=network,
        homophily_strength=homophily,
        average_connections=max(3, min(10, agents // 10))  # Reasonable default
    )
    
    config = ExperimentConfig(
        name=f"Custom_{topic}_{timestamp}",
        description=f"Custom experiment: {agents} agents, {rounds} rounds, {topic} topic",
        num_agents=agents,
        topic=topic_map[topic],
        belief_distribution=belief_dist,
        network_config=network_config,
        num_rounds=rounds,
        interactions_per_round=max(50, agents * 3),  # Scale interactions with population
        intervention_type=intervention,
        intervention_round=intervention_round,
        random_seed=seed
    )
    
    return config


def load_config_from_file(config_file):
    """Load experiment configuration from JSON file"""
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Convert back to proper types
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
    
    if 'network_config' in config_data:
        config_data['network_config'] = NetworkConfig(**config_data['network_config'])
    
    return ExperimentConfig(**config_data)


@click.command()
@click.option('--name', '-n', required=True, help='Experiment name')
@click.option('--agents', type=int, default=50, help='Number of agents')
@click.option('--rounds', type=int, default=15, help='Number of rounds')
@click.option('--output', '-o', type=str, default='configs', help='Output directory for config file')
def create_config(name, agents, rounds, output):
    """Create a configuration file template for custom experiments"""
    
    config_template = {
        "name": name,
        "description": f"Custom experiment configuration for {name}",
        "num_agents": agents,
        "topic": "gun_control",
        "belief_distribution": "polarized",
        "network_config": {
            "network_type": "preferential_attachment",
            "homophily_strength": 0.7,
            "average_connections": 5
        },
        "num_rounds": rounds,
        "interactions_per_round": agents * 3,
        "intervention_round": None,
        "intervention_type": None,
        "random_seed": None
    }
    
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = output_dir / f"{name.lower().replace(' ', '_')}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print(f"📝 Configuration template created: {config_file}")
    print(f"   Edit this file and use --config-file to run your custom experiment")


@click.group()
def cli():
    """Echo Chamber Social Dynamics Experiments
    
    Study how beliefs spread and echo chambers form in social networks of AI agents.
    """
    pass


cli.add_command(main, name='run')
cli.add_command(create_config, name='create-config')


if __name__ == '__main__':
    cli()