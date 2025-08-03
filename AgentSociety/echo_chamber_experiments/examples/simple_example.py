#!/usr/bin/env python3
"""
Simple Example - Echo Chamber Experiments

This script demonstrates basic usage of the echo chamber experiment framework.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from core.experiment import run_predefined_experiment, EchoChamberExperiment, ExperimentConfig
    from core.network import NetworkConfig
    from core.agent import TopicType
    from visualizations.plots import EchoChamberVisualizer
except ImportError:
    # Fallback for different directory structures
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from echo_chamber_experiments.core.experiment import run_predefined_experiment, EchoChamberExperiment, ExperimentConfig
    from echo_chamber_experiments.core.network import NetworkConfig
    from echo_chamber_experiments.core.agent import TopicType
    from echo_chamber_experiments.visualizations.plots import EchoChamberVisualizer


def main():
    print("🏛️  ECHO CHAMBER EXPERIMENTS - SIMPLE EXAMPLE")
    print("=" * 50)
    
    # 1. Run a predefined experiment
    print("\n1️⃣  Running basic polarization experiment...")
    results = run_predefined_experiment("basic_polarization", random_seed=42)
    
    print(f"✅ Experiment completed!")
    print(f"📊 Final polarization: {results.polarization_over_time[-1]:.3f}")
    print(f"🏠 Echo chambers formed: {len(results.final_echo_chambers)}")
    print(f"🌉 Bridge agents: {len(results.bridge_agents)}")
    
    # 2. Create custom experiment
    print("\n2️⃣  Running custom experiment...")
    
    config = ExperimentConfig(
        name="Custom Climate Study",
        description="Small custom experiment on climate change",
        num_agents=30,
        topic=TopicType.CLIMATE_CHANGE,
        belief_distribution="polarized",
        network_config=NetworkConfig(
            network_type="small_world",
            homophily_strength=0.8
        ),
        num_rounds=10,
        interactions_per_round=100,
        random_seed=123
    )
    
    experiment = EchoChamberExperiment(config)
    custom_results = experiment.run_full_experiment()
    
    print(f"✅ Custom experiment completed!")
    print(f"📊 Final polarization: {custom_results.polarization_over_time[-1]:.3f}")
    print(f"🏠 Echo chambers formed: {len(custom_results.final_echo_chambers)}")
    
    # 3. Generate analysis
    print("\n3️⃣  Generating analysis...")
    
    visualizer = EchoChamberVisualizer(custom_results)
    summary = visualizer.generate_summary_report()
    
    print("\n📋 EXPERIMENT SUMMARY:")
    print("-" * 30)
    print(summary[:500] + "..." if len(summary) > 500 else summary)
    
    # 4. Export data
    if custom_results.agents_history:
        df = custom_results.to_dataframe()
        print(f"\n📈 Data exported: {len(df)} rows of agent data")
        print(f"Columns: {list(df.columns)}")
    
    print("\n🎉 Example completed successfully!")
    print("💡 Try running the CLI for more options:")
    print("   python run_experiment.py run --experiment intervention_study")


if __name__ == "__main__":
    main()