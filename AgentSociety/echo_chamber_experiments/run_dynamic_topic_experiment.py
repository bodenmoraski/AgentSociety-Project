#!/usr/bin/env python3
"""
Dynamic Topic Experiment Runner

This script demonstrates how to run experiments with the new dynamic topic
system generator, showing integration with the existing experiment framework.
"""

import sys
import os
import random
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.topic_generator import TopicGenerator, TopicComplexity, generate_topic_system
from core.dynamic_agent import create_dynamic_agent_population, DynamicExperimentConfig
from core.network import SocialNetwork, NetworkConfig
from core.experiment import EchoChamberExperiment, ExperimentConfig


def run_dynamic_topic_experiment(
    topic_input: str,
    num_agents: int = 50,
    num_rounds: int = 10,
    complexity: TopicComplexity = TopicComplexity.MODERATE,
    belief_distribution: str = "polarized",
    use_ai: bool = False,
    api_key: str = None
):
    """
    Run a complete experiment with a dynamic topic
    
    Args:
        topic_input: Topic string (e.g., "climate change", "vaccination")
        num_agents: Number of agents in the simulation
        num_rounds: Number of simulation rounds
        complexity: Topic complexity level
        belief_distribution: Belief distribution type
        use_ai: Whether to use AI for topic generation
        api_key: OpenAI API key for AI generation
    """
    
    print(f"🚀 Starting Dynamic Topic Experiment")
    print(f"📋 Topic: {topic_input}")
    print(f"👥 Agents: {num_agents}")
    print(f"🔄 Rounds: {num_rounds}")
    print(f"🧠 Complexity: {complexity.value}")
    print(f"📊 Distribution: {belief_distribution}")
    print(f"🤖 AI Enabled: {use_ai}")
    print("=" * 60)
    
    # Generate topic system
    print("\n🔧 Generating topic system...")
    generator = TopicGenerator(use_ai=use_ai, api_key=api_key)
    topic_system = generator.generate_topic_system(topic_input, complexity)
    
    print(f"✅ Generated topic system for: {topic_system.topic_name}")
    print(f"   Description: {topic_system.topic_description}")
    print(f"   Dimensions: {topic_system.belief_dimensions}")
    print(f"   Controversy Level: {topic_system.controversy_level:.2f}")
    print(f"   Key Terms: {', '.join(topic_system.key_terms)}")
    
    # Create agents
    print(f"\n👥 Creating {num_agents} agents...")
    agents = create_dynamic_agent_population(
        num_agents=num_agents,
        topic_input=topic_input,
        belief_distribution=belief_distribution,
        complexity=complexity,
        use_ai=use_ai,
        api_key=api_key
    )
    
    print(f"✅ Created {len(agents)} agents")
    
    # Show sample agent messages
    print(f"\n💬 Sample Agent Messages:")
    for i, agent in enumerate(agents[:3]):
        message = agent.generate_message()
        belief = agent.belief_strength
        print(f"   Agent {i+1} (belief: {belief:.2f}): '{message}'")
    
    # Create network
    print(f"\n🌐 Creating social network...")
    network_config = NetworkConfig(
        network_type="scale_free",
        average_connections=4
    )
    
    network = SocialNetwork(agents, network_config)
    
    print(f"✅ Network created with {len(network.graph.edges)} connections")
    
    # Create experiment configuration
    experiment_config = ExperimentConfig(
        name=f"dynamic_{topic_input.replace(' ', '_')}_experiment",
        description=f"Dynamic topic experiment for {topic_input}",
        num_agents=num_agents,
        num_rounds=num_rounds,
        interactions_per_round=100,
        network_config=network_config,
        save_detailed_history=True,
        output_directory=f"dynamic_experiment_results/{topic_input.replace(' ', '_')}"
    )
    
    # Run experiment
    print(f"\n🔬 Running experiment...")
    experiment = EchoChamberExperiment(experiment_config)
    experiment.agents = agents
    experiment.network = network
    
    results = experiment.run_full_experiment()
    
    # Display results
    print(f"\n📊 Experiment Results:")
    if results.polarization_over_time:
        print(f"   Final Polarization: {results.polarization_over_time[-1]:.3f}")
    print(f"   Echo Chambers: {len(results.final_echo_chambers)}")
    print(f"   Bridge Agents: {len(results.bridge_agents)}")
    print(f"   Most Influential: {len(results.most_influential_agents)}")
    
    # Show belief evolution
    print(f"\n📈 Belief Evolution Summary:")
    initial_beliefs = [agent.belief_history[0] for agent in agents]
    final_beliefs = [agent.belief_strength for agent in agents]
    
    print(f"   Initial Belief Range: {min(initial_beliefs):.3f} to {max(initial_beliefs):.3f}")
    print(f"   Final Belief Range: {min(final_beliefs):.3f} to {max(final_beliefs):.3f}")
    print(f"   Belief Change: {np.mean(final_beliefs) - np.mean(initial_beliefs):.3f}")
    
    return results, topic_system


def run_multi_topic_comparison(topics: list, num_agents: int = 30, num_rounds: int = 5):
    """
    Run experiments for multiple topics and compare results
    
    Args:
        topics: List of topic strings
        num_agents: Number of agents per topic
        num_rounds: Number of rounds per experiment
    """
    
    print(f"🔬 Multi-Topic Comparison Experiment")
    print(f"📋 Topics: {', '.join(topics)}")
    print(f"👥 Agents per topic: {num_agents}")
    print(f"🔄 Rounds per experiment: {num_rounds}")
    print("=" * 60)
    
    results_summary = {}
    
    for topic in topics:
        print(f"\n🔬 Running experiment for: {topic}")
        
        try:
            results, topic_system = run_dynamic_topic_experiment(
                topic_input=topic,
                num_agents=num_agents,
                num_rounds=num_rounds,
                complexity=TopicComplexity.MODERATE
            )
            
            results_summary[topic] = {
                'final_polarization': results.polarization_over_time[-1] if results.polarization_over_time else 0.0,
                'echo_chamber_count': len(results.final_echo_chambers),
                'bridge_agents': len(results.bridge_agents),
                'influential_agents': len(results.most_influential_agents),
                'controversy_level': topic_system.controversy_level,
                'complexity': topic_system.complexity.value
            }
            
        except Exception as e:
            print(f"❌ Error running experiment for {topic}: {e}")
            results_summary[topic] = {'error': str(e)}
    
    # Display comparison
    print(f"\n📊 Multi-Topic Comparison Results:")
    print("=" * 60)
    
    for topic, results in results_summary.items():
        if 'error' not in results:
            print(f"\n📋 {topic}:")
            print(f"   Controversy Level: {results['controversy_level']:.2f}")
            print(f"   Complexity: {results['complexity']}")
            print(f"   Final Polarization: {results['final_polarization']:.3f}")
            print(f"   Echo Chambers: {results['echo_chamber_count']}")
            print(f"   Bridge Agents: {results['bridge_agents']}")
            print(f"   Influential Agents: {results['influential_agents']}")
        else:
            print(f"\n❌ {topic}: {results['error']}")
    
    return results_summary


def main():
    """Main function to run experiments"""
    
    print("🚀 Dynamic Topic System Experiment Runner")
    print("=" * 60)
    
    # Example 1: Single topic experiment
    print("\n🎯 Example 1: Single Topic Experiment")
    print("-" * 40)
    
    results, topic_system = run_dynamic_topic_experiment(
        topic_input="vaccination",
        num_agents=30,
        num_rounds=5,
        complexity=TopicComplexity.MODERATE
    )
    
    # Example 2: Multi-topic comparison
    print("\n🎯 Example 2: Multi-Topic Comparison")
    print("-" * 40)
    
    topics = ["climate change", "vaccination", "AI regulation"]
    comparison_results = run_multi_topic_comparison(
        topics=topics,
        num_agents=20,
        num_rounds=3
    )
    
    # Example 3: Unknown topic
    print("\n🎯 Example 3: Unknown Topic Experiment")
    print("-" * 40)
    
    unknown_results, unknown_topic_system = run_dynamic_topic_experiment(
        topic_input="space exploration",
        num_agents=25,
        num_rounds=4,
        complexity=TopicComplexity.SIMPLE
    )
    
    print(f"\n✅ All experiments completed!")
    print(f"📁 Results saved to: dynamic_experiment_results/")
    
    # Summary
    print(f"\n📊 Experiment Summary:")
    if results.polarization_over_time:
        print(f"   • Single topic (vaccination): {results.polarization_over_time[-1]:.3f} polarization")
    print(f"   • Multi-topic comparison: {len(topics)} topics tested")
    if unknown_results.polarization_over_time:
        print(f"   • Unknown topic (space exploration): {unknown_results.polarization_over_time[-1]:.3f} polarization")
    
    print(f"\n🎯 Key Benefits Demonstrated:")
    print(f"   • Dynamic topic generation from string inputs")
    print(f"   • Seamless integration with existing experiment framework")
    print(f"   • Support for unknown topics with generic templates")
    print(f"   • Multi-topic comparison capabilities")
    print(f"   • Backward compatibility with existing code")


if __name__ == "__main__":
    main() 