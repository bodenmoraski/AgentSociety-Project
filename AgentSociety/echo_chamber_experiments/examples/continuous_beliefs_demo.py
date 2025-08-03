"""
Demonstration of Continuous Belief Distributions

This script shows how the new continuous parameter system provides much more
flexibility and control compared to the discrete "polarized", "normal", "uniform" options.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pandas as pd

from core.continuous_beliefs import (
    BeliefDistributionParams, 
    ContinuousBeliefGenerator,
    DistributionType,
    create_polarized_params,
    create_moderate_params,
    create_uniform_params
)
from core.continuous_integration import (
    ContinuousAgentConfig,
    create_continuous_agent_population,
    create_highly_polarized_population,
    create_tri_modal_population,
    compare_distributions
)
from core.agent import TopicType


def demo_basic_distributions():
    """Demonstrate the basic distribution types and how to customize them"""
    
    print("=== BASIC DISTRIBUTION TYPES ===\n")
    
    # Create different distribution configurations
    distributions = {
        "Legacy Polarized (Fixed)": BeliefDistributionParams(
            distribution_type=DistributionType.BIMODAL,
            polarization_strength=0.8,  # Fixed in legacy system
            gap_size=0.4,               # Fixed in legacy system
            spread=0.3                  # Fixed in legacy system
        ),
        
        "Continuous Polarized (Tunable)": BeliefDistributionParams(
            distribution_type=DistributionType.BIMODAL,
            polarization_strength=0.6,   # Can be adjusted continuously
            polarization_asymmetry=0.3,  # NEW: right-leaning bias
            gap_size=0.7,                # Can be adjusted
            spread=0.25                  # Can be adjusted
        ),
        
        "Moderate with Skew": BeliefDistributionParams(
            distribution_type=DistributionType.NORMAL,
            center=0.1,                  # NEW: slight right bias
            spread=0.4,
            skewness=0.8                 # NEW: right-skewed
        ),
        
        "Multi-Modal (Impossible in Legacy)": BeliefDistributionParams(
            distribution_type=DistributionType.MIXTURE,
            mixture_components=[
                {'type': 'normal', 'mean': -0.8, 'std': 0.1},  # Far left
                {'type': 'normal', 'mean': -0.2, 'std': 0.15}, # Center-left
                {'type': 'normal', 'mean': 0.6, 'std': 0.2},   # Right
            ],
            mode_weights=[0.2, 0.3, 0.5]  # Asymmetric weights
        )
    }
    
    # Generate and visualize each distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, params) in enumerate(distributions.items()):
        generator = ContinuousBeliefGenerator(params)
        beliefs = generator.generate_beliefs(2000)
        stats = generator.get_distribution_stats(2000)
        
        # Plot histogram
        axes[i].hist(beliefs, bins=40, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black')
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Belief Strength')
        axes[i].set_ylabel('Density')
        axes[i].set_xlim(-1, 1)
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Mean: {stats['mean']:.2f}
Polarization: {stats['polarization_index']:.2f}
Diversity: {stats['diversity_index']:.2f}
Skewness: {stats['skewness']:.2f}"""
        
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        print(f"{name}:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std: {stats['std']:.3f}")
        print(f"  Polarization Index: {stats['polarization_index']:.3f}")
        print(f"  Diversity Index: {stats['diversity_index']:.3f}")
        print()
    
    plt.tight_layout()
    plt.savefig('continuous_vs_legacy_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_parameter_sensitivity():
    """Demonstrate how continuous parameters affect distribution shape"""
    
    print("=== PARAMETER SENSITIVITY ANALYSIS ===\n")
    
    # Test how different polarization strengths affect the distribution
    polarization_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    fig, axes = plt.subplots(1, len(polarization_values), figsize=(20, 4))
    
    results = []
    
    for i, pol_strength in enumerate(polarization_values):
        params = create_polarized_params(polarization_strength=pol_strength)
        generator = ContinuousBeliefGenerator(params)
        beliefs = generator.generate_beliefs(1000)
        stats = generator.get_distribution_stats(1000)
        
        # Plot
        axes[i].hist(beliefs, bins=30, density=True, alpha=0.7, color='lightcoral')
        axes[i].set_title(f'Polarization = {pol_strength}')
        axes[i].set_xlabel('Belief')
        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(0, 2.5)
        
        # Store results
        results.append({
            'polarization_param': pol_strength,
            'measured_polarization': stats['polarization_index'],
            'diversity': stats['diversity_index'],
            'std': stats['std']
        })
        
        print(f"Polarization Parameter: {pol_strength}")
        print(f"  Measured Polarization: {stats['polarization_index']:.3f}")
        print(f"  Belief Diversity: {stats['diversity_index']:.3f}")
        print(f"  Standard Deviation: {stats['std']:.3f}")
        print()
    
    plt.tight_layout()
    plt.savefig('polarization_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Show how the parameter maps to actual measured polarization
    df_results = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df_results['polarization_param'], df_results['measured_polarization'], 'o-')
    plt.xlabel('Polarization Parameter')
    plt.ylabel('Measured Polarization Index')
    plt.title('Parameter vs Measured Polarization')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(df_results['polarization_param'], df_results['diversity'], 'o-', color='green')
    plt.xlabel('Polarization Parameter') 
    plt.ylabel('Belief Diversity Index')
    plt.title('Polarization vs Diversity Trade-off')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('parameter_relationships.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_personality_correlations():
    """Demonstrate how belief-personality correlations work"""
    
    print("=== PERSONALITY-BELIEF CORRELATIONS ===\n")
    
    # Create configurations with different correlation patterns
    configs = {
        "No Correlations": ContinuousAgentConfig(
            num_agents=200,
            topic=TopicType.GUN_CONTROL,
            belief_params=create_moderate_params(spread=0.6),
            trait_correlations={}  # No correlations
        ),
        
        "Strong Beliefs → High Confidence": ContinuousAgentConfig(
            num_agents=200,
            topic=TopicType.GUN_CONTROL,
            belief_params=create_moderate_params(spread=0.6),
            trait_correlations={
                'belief_strength': {
                    'confidence': 0.6,  # Strong correlation
                    'openness': -0.4    # Negative correlation
                }
            }
        ),
        
        "Extreme Beliefs → Low Openness": ContinuousAgentConfig(
            num_agents=200,
            topic=TopicType.GUN_CONTROL,
            belief_params=create_polarized_params(polarization_strength=0.8),
            trait_correlations={
                'belief_strength': {
                    'openness': -0.7,           # Very negative correlation
                    'confirmation_bias': 0.5    # Positive correlation
                }
            }
        )
    }
    
    fig, axes = plt.subplots(len(configs), 2, figsize=(12, 4*len(configs)))
    if len(configs) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (config_name, config) in enumerate(configs.items()):
        # Generate agents
        agents = create_continuous_agent_population(config)
        
        # Extract data
        beliefs = [abs(agent.belief_strength) for agent in agents]  # Use absolute for correlation
        confidence = [agent.confidence for agent in agents]
        openness = [agent.openness for agent in agents]
        
        # Plot belief vs confidence correlation
        axes[i, 0].scatter(beliefs, confidence, alpha=0.6, s=20)
        axes[i, 0].set_xlabel('|Belief Strength|')
        axes[i, 0].set_ylabel('Confidence')
        axes[i, 0].set_title(f'{config_name}\nBelief vs Confidence')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_conf = np.corrcoef(beliefs, confidence)[0, 1]
        axes[i, 0].text(0.05, 0.95, f'r = {corr_conf:.3f}', 
                       transform=axes[i, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot belief vs openness correlation
        axes[i, 1].scatter(beliefs, openness, alpha=0.6, s=20, color='orange')
        axes[i, 1].set_xlabel('|Belief Strength|')
        axes[i, 1].set_ylabel('Openness')
        axes[i, 1].set_title(f'{config_name}\nBelief vs Openness')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_open = np.corrcoef(beliefs, openness)[0, 1]
        axes[i, 1].text(0.05, 0.95, f'r = {corr_open:.3f}', 
                       transform=axes[i, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        print(f"{config_name}:")
        print(f"  Belief-Confidence correlation: {corr_conf:.3f}")
        print(f"  Belief-Openness correlation: {corr_open:.3f}")
        print()
    
    plt.tight_layout()
    plt.savefig('personality_correlations.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_complex_scenarios():
    """Demonstrate complex scenarios impossible with discrete system"""
    
    print("=== COMPLEX SCENARIOS (IMPOSSIBLE WITH DISCRETE SYSTEM) ===\n")
    
    # Scenario 1: Gradual polarization over time simulation
    print("Scenario 1: Simulating gradual polarization increase")
    
    polarization_timeline = [0.1, 0.3, 0.5, 0.7, 0.9]
    population_snapshots = []
    
    for t, pol_strength in enumerate(polarization_timeline):
        params = create_polarized_params(
            polarization_strength=pol_strength,
            asymmetry=0.1 * t  # Increasing right bias over time
        )
        
        config = ContinuousAgentConfig(
            num_agents=100,
            topic=TopicType.CLIMATE_CHANGE,
            belief_params=params,
            random_seed=42  # Same seed for consistency
        )
        
        agents = create_continuous_agent_population(config)
        beliefs = [agent.belief_strength for agent in agents]
        population_snapshots.append(beliefs)
        
        print(f"  Time {t}: Polarization = {pol_strength:.1f}, Mean = {np.mean(beliefs):.3f}")
    
    # Visualize polarization evolution
    plt.figure(figsize=(15, 5))
    
    for t, beliefs in enumerate(population_snapshots):
        plt.subplot(1, len(population_snapshots), t+1)
        plt.hist(beliefs, bins=20, density=True, alpha=0.7, 
                color=plt.cm.Reds(0.3 + 0.6*t/len(population_snapshots)))
        plt.title(f'Time {t}\nPol={polarization_timeline[t]:.1f}')
        plt.xlabel('Belief')
        plt.xlim(-1, 1)
        plt.ylim(0, 3)
    
    plt.tight_layout()
    plt.savefig('polarization_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Scenario 2: Population with realistic belief clusters
    print("\nScenario 2: Realistic multi-modal population")
    
    # Create a population representing a real political distribution
    # Far-left, center-left, center, center-right, far-right
    realistic_components = [
        {'type': 'normal', 'mean': -0.85, 'std': 0.08},  # Far-left (8%)
        {'type': 'normal', 'mean': -0.45, 'std': 0.12},  # Center-left (25%)
        {'type': 'normal', 'mean': 0.05, 'std': 0.15},   # Center (34%)
        {'type': 'normal', 'mean': 0.55, 'std': 0.12},   # Center-right (23%)
        {'type': 'normal', 'mean': 0.85, 'std': 0.08},   # Far-right (10%)
    ]
    
    realistic_weights = [0.08, 0.25, 0.34, 0.23, 0.10]
    
    realistic_params = BeliefDistributionParams(
        distribution_type=DistributionType.MIXTURE,
        mixture_components=realistic_components,
        mode_weights=realistic_weights,
        personality_correlations={
            'openness': -0.3,      # More extreme -> less open
            'confidence': 0.4,     # More extreme -> more confident
            'confirmation_bias': 0.5  # More extreme -> more biased
        }
    )
    
    config = ContinuousAgentConfig(
        num_agents=1000,
        topic=TopicType.IMMIGRATION,
        belief_params=realistic_params,
        random_seed=123
    )
    
    agents = create_continuous_agent_population(config)
    beliefs = [agent.belief_strength for agent in agents]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(beliefs, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    plt.title('Realistic Multi-Modal Population\n(Impossible with discrete system)')
    plt.xlabel('Belief Strength')
    plt.ylabel('Density')
    plt.xlim(-1, 1)
    
    # Add cluster labels
    cluster_positions = [-0.85, -0.45, 0.05, 0.55, 0.85]
    cluster_labels = ['Far-Left', 'Center-Left', 'Center', 'Center-Right', 'Far-Right']
    
    for pos, label in zip(cluster_positions, cluster_labels):
        plt.axvline(pos, color='red', linestyle='--', alpha=0.7)
        plt.text(pos, plt.ylim()[1]*0.9, label, rotation=90, 
                verticalalignment='top', fontsize=8)
    
    # Show personality trait distributions by belief extremity
    plt.subplot(1, 2, 2)
    
    # Group agents by belief extremity
    moderate_agents = [a for a in agents if abs(a.belief_strength) < 0.3]
    extreme_agents = [a for a in agents if abs(a.belief_strength) > 0.7]
    
    if moderate_agents and extreme_agents:
        moderate_openness = [a.openness for a in moderate_agents]
        extreme_openness = [a.openness for a in extreme_agents]
        
        plt.hist(moderate_openness, bins=20, alpha=0.7, label='Moderate Beliefs', 
                density=True, color='green')
        plt.hist(extreme_openness, bins=20, alpha=0.7, label='Extreme Beliefs', 
                density=True, color='red')
        
        plt.xlabel('Openness')
        plt.ylabel('Density')
        plt.title('Openness by Belief Extremity')
        plt.legend()
        
        print(f"  Moderate agents (|belief| < 0.3): {len(moderate_agents)} ({len(moderate_agents)/len(agents):.1%})")
        print(f"  Extreme agents (|belief| > 0.7): {len(extreme_agents)} ({len(extreme_agents)/len(agents):.1%})")
        print(f"  Average openness - Moderate: {np.mean(moderate_openness):.3f}")
        print(f"  Average openness - Extreme: {np.mean(extreme_openness):.3f}")
    
    plt.tight_layout()
    plt.savefig('realistic_population.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_parameter_optimization():
    """Demonstrate how continuous parameters enable optimization"""
    
    print("=== PARAMETER OPTIMIZATION POTENTIAL ===\n")
    
    # Simulate optimizing for specific population characteristics
    print("Objective: Find parameters that create 60% moderate, 40% polarized population")
    
    target_moderate_ratio = 0.6
    
    # Test different parameter combinations
    test_configs = []
    results = []
    
    for pol_strength in np.linspace(0.1, 0.9, 9):
        for gap_size in np.linspace(0.2, 0.8, 5):
            
            params = create_polarized_params(
                polarization_strength=pol_strength,
                gap_size=gap_size
            )
            
            config = ContinuousAgentConfig(
                num_agents=500,
                topic=TopicType.GUN_CONTROL,
                belief_params=params,
                random_seed=456
            )
            
            agents = create_continuous_agent_population(config)
            beliefs = [agent.belief_strength for agent in agents]
            
            # Calculate moderate ratio (|belief| < 0.4)
            moderate_count = sum(1 for b in beliefs if abs(b) < 0.4)
            moderate_ratio = moderate_count / len(beliefs)
            
            # Calculate how close to target
            error = abs(moderate_ratio - target_moderate_ratio)
            
            test_configs.append({
                'polarization_strength': pol_strength,
                'gap_size': gap_size,
                'moderate_ratio': moderate_ratio,
                'error': error
            })
            
            results.append([pol_strength, gap_size, moderate_ratio, error])
    
    # Find best configuration
    best_config = min(test_configs, key=lambda x: x['error'])
    
    print(f"Best configuration found:")
    print(f"  Polarization Strength: {best_config['polarization_strength']:.2f}")
    print(f"  Gap Size: {best_config['gap_size']:.2f}")
    print(f"  Achieved Moderate Ratio: {best_config['moderate_ratio']:.3f}")
    print(f"  Target Moderate Ratio: {target_moderate_ratio:.3f}")
    print(f"  Error: {best_config['error']:.3f}")
    
    # Visualize parameter space
    results_array = np.array(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create meshgrid for contour plot
    pol_values = np.unique(results_array[:, 0])
    gap_values = np.unique(results_array[:, 1])
    
    moderate_grid = results_array[:, 2].reshape(len(pol_values), len(gap_values))
    error_grid = results_array[:, 3].reshape(len(pol_values), len(gap_values))
    
    # Plot moderate ratio surface
    c1 = ax1.contourf(pol_values, gap_values, moderate_grid.T, levels=20, cmap='viridis')
    ax1.contour(pol_values, gap_values, moderate_grid.T, levels=[target_moderate_ratio], colors='red', linewidths=2)
    ax1.scatter(best_config['polarization_strength'], best_config['gap_size'], 
               color='red', s=100, marker='*', label='Best Config')
    ax1.set_xlabel('Polarization Strength')
    ax1.set_ylabel('Gap Size')
    ax1.set_title('Moderate Ratio Across Parameter Space')
    ax1.legend()
    plt.colorbar(c1, ax=ax1, label='Moderate Ratio')
    
    # Plot error surface
    c2 = ax2.contourf(pol_values, gap_values, error_grid.T, levels=20, cmap='plasma')
    ax2.scatter(best_config['polarization_strength'], best_config['gap_size'], 
               color='white', s=100, marker='*', label='Best Config')
    ax2.set_xlabel('Polarization Strength')
    ax2.set_ylabel('Gap Size')
    ax2.set_title('Error from Target')
    ax2.legend()
    plt.colorbar(c2, ax=ax2, label='Error')
    
    plt.tight_layout()
    plt.savefig('parameter_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nThis demonstrates how continuous parameters enable:")
    print("  - Systematic parameter space exploration")
    print("  - Optimization for specific population characteristics")
    print("  - Fine-tuning that's impossible with discrete categories")


if __name__ == "__main__":
    print("CONTINUOUS BELIEF DISTRIBUTION DEMONSTRATION")
    print("=" * 60)
    print()
    
    print("This demo shows how continuous parameters provide much more")
    print("flexibility than the current discrete 'polarized', 'normal', 'uniform' system.\n")
    
    try:
        demo_basic_distributions()
        print("\n" + "="*60 + "\n")
        
        demo_parameter_sensitivity()
        print("\n" + "="*60 + "\n")
        
        demo_personality_correlations()
        print("\n" + "="*60 + "\n")
        
        demo_complex_scenarios()
        print("\n" + "="*60 + "\n")
        
        demo_parameter_optimization()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\nKey Benefits of Continuous Approach:")
        print("✓ Fine-grained control over distribution shape")
        print("✓ Support for complex multi-modal distributions") 
        print("✓ Personality-belief correlations")
        print("✓ Parameter optimization and sensitivity analysis")
        print("✓ Realistic modeling of belief distributions")
        print("✓ Backward compatibility with existing code")
        
    except Exception as e:
        print(f"Error running demonstration: {e}")
        print("Make sure you have matplotlib, scipy, and pandas installed:")
        print("pip install matplotlib scipy pandas")