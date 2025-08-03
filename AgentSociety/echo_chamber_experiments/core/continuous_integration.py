"""
Integration module for continuous belief distributions with the existing agent framework.

This module provides backward-compatible integration while enabling the new
continuous parameter system.
"""

import random
import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, replace

from .agent import Agent, TopicType, PersonalityType, create_diverse_agent_population
from .continuous_beliefs import (
    BeliefDistributionParams, 
    ContinuousBeliefGenerator, 
    DistributionType,
    create_polarized_params,
    create_moderate_params, 
    create_uniform_params
)


@dataclass
class ContinuousAgentConfig:
    """
    Extended agent configuration that supports continuous belief parameters.
    
    This replaces the simple string-based belief_distribution with rich
    continuous parameters while maintaining backward compatibility.
    """
    
    # Basic parameters (same as before)
    num_agents: int = 50
    topic: TopicType = TopicType.GUN_CONTROL
    
    # New continuous belief parameters
    belief_params: BeliefDistributionParams = None
    
    # Legacy support
    belief_distribution: Optional[str] = None  # For backward compatibility
    
    # Trait generation parameters
    trait_correlations: Dict[str, Dict[str, float]] = None
    personality_distribution: Dict[PersonalityType, float] = None
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Initialize default parameters and handle backward compatibility"""
        
        # Handle backward compatibility with old string-based distributions
        if self.belief_params is None:
            if self.belief_distribution == "polarized":
                self.belief_params = create_polarized_params()
            elif self.belief_distribution == "normal":
                self.belief_params = create_moderate_params()
            elif self.belief_distribution == "uniform":
                self.belief_params = create_uniform_params()
            else:
                # Default to moderate polarization
                self.belief_params = create_polarized_params(polarization_strength=0.7)
        
        # Set default trait correlations if not provided
        if self.trait_correlations is None:
            self.trait_correlations = {
                'belief_strength': {
                    'confidence': 0.3,      # Strong beliefs -> more confident
                    'openness': -0.2,       # Strong beliefs -> less open
                    'confirmation_bias': 0.4 # Strong beliefs -> more biased
                }
            }
        
        # Set default personality distribution if not provided
        if self.personality_distribution is None:
            self.personality_distribution = {
                PersonalityType.CONFORMIST: 0.3,
                PersonalityType.CONTRARIAN: 0.15,
                PersonalityType.INDEPENDENT: 0.4,
                PersonalityType.AMPLIFIER: 0.15
            }


def create_continuous_agent_population(config: ContinuousAgentConfig) -> List[Agent]:
    """
    Create agent population using continuous belief distribution parameters.
    
    This is the new main function that replaces create_diverse_agent_population
    when using continuous parameters.
    
    Args:
        config: Configuration object with continuous parameters
        
    Returns:
        List of agents with beliefs generated from continuous distributions
    """
    
    # Set random seed if provided
    if config.random_seed is not None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    # Create belief generator
    belief_generator = ContinuousBeliefGenerator(config.belief_params)
    if config.random_seed is not None:
        belief_generator.set_seed(config.random_seed)
    
    # Pre-generate personality traits to use for belief correlations
    agent_traits = []
    personality_types = []
    
    for i in range(config.num_agents):
        # Generate base personality traits
        traits = _generate_personality_traits()
        agent_traits.append(traits)
        
        # Select personality type based on distribution
        personality_type = _select_personality_type(config.personality_distribution)
        personality_types.append(personality_type)
    
    # Generate beliefs with personality correlations
    beliefs = belief_generator.generate_beliefs(
        config.num_agents, 
        agent_traits=agent_traits
    )
    
    # Create agents
    agents = []
    for i in range(config.num_agents):
        # Get traits and apply correlations with belief strength
        traits = agent_traits[i].copy()
        belief_strength = beliefs[i]
        personality_type = personality_types[i]
        
        # Apply trait correlations based on belief strength
        traits = _apply_trait_correlations(
            traits, 
            belief_strength, 
            config.trait_correlations
        )
        
        # Adjust traits based on personality type (existing logic)
        traits = _adjust_traits_for_personality(traits, personality_type)
        
        # Create agent
        agent = Agent(
            id=i,
            name=f"Agent_{i:03d}",
            belief_strength=belief_strength,
            topic=config.topic,
            openness=traits['openness'],
            confidence=traits['confidence'],
            sociability=traits['sociability'],
            confirmation_bias=traits['confirmation_bias'],
            personality_type=personality_type,
            influence_power=traits['influence_power'],
            network_centrality=traits['network_centrality']
        )
        
        agents.append(agent)
    
    return agents


def _generate_personality_traits() -> Dict[str, float]:
    """Generate base personality traits using realistic distributions"""
    return {
        'openness': np.random.beta(2, 2),
        'confidence': np.random.beta(2, 3),
        'sociability': np.random.beta(2, 2),
        'confirmation_bias': np.random.beta(3, 2),
        'influence_power': np.random.beta(2, 3),
        'network_centrality': np.random.beta(2, 2)
    }


def _select_personality_type(distribution: Dict[PersonalityType, float]) -> PersonalityType:
    """Select personality type based on probability distribution"""
    types = list(distribution.keys())
    weights = list(distribution.values())
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    return random.choices(types, weights=weights)[0]


def _apply_trait_correlations(traits: Dict[str, float], 
                            belief_strength: float,
                            correlations: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Apply correlations between belief strength and personality traits"""
    
    modified_traits = traits.copy()
    
    if 'belief_strength' in correlations:
        belief_correlations = correlations['belief_strength']
        
        # Apply each correlation
        for trait_name, correlation in belief_correlations.items():
            if trait_name in modified_traits:
                # Use absolute belief strength for correlation with traits like confidence
                # (strong beliefs in either direction correlate with confidence)
                belief_magnitude = abs(belief_strength)
                
                # Calculate adjustment based on belief magnitude and correlation
                # Scale the effect to be reasonable (max ~0.3 adjustment)
                adjustment = belief_magnitude * correlation * 0.3
                
                # Apply adjustment while keeping traits in [0, 1] range
                current_value = modified_traits[trait_name]
                new_value = current_value + adjustment
                modified_traits[trait_name] = np.clip(new_value, 0.0, 1.0)
    
    return modified_traits


def _adjust_traits_for_personality(traits: Dict[str, float], 
                                 personality_type: PersonalityType) -> Dict[str, float]:
    """Apply personality type adjustments (existing logic from agent.py)"""
    
    modified_traits = traits.copy()
    
    if personality_type == PersonalityType.CONFORMIST:
        modified_traits['openness'] = min(1.0, modified_traits['openness'] * 1.5)
        modified_traits['confidence'] *= 0.8
    elif personality_type == PersonalityType.CONTRARIAN:
        modified_traits['confirmation_bias'] *= 0.5
        modified_traits['confidence'] = min(1.0, modified_traits['confidence'] * 1.3)
    elif personality_type == PersonalityType.INDEPENDENT:
        modified_traits['openness'] *= 0.6
        modified_traits['confirmation_bias'] *= 0.7
    elif personality_type == PersonalityType.AMPLIFIER:
        modified_traits['influence_power'] = min(1.0, modified_traits['influence_power'] * 1.4)
        modified_traits['sociability'] = min(1.0, modified_traits['sociability'] * 1.2)
    
    return modified_traits


# === Backward Compatibility Functions ===

def create_continuous_population_from_legacy(num_agents: int,
                                           topic: TopicType,
                                           belief_distribution: str = "polarized") -> List[Agent]:
    """
    Create population using the new continuous system but with legacy string parameters.
    
    This provides a drop-in replacement for create_diverse_agent_population
    while using the more flexible continuous backend.
    """
    
    config = ContinuousAgentConfig(
        num_agents=num_agents,
        topic=topic,
        belief_distribution=belief_distribution
    )
    
    return create_continuous_agent_population(config)


# === Convenience Functions for Common Configurations ===

def create_highly_polarized_population(num_agents: int, 
                                     topic: TopicType,
                                     polarization_strength: float = 0.9,
                                     asymmetry: float = 0.0) -> List[Agent]:
    """Create a highly polarized population with adjustable parameters"""
    
    belief_params = create_polarized_params(
        polarization_strength=polarization_strength,
        asymmetry=asymmetry
    )
    
    config = ContinuousAgentConfig(
        num_agents=num_agents,
        topic=topic,
        belief_params=belief_params
    )
    
    return create_continuous_agent_population(config)


def create_moderate_population(num_agents: int,
                             topic: TopicType, 
                             center_bias: float = 0.0,
                             spread: float = 0.4) -> List[Agent]:
    """Create a moderate/centrist population"""
    
    belief_params = create_moderate_params(
        center_bias=center_bias,
        spread=spread
    )
    
    config = ContinuousAgentConfig(
        num_agents=num_agents,
        topic=topic,
        belief_params=belief_params
    )
    
    return create_continuous_agent_population(config)


def create_tri_modal_population(num_agents: int,
                              topic: TopicType,
                              left_weight: float = 0.3,
                              center_weight: float = 0.4,
                              right_weight: float = 0.3) -> List[Agent]:
    """Create a population with three belief clusters (left, center, right)"""
    
    # Create tri-modal mixture
    mixture_components = [
        {'type': 'normal', 'mean': -0.7, 'std': 0.15},  # Left cluster
        {'type': 'normal', 'mean': 0.0, 'std': 0.1},    # Center cluster  
        {'type': 'normal', 'mean': 0.7, 'std': 0.15},   # Right cluster
    ]
    
    belief_params = BeliefDistributionParams(
        distribution_type=DistributionType.MIXTURE,
        mixture_components=mixture_components,
        mode_weights=[left_weight, center_weight, right_weight]
    )
    
    config = ContinuousAgentConfig(
        num_agents=num_agents,
        topic=topic,
        belief_params=belief_params
    )
    
    return create_continuous_agent_population(config)


def create_custom_population(num_agents: int,
                           topic: TopicType,
                           belief_params: BeliefDistributionParams,
                           personality_correlations: Optional[Dict[str, Dict[str, float]]] = None,
                           random_seed: Optional[int] = None) -> List[Agent]:
    """Create a population with fully custom parameters"""
    
    config = ContinuousAgentConfig(
        num_agents=num_agents,
        topic=topic,
        belief_params=belief_params,
        trait_correlations=personality_correlations,
        random_seed=random_seed
    )
    
    return create_continuous_agent_population(config)


# === Analysis and Comparison Tools ===

def compare_distributions(configs: List[ContinuousAgentConfig], 
                         num_samples: int = 1000) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple belief distribution configurations.
    
    Useful for exploring how different parameters affect population characteristics.
    """
    
    results = {}
    
    for i, config in enumerate(configs):
        # Generate sample population
        agents = create_continuous_agent_population(
            replace(config, num_agents=num_samples)
        )
        
        # Extract beliefs
        beliefs = [agent.belief_strength for agent in agents]
        
        # Calculate statistics
        belief_generator = ContinuousBeliefGenerator(config.belief_params)
        stats = belief_generator.get_distribution_stats(num_samples)
        
        # Add population-level stats
        stats['extreme_agents'] = sum(1 for b in beliefs if abs(b) > 0.8) / len(beliefs)
        stats['moderate_agents'] = sum(1 for b in beliefs if abs(b) < 0.3) / len(beliefs)
        
        results[f'config_{i}'] = stats
    
    return results


# === Usage Examples ===

if __name__ == "__main__":
    # Example 1: Drop-in replacement for legacy function
    legacy_agents = create_continuous_population_from_legacy(
        num_agents=100,
        topic=TopicType.GUN_CONTROL,
        belief_distribution="polarized"
    )
    
    print(f"Created {len(legacy_agents)} agents with legacy interface")
    print(f"Belief range: {min(a.belief_strength for a in legacy_agents):.2f} to {max(a.belief_strength for a in legacy_agents):.2f}")
    
    # Example 2: Highly customized population
    custom_belief_params = BeliefDistributionParams(
        distribution_type=DistributionType.BIMODAL,
        polarization_strength=0.8,
        polarization_asymmetry=0.3,  # Right-leaning
        gap_size=0.6,
        personality_correlations={
            'openness': -0.4,    # Strong beliefs -> less open
            'confidence': 0.5,   # Strong beliefs -> more confident
        }
    )
    
    custom_agents = create_custom_population(
        num_agents=50,
        topic=TopicType.CLIMATE_CHANGE,
        belief_params=custom_belief_params,
        random_seed=42
    )
    
    print(f"\nCreated {len(custom_agents)} agents with custom parameters")
    
    # Example 3: Compare different configurations
    configs_to_compare = [
        ContinuousAgentConfig(
            num_agents=100,
            topic=TopicType.GUN_CONTROL,
            belief_params=create_polarized_params(polarization_strength=0.9)
        ),
        ContinuousAgentConfig(
            num_agents=100,
            topic=TopicType.GUN_CONTROL,
            belief_params=create_moderate_params(spread=0.3)
        )
    ]
    
    comparison = compare_distributions(configs_to_compare)
    
    print("\nDistribution Comparison:")
    for config_name, stats in comparison.items():
        print(f"{config_name}:")
        print(f"  Polarization: {stats['polarization_index']:.3f}")
        print(f"  Diversity: {stats['diversity_index']:.3f}")
        print(f"  Extreme agents: {stats['extreme_agents']:.1%}")