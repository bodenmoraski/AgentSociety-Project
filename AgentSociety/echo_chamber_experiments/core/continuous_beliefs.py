"""
Continuous Parameter-Based Belief Distribution System

This module provides a flexible framework for generating belief distributions
with continuous parameters instead of discrete categorical options.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.stats import norm, beta, gamma, uniform, truncnorm
import matplotlib.pyplot as plt


class DistributionType(Enum):
    """Supported distribution types for belief generation"""
    NORMAL = "normal"
    BETA = "beta" 
    GAMMA = "gamma"
    UNIFORM = "uniform"
    TRUNCATED_NORMAL = "truncated_normal"
    BIMODAL = "bimodal"
    MIXTURE = "mixture"
    CUSTOM = "custom"


@dataclass
class BeliefDistributionParams:
    """
    Comprehensive parameter set for flexible belief distribution generation.
    
    This replaces the simple string-based approach with rich continuous parameters
    that allow fine-grained control over belief distribution shapes.
    """
    
    # === Core Distribution Parameters ===
    distribution_type: DistributionType = DistributionType.BIMODAL
    
    # === Central Tendency Parameters ===
    center: float = 0.0                    # Distribution center (-1 to 1)
    spread: float = 0.6                    # Overall spread/variance 
    skewness: float = 0.0                  # Left/right skew (-2 to 2)
    
    # === Polarization Parameters ===
    polarization_strength: float = 0.7     # 0=uniform, 1=maximally polarized
    polarization_asymmetry: float = 0.0    # -1=left heavy, +1=right heavy
    gap_size: float = 0.4                  # Size of gap between poles (0-1)
    
    # === Multi-Modal Parameters ===
    num_modes: int = 2                     # Number of belief clusters
    mode_separation: float = 1.2           # Distance between modes
    mode_weights: Optional[List[float]] = None  # Relative size of each mode
    
    # === Advanced Shape Parameters ===
    tail_heaviness: float = 0.5            # 0=light tails, 1=heavy tails
    concentration: float = 0.5             # Concentration around modes
    
    # === Mixture Model Parameters ===
    mixture_components: Optional[List[Dict]] = None  # For custom mixtures
    
    # === Correlation Parameters ===
    personality_correlations: Dict[str, float] = field(default_factory=lambda: {
        'openness': 0.0,        # More open -> more moderate beliefs
        'confidence': 0.0,      # More confident -> more extreme beliefs  
        'contrarianism': 0.0,   # More contrarian -> opposite of majority
    })
    
    # === Bounds and Constraints ===
    min_belief: float = -1.0
    max_belief: float = 1.0
    
    # === Random Variation ===
    noise_level: float = 0.05              # Random noise added to beliefs
    
    def __post_init__(self):
        """Validate and normalize parameters"""
        # Ensure valid ranges
        self.center = np.clip(self.center, self.min_belief, self.max_belief)
        self.spread = np.clip(self.spread, 0.01, 2.0)
        self.polarization_strength = np.clip(self.polarization_strength, 0.0, 1.0)
        
        # Set default mode weights if not provided
        if self.mode_weights is None:
            self.mode_weights = [1.0 / self.num_modes] * self.num_modes


class ContinuousBeliefGenerator:
    """
    Advanced belief distribution generator with continuous parameters.
    
    This class replaces the simple categorical approach with a flexible
    parameter-based system that can generate a wide variety of belief
    distribution shapes.
    """
    
    def __init__(self, params: BeliefDistributionParams):
        self.params = params
        self.rng = np.random.RandomState()  # For reproducibility
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible generation"""
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
    
    def generate_beliefs(self, num_agents: int, 
                        agent_traits: Optional[List[Dict]] = None) -> List[float]:
        """
        Generate belief values for a population of agents.
        
        Args:
            num_agents: Number of agents to generate beliefs for
            agent_traits: Optional list of agent personality traits for correlation
            
        Returns:
            List of belief values in range [min_belief, max_belief]
        """
        
        if self.params.distribution_type == DistributionType.BIMODAL:
            beliefs = self._generate_bimodal_beliefs(num_agents)
        elif self.params.distribution_type == DistributionType.NORMAL:
            beliefs = self._generate_normal_beliefs(num_agents)
        elif self.params.distribution_type == DistributionType.BETA:
            beliefs = self._generate_beta_beliefs(num_agents)
        elif self.params.distribution_type == DistributionType.MIXTURE:
            beliefs = self._generate_mixture_beliefs(num_agents)
        elif self.params.distribution_type == DistributionType.UNIFORM:
            beliefs = self._generate_uniform_beliefs(num_agents)
        else:
            # Default to bimodal
            beliefs = self._generate_bimodal_beliefs(num_agents)
        
        # Apply personality correlations if provided
        if agent_traits:
            beliefs = self._apply_personality_correlations(beliefs, agent_traits)
        
        # Add noise and clip to bounds
        beliefs = self._add_noise_and_clip(beliefs)
        
        return beliefs.tolist()
    
    def _generate_bimodal_beliefs(self, num_agents: int) -> np.ndarray:
        """Generate bimodal (polarized) belief distribution"""
        
        # Calculate mode positions based on parameters
        gap_half = self.params.gap_size / 2
        left_mode = self.params.center - self.params.mode_separation/2 - gap_half
        right_mode = self.params.center + self.params.mode_separation/2 + gap_half
        
        # Adjust for asymmetry
        asymmetry_shift = self.params.polarization_asymmetry * 0.3
        left_mode -= asymmetry_shift
        right_mode += asymmetry_shift
        
        # Determine assignment to modes based on polarization strength
        mode_assignment_prob = 0.5 + self.params.polarization_asymmetry * 0.3
        
        beliefs = np.zeros(num_agents)
        
        for i in range(num_agents):
            if self.rng.random() < mode_assignment_prob:
                # Right mode
                mode_std = self.params.spread * (1 - self.params.polarization_strength * 0.7)
                belief = self.rng.normal(right_mode, mode_std)
            else:
                # Left mode  
                mode_std = self.params.spread * (1 - self.params.polarization_strength * 0.7)
                belief = self.rng.normal(left_mode, mode_std)
            
            beliefs[i] = belief
        
        return beliefs
    
    def _generate_normal_beliefs(self, num_agents: int) -> np.ndarray:
        """Generate normal distribution beliefs"""
        
        # Adjust standard deviation based on spread and concentration
        std = self.params.spread * (1.0 + self.params.concentration)
        
        # Generate beliefs with skewness
        if abs(self.params.skewness) < 0.01:
            # Regular normal distribution
            beliefs = self.rng.normal(self.params.center, std, num_agents)
        else:
            # Skewed normal using skewnorm
            from scipy.stats import skewnorm
            beliefs = skewnorm.rvs(
                a=self.params.skewness, 
                loc=self.params.center,
                scale=std,
                size=num_agents,
                random_state=self.rng
            )
        
        return beliefs
    
    def _generate_beta_beliefs(self, num_agents: int) -> np.ndarray:
        """Generate beta distribution beliefs (good for bounded distributions)"""
        
        # Convert center and spread to beta parameters
        mean = (self.params.center - self.params.min_belief) / (self.params.max_belief - self.params.min_belief)
        
        # Derive alpha and beta parameters from mean and concentration
        if self.params.concentration > 0:
            total = 2 + self.params.concentration * 10  # Higher concentration = more peaked
            alpha = mean * total
            beta = (1 - mean) * total
        else:
            alpha = beta = 1  # Uniform beta
        
        # Generate from beta and scale to belief range
        beta_samples = self.rng.beta(alpha, beta, num_agents)
        beliefs = beta_samples * (self.params.max_belief - self.params.min_belief) + self.params.min_belief
        
        return beliefs
    
    def _generate_mixture_beliefs(self, num_agents: int) -> np.ndarray:
        """Generate mixture of multiple distributions"""
        
        if not self.params.mixture_components:
            # Default to bimodal if no mixture specified
            return self._generate_bimodal_beliefs(num_agents)
        
        beliefs = np.zeros(num_agents)
        
        for i in range(num_agents):
            # Choose component based on weights
            component_idx = self.rng.choice(
                len(self.params.mixture_components),
                p=self.params.mode_weights[:len(self.params.mixture_components)]
            )
            
            component = self.params.mixture_components[component_idx]
            
            # Generate from chosen component
            if component['type'] == 'normal':
                belief = self.rng.normal(component['mean'], component['std'])
            elif component['type'] == 'uniform':
                belief = self.rng.uniform(component['min'], component['max'])
            else:
                belief = self.rng.normal(0, self.params.spread)  # Fallback
            
            beliefs[i] = belief
        
        return beliefs
    
    def _generate_uniform_beliefs(self, num_agents: int) -> np.ndarray:
        """Generate uniform distribution beliefs"""
        
        # Calculate range based on center and spread
        range_size = self.params.spread * (self.params.max_belief - self.params.min_belief)
        min_val = max(self.params.min_belief, self.params.center - range_size/2)
        max_val = min(self.params.max_belief, self.params.center + range_size/2)
        
        beliefs = self.rng.uniform(min_val, max_val, num_agents)
        return beliefs
    
    def _apply_personality_correlations(self, beliefs: np.ndarray, 
                                      agent_traits: List[Dict]) -> np.ndarray:
        """Apply correlations between beliefs and personality traits"""
        
        if not self.params.personality_correlations:
            return beliefs
        
        modified_beliefs = beliefs.copy()
        
        for i, traits in enumerate(agent_traits):
            adjustment = 0.0
            
            # Apply each correlation
            for trait_name, correlation in self.params.personality_correlations.items():
                if trait_name in traits:
                    trait_value = traits[trait_name]
                    # Convert trait to adjustment (-1 to 1 range)
                    trait_effect = (trait_value - 0.5) * 2  # Scale 0-1 to -1 to 1
                    adjustment += correlation * trait_effect * 0.3  # Scale effect
            
            modified_beliefs[i] += adjustment
        
        return modified_beliefs
    
    def _add_noise_and_clip(self, beliefs: np.ndarray) -> np.ndarray:
        """Add random noise and clip to valid range"""
        
        # Add noise
        if self.params.noise_level > 0:
            noise = self.rng.normal(0, self.params.noise_level, len(beliefs))
            beliefs += noise
        
        # Clip to bounds
        beliefs = np.clip(beliefs, self.params.min_belief, self.params.max_belief)
        
        return beliefs
    
    def plot_distribution(self, num_samples: int = 1000, 
                         show_params: bool = True) -> plt.Figure:
        """
        Visualize the belief distribution with current parameters.
        
        Useful for exploring and tuning distribution parameters.
        """
        
        # Generate sample beliefs
        sample_beliefs = self.generate_beliefs(num_samples)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(sample_beliefs, bins=50, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax1.set_xlabel('Belief Strength')
        ax1.set_ylabel('Density')
        ax1.set_title('Belief Distribution')
        ax1.set_xlim(self.params.min_belief, self.params.max_belief)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(sample_beliefs, vert=True)
        ax2.set_ylabel('Belief Strength')
        ax2.set_title('Belief Distribution Summary')
        ax2.grid(True, alpha=0.3)
        
        # Add parameter info
        if show_params:
            param_text = f"""Parameters:
Type: {self.params.distribution_type.value}
Center: {self.params.center:.2f}
Spread: {self.params.spread:.2f}
Polarization: {self.params.polarization_strength:.2f}
Skewness: {self.params.skewness:.2f}
Modes: {self.params.num_modes}"""
            
            fig.text(0.02, 0.98, param_text, fontsize=9, 
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def get_distribution_stats(self, num_samples: int = 10000) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for the current distribution.
        
        Useful for understanding and comparing different parameter settings.
        """
        
        sample_beliefs = self.generate_beliefs(num_samples)
        
        stats_dict = {
            'mean': np.mean(sample_beliefs),
            'std': np.std(sample_beliefs),
            'skewness': stats.skew(sample_beliefs),
            'kurtosis': stats.kurtosis(sample_beliefs),
            'min': np.min(sample_beliefs),
            'max': np.max(sample_beliefs),
            'median': np.median(sample_beliefs),
            'q25': np.percentile(sample_beliefs, 25),
            'q75': np.percentile(sample_beliefs, 75),
            'iqr': np.percentile(sample_beliefs, 75) - np.percentile(sample_beliefs, 25),
            'polarization_index': self._calculate_polarization_index(sample_beliefs),
            'diversity_index': self._calculate_diversity_index(sample_beliefs)
        }
        
        return stats_dict
    
    def _calculate_polarization_index(self, beliefs: np.ndarray) -> float:
        """Calculate a polarization index (0=uniform, 1=maximally polarized)"""
        
        # Count beliefs in extreme regions vs center
        extreme_threshold = 0.6
        center_threshold = 0.2
        
        extreme_count = np.sum((np.abs(beliefs) > extreme_threshold))
        center_count = np.sum((np.abs(beliefs) < center_threshold))
        total = len(beliefs)
        
        if total == 0:
            return 0.0
        
        # High polarization = many extremes, few moderates
        polarization = (extreme_count / total) - (center_count / total)
        return np.clip(polarization, 0.0, 1.0)
    
    def _calculate_diversity_index(self, beliefs: np.ndarray) -> float:
        """Calculate belief diversity using Shannon entropy"""
        
        # Bin beliefs and calculate entropy
        hist, _ = np.histogram(beliefs, bins=20, range=(self.params.min_belief, self.params.max_belief))
        hist = hist + 1e-10  # Avoid log(0)
        probs = hist / np.sum(hist)
        
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs))  # Maximum possible entropy
        
        return entropy / max_entropy  # Normalize to 0-1


# === Convenient Preset Configurations ===

def create_polarized_params(polarization_strength: float = 0.8,
                           asymmetry: float = 0.0,
                           gap_size: float = 0.4) -> BeliefDistributionParams:
    """Create parameters for polarized belief distribution"""
    return BeliefDistributionParams(
        distribution_type=DistributionType.BIMODAL,
        polarization_strength=polarization_strength,
        polarization_asymmetry=asymmetry,
        gap_size=gap_size,
        spread=0.3
    )


def create_moderate_params(center_bias: float = 0.0,
                          spread: float = 0.4) -> BeliefDistributionParams:
    """Create parameters for moderate/centrist belief distribution"""
    return BeliefDistributionParams(
        distribution_type=DistributionType.NORMAL,
        center=center_bias,
        spread=spread,
        concentration=0.7
    )


def create_uniform_params(range_min: float = -1.0,
                         range_max: float = 1.0) -> BeliefDistributionParams:
    """Create parameters for uniform belief distribution"""
    return BeliefDistributionParams(
        distribution_type=DistributionType.UNIFORM,
        center=(range_min + range_max) / 2,
        spread=range_max - range_min,
        min_belief=range_min,
        max_belief=range_max
    )


def create_custom_mixture_params(components: List[Dict],
                                weights: List[float]) -> BeliefDistributionParams:
    """Create parameters for custom mixture distribution"""
    return BeliefDistributionParams(
        distribution_type=DistributionType.MIXTURE,
        mixture_components=components,
        mode_weights=weights
    )


# === Usage Examples ===

if __name__ == "__main__":
    # Example 1: Highly polarized distribution
    polarized_params = create_polarized_params(polarization_strength=0.9, asymmetry=0.2)
    generator = ContinuousBeliefGenerator(polarized_params)
    beliefs = generator.generate_beliefs(1000)
    
    print("Polarized Distribution Stats:")
    stats = generator.get_distribution_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    # Example 2: Moderate distribution with personality correlations
    moderate_params = create_moderate_params(center_bias=0.1, spread=0.5)
    moderate_params.personality_correlations = {
        'openness': -0.3,     # More open -> more moderate
        'confidence': 0.4,    # More confident -> more extreme
    }
    
    # Example 3: Custom mixture
    mixture_components = [
        {'type': 'normal', 'mean': -0.6, 'std': 0.2},  # Left cluster
        {'type': 'normal', 'mean': 0.0, 'std': 0.1},   # Center cluster  
        {'type': 'normal', 'mean': 0.7, 'std': 0.25},  # Right cluster
    ]
    mixture_params = create_custom_mixture_params(
        components=mixture_components,
        weights=[0.3, 0.4, 0.3]
    )
    
    print("\nExample parameter configurations created successfully!")