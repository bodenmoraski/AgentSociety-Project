# Continuous Belief Distribution System

This guide explains how to use the new continuous parameter system for belief distributions, which provides much more flexibility than the previous discrete `"polarized"`, `"normal"`, `"uniform"` options.

## Quick Start

### Before (Discrete System)
```python
# Limited to 3 fixed options with hardcoded parameters
agents = create_diverse_agent_population(
    num_agents=100,
    topic=TopicType.GUN_CONTROL,
    belief_distribution="polarized"  # Only 3 choices, fixed parameters
)
```

### After (Continuous System)
```python
from core.continuous_integration import create_highly_polarized_population

# Much more control and customization
agents = create_highly_polarized_population(
    num_agents=100,
    topic=TopicType.GUN_CONTROL,
    polarization_strength=0.75,  # Continuously adjustable 0-1
    asymmetry=0.2                # NEW: right-leaning bias
)
```

## Key Improvements

### 1. **Continuous Parameter Control**
- **Polarization Strength**: 0.0 (uniform) → 1.0 (maximally polarized)
- **Asymmetry**: -1.0 (left-biased) → +1.0 (right-biased)
- **Gap Size**: 0.0 (overlapping) → 1.0 (complete separation)
- **Spread/Concentration**: Fine-tune distribution width

### 2. **Multiple Distribution Types**
- **Bimodal**: Polarized populations with adjustable parameters
- **Normal**: Gaussian distributions with skewness control
- **Beta**: Bounded distributions with shape control
- **Mixture**: Combine multiple distributions with custom weights
- **Custom**: Define your own distribution functions

### 3. **Personality-Belief Correlations**
```python
# Correlate belief strength with personality traits
belief_params.personality_correlations = {
    'openness': -0.4,     # Strong beliefs → less open
    'confidence': 0.5,    # Strong beliefs → more confident
    'confirmation_bias': 0.6  # Strong beliefs → more biased
}
```

### 4. **Multi-Modal Populations**
```python
# Create populations with multiple belief clusters
agents = create_tri_modal_population(
    num_agents=200,
    topic=TopicType.CLIMATE_CHANGE,
    left_weight=0.3,    # 30% left-leaning
    center_weight=0.4,  # 40% moderate
    right_weight=0.3    # 30% right-leaning
)
```

## Usage Examples

### Example 1: Basic Continuous Polarization
```python
from core.continuous_integration import ContinuousAgentConfig, create_continuous_agent_population
from core.continuous_beliefs import create_polarized_params

# Create custom polarized parameters
belief_params = create_polarized_params(
    polarization_strength=0.8,   # Strong polarization
    asymmetry=0.3,              # Right-leaning
    gap_size=0.6                # Large gap between poles
)

config = ContinuousAgentConfig(
    num_agents=150,
    topic=TopicType.HEALTHCARE,
    belief_params=belief_params,
    random_seed=42
)

agents = create_continuous_agent_population(config)
```

### Example 2: Realistic Multi-Modal Distribution
```python
from core.continuous_beliefs import BeliefDistributionParams, DistributionType

# Create a realistic political distribution
realistic_params = BeliefDistributionParams(
    distribution_type=DistributionType.MIXTURE,
    mixture_components=[
        {'type': 'normal', 'mean': -0.8, 'std': 0.1},  # Far-left (10%)
        {'type': 'normal', 'mean': -0.3, 'std': 0.15}, # Center-left (30%)
        {'type': 'normal', 'mean': 0.1, 'std': 0.2},   # Center (35%)
        {'type': 'normal', 'mean': 0.6, 'std': 0.12},  # Center-right (20%)
        {'type': 'normal', 'mean': 0.9, 'std': 0.08},  # Far-right (5%)
    ],
    mode_weights=[0.1, 0.3, 0.35, 0.2, 0.05],
    personality_correlations={
        'openness': -0.5,           # Extreme beliefs → less open
        'confidence': 0.4,          # Strong beliefs → more confident
        'confirmation_bias': 0.6    # Extreme beliefs → more biased
    }
)

config = ContinuousAgentConfig(
    num_agents=500,
    topic=TopicType.IMMIGRATION,
    belief_params=realistic_params
)

agents = create_continuous_agent_population(config)
```

### Example 3: Parameter Exploration
```python
import numpy as np
from core.continuous_beliefs import ContinuousBeliefGenerator

# Explore how polarization strength affects distribution
for pol_strength in np.linspace(0.1, 0.9, 5):
    params = create_polarized_params(polarization_strength=pol_strength)
    generator = ContinuousBeliefGenerator(params)
    
    # Generate sample and analyze
    beliefs = generator.generate_beliefs(1000)
    stats = generator.get_distribution_stats()
    
    print(f"Polarization {pol_strength:.1f}: "
          f"Measured polarization = {stats['polarization_index']:.3f}, "
          f"Diversity = {stats['diversity_index']:.3f}")
```

### Example 4: Backward Compatibility
```python
from core.continuous_integration import create_continuous_population_from_legacy

# Drop-in replacement for old function - uses continuous backend
agents = create_continuous_population_from_legacy(
    num_agents=100,
    topic=TopicType.GUN_CONTROL,
    belief_distribution="polarized"  # Automatically converted to continuous params
)
```

## Advanced Features

### Distribution Analysis
```python
# Analyze distribution characteristics
generator = ContinuousBeliefGenerator(params)
stats = generator.get_distribution_stats(num_samples=5000)

print(f"Polarization Index: {stats['polarization_index']:.3f}")  # 0=uniform, 1=maximally polarized
print(f"Diversity Index: {stats['diversity_index']:.3f}")        # Shannon entropy-based
print(f"Skewness: {stats['skewness']:.3f}")                      # Left/right bias
```

### Visualization
```python
# Visualize distribution with parameters
fig = generator.plot_distribution(num_samples=2000, show_params=True)
fig.savefig('my_distribution.png')
```

### Parameter Optimization
```python
# Find parameters that achieve specific population characteristics
def objective_function(pol_strength, gap_size):
    params = create_polarized_params(
        polarization_strength=pol_strength,
        gap_size=gap_size
    )
    
    config = ContinuousAgentConfig(num_agents=500, belief_params=params)
    agents = create_continuous_agent_population(config)
    
    # Calculate moderate ratio
    moderate_count = sum(1 for a in agents if abs(a.belief_strength) < 0.3)
    moderate_ratio = moderate_count / len(agents)
    
    # Objective: achieve 60% moderate agents
    target_moderate_ratio = 0.6
    return abs(moderate_ratio - target_moderate_ratio)

# Use optimization library to find best parameters
from scipy.optimize import minimize

result = minimize(
    lambda x: objective_function(x[0], x[1]),
    x0=[0.5, 0.5],  # Initial guess
    bounds=[(0.1, 0.9), (0.1, 0.9)]  # Parameter bounds
)

print(f"Optimal parameters: polarization={result.x[0]:.3f}, gap_size={result.x[1]:.3f}")
```

## Integration with Experiments

### Modifying Existing Experiments
```python
# In your experiment configuration, replace:
# OLD:
# belief_distribution="polarized"

# NEW:
from core.continuous_integration import ContinuousAgentConfig
from core.continuous_beliefs import create_polarized_params

belief_params = create_polarized_params(
    polarization_strength=0.75,  # Adjustable
    asymmetry=0.1,              # Slight right bias
    gap_size=0.5                # Medium gap
)

config = ContinuousAgentConfig(
    num_agents=self.config.num_agents,
    topic=self.config.topic,
    belief_params=belief_params,
    random_seed=self.config.random_seed
)

# Use in your experiment
self.agents = create_continuous_agent_population(config)
```

### Experiment Parameter Sweeps
```python
# Test multiple polarization levels in one experiment run
polarization_levels = [0.2, 0.4, 0.6, 0.8]
results_by_polarization = {}

for pol_level in polarization_levels:
    belief_params = create_polarized_params(polarization_strength=pol_level)
    config = ContinuousAgentConfig(belief_params=belief_params, ...)
    
    # Run experiment with this configuration
    experiment = EchoChamberExperiment(config)
    results = experiment.run_full_experiment()
    
    results_by_polarization[pol_level] = results

# Analyze how polarization affects experiment outcomes
```

## Benefits Over Discrete System

1. **Fine-Grained Control**: Adjust parameters continuously instead of 3 fixed options
2. **Parameter Optimization**: Use optimization algorithms to find ideal configurations
3. **Realistic Modeling**: Model real-world belief distributions with multiple clusters
4. **Sensitivity Analysis**: Understand how small parameter changes affect outcomes
5. **Reproducible Research**: Precisely specify and reproduce experimental conditions
6. **Correlation Modeling**: Model relationships between beliefs and personality traits
7. **Backward Compatibility**: Existing code continues to work unchanged

## Migration Guide

### Step 1: Update Imports
```python
# Add these imports
from core.continuous_integration import (
    ContinuousAgentConfig, 
    create_continuous_agent_population,
    create_highly_polarized_population
)
from core.continuous_beliefs import create_polarized_params, create_moderate_params
```

### Step 2: Replace Agent Creation
```python
# OLD:
agents = create_diverse_agent_population(num_agents, topic, "polarized")

# NEW (drop-in replacement):
agents = create_continuous_population_from_legacy(num_agents, topic, "polarized")

# NEW (with customization):
agents = create_highly_polarized_population(
    num_agents, topic, 
    polarization_strength=0.8, 
    asymmetry=0.1
)
```

### Step 3: Experiment with Parameters
Start with the convenience functions, then gradually explore more advanced features:

1. `create_highly_polarized_population()` - Adjustable polarization
2. `create_tri_modal_population()` - Three belief clusters
3. `create_custom_population()` - Full parameter control
4. Custom `BeliefDistributionParams` - Maximum flexibility

## Performance Notes

- **Generation Speed**: Similar to discrete system for basic distributions
- **Memory Usage**: Negligible increase
- **Flexibility**: Dramatically increased with continuous parameters
- **Compatibility**: 100% backward compatible with existing experiments

## Troubleshooting

### Common Issues

1. **Parameters out of range**: Most parameters are automatically clipped to valid ranges
2. **Mixture weights don't sum to 1**: Automatically normalized
3. **Correlation values too strong**: Keep correlations in [-1, 1] range, typically [-0.8, 0.8]

### Debugging

```python
# Check distribution statistics
generator = ContinuousBeliefGenerator(params)
stats = generator.get_distribution_stats()
print(stats)

# Visualize distribution
fig = generator.plot_distribution()
fig.show()

# Test with small population first
test_agents = create_continuous_agent_population(
    replace(config, num_agents=50)
)
```

## Examples and Demos

Run the comprehensive demonstration:
```bash
cd AgentSociety/echo_chamber_experiments/examples
python continuous_beliefs_demo.py
```

This will generate visualizations showing:
- Comparison with discrete system
- Parameter sensitivity analysis
- Personality correlations
- Complex multi-modal scenarios
- Parameter optimization examples

---

**Next Steps**: Start with the convenience functions, then gradually explore the full parameter space to find configurations that best match your experimental needs!