# Echo Chamber Social Dynamics Experiments

A comprehensive framework for studying belief propagation, polarization, and echo chamber formation in AI agent societies. This project simulates realistic social dynamics to understand how opinions spread, how echo chambers form, and how interventions might reduce harmful polarization.

## 🎯 Features

### Core Capabilities
- **Diverse Agent Population**: Agents with different personalities, openness levels, and social traits
- **Multiple Network Types**: Random, small-world, scale-free, and preferential attachment networks
- **Realistic Social Dynamics**: Homophily-based connections, belief-based influence, and dynamic rewiring
- **Intervention Testing**: Fact-checking, diverse exposure, and bridge-building interventions
- **Rich Analysis**: Comprehensive metrics, visualizations, and statistical analysis

### Belief Distribution System (NEW!)
- **Continuous Parameters**: Fine-grained control over belief distributions instead of discrete categories
- **Adjustable Polarization**: Polarization strength from 0.0 (uniform) to 1.0 (maximally polarized)
- **Asymmetric Bias**: Left/right bias control (-1.0 to +1.0)
- **Multi-Modal Populations**: Support for complex realistic distributions with multiple belief clusters
- **Personality Correlations**: Link belief strength with personality traits (confidence, openness, etc.)
- **Backward Compatible**: Drop-in replacement for existing experiments

### Topics Available
- Gun Control
- Climate Change
- Healthcare Policy
- Taxation
- Immigration
- Technology Regulation

### Agent Personality Types
- **Conformist**: Easily influenced by confident agents and majority opinions
- **Contrarian**: Tends to oppose popular opinions and resist influence
- **Independent**: Less influenced by others, makes decisions autonomously
- **Amplifier**: Spreads beliefs with high intensity and social activity

## 🚀 Quick Start

### Installation

1. **Clone and set up environment**:
```bash
cd AgentSociety/echo_chamber_experiments
pip install -r ../requirements.txt
```

2. **Run a predefined experiment**:
```bash
python run_experiment.py run --experiment basic_polarization
```

3. **View results**:
- Check the `results/` directory for JSON data, summary reports, and visualizations
- Open `interactive_dashboard.html` in your browser for interactive exploration

### Basic Usage Examples

#### 1. Run Predefined Experiments
```bash
# Basic polarization study
python run_experiment.py run --experiment basic_polarization

# Intervention effectiveness study  
python run_experiment.py run --experiment intervention_study

# Bridge building experiment
python run_experiment.py run --experiment bridge_building

# Large-scale population dynamics
python run_experiment.py run --experiment large_scale
```

#### 2. Custom Experiments
```bash
# Custom experiment with specific parameters
python run_experiment.py run --custom \
    --agents 100 \
    --rounds 20 \
    --topic climate_change \
    --network small_world \
    --intervention fact_check \
    --intervention-round 10 \
    --interactive

# With reproducible seed
python run_experiment.py run --custom \
    --agents 50 \
    --rounds 15 \
    --topic gun_control \
    --seed 42
```

#### 3. Configuration File Based
```bash
# Create a configuration template
python run_experiment.py create-config --name "my_study" --agents 75

# Run from configuration file
python run_experiment.py run --config-file configs/climate_debate_config.json
```

#### 4. Continuous Belief Distributions (NEW!)
```python
from core.continuous_integration import create_highly_polarized_population, create_tri_modal_population
from core.agent import TopicType

# Adjustable polarization with bias control
agents = create_highly_polarized_population(
    num_agents=100,
    topic=TopicType.CLIMATE_CHANGE,
    polarization_strength=0.75,  # 0.0-1.0 continuous control
    asymmetry=0.2                # Right-leaning bias
)

# Realistic multi-modal population
agents = create_tri_modal_population(
    num_agents=200,
    topic=TopicType.HEALTHCARE,
    left_weight=0.3,    # 30% left-leaning
    center_weight=0.4,  # 40% moderate  
    right_weight=0.3    # 30% right-leaning
)
```

## 📊 Understanding Results

### Key Metrics

1. **Polarization**: Average distance of beliefs from center (0-1)
2. **Echo Chamber Count**: Number of isolated belief communities
3. **Network Fragmentation**: Proportion of disconnected components
4. **Bridge Agents**: Agents connecting different belief groups

### Output Files

Each experiment generates:
- `results_{timestamp}.json`: Complete experimental data
- `summary_{timestamp}.txt`: Human-readable summary report
- `agent_data_{timestamp}.csv`: Agent-level data for analysis
- `visualizations_{timestamp}/`: Plots and interactive dashboard

### Visualization Types

1. **Belief Evolution**: How individual beliefs change over time
2. **Network Analysis**: Network structure and community formation
3. **Agent Analysis**: Personality effects and influence patterns
4. **Interactive Dashboard**: Comprehensive explorable interface

## 🔬 Experiment Types

### 1. Basic Polarization Study
Studies natural echo chamber formation without interventions.

**Best for**: Understanding baseline dynamics, testing network effects

### 2. Intervention Studies
Tests the effectiveness of different intervention strategies:
- **Fact Checking**: Moderation of extreme beliefs
- **Diverse Exposure**: Cross-cutting interactions
- **Bridge Building**: Strategic connection formation

**Best for**: Policy research, platform design decisions

### 3. Large-Scale Dynamics
High agent count simulations for population-level insights.

**Best for**: Generalizability testing, emergent behavior study

### 4. Custom Studies
Fully configurable experiments for specific research questions.

**Best for**: Novel hypotheses, parameter sensitivity analysis

## 🛠️ Advanced Usage

### Programming Interface

```python
from echo_chamber_experiments import run_predefined_experiment, EchoChamberExperiment
from echo_chamber_experiments.core import ExperimentConfig, NetworkConfig, TopicType

# Run predefined experiment
results = run_predefined_experiment("basic_polarization", random_seed=42)

# Create custom experiment
config = ExperimentConfig(
    name="Custom Study",
    num_agents=100,
    topic=TopicType.CLIMATE_CHANGE,
    num_rounds=20,
    intervention_type="diverse_exposure",
    intervention_round=10
)

experiment = EchoChamberExperiment(config)
results = experiment.run_full_experiment()

# Analyze results
print(f"Final polarization: {results.polarization_over_time[-1]}")
print(f"Echo chambers: {len(results.final_echo_chambers)}")
```

### Continuous Belief Distribution Programming Interface (NEW!)

```python
from core.continuous_integration import ContinuousAgentConfig, create_continuous_agent_population
from core.continuous_beliefs import BeliefDistributionParams, DistributionType

# Full custom configuration with continuous parameters
belief_params = BeliefDistributionParams(
    distribution_type=DistributionType.BIMODAL,
    polarization_strength=0.8,      # 0.0-1.0 continuous
    polarization_asymmetry=0.3,     # -1.0 to +1.0 bias
    gap_size=0.6,                   # Gap between belief clusters
    personality_correlations={
        'openness': -0.4,           # Strong beliefs → less open
        'confidence': 0.5,          # Strong beliefs → more confident
    }
)

config = ContinuousAgentConfig(
    num_agents=150,
    topic=TopicType.IMMIGRATION,
    belief_params=belief_params,
    random_seed=42
)

agents = create_continuous_agent_population(config)

# Parameter space exploration
import numpy as np
for pol_strength in np.linspace(0.1, 0.9, 5):
    # Test different polarization levels
    params = create_polarized_params(polarization_strength=pol_strength)
    agents = create_custom_population(100, TopicType.GUN_CONTROL, params)
    # ... analyze outcomes ...
```

### Custom Analysis

```python
from echo_chamber_experiments.visualizations import EchoChamberVisualizer
import pandas as pd

# Load results and create visualizer
visualizer = EchoChamberVisualizer(results)

# Generate custom plots
belief_plot = visualizer.plot_belief_evolution()
network_plot = visualizer.plot_network_analysis()

# Export data for external analysis
df = results.to_dataframe()
df.to_csv("my_analysis_data.csv")

# Statistical analysis
correlation = df.groupby('id').apply(
    lambda x: x['belief_strength'].corr(x['round'])
).mean()
print(f"Average belief trajectory correlation: {correlation}")
```

### Configuration Customization

Create detailed configuration files for reproducible research:

```json
{
  "name": "My Research Study",
  "description": "Testing hypothesis about personality effects",
  "num_agents": 150,
  "topic": "immigration",
  "belief_distribution": "polarized",
  "network_config": {
    "network_type": "preferential_attachment",
    "homophily_strength": 0.75,
    "average_connections": 7,
    "dynamic_rewiring": true,
    "bridge_probability": 0.04
  },
  "num_rounds": 25,
  "interactions_per_round": 400,
  "intervention_round": 15,
  "intervention_type": "bridge_building",
  "random_seed": 12345
}
```

## 📈 Research Applications

### Academic Research
- **Social Psychology**: Study of belief formation and change
- **Network Science**: Community detection and influence propagation  
- **Political Science**: Polarization and echo chamber dynamics
- **Computer Science**: Multi-agent systems and social simulation

### Policy Applications
- **Platform Design**: Testing content recommendation algorithms
- **Intervention Design**: Evaluating depolarization strategies
- **Public Health**: Understanding misinformation spread
- **Education**: Designing bias-resistant communication strategies

### Business Applications
- **Market Research**: Opinion dynamics in consumer behavior
- **Product Development**: Social feature design and testing
- **Risk Assessment**: Understanding viral misinformation patterns

## 🔧 Technical Details

### System Requirements
- Python 3.8+
- 4GB+ RAM (for large experiments)
- Modern CPU (experiments are CPU-intensive)

### Dependencies
- numpy, pandas: Data processing
- matplotlib, seaborn, plotly: Visualization
- networkx: Network analysis
- click: Command-line interface
- jupyter: Optional notebook interface

### Performance Notes
- Agent count scales roughly O(n²) for interactions
- Network formation is O(n log n) for most types
- Memory usage is proportional to `agents × rounds` for detailed history
- Use `save_detailed_history: false` for large experiments

### Platform Compatibility
✅ **Fully Cross-Platform** - Works on Windows, macOS, and Linux
✅ **No Binary Dependencies** - Pure Python implementation
✅ **Lightweight** - No GPU or special hardware requirements

## 📚 Examples and Tutorials

### Research Scenarios

#### 1. Social Media Platform Study
```bash
# Simulate Twitter-like network with intervention
python run_experiment.py run --custom \
    --agents 200 \
    --rounds 30 \
    --topic tech_regulation \
    --network scale_free \
    --intervention diverse_exposure \
    --intervention-round 15 \
    --output social_media_study/
```

#### 2. Political Polarization Research
```bash
# Study polarization with different network structures
for network in random small_world scale_free preferential_attachment; do
    python run_experiment.py run --custom \
        --agents 100 \
        --rounds 25 \
        --topic gun_control \
        --network $network \
        --output "polarization_study/network_$network/" \
        --seed 42
done
```

#### 3. Intervention Comparison
```bash
# Compare all intervention types
for intervention in fact_check diverse_exposure bridge_building; do
    python run_experiment.py run --custom \
        --agents 80 \
        --rounds 20 \
        --topic climate_change \
        --intervention $intervention \
        --intervention-round 10 \
        --output "intervention_study/type_$intervention/" \
        --seed 123
done
```

## 📚 Continuous Belief Distribution System

The new continuous belief system provides unprecedented flexibility in modeling agent populations. For comprehensive documentation, see:

### Quick Reference

- **📖 [CONTINUOUS_BELIEFS_GUIDE.md](CONTINUOUS_BELIEFS_GUIDE.md)** - Complete usage guide and examples
- **🧪 Test the system**: `python test_continuous_beliefs.py`
- **🎨 See demonstrations**: `python examples/continuous_beliefs_demo.py`

### Key Benefits Over Discrete System

| Feature | Before (Discrete) | After (Continuous) |
|---------|-------------------|-------------------|
| **Parameter Control** | 3 fixed options | Infinite continuous combinations |
| **Polarization** | Fixed strength | 0.0-1.0 adjustable |
| **Population Bias** | None | -1.0 to +1.0 asymmetry |
| **Multi-Modal** | Impossible | Support for any number of clusters |
| **Correlations** | None | Belief-personality correlations |
| **Optimization** | Manual tuning | Algorithmic parameter search |

### Migration from Discrete System

```python
# OLD (Discrete):
agents = create_diverse_agent_population(100, TopicType.GUN_CONTROL, "polarized")

# NEW (Drop-in replacement):
agents = create_continuous_population_from_legacy(100, TopicType.GUN_CONTROL, "polarized")

# NEW (With customization):
agents = create_highly_polarized_population(
    100, TopicType.GUN_CONTROL, 
    polarization_strength=0.8, 
    asymmetry=0.2
)
```

## 🤝 Contributing

This project is designed for collaborative research. To contribute:

1. **Fork the repository** and create feature branches
2. **Add new experiment types** in `core/experiment.py`
3. **Create new visualization types** in `visualizations/plots.py`
4. **Submit pull requests** with clear descriptions
5. **Share interesting findings** and use cases

### Adding New Features

#### New Agent Behaviors
```python
# In core/agent.py
class NewPersonalityType(Enum):
    SKEPTICAL = "skeptical"

# Add behavior in Agent.calculate_influence_susceptibility()
if self.personality_type == PersonalityType.SKEPTICAL:
    base_susceptibility *= 0.4  # Very resistant to influence
```

#### New Network Types
```python
# In core/network.py  
def _create_new_network_type(self):
    """Implement custom network formation algorithm"""
    # Your network formation logic here
    pass
```

#### New Interventions
```python
# In core/experiment.py
def _apply_new_intervention(self):
    """Implement custom intervention strategy"""
    # Your intervention logic here
    pass
```

## 📄 License and Citation

This project is open source. If you use it in research, please cite:

```
Echo Chamber Social Dynamics Experiments (2024)
AgentSociety Research Framework
https://github.com/your-repo/echo-chamber-experiments
```

## 🆘 Support and Troubleshooting

### Common Issues

1. **"Module not found" errors**: Ensure you're running from the correct directory
2. **Memory issues with large experiments**: Reduce agent count or disable detailed history
3. **Slow performance**: Use fewer rounds or lower interaction counts for testing
4. **Visualization errors**: Install all optional dependencies

### Getting Help

- **Documentation**: This README and inline code comments
- **Examples**: Check the `configs/` directory for working examples
- **Issues**: Report bugs and request features via GitHub issues
- **Community**: Join discussions in project forums

---

**Ready to explore social dynamics?** Start with a basic experiment and dive into the fascinating world of opinion formation and echo chamber dynamics! 🚀