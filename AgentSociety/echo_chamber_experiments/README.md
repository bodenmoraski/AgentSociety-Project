# 🌊 Dynamic Belief Evolution Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research: Active](https://img.shields.io/badge/research-active-green.svg)](https://github.com/your-repo)

**A sophisticated framework for studying belief dynamics during crisis scenarios using agent-based modeling with continuous parameter spaces.**

## 🎯 Overview

The Dynamic Belief Evolution Framework enables researchers to study how beliefs, polarization, and echo chambers evolve during crisis events. It features:

- **🌊 Crisis Scenario Modeling** - Pandemic, election, economic shock scenarios
- **📈 Time-Varying Parameters** - Smooth interpolation between belief states
- **👥 Agent-Based Simulation** - Individual belief trajectories and interactions
- **🔍 Advanced Analysis** - Phase transition detection, crisis impact quantification
- **📊 Professional Visualizations** - Publication-ready static and interactive plots
- **🧪 Comprehensive Testing** - Sanity checks and anomaly detection

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd echo_chamber_experiments

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

### Run Your First Experiment

```python
from experiments.dynamic_evolution.experiment import DynamicEvolutionExperiment, DynamicEvolutionConfig
from core.dynamic_parameters import CrisisType
from core.agent import TopicType

# Configure experiment
config = DynamicEvolutionConfig(
    name="My First Crisis Experiment",
    num_agents=50,
    num_rounds=20,
    crisis_scenario=CrisisType.PANDEMIC,
    crisis_severity=0.8,
    topic=TopicType.HEALTHCARE
)

# Run experiment
experiment = DynamicEvolutionExperiment(config)
results = experiment.run_full_experiment()

# Analyze results
print(f"Crisis impact: {results.crisis_impact_metrics['polarization_increase']:.3f}")
```

### Create Visualizations

```python
from experiments.dynamic_evolution.visualizations import DynamicEvolutionVisualizer

# Create comprehensive overview
visualizer = DynamicEvolutionVisualizer(results)
fig = visualizer.plot_dynamic_evolution_overview()
fig.savefig("crisis_analysis.png", dpi=300)

# Create interactive dashboard
dashboard = visualizer.create_interactive_dashboard()
dashboard.write_html("interactive_analysis.html")
```

## 📊 Demo & Examples

### Quick Demos
```bash
# Visual demonstration
python demos/simple_visualization_demo.py

# System health check  
python tests/test_sanity_checks.py

# Anomaly detection demo
python demos/enhanced_anomaly_detection.py
```

### Pre-built Crisis Scenarios
```bash
# Pandemic scenario
python run_dynamic_evolution.py --scenario pandemic --severity 0.8 --agents 100

# Election cycle
python run_dynamic_evolution.py --scenario election --agents 75 --duration 25

# Economic shock
python run_dynamic_evolution.py --scenario economic_shock --severity 0.6
```

## 🏗️ Framework Architecture

```
echo_chamber_experiments/
├── 📁 core/                          # Core framework components
│   ├── agent.py                      # Agent behavior and beliefs
│   ├── network.py                    # Social network dynamics
│   ├── experiment.py                 # Base experiment framework
│   ├── dynamic_parameters.py         # Time-varying parameters
│   ├── continuous_beliefs.py         # Continuous belief distributions
│   └── continuous_integration.py     # Integration utilities
├── 📁 experiments/                   # Specialized experiments
│   └── dynamic_evolution/            # Crisis-driven belief evolution
│       ├── experiment.py             # Main experiment implementation
│       ├── analysis.py               # Mathematical analysis tools
│       └── visualizations.py         # Comprehensive visualization suite
├── 📁 configs/                       # Pre-built configurations
│   ├── dynamic_pandemic.json         # Pandemic scenario
│   ├── dynamic_election.json         # Election cycle scenario
│   └── dynamic_economic_shock.json   # Economic crisis scenario
├── 📁 demos/                         # Demonstration scripts
│   ├── simple_visualization_demo.py  # Basic visualization demo
│   └── enhanced_anomaly_detection.py # Advanced anomaly detection
├── 📁 tests/                         # Comprehensive test suite
│   ├── test_sanity_checks.py         # System sanity validation
│   ├── test_dynamic_evolution.py     # B1 experiment tests
│   └── test_continuous_beliefs.py    # Continuous belief system tests
└── 📁 docs/                          # Documentation
    ├── DYNAMIC_BELIEF_EVOLUTION_GUIDE.md
    └── CONTINUOUS_BELIEFS_GUIDE.md
```

## 🔬 Research Applications

### Crisis Communication
- **Pandemic Response** - Model belief evolution during health crises
- **Risk Communication** - Study effective messaging strategies
- **Public Health Campaigns** - Optimize intervention timing

### Political Dynamics  
- **Election Cycles** - Analyze polarization during campaigns
- **Social Media Impact** - Study echo chamber formation
- **Policy Communication** - Model public opinion dynamics

### Platform Design
- **Recommendation Algorithms** - Test bias mitigation strategies
- **Content Moderation** - Optimize intervention policies  
- **Community Guidelines** - Design healthy discourse environments

## 📈 Key Features

### 🌊 Crisis Scenario Modeling
```python
# Built-in crisis scenarios with empirically-grounded parameters
scenarios = {
    'pandemic': CrisisScenarioGenerator.pandemic_scenario(severity=0.8),
    'election': CrisisScenarioGenerator.election_scenario(polarization_peak=0.9),
    'economic': CrisisScenarioGenerator.economic_shock_scenario(severity=0.6)
}
```

### 📊 Advanced Analytics
- **Phase Transition Detection** - Identify critical belief shifts
- **Trajectory Modeling** - Fit mathematical models to belief evolution
- **Crisis Impact Quantification** - Measure polarization changes
- **Recovery Pattern Analysis** - Study post-crisis dynamics

### 🎨 Publication-Ready Visualizations
- **Multi-panel Overviews** - Comprehensive analysis dashboards
- **Interactive Dashboards** - Real-time exploration tools
- **Agent Heatmaps** - Belief evolution visualization
- **Crisis Timelines** - Annotated parameter evolution
- **Network Analysis** - Agent correlation networks

### 🧪 Quality Assurance
- **Automated Sanity Checks** - System health validation
- **Anomaly Detection** - Unusual pattern identification  
- **Performance Monitoring** - Execution speed tracking
- **Mathematical Validation** - Bounds and consistency checking

## 📋 Configuration Options

### Basic Experiment Config
```python
config = DynamicEvolutionConfig(
    name="Custom Experiment",
    num_agents=100,                    # Population size
    num_rounds=25,                     # Simulation duration
    topic=TopicType.HEALTHCARE,        # Discussion topic
    interactions_per_round=200,        # Social interactions
    
    # Crisis parameters
    crisis_scenario=CrisisType.PANDEMIC,
    crisis_severity=0.7,               # Impact strength (0-1)
    
    # Analysis options
    belief_history_tracking=True,      # Track individual trajectories
    optimize_intervention_timing=True, # Find optimal intervention points
    
    # Reproducibility
    random_seed=42
)
```

### Advanced Parameter Control
```python
from core.dynamic_parameters import DynamicBeliefParameters, ParameterKeyframe

# Custom time-varying parameters
custom_scenario = DynamicBeliefParameters(
    keyframes=[
        ParameterKeyframe(time_point=0, parameters=pre_crisis_params),
        ParameterKeyframe(time_point=10, parameters=peak_crisis_params),
        ParameterKeyframe(time_point=20, parameters=recovery_params)
    ],
    default_interpolation=InterpolationMethod.CUBIC_SPLINE
)
```

## 🔍 Quality Assurance

### Automated Testing
```bash
# Run comprehensive test suite
python -m pytest tests/ -v

# Quick system health check
python tests/test_sanity_checks.py

# Performance benchmarking  
python tests/test_performance.py
```

### Anomaly Detection
```python
from demos.enhanced_anomaly_detection import EnhancedAnomalyDetector

detector = EnhancedAnomalyDetector(sensitivity='medium')
alerts = detector.detect_all_anomalies(results)

# Automatic report generation
detector.create_anomaly_report("anomaly_analysis.md")
```

## 📚 Documentation

- **[📖 Dynamic Belief Evolution Guide](DYNAMIC_BELIEF_EVOLUTION_GUIDE.md)** - Complete system documentation
- **[🧮 Continuous Beliefs Guide](CONTINUOUS_BELIEFS_GUIDE.md)** - Mathematical foundations  
- **[📋 Implementation Plan](CONTINUOUS_EXPERIMENTS_IMPLEMENTATION_PLAN.md)** - Development roadmap
- **[🚀 Installation Guide](INSTALL.md)** - Setup instructions

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)  
3. **Run tests** (`python tests/test_sanity_checks.py`)
4. **Commit** your changes (`git commit -m 'Add amazing feature'`)
5. **Push** to the branch (`git push origin feature/amazing-feature`)
6. **Open** a Pull Request

### Development Setup
```bash
# Development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Run full test suite
python -m pytest tests/ --cov=core --cov=experiments
```

## 📊 Benchmarks & Performance

| Experiment Size | Agents | Rounds | Avg. Runtime | Memory Usage |
|----------------|--------|--------|--------------|--------------|
| Small          | 25     | 10     | 0.15s        | ~50MB        |
| Medium         | 100    | 25     | 1.2s         | ~150MB       |
| Large          | 500    | 50     | 12s          | ~500MB       |
| Enterprise     | 2000   | 100    | 180s         | ~2GB         |

*Benchmarks run on MacBook Pro M1, 16GB RAM*

## 📈 Research Impact

### Publications Using This Framework
- *Coming soon - Submit your papers using this framework!*

### Citation
If you use this framework in your research, please cite:
```bibtex
@software{dynamic_belief_evolution,
  title={Dynamic Belief Evolution Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## ⚡ Performance Tips

- **Use appropriate population sizes** - Start with 50-100 agents
- **Monitor memory usage** - Large experiments can use significant RAM
- **Enable caching** - Reuse computed network structures when possible
- **Parallelize analysis** - Use built-in batch processing for parameter sweeps
- **Profile bottlenecks** - Use the built-in performance monitoring

## 🐛 Troubleshooting

### Common Issues

**Q: Experiment runs slowly with many agents**
```python
# A: Reduce interaction density
config.interactions_per_round = config.num_agents * 2  # Instead of default 5x
```

**Q: Visualizations fail to generate**  
```bash
# A: Check display backend
python -c "import matplotlib; print(matplotlib.get_backend())"
export MPLBACKEND=Agg  # For headless environments
```

**Q: Memory usage too high**
```python
# A: Disable trajectory tracking for large experiments
config.belief_history_tracking = False
```

### Getting Help
- **📖 Check the documentation** in the `docs/` folder
- **🧪 Run sanity checks** with `python tests/test_sanity_checks.py`
- **🔍 Use anomaly detection** to identify issues
- **💬 Open an issue** on GitHub with detailed error logs

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AgentSociety Framework** - Base agent simulation capabilities
- **Research Community** - Empirical grounding and validation
- **Open Source Libraries** - NumPy, matplotlib, plotly, scikit-learn, networkx

---

<div align="center">

**🌊 Ready to model the future of belief dynamics? Get started today! 🚀**

[📖 Documentation](docs/) • [🎯 Examples](demos/) • [🧪 Tests](tests/) • [📊 Benchmarks](#benchmarks--performance)

</div>