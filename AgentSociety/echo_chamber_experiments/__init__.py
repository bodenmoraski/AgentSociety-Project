"""
Echo Chamber Social Dynamics Experiments

A comprehensive framework for studying belief propagation, polarization, and echo chamber
formation in AI agent societies.

Key Features:
- Diverse agent personalities and traits
- Multiple network formation models
- Intervention testing capabilities
- Rich visualization and analysis tools
- Reproducible experimental framework

Example Usage:
    from echo_chamber_experiments.core.experiment import run_predefined_experiment
    
    results = run_predefined_experiment("basic_polarization")
    print(f"Final polarization: {results.polarization_over_time[-1]}")
"""

from .core.agent import Agent, TopicType, PersonalityType, create_diverse_agent_population
from .core.network import SocialNetwork, NetworkConfig
from .core.experiment import EchoChamberExperiment, ExperimentConfig, ExperimentResults, run_predefined_experiment
from .visualizations.plots import EchoChamberVisualizer

__version__ = "1.0.0"
__author__ = "Echo Chamber Research Team"

__all__ = [
    'Agent', 'TopicType', 'PersonalityType', 'create_diverse_agent_population',
    'SocialNetwork', 'NetworkConfig',
    'EchoChamberExperiment', 'ExperimentConfig', 'ExperimentResults', 'run_predefined_experiment',
    'EchoChamberVisualizer'
]