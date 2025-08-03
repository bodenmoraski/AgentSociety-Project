"""
Dynamic Belief Evolution Experiment (B1)

This module implements advanced experiments for studying belief dynamics
during crisis scenarios using time-varying parameters and mathematical
trajectory analysis.

Key Components:
- DynamicBeliefParameters: Time-varying parameter interpolation system
- DynamicEvolutionExperiment: Main experiment implementation  
- TrajectoryAnalyzer: Mathematical analysis of belief trajectories
- CrisisScenarioGenerator: Realistic crisis scenario modeling
- DynamicEvolutionVisualizer: Specialized visualization tools

Mathematical Foundation:
- Parameter interpolation using multiple algorithms (linear, cubic spline, sigmoid)
- Change point detection using CUSUM and Bayesian methods
- Trajectory modeling with multiple mathematical families
- Phase transition analysis and crisis impact quantification
- Intervention timing optimization through systematic evaluation

Usage:
    from experiments.dynamic_evolution import DynamicEvolutionExperiment, DynamicEvolutionConfig
    from experiments.dynamic_evolution import run_pandemic_experiment, run_election_experiment
    from experiments.dynamic_evolution.visualizations import DynamicEvolutionVisualizer
"""

from .experiment import (
    DynamicEvolutionExperiment,
    DynamicEvolutionConfig, 
    DynamicEvolutionResults,
    run_pandemic_experiment,
    run_election_experiment,
    run_economic_shock_experiment,
    compare_crisis_scenarios
)

from .analysis import (
    TrajectoryAnalyzer,
    CrisisAnalyzer,
    TrajectoryModel,
    ChangePoint,
    ModelType,
    ChangePointMethod
)

from .visualizations import DynamicEvolutionVisualizer

__all__ = [
    'DynamicEvolutionExperiment',
    'DynamicEvolutionConfig', 
    'DynamicEvolutionResults',
    'run_pandemic_experiment',
    'run_election_experiment', 
    'run_economic_shock_experiment',
    'compare_crisis_scenarios',
    'TrajectoryAnalyzer',
    'CrisisAnalyzer',
    'TrajectoryModel',
    'ChangePoint',
    'ModelType',
    'ChangePointMethod',
    'DynamicEvolutionVisualizer'
]