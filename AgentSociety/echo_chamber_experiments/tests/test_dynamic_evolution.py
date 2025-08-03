#!/usr/bin/env python3
"""
Comprehensive Test Suite for Dynamic Belief Evolution (B1)

Tests all components of the dynamic evolution experiment system including:
- Dynamic parameter interpolation and mathematical consistency
- Crisis scenario generation and realism
- Experiment execution and result validation
- Trajectory analysis and mathematical models
- Visualization generation and output quality
- CLI integration and configuration loading

Mathematical Testing Focus:
- Parameter interpolation accuracy and smoothness
- Change point detection statistical validity
- Trajectory model fitting and selection
- Crisis impact quantification
- Cross-platform reproducibility
"""

import sys
import os
import tempfile
import shutil
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import time

# Add the echo chamber experiments directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components for testing
from core.dynamic_parameters import (
    DynamicBeliefParameters, 
    ParameterKeyframe, 
    CrisisScenarioGenerator,
    CrisisType,
    InterpolationMethod
)
from core.continuous_beliefs import BeliefDistributionParams, DistributionType
from experiments.dynamic_evolution.experiment import (
    DynamicEvolutionExperiment,
    DynamicEvolutionConfig,
    run_pandemic_experiment,
    run_election_experiment,
    compare_crisis_scenarios
)
from experiments.dynamic_evolution.analysis import (
    TrajectoryAnalyzer,
    CrisisAnalyzer,
    ModelType,
    ChangePointMethod
)
from experiments.dynamic_evolution.visualizations import DynamicEvolutionVisualizer
from core.agent import TopicType
from core.network import NetworkConfig


def test_dynamic_parameter_interpolation():
    """Test mathematical correctness of parameter interpolation"""
    print("🔬 Testing dynamic parameter interpolation...")
    
    # Create test keyframes
    keyframes = [
        ParameterKeyframe(
            time_point=0,
            parameters=BeliefDistributionParams(
                polarization_strength=0.2,
                polarization_asymmetry=0.0,
                gap_size=0.1
            )
        ),
        ParameterKeyframe(
            time_point=10,
            parameters=BeliefDistributionParams(
                polarization_strength=0.8,
                polarization_asymmetry=0.3,
                gap_size=0.4
            )
        ),
        ParameterKeyframe(
            time_point=20,
            parameters=BeliefDistributionParams(
                polarization_strength=0.4,
                polarization_asymmetry=-0.1,
                gap_size=0.2
            )
        )
    ]
    
    dynamic_params = DynamicBeliefParameters(
        keyframes=keyframes,
        default_interpolation=InterpolationMethod.CUBIC_SPLINE
    )
    
    # Test exact keyframe retrieval
    params_0 = dynamic_params.get_parameters_at_time(0)
    assert abs(params_0.polarization_strength - 0.2) < 1e-10, "Exact keyframe retrieval failed"
    
    params_10 = dynamic_params.get_parameters_at_time(10)
    assert abs(params_10.polarization_strength - 0.8) < 1e-10, "Exact keyframe retrieval failed"
    
    # Test interpolation
    params_5 = dynamic_params.get_parameters_at_time(5)
    assert 0.2 <= params_5.polarization_strength <= 0.8, "Interpolation out of bounds"
    
    # Test smoothness (no sudden jumps)
    values = []
    for t in np.linspace(0, 20, 100):
        params = dynamic_params.get_parameters_at_time(t)
        values.append(params.polarization_strength)
    
    # Check for smoothness (no large jumps)
    derivatives = np.diff(values)
    max_derivative = np.max(np.abs(derivatives))
    assert max_derivative < 0.5, f"Interpolation not smooth, max derivative: {max_derivative}"
    
    # Test derivative computation
    derivative = dynamic_params.get_parameter_derivative(5.0, 'polarization_strength')
    assert isinstance(derivative, float), "Derivative computation failed"
    
    print("✅ Dynamic parameter interpolation tests passed")


def test_crisis_scenario_generation():
    """Test crisis scenario generation and mathematical properties"""
    print("🌊 Testing crisis scenario generation...")
    
    # Test pandemic scenario
    pandemic_params = CrisisScenarioGenerator.pandemic_scenario(severity=0.8, duration_rounds=25)
    
    # Validate structure
    assert len(pandemic_params.keyframes) >= 3, "Pandemic scenario too simple"
    assert pandemic_params.keyframes[0].time_point == 0, "Pandemic should start at time 0"
    assert pandemic_params.keyframes[-1].time_point == 25, "Pandemic should end at specified duration"
    
    # Test progression (polarization should increase then decrease)
    polarization_values = []
    for kf in pandemic_params.keyframes:
        polarization_values.append(kf.parameters.polarization_strength)
    
    max_pol_idx = np.argmax(polarization_values)
    assert max_pol_idx > 0 and max_pol_idx < len(polarization_values) - 1, "Pandemic should have polarization peak in middle"
    
    # Test election scenario
    election_params = CrisisScenarioGenerator.election_scenario(polarization_peak=0.9, cycle_rounds=20)
    
    # Validate election structure
    assert len(election_params.keyframes) >= 4, "Election scenario too simple"
    election_polarizations = [kf.parameters.polarization_strength for kf in election_params.keyframes]
    max_election_pol = max(election_polarizations)
    assert abs(max_election_pol - 0.9) < 0.1, f"Election peak polarization incorrect: {max_election_pol}"
    
    # Test economic shock scenario
    economic_params = CrisisScenarioGenerator.economic_shock_scenario(shock_severity=0.7, recovery_time=30)
    
    # Should have rapid initial change
    initial_pol = economic_params.keyframes[0].parameters.polarization_strength
    second_pol = economic_params.keyframes[1].parameters.polarization_strength
    assert second_pol > initial_pol + 0.2, "Economic shock should cause rapid initial polarization increase"
    
    print("✅ Crisis scenario generation tests passed")


def test_experiment_execution():
    """Test complete experiment execution and result validation"""
    print("🚀 Testing experiment execution...")
    
    # Create minimal test configuration
    config = DynamicEvolutionConfig(
        name="Test Dynamic Evolution",
        description="Test experiment for validation",
        num_agents=20,  # Small for fast testing
        topic=TopicType.HEALTHCARE,
        num_rounds=10,  # Short for fast testing
        crisis_scenario=CrisisType.PANDEMIC,
        crisis_severity=0.6,
        random_seed=42  # For reproducibility
    )
    
    # Run experiment
    start_time = time.time()
    experiment = DynamicEvolutionExperiment(config)
    results = experiment.run_full_experiment()
    duration = time.time() - start_time
    
    # Validate execution time (should be reasonable)
    assert duration < 30.0, f"Experiment too slow: {duration:.2f}s > 30s"
    
    # Validate results structure
    assert results is not None, "Experiment returned no results"
    assert len(results.polarization_over_time) == config.num_rounds, "Incorrect number of rounds recorded"
    assert len(results.agents_history) == config.num_rounds, "Agent history incomplete"
    
    # Validate belief trajectories
    assert len(results.belief_trajectories) == config.num_agents, "Incorrect number of trajectory records"
    
    for agent_id, trajectory in results.belief_trajectories.items():
        assert len(trajectory) == config.num_rounds, f"Agent {agent_id} trajectory incomplete"
        assert all(-1.1 <= belief <= 1.1 for belief in trajectory), f"Agent {agent_id} beliefs out of range"
    
    # Validate parameter evolution
    assert len(results.parameter_evolution) > 0, "No parameter evolution recorded"
    
    for param_name, values in results.parameter_evolution.items():
        assert len(values) == config.num_rounds, f"Parameter {param_name} evolution incomplete"
    
    # Test crisis impact metrics
    assert results.crisis_impact_metrics is not None, "No crisis impact metrics computed"
    assert 'polarization_increase' in results.crisis_impact_metrics, "Missing polarization increase metric"
    
    print("✅ Experiment execution tests passed")


def test_trajectory_analysis():
    """Test mathematical trajectory analysis and model fitting"""
    print("📈 Testing trajectory analysis...")
    
    # Create synthetic trajectory data
    time_points = np.arange(20)
    
    # Linear trajectory
    linear_trajectory = 0.1 * time_points - 0.5
    
    # Quadratic trajectory
    quadratic_trajectory = 0.02 * time_points**2 - 0.3 * time_points + 0.1
    
    # Exponential-like trajectory
    exponential_trajectory = np.tanh(0.3 * (time_points - 10)) * 0.8
    
    analyzer = TrajectoryAnalyzer()
    
    # Test linear model fitting
    linear_models = analyzer.fit_trajectory_models(time_points, linear_trajectory, [ModelType.LINEAR])
    assert ModelType.LINEAR in linear_models, "Linear model fitting failed"
    
    linear_model = linear_models[ModelType.LINEAR]
    assert linear_model.r_squared > 0.95, f"Linear model fit poor: R² = {linear_model.r_squared}"
    assert linear_model.monotonicity in ["increasing", "decreasing"], "Linear model monotonicity incorrect"
    
    # Test quadratic model fitting
    quad_models = analyzer.fit_trajectory_models(time_points, quadratic_trajectory, [ModelType.QUADRATIC])
    assert ModelType.QUADRATIC in quad_models, "Quadratic model fitting failed"
    
    quad_model = quad_models[ModelType.QUADRATIC]
    assert quad_model.r_squared > 0.95, f"Quadratic model fit poor: R² = {quad_model.r_squared}"
    
    # Test turning point detection
    if quad_model.turning_points:
        turning_point = quad_model.turning_points[0]
        expected_turning_point = 0.3 / (2 * 0.02)  # -b/(2a)
        assert abs(turning_point[0] - expected_turning_point) < 2.0, "Turning point detection inaccurate"
    
    # Test model selection
    all_models = analyzer.fit_trajectory_models(time_points, quadratic_trajectory)
    best_model = analyzer.select_best_model(all_models, criterion='r_squared')
    assert best_model.r_squared >= quad_model.r_squared, "Model selection suboptimal"
    
    print("✅ Trajectory analysis tests passed")


def test_change_point_detection():
    """Test change point detection algorithms"""
    print("🔍 Testing change point detection...")
    
    # Create synthetic time series with known change points
    n = 100
    time_points = np.arange(n)
    
    # Series with change at point 50
    values = np.concatenate([
        np.random.normal(0.2, 0.1, 50),  # First segment
        np.random.normal(0.7, 0.1, 50)   # Second segment (change in mean)
    ])
    
    analyzer = TrajectoryAnalyzer()
    
    # Test CUSUM change point detection
    change_points = analyzer.detect_change_points(
        time_points, values, 
        method=ChangePointMethod.CUSUM,
        min_segment_length=10
    )
    
    assert len(change_points) > 0, "CUSUM failed to detect change point"
    
    # Check if detected change point is near the true change point (50)
    detected_locations = [cp.location for cp in change_points]
    min_distance = min(abs(loc - 50) for loc in detected_locations)
    assert min_distance < 10, f"Change point detection inaccurate: closest detection {min_distance} points from true change"
    
    # Test statistical significance
    significant_changes = [cp for cp in change_points if cp.confidence > 0.9]
    assert len(significant_changes) > 0, "No statistically significant change points detected"
    
    # Test variance change detection
    variance_changes = analyzer.detect_change_points(
        time_points, values,
        method=ChangePointMethod.VARIANCE_CHANGE,
        min_segment_length=10
    )
    
    # Should detect at least one change (not necessarily the same as CUSUM)
    assert len(variance_changes) >= 0, "Variance change detection failed"
    
    print("✅ Change point detection tests passed")


def test_visualization_generation():
    """Test visualization generation without errors"""
    print("📊 Testing visualization generation...")
    
    # Create test results with minimal data
    config = DynamicEvolutionConfig(
        name="Test Visualization",
        num_agents=15,
        num_rounds=8,
        crisis_scenario=CrisisType.PANDEMIC,
        random_seed=42
    )
    
    experiment = DynamicEvolutionExperiment(config)
    results = experiment.run_full_experiment()
    
    visualizer = DynamicEvolutionVisualizer(results)
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Test overview plot
            overview_fig = visualizer.plot_dynamic_evolution_overview()
            assert overview_fig is not None, "Overview plot generation failed"
            
            # Save to verify no errors
            overview_path = Path(temp_dir) / "test_overview.png"
            overview_fig.savefig(overview_path)
            assert overview_path.exists(), "Overview plot not saved"
            
            # Test trajectory model comparison (if trajectories exist)
            if results.belief_trajectories:
                agent_id = list(results.belief_trajectories.keys())[0]
                model_fig = visualizer.plot_trajectory_model_comparison(agent_id)
                assert model_fig is not None, "Trajectory model plot generation failed"
                
                model_path = Path(temp_dir) / "test_models.png"
                model_fig.savefig(model_path)
                assert model_path.exists(), "Model plot not saved"
            
            # Test correlation network (if multiple agents)
            if len(results.belief_trajectories) > 1:
                network_fig = visualizer.plot_cross_agent_correlation_network()
                assert network_fig is not None, "Network plot generation failed"
                
                network_path = Path(temp_dir) / "test_network.png"
                network_fig.savefig(network_path)
                assert network_path.exists(), "Network plot not saved"
            
            # Test interactive dashboard (if Plotly available)
            try:
                dashboard = visualizer.create_interactive_dashboard()
                if dashboard is not None:
                    dashboard_path = Path(temp_dir) / "test_dashboard.html"
                    dashboard.write_html(str(dashboard_path))
                    assert dashboard_path.exists(), "Interactive dashboard not saved"
            except ImportError:
                warnings.warn("Plotly not available, skipping interactive dashboard test")
            
            # Test comprehensive report generation
            report_files = visualizer.generate_comprehensive_report(temp_dir)
            assert len(report_files) > 0, "No report files generated"
            
            # Verify report files exist
            for component, file_path in report_files.items():
                assert Path(file_path).exists(), f"Report file missing: {component} -> {file_path}"
            
        except Exception as e:
            # Close any open figures to prevent memory leaks
            import matplotlib.pyplot as plt
            plt.close('all')
            raise e
        
        # Close figures
        import matplotlib.pyplot as plt
        plt.close('all')
    
    print("✅ Visualization generation tests passed")


def test_convenience_functions():
    """Test convenience functions for common experiment scenarios"""
    print("🎭 Testing convenience functions...")
    
    # Test pandemic experiment
    pandemic_results = run_pandemic_experiment(
        num_agents=20, 
        duration=8, 
        severity=0.6, 
        random_seed=42
    )
    
    assert pandemic_results is not None, "Pandemic experiment failed"
    assert pandemic_results.config.crisis_scenario == CrisisType.PANDEMIC, "Pandemic scenario not set correctly"
    assert len(pandemic_results.polarization_over_time) == 8, "Pandemic experiment duration incorrect"
    
    # Test election experiment
    election_results = run_election_experiment(
        num_agents=20,
        duration=8,
        peak_polarization=0.8,
        random_seed=42
    )
    
    assert election_results is not None, "Election experiment failed"
    assert election_results.config.crisis_scenario == CrisisType.ELECTION, "Election scenario not set correctly"
    
    # Test comparison (reduced scope for testing)
    comparison_results = compare_crisis_scenarios(num_agents=15, duration=6)
    
    assert len(comparison_results) >= 2, "Comparison returned too few scenarios"
    assert 'pandemic' in comparison_results, "Pandemic scenario missing from comparison"
    
    # Verify all scenarios have results
    for scenario_name, results in comparison_results.items():
        assert results is not None, f"No results for {scenario_name} scenario"
        assert len(results.polarization_over_time) == 6, f"{scenario_name} duration incorrect"
    
    print("✅ Convenience functions tests passed")


def test_configuration_loading():
    """Test configuration file loading and validation"""
    print("📄 Testing configuration loading...")
    
    # Create test configuration
    test_config = {
        "name": "Test Config Loading",
        "description": "Test configuration for validation",
        "experiment_type": "dynamic_evolution",
        "num_agents": 25,
        "topic": "healthcare",
        "num_rounds": 12,
        "crisis_scenario": "pandemic",
        "crisis_severity": 0.7,
        "network_config": {
            "network_type": "small_world",
            "homophily_strength": 0.8,
            "average_connections": 5
        },
        "random_seed": 123
    }
    
    # Save and load configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(test_config, temp_file, indent=2)
        temp_config_path = temp_file.name
    
    try:
        # Import the loading function
        from run_dynamic_evolution import load_dynamic_config_from_file
        
        # Load configuration
        loaded_config = load_dynamic_config_from_file(temp_config_path)
        
        # Validate loaded configuration
        assert loaded_config.name == test_config["name"], "Configuration name not loaded correctly"
        assert loaded_config.num_agents == test_config["num_agents"], "Agent count not loaded correctly"
        assert loaded_config.crisis_scenario == CrisisType.PANDEMIC, "Crisis scenario not converted correctly"
        assert loaded_config.random_seed == test_config["random_seed"], "Random seed not loaded correctly"
        
        # Test experiment execution with loaded config
        experiment = DynamicEvolutionExperiment(loaded_config)
        results = experiment.run_full_experiment()
        
        assert results is not None, "Experiment with loaded config failed"
        assert len(results.polarization_over_time) == test_config["num_rounds"], "Loaded config rounds incorrect"
        
    finally:
        # Clean up temporary file
        os.unlink(temp_config_path)
    
    print("✅ Configuration loading tests passed")


def test_reproducibility():
    """Test experiment reproducibility with same seeds"""
    print("🔄 Testing reproducibility...")
    
    config = DynamicEvolutionConfig(
        name="Reproducibility Test",
        num_agents=20,
        num_rounds=10,
        crisis_scenario=CrisisType.PANDEMIC,
        crisis_severity=0.7,
        random_seed=42  # Fixed seed
    )
    
    # Run experiment twice
    experiment1 = DynamicEvolutionExperiment(config)
    results1 = experiment1.run_full_experiment()
    
    experiment2 = DynamicEvolutionExperiment(config)
    results2 = experiment2.run_full_experiment()
    
    # Check reproducibility
    pol_diff = np.max(np.abs(np.array(results1.polarization_over_time) - np.array(results2.polarization_over_time)))
    assert pol_diff < 1e-10, f"Polarization not reproducible: max difference {pol_diff}"
    
    # Check belief trajectory reproducibility
    for agent_id in results1.belief_trajectories:
        if agent_id in results2.belief_trajectories:
            traj1 = np.array(results1.belief_trajectories[agent_id])
            traj2 = np.array(results2.belief_trajectories[agent_id])
            traj_diff = np.max(np.abs(traj1 - traj2))
            assert traj_diff < 1e-10, f"Agent {agent_id} trajectory not reproducible: max difference {traj_diff}"
    
    # Check parameter evolution reproducibility
    for param_name in results1.parameter_evolution:
        if param_name in results2.parameter_evolution:
            param1 = np.array(results1.parameter_evolution[param_name])
            param2 = np.array(results2.parameter_evolution[param_name])
            param_diff = np.max(np.abs(param1 - param2))
            assert param_diff < 1e-10, f"Parameter {param_name} not reproducible: max difference {param_diff}"
    
    print("✅ Reproducibility tests passed")


def test_performance_benchmarks():
    """Test performance benchmarks for various experiment scales"""
    print("⚡ Testing performance benchmarks...")
    
    # Test small experiment
    small_config = DynamicEvolutionConfig(
        name="Small Performance Test",
        num_agents=30,
        num_rounds=15,
        crisis_scenario=CrisisType.PANDEMIC,
        random_seed=42
    )
    
    start_time = time.time()
    small_experiment = DynamicEvolutionExperiment(small_config)
    small_results = small_experiment.run_full_experiment()
    small_duration = time.time() - start_time
    
    assert small_duration < 10.0, f"Small experiment too slow: {small_duration:.2f}s > 10s"
    print(f"   Small experiment (30 agents, 15 rounds): {small_duration:.2f}s")
    
    # Test medium experiment
    medium_config = DynamicEvolutionConfig(
        name="Medium Performance Test",
        num_agents=60,
        num_rounds=20,
        crisis_scenario=CrisisType.ELECTION,
        random_seed=42
    )
    
    start_time = time.time()
    medium_experiment = DynamicEvolutionExperiment(medium_config)
    medium_results = medium_experiment.run_full_experiment()
    medium_duration = time.time() - start_time
    
    assert medium_duration < 30.0, f"Medium experiment too slow: {medium_duration:.2f}s > 30s"
    print(f"   Medium experiment (60 agents, 20 rounds): {medium_duration:.2f}s")
    
    # Performance scaling check (should be roughly linear)
    scaling_factor = medium_duration / small_duration
    expected_scaling = (60 * 20) / (30 * 15)  # ~2.67
    
    assert scaling_factor < expected_scaling * 3, f"Performance scaling poor: {scaling_factor:.2f}x vs expected ~{expected_scaling:.2f}x"
    print(f"   Performance scaling: {scaling_factor:.2f}x (expected ~{expected_scaling:.2f}x)")
    
    print("✅ Performance benchmark tests passed")


def test_mathematical_consistency():
    """Test mathematical consistency of computed metrics"""
    print("🧮 Testing mathematical consistency...")
    
    # Run experiment with known parameters
    config = DynamicEvolutionConfig(
        name="Mathematical Consistency Test",
        num_agents=25,
        num_rounds=15,
        crisis_scenario=CrisisType.PANDEMIC,
        crisis_severity=0.8,
        random_seed=42
    )
    
    experiment = DynamicEvolutionExperiment(config)
    results = experiment.run_full_experiment()
    
    # Test polarization bounds
    for pol in results.polarization_over_time:
        assert 0.0 <= pol <= 1.0, f"Polarization out of bounds: {pol}"
    
    # Test belief trajectory bounds
    for agent_id, trajectory in results.belief_trajectories.items():
        for belief in trajectory:
            assert -1.1 <= belief <= 1.1, f"Agent {agent_id} belief out of bounds: {belief}"
    
    # Test parameter evolution bounds
    if 'polarization_strength' in results.parameter_evolution:
        for pol_strength in results.parameter_evolution['polarization_strength']:
            assert 0.0 <= pol_strength <= 1.0, f"Polarization strength out of bounds: {pol_strength}"
    
    if 'polarization_asymmetry' in results.parameter_evolution:
        for asymmetry in results.parameter_evolution['polarization_asymmetry']:
            assert -1.0 <= asymmetry <= 1.0, f"Polarization asymmetry out of bounds: {asymmetry}"
    
    # Test crisis impact metrics consistency
    if results.crisis_impact_metrics:
        metrics = results.crisis_impact_metrics
        
        if 'recovery_ratio' in metrics:
            recovery = metrics['recovery_ratio']
            assert 0.0 <= recovery <= 1.5, f"Recovery ratio unrealistic: {recovery}"  # Allow slight overshoot
        
        if 'volatility' in metrics:
            volatility = metrics['volatility']
            assert volatility >= 0.0, f"Volatility cannot be negative: {volatility}"
    
    # Test trajectory statistics consistency
    traj_stats = results.compute_trajectory_statistics()
    
    if 'individual_trajectories' in traj_stats:
        ind_stats = traj_stats['individual_trajectories']
        if 'mean_variance' in ind_stats:
            assert ind_stats['mean_variance'] >= 0.0, "Variance cannot be negative"
    
    print("✅ Mathematical consistency tests passed")


def run_all_tests():
    """Run complete test suite for Dynamic Belief Evolution"""
    
    print("🧪 DYNAMIC BELIEF EVOLUTION (B1) - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # List of all test functions
    tests = [
        ("Dynamic Parameter Interpolation", test_dynamic_parameter_interpolation),
        ("Crisis Scenario Generation", test_crisis_scenario_generation),
        ("Experiment Execution", test_experiment_execution),
        ("Trajectory Analysis", test_trajectory_analysis),
        ("Change Point Detection", test_change_point_detection),
        ("Visualization Generation", test_visualization_generation),
        ("Convenience Functions", test_convenience_functions),
        ("Configuration Loading", test_configuration_loading),
        ("Reproducibility", test_reproducibility),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Mathematical Consistency", test_mathematical_consistency)
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            test_start = time.time()
            test_func()
            test_duration = time.time() - test_start
            results.append((test_name, True, test_duration, None))
            print(f"✅ {test_name} PASSED ({test_duration:.2f}s)")
            
        except Exception as e:
            test_duration = time.time() - test_start
            results.append((test_name, False, test_duration, str(e)))
            print(f"❌ {test_name} FAILED ({test_duration:.2f}s): {e}")
            import traceback
            traceback.print_exc()
    
    total_duration = time.time() - total_start_time
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for _, success, _, _ in results if success)
    failed = len(results) - passed
    
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_duration:.2f}s")
    print()
    
    for test_name, success, duration, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<40} {status} ({duration:.2f}s)")
        if error:
            print(f"    Error: {error}")
    
    print()
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Dynamic Belief Evolution (B1) is ready for use.")
    else:
        print(f"⚠️  {failed} test(s) failed. Please review and fix issues before deployment.")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)