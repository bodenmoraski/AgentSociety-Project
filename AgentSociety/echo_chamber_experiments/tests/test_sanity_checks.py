"""
Comprehensive Sanity Checks and Quality Assurance for Dynamic Belief Evolution

This module provides common-sense tests, anomaly detection, and quality measures
to ensure the experimental system behaves correctly and spot any odd behavior.

Test Categories:
1. Mathematical Consistency Tests
2. Behavioral Sanity Checks  
3. Performance and Stability Tests
4. Data Quality Validation
5. Anomaly Detection
6. Crisis Scenario Validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Tuple, Optional, Any
import time
from dataclasses import dataclass
from pathlib import Path

# Import experiment components
try:
    from experiments.dynamic_evolution.experiment import DynamicEvolutionExperiment, DynamicEvolutionConfig
    from experiments.dynamic_evolution.analysis import TrajectoryAnalyzer
    from core.dynamic_parameters import CrisisType, CrisisScenarioGenerator
    from core.agent import TopicType
except ImportError:
    print("⚠️ Warning: Could not import all experiment components")


@dataclass
class SanityCheckResult:
    """Container for sanity check results"""
    test_name: str
    passed: bool
    score: float  # 0-1, higher is better
    details: Dict[str, Any]
    warnings: List[str]
    recommendations: List[str]


class DynamicEvolutionSanityChecker:
    """
    Comprehensive sanity checking system for dynamic belief evolution experiments.
    
    Provides automatic detection of:
    - Mathematical inconsistencies
    - Behavioral anomalies
    - Performance issues
    - Data quality problems
    - Crisis scenario validity
    """
    
    def __init__(self, tolerance_strict: float = 0.01, tolerance_loose: float = 0.1):
        self.tolerance_strict = tolerance_strict
        self.tolerance_loose = tolerance_loose
        self.results = []
        
    def run_full_sanity_check(self, config: DynamicEvolutionConfig, 
                             run_experiment: bool = True) -> Dict[str, SanityCheckResult]:
        """
        Run comprehensive sanity check suite.
        
        Args:
            config: Experiment configuration to test
            run_experiment: Whether to actually run the experiment (vs test config only)
            
        Returns:
            Dictionary of test results
        """
        
        print("🧪 Running Comprehensive Sanity Check Suite")
        print("=" * 60)
        
        results = {}
        
        # 1. Configuration Validation
        print("📋 Testing Configuration...")
        results['config_validation'] = self._test_configuration_sanity(config)
        
        # 2. Crisis Scenario Validation  
        print("🌊 Testing Crisis Scenarios...")
        results['crisis_validation'] = self._test_crisis_scenario_sanity(config)
        
        if run_experiment:
            # 3. Run Small Test Experiment
            print("🏃 Running Test Experiment...")
            test_results = self._run_test_experiment(config)
            
            if test_results:
                # 4. Mathematical Consistency
                print("🧮 Testing Mathematical Consistency...")
                results['math_consistency'] = self._test_mathematical_consistency(test_results)
                
                # 5. Behavioral Sanity
                print("🎭 Testing Behavioral Sanity...")
                results['behavioral_sanity'] = self._test_behavioral_sanity(test_results)
                
                # 6. Data Quality
                print("📊 Testing Data Quality...")
                results['data_quality'] = self._test_data_quality(test_results)
                
                # 7. Performance Check
                print("⚡ Testing Performance...")
                results['performance'] = self._test_performance(config)
                
                # 8. Anomaly Detection
                print("🔍 Running Anomaly Detection...")
                results['anomaly_detection'] = self._test_anomaly_detection(test_results)
        
        # Summary
        self._print_sanity_summary(results)
        return results
    
    def _test_configuration_sanity(self, config: DynamicEvolutionConfig) -> SanityCheckResult:
        """Test if configuration makes common sense"""
        
        warnings_list = []
        recommendations = []
        score = 1.0
        
        # Basic bounds checking
        if config.num_agents < 5:
            warnings_list.append("Very few agents - results may be unreliable")
            score -= 0.2
            
        if config.num_agents > 1000:
            warnings_list.append("Many agents - may be slow")
            recommendations.append("Consider reducing agents for testing")
            
        if config.num_rounds < 3:
            warnings_list.append("Very few rounds - insufficient for dynamics")
            score -= 0.3
            
        if config.num_rounds > 100:
            warnings_list.append("Many rounds - may be slow")
            
        # Interaction density check
        interactions_per_agent = config.interactions_per_round / config.num_agents
        if interactions_per_agent < 0.5:
            warnings_list.append("Low interaction density - agents may not influence each other")
            recommendations.append("Increase interactions_per_round")
            score -= 0.1
            
        if interactions_per_agent > 10:
            warnings_list.append("High interaction density - may be unrealistic")
            
        # Crisis severity bounds
        if hasattr(config, 'crisis_severity'):
            if config.crisis_severity < 0 or config.crisis_severity > 1:
                warnings_list.append("Crisis severity outside [0,1] range")
                score -= 0.3
                
        details = {
            'agents': config.num_agents,
            'rounds': config.num_rounds,
            'interactions_per_agent': interactions_per_agent,
            'crisis_severity': getattr(config, 'crisis_severity', None)
        }
        
        return SanityCheckResult(
            test_name="Configuration Validation",
            passed=score > 0.6,
            score=score,
            details=details,
            warnings=warnings_list,
            recommendations=recommendations
        )
    
    def _test_crisis_scenario_sanity(self, config: DynamicEvolutionConfig) -> SanityCheckResult:
        """Test if crisis scenario parameters make sense"""
        
        warnings_list = []
        recommendations = []
        score = 1.0
        
        try:
            # Generate crisis scenario
            if config.crisis_scenario:
                scenario = CrisisScenarioGenerator.custom_scenario(
                    crisis_type=config.crisis_scenario,
                    severity=getattr(config, 'crisis_severity', 0.7),
                    duration=config.num_rounds
                )
                
                keyframes = scenario.keyframes
                
                # Check keyframe sanity
                if len(keyframes) < 3:
                    warnings_list.append("Too few keyframes for realistic crisis")
                    score -= 0.2
                    
                # Check parameter ranges
                pol_strengths = [kf.parameters.polarization_strength for kf in keyframes]
                if max(pol_strengths) - min(pol_strengths) < 0.1:
                    warnings_list.append("Crisis causes minimal polarization change")
                    score -= 0.3
                    
                # Check timeline sanity
                times = [kf.time_point for kf in keyframes]
                if times != sorted(times):
                    warnings_list.append("Keyframe times not in order")
                    score -= 0.5
                    
                # Check for realistic crisis arc
                early_pol = pol_strengths[0]
                peak_pol = max(pol_strengths)
                final_pol = pol_strengths[-1]
                
                if peak_pol <= early_pol:
                    warnings_list.append("Crisis doesn't increase polarization")
                    recommendations.append("Check crisis severity parameter")
                    score -= 0.4
                    
                details = {
                    'keyframes': len(keyframes),
                    'polarization_range': (min(pol_strengths), max(pol_strengths)),
                    'polarization_change': peak_pol - early_pol,
                    'recovery_ratio': (peak_pol - final_pol) / (peak_pol - early_pol) if peak_pol > early_pol else 0
                }
                
            else:
                warnings_list.append("No crisis scenario specified")
                score -= 0.1
                details = {'scenario': 'None'}
                
        except Exception as e:
            warnings_list.append(f"Crisis scenario generation failed: {e}")
            score = 0.0
            details = {'error': str(e)}
            
        return SanityCheckResult(
            test_name="Crisis Scenario Validation",
            passed=score > 0.6,
            score=score,
            details=details,
            warnings=warnings_list,
            recommendations=recommendations
        )
    
    def _run_test_experiment(self, config: DynamicEvolutionConfig) -> Optional[Any]:
        """Run small test experiment for validation"""
        
        try:
            # Create scaled-down version for testing
            test_config = DynamicEvolutionConfig(
                name=f"Sanity Test: {config.name}",
                description="Automated sanity check run",
                num_agents=min(config.num_agents, 20),  # Cap at 20 agents
                topic=config.topic,
                num_rounds=min(config.num_rounds, 10),  # Cap at 10 rounds
                crisis_scenario=config.crisis_scenario,
                crisis_severity=getattr(config, 'crisis_severity', 0.7),
                interactions_per_round=min(config.interactions_per_round, 50),
                belief_history_tracking=True,
                random_seed=42  # Fixed for reproducibility
            )
            
            # Run experiment
            experiment = DynamicEvolutionExperiment(test_config)
            results = experiment.run_full_experiment()
            return results
            
        except Exception as e:
            print(f"❌ Test experiment failed: {e}")
            return None
    
    def _test_mathematical_consistency(self, results) -> SanityCheckResult:
        """Test mathematical consistency of results"""
        
        warnings_list = []
        recommendations = []
        score = 1.0
        
        # Check polarization bounds
        pol_values = results.polarization_over_time
        if not all(0 <= p <= 1.5 for p in pol_values):  # Allow slight overshoot
            invalid_count = sum(1 for p in pol_values if p < 0 or p > 1.5)
            warnings_list.append(f"Polarization values outside bounds: {invalid_count} violations")
            score -= 0.4
            
        # Check belief trajectory bounds
        belief_violations = 0
        if results.belief_trajectories:
            for agent_id, trajectory in results.belief_trajectories.items():
                violations = sum(1 for b in trajectory if b < -1.2 or b > 1.2)
                belief_violations += violations
                
        if belief_violations > 0:
            warnings_list.append(f"Belief values outside bounds: {belief_violations} violations")
            score -= 0.3
            
        # Check for NaN or infinite values
        nan_count = sum(1 for p in pol_values if np.isnan(p) or np.isinf(p))
        if nan_count > 0:
            warnings_list.append(f"Invalid numeric values: {nan_count} NaN/Inf")
            score -= 0.5
            
        # Check trajectory completeness
        if results.belief_trajectories:
            expected_length = len(pol_values)
            incomplete_trajectories = sum(
                1 for traj in results.belief_trajectories.values() 
                if len(traj) != expected_length
            )
            if incomplete_trajectories > 0:
                warnings_list.append(f"Incomplete trajectories: {incomplete_trajectories}")
                score -= 0.2
                
        # Check parameter evolution consistency
        if results.parameter_evolution:
            for param, values in results.parameter_evolution.items():
                if len(values) != len(pol_values):
                    warnings_list.append(f"Parameter {param} has inconsistent length")
                    score -= 0.1
                    
        details = {
            'polarization_range': (min(pol_values), max(pol_values)),
            'belief_violations': belief_violations,
            'nan_count': nan_count,
            'trajectory_count': len(results.belief_trajectories) if results.belief_trajectories else 0
        }
        
        return SanityCheckResult(
            test_name="Mathematical Consistency",
            passed=score > 0.7,
            score=score,
            details=details,
            warnings=warnings_list,
            recommendations=recommendations
        )
    
    def _test_behavioral_sanity(self, results) -> SanityCheckResult:
        """Test if agent behavior makes common sense"""
        
        warnings_list = []
        recommendations = []
        score = 1.0
        
        pol_values = results.polarization_over_time
        
        # Check if polarization changes over time (shouldn't be static)
        pol_variance = np.var(pol_values)
        if pol_variance < 1e-6:
            warnings_list.append("Polarization remains static - no dynamics")
            score -= 0.4
            recommendations.append("Check interaction mechanisms")
            
        # Check if crisis has observable impact
        if len(pol_values) > 3:
            early_pol = np.mean(pol_values[:2])
            late_pol = np.mean(pol_values[-2:])
            crisis_impact = abs(late_pol - early_pol)
            
            if crisis_impact < 0.05:
                warnings_list.append("Crisis has minimal observable impact")
                score -= 0.3
                recommendations.append("Increase crisis severity or check implementation")
                
        # Check agent diversity
        if results.belief_trajectories:
            final_beliefs = [traj[-1] for traj in results.belief_trajectories.values()]
            belief_diversity = np.std(final_beliefs)
            
            if belief_diversity < 0.1:
                warnings_list.append("Agents converge to identical beliefs")
                score -= 0.2
                recommendations.append("Check interaction diversity parameters")
                
            if belief_diversity > 1.5:
                warnings_list.append("Agents remain completely isolated")
                score -= 0.2
                recommendations.append("Increase interaction strength")
                
        # Check for phase transitions
        phase_count = len(results.phase_transitions) if hasattr(results, 'phase_transitions') else 0
        if phase_count == 0:
            warnings_list.append("No phase transitions detected")
            score -= 0.1
            
        # Check crisis impact metrics
        has_crisis_metrics = hasattr(results, 'crisis_impact_metrics') and len(results.crisis_impact_metrics) > 0
        if not has_crisis_metrics:
            warnings_list.append("No crisis impact metrics computed")
            score -= 0.1
            
        details = {
            'polarization_variance': pol_variance,
            'crisis_impact': crisis_impact if 'crisis_impact' in locals() else 0,
            'belief_diversity': belief_diversity if 'belief_diversity' in locals() else 0,
            'phase_transitions': phase_count,
            'has_crisis_metrics': has_crisis_metrics
        }
        
        return SanityCheckResult(
            test_name="Behavioral Sanity",
            passed=score > 0.6,
            score=score,
            details=details,
            warnings=warnings_list,
            recommendations=recommendations
        )
    
    def _test_data_quality(self, results) -> SanityCheckResult:
        """Test data quality and completeness"""
        
        warnings_list = []
        recommendations = []
        score = 1.0
        
        # Check data completeness
        has_polarization = hasattr(results, 'polarization_over_time') and len(results.polarization_over_time) > 0
        has_trajectories = hasattr(results, 'belief_trajectories') and len(results.belief_trajectories) > 0
        has_parameters = hasattr(results, 'parameter_evolution') and len(results.parameter_evolution) > 0
        
        if not has_polarization:
            warnings_list.append("Missing polarization data")
            score -= 0.3
            
        if not has_trajectories:
            warnings_list.append("Missing belief trajectories")
            score -= 0.3
            
        if not has_parameters:
            warnings_list.append("Missing parameter evolution data")
            score -= 0.2
            
        # Check data resolution
        if has_polarization:
            data_points = len(results.polarization_over_time)
            if data_points < 3:
                warnings_list.append("Insufficient data resolution")
                score -= 0.2
                
        # Check agent coverage
        if has_trajectories:
            expected_agents = results.config.num_agents
            tracked_agents = len(results.belief_trajectories)
            coverage = tracked_agents / expected_agents
            
            if coverage < 0.8:
                warnings_list.append(f"Low agent tracking coverage: {coverage:.1%}")
                score -= 0.2
                
        details = {
            'has_polarization': has_polarization,
            'has_trajectories': has_trajectories,
            'has_parameters': has_parameters,
            'data_points': len(results.polarization_over_time) if has_polarization else 0,
            'agent_coverage': coverage if 'coverage' in locals() else 0
        }
        
        return SanityCheckResult(
            test_name="Data Quality",
            passed=score > 0.7,
            score=score,
            details=details,
            warnings=warnings_list,
            recommendations=recommendations
        )
    
    def _test_performance(self, config: DynamicEvolutionConfig) -> SanityCheckResult:
        """Test performance characteristics"""
        
        warnings_list = []
        recommendations = []
        score = 1.0
        
        # Run timed test
        start_time = time.time()
        
        try:
            # Small performance test
            perf_config = DynamicEvolutionConfig(
                name="Performance Test",
                description="Performance measurement",
                num_agents=20,
                topic=TopicType.HEALTHCARE,
                num_rounds=5,
                crisis_scenario=CrisisType.PANDEMIC,
                crisis_severity=0.7,
                interactions_per_round=30,
                random_seed=42
            )
            
            experiment = DynamicEvolutionExperiment(perf_config)
            results = experiment.run_full_experiment()
            
            duration = time.time() - start_time
            
            # Performance benchmarks
            time_per_agent_round = duration / (perf_config.num_agents * perf_config.num_rounds)
            
            if duration > 30:
                warnings_list.append(f"Slow execution: {duration:.1f}s for small test")
                score -= 0.3
                recommendations.append("Consider reducing agent count or rounds")
                
            if time_per_agent_round > 0.1:
                warnings_list.append(f"High per-agent-round time: {time_per_agent_round:.3f}s")
                score -= 0.2
                
        except Exception as e:
            warnings_list.append(f"Performance test failed: {e}")
            duration = float('inf')
            time_per_agent_round = float('inf')
            score = 0.0
            
        details = {
            'total_duration': duration,
            'time_per_agent_round': time_per_agent_round,
            'estimated_full_time': duration * (config.num_agents * config.num_rounds) / (20 * 5)
        }
        
        return SanityCheckResult(
            test_name="Performance",
            passed=score > 0.6,
            score=score,
            details=details,
            warnings=warnings_list,
            recommendations=recommendations
        )
    
    def _test_anomaly_detection(self, results) -> SanityCheckResult:
        """Detect anomalous patterns in results"""
        
        warnings_list = []
        recommendations = []
        score = 1.0
        
        pol_values = results.polarization_over_time
        
        # Detect sudden spikes or drops
        if len(pol_values) > 2:
            pol_diff = np.diff(pol_values)
            large_changes = np.abs(pol_diff) > 0.3
            
            if np.any(large_changes):
                spike_count = np.sum(large_changes)
                warnings_list.append(f"Sudden polarization changes detected: {spike_count}")
                score -= 0.1
                
        # Detect oscillations
        if len(pol_values) > 4:
            # Simple oscillation detection
            sign_changes = np.sum(np.diff(np.sign(np.diff(pol_values))) != 0)
            if sign_changes > len(pol_values) * 0.6:
                warnings_list.append("High oscillation in polarization")
                score -= 0.1
                
        # Detect impossible belief distributions
        if results.belief_trajectories:
            for agent_id, trajectory in results.belief_trajectories.items():
                # Check for impossible jumps
                if len(trajectory) > 1:
                    jumps = np.abs(np.diff(trajectory))
                    impossible_jumps = np.sum(jumps > 1.5)
                    if impossible_jumps > 0:
                        warnings_list.append(f"Agent {agent_id} has impossible belief jumps")
                        score -= 0.2
                        break
                        
        details = {
            'polarization_spikes': np.sum(large_changes) if 'large_changes' in locals() else 0,
            'oscillation_ratio': sign_changes / len(pol_values) if 'sign_changes' in locals() else 0,
            'impossible_jumps': any('impossible_jumps' in locals() for _ in [None])
        }
        
        return SanityCheckResult(
            test_name="Anomaly Detection",
            passed=score > 0.8,
            score=score,
            details=details,
            warnings=warnings_list,
            recommendations=recommendations
        )
    
    def _print_sanity_summary(self, results: Dict[str, SanityCheckResult]):
        """Print comprehensive summary of sanity check results"""
        
        print("\n" + "="*60)
        print("🎯 SANITY CHECK SUMMARY")
        print("="*60)
        
        overall_score = np.mean([r.score for r in results.values()])
        passed_tests = sum(1 for r in results.values() if r.passed)
        total_tests = len(results)
        
        # Overall status
        if overall_score >= 0.8:
            status = "🟢 EXCELLENT"
        elif overall_score >= 0.6:
            status = "🟡 ACCEPTABLE"
        else:
            status = "🔴 NEEDS ATTENTION"
            
        print(f"\n📊 Overall Score: {overall_score:.2f}/1.00 ({status})")
        print(f"✅ Passed Tests: {passed_tests}/{total_tests}")
        
        # Individual test results
        print(f"\n📋 Test Results:")
        for test_name, result in results.items():
            status_icon = "✅" if result.passed else "❌"
            print(f"   {status_icon} {result.test_name}: {result.score:.2f}")
            
        # Collect all warnings and recommendations
        all_warnings = []
        all_recommendations = []
        
        for result in results.values():
            all_warnings.extend(result.warnings)
            all_recommendations.extend(result.recommendations)
            
        # Print prioritized issues
        if all_warnings:
            print(f"\n⚠️  Warnings ({len(all_warnings)}):")
            for warning in all_warnings[:5]:  # Top 5
                print(f"   • {warning}")
            if len(all_warnings) > 5:
                print(f"   ... and {len(all_warnings) - 5} more")
                
        if all_recommendations:
            print(f"\n💡 Recommendations ({len(all_recommendations)}):")
            for rec in all_recommendations[:5]:  # Top 5
                print(f"   • {rec}")
            if len(all_recommendations) > 5:
                print(f"   ... and {len(all_recommendations) - 5} more")
                
        print("\n" + "="*60)


def run_quick_sanity_check():
    """Run a quick sanity check with default parameters"""
    
    print("🚀 Quick Sanity Check")
    
    # Create test configuration
    config = DynamicEvolutionConfig(
        name="Quick Sanity Test",
        description="Automated sanity validation",
        num_agents=15,
        topic=TopicType.HEALTHCARE,
        num_rounds=8,
        crisis_scenario=CrisisType.PANDEMIC,
        crisis_severity=0.7,
        interactions_per_round=25,
        belief_history_tracking=True,
        random_seed=42
    )
    
    # Run sanity checks
    checker = DynamicEvolutionSanityChecker()
    results = checker.run_full_sanity_check(config, run_experiment=True)
    
    return results


if __name__ == "__main__":
    # Run when executed directly
    results = run_quick_sanity_check()