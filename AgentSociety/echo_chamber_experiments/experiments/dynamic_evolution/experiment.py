"""
Dynamic Belief Evolution Experiment (B1)

This experiment studies how beliefs evolve during crisis scenarios using
time-varying parameters. It models realistic crisis dynamics including
pandemic responses, election cycles, economic shocks, and social unrest.

Mathematical Foundation:
- Time-varying belief distributions: P(t) = φ(P₀, P₁, ..., Pₙ, t)  
- Crisis trajectory modeling with empirically-grounded scenarios
- Phase transition detection and intervention timing optimization
- Belief velocity and acceleration analysis
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, replace
from pathlib import Path

# Import core experiment framework
try:
    from ...core.experiment import ExperimentConfig, ExperimentResults, EchoChamberExperiment
    from ...core.continuous_integration import ContinuousAgentConfig, create_continuous_agent_population
    from ...core.network import SocialNetwork, NetworkConfig
    from ...core.agent import Agent, TopicType
    from ...core.dynamic_parameters import (
        DynamicBeliefParameters, 
        ParameterKeyframe, 
        CrisisScenarioGenerator,
        CrisisType,
        InterpolationMethod
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from core.experiment import ExperimentConfig, ExperimentResults, EchoChamberExperiment
    from core.continuous_integration import ContinuousAgentConfig, create_continuous_agent_population
    from core.network import SocialNetwork, NetworkConfig
    from core.agent import Agent, TopicType
    from core.dynamic_parameters import (
        DynamicBeliefParameters, 
        ParameterKeyframe, 
        CrisisScenarioGenerator,
        CrisisType,
        InterpolationMethod
    )


@dataclass
class DynamicEvolutionConfig(ExperimentConfig):
    """
    Configuration for Dynamic Belief Evolution experiments.
    
    Extends the base ExperimentConfig with dynamic parameter capabilities
    while maintaining full backward compatibility.
    """
    
    # Dynamic belief parameters
    dynamic_parameters: Optional[DynamicBeliefParameters] = None
    crisis_scenario: Optional[CrisisType] = None
    crisis_severity: float = 0.7
    
    # Analysis parameters
    sample_interval: int = 1                    # How often to sample beliefs
    belief_history_tracking: bool = True        # Track individual belief trajectories
    phase_detection_threshold: float = 0.1      # Threshold for detecting rapid changes
    
    # Intervention optimization
    optimize_intervention_timing: bool = False  # Find optimal intervention timing
    intervention_candidates: List[int] = field(default_factory=list)  # Candidate intervention rounds
    
    # Mathematical analysis settings
    derivative_epsilon: float = 0.1            # Step size for derivative computation
    trajectory_smoothing: float = 0.0          # Smoothing factor for trajectory analysis
    
    def __post_init__(self):
        """Initialize dynamic evolution configuration"""
        super().__post_init__()
        
        # If crisis scenario specified but no dynamic parameters, generate them
        if self.crisis_scenario and not self.dynamic_parameters:
            self.dynamic_parameters = CrisisScenarioGenerator.custom_scenario(
                crisis_type=self.crisis_scenario,
                severity=self.crisis_severity,
                duration=self.num_rounds
            )
        
        # Set up intervention candidates if optimization requested
        if self.optimize_intervention_timing and not self.intervention_candidates:
            # Default candidates: every 3 rounds from round 3 to 3/4 through experiment
            end_round = int(0.75 * self.num_rounds)
            self.intervention_candidates = list(range(3, end_round, 3))


class DynamicEvolutionResults(ExperimentResults):
    """
    Extended results container for dynamic belief evolution experiments.
    
    Includes trajectory analysis, phase transition detection, and
    intervention timing optimization results.
    """
    
    def __init__(self, config: DynamicEvolutionConfig):
        # Initialize parent class
        super().__init__(config)
        
        # Dynamic evolution specific results
        self.belief_trajectories: Dict[int, List[float]] = {}  # agent_id -> belief over time
        self.parameter_evolution: Dict[str, List[float]] = {}  # parameter -> value over time
        self.belief_velocities: List[float] = []              # Rate of belief change per round
        self.belief_accelerations: List[float] = []           # Acceleration of belief change
        
        # Phase transition analysis
        self.phase_transitions: List[Tuple[int, str, float]] = []  # (round, parameter, magnitude)
        self.rapid_change_periods: Dict[str, List[Tuple[int, int]]] = {}  # parameter -> [(start, end)]
        
        # Crisis scenario analysis
        self.crisis_impact_metrics: Dict[str, float] = {}
        self.recovery_timeline: Optional[Dict[str, int]] = None  # When parameters return to baseline
        
        # Intervention timing analysis
        self.optimal_intervention_round: Optional[int] = None
        self.intervention_effectiveness_by_timing: Dict[int, float] = {}
    
    def compute_trajectory_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive statistics about belief trajectories.
        
        Mathematical Analysis:
        - Trajectory variance: σ²(trajectory) for each agent
        - Cross-agent correlation: correlation matrix of trajectories  
        - Synchronization index: how synchronized belief changes are
        - Volatility metrics: standard deviation of belief velocities
        
        Returns:
            Dictionary of trajectory statistics
        """
        
        if not self.belief_trajectories:
            return {}
        
        # Convert trajectories to matrix for analysis
        agent_ids = sorted(self.belief_trajectories.keys())
        trajectory_matrix = np.array([
            self.belief_trajectories[agent_id] for agent_id in agent_ids
        ])
        
        stats = {}
        
        # Individual trajectory statistics
        trajectory_variances = np.var(trajectory_matrix, axis=1)
        trajectory_ranges = np.ptp(trajectory_matrix, axis=1)  # Peak-to-peak
        
        stats['individual_trajectories'] = {
            'mean_variance': np.mean(trajectory_variances),
            'std_variance': np.std(trajectory_variances),
            'mean_range': np.mean(trajectory_ranges),
            'max_range': np.max(trajectory_ranges),
            'min_range': np.min(trajectory_ranges)
        }
        
        # Cross-agent correlation analysis
        if len(agent_ids) > 1:
            correlation_matrix = np.corrcoef(trajectory_matrix)
            upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
            
            stats['cross_agent_correlation'] = {
                'mean_correlation': np.mean(upper_triangle),
                'std_correlation': np.std(upper_triangle),
                'max_correlation': np.max(upper_triangle),
                'min_correlation': np.min(upper_triangle)
            }
        
        # Synchronization analysis
        if len(self.belief_velocities) > 1:
            # Compute synchronization index: variance of population mean over time
            population_means = np.mean(trajectory_matrix, axis=0)
            synchronization_index = np.var(population_means)
            
            stats['synchronization'] = {
                'synchronization_index': synchronization_index,
                'population_trajectory_range': np.ptp(population_means)
            }
        
        # Volatility analysis
        if self.belief_velocities:
            velocity_volatility = np.std(self.belief_velocities)
            stats['volatility'] = {
                'velocity_volatility': velocity_volatility,
                'max_velocity': np.max(np.abs(self.belief_velocities)),
                'velocity_autocorrelation': self._compute_autocorrelation(self.belief_velocities)
            }
        
        return stats
    
    def _compute_autocorrelation(self, series: List[float], max_lag: int = 5) -> List[float]:
        """Compute autocorrelation function of a time series"""
        if len(series) < 2:
            return []
        
        series_array = np.array(series)
        n = len(series_array)
        autocorr = []
        
        for lag in range(1, min(max_lag + 1, n)):
            if n - lag <= 1:
                break
            
            # Compute correlation between series and lagged version
            corr = np.corrcoef(series_array[:-lag], series_array[lag:])[0, 1]
            if not np.isnan(corr):
                autocorr.append(corr)
        
        return autocorr


class DynamicEvolutionExperiment:
    """
    Dynamic Belief Evolution Experiment Implementation.
    
    This experiment runs echo chamber simulations with time-varying belief
    parameters to study crisis-driven dynamics and optimal intervention timing.
    
    Mathematical Foundation:
    - Integrates dynamic parameter system with agent-based modeling
    - Tracks belief evolution trajectories P_i(t) for each agent i
    - Analyzes population dynamics using statistical measures
    - Optimizes intervention timing through systematic evaluation
    """
    
    def __init__(self, config: DynamicEvolutionConfig):
        self.config = config
        self.results = DynamicEvolutionResults(config)
        
        # Validate configuration
        if not config.dynamic_parameters:
            raise ValueError("Dynamic parameters must be specified for dynamic evolution experiment")
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Initialize tracking variables
        self.round_number = 0
        self.agents = []
        self.network = None
        
        # Analysis tracking
        self.parameter_history = {
            'polarization_strength': [],
            'polarization_asymmetry': [],
            'gap_size': [],
            'center': [],
            'spread': []
        }
        
        print(f"🌀 Initializing Dynamic Belief Evolution Experiment")
        print(f"   Crisis scenario: {config.crisis_scenario.value if config.crisis_scenario else 'Custom'}")
        print(f"   Duration: {config.num_rounds} rounds")
        print(f"   Dynamic parameters: {len(config.dynamic_parameters.keyframes)} keyframes")
    
    def run_full_experiment(self) -> DynamicEvolutionResults:
        """
        Run the complete dynamic belief evolution experiment.
        
        Process:
        1. Initialize population with initial belief parameters
        2. For each round:
           - Update belief parameters based on timeline
           - Regenerate agent population with new parameters  
           - Run interaction dynamics
           - Track belief trajectories and parameter evolution
        3. Analyze trajectories and detect phase transitions
        4. Optimize intervention timing if requested
        
        Returns:
            Comprehensive results including trajectory analysis
        """
        
        print(f"🚀 Starting Dynamic Belief Evolution: {self.config.name}")
        start_time = time.time()
        
        # Run main experiment loop
        for round_num in range(self.config.num_rounds):
            self.round_number = round_num
            print(f"\n🔄 Round {round_num + 1}/{self.config.num_rounds}")
            
            # Get parameters for this round
            current_params = self.config.dynamic_parameters.get_parameters_at_time(round_num)
            
            # Track parameter evolution
            self._record_parameter_state(current_params)
            
            # Initialize or update agent population
            if round_num == 0:
                self._initialize_population(current_params)
            else:
                self._update_population_beliefs(current_params)
            
            # Check for intervention
            if round_num == self.config.intervention_round and self.config.intervention_type:
                self._apply_intervention()
            
            # Run interaction round
            self._run_interaction_round()
            
            # Track belief trajectories
            if self.config.belief_history_tracking:
                self._record_belief_trajectories()
            
            # Update network dynamics
            if self.network:
                self.network.update_network_dynamics(round_num)
            
            # Record experiment state
            self._record_round_state()
            
            # Print progress
            current_polarization = self.results.polarization_over_time[-1] if self.results.polarization_over_time else 0
            print(f"   Polarization: {current_polarization:.3f}")
            print(f"   Belief center: {current_params.center:.3f}")
            print(f"   Polarization strength: {current_params.polarization_strength:.3f}")
        
        # Post-experiment analysis
        self._perform_trajectory_analysis()
        self._detect_phase_transitions()
        self._analyze_crisis_impact()
        
        # Intervention timing optimization
        if self.config.optimize_intervention_timing:
            self._optimize_intervention_timing()
        
        duration = time.time() - start_time
        print(f"\n✅ Dynamic Evolution completed in {duration:.2f} seconds")
        
        return self.results
    
    def _initialize_population(self, initial_params):
        """Initialize agent population with initial belief parameters"""
        
        continuous_config = ContinuousAgentConfig(
            num_agents=self.config.num_agents,
            topic=self.config.topic,
            belief_params=initial_params,
            random_seed=self.config.random_seed
        )
        
        self.agents = create_continuous_agent_population(continuous_config)
        self.network = SocialNetwork(self.agents, self.config.network_config)
        
        # Initialize trajectory tracking
        self.results.belief_trajectories = {
            agent.id: [agent.belief_strength] for agent in self.agents
        }
        
        # Record initial state
        self._record_round_state()
    
    def _update_population_beliefs(self, new_params):
        """
        Update agent beliefs based on new parameters while preserving agent identities.
        
        Mathematical Approach:
        - Compute belief shift: Δ = P_new - P_current  
        - Apply proportional adjustment to each agent
        - Preserve relative belief positions within new distribution
        - Add noise for realistic dynamics
        """
        
        if not self.agents:
            return
        
        # Get current belief distribution statistics
        current_beliefs = [agent.belief_strength for agent in self.agents]
        current_mean = np.mean(current_beliefs)
        current_std = np.std(current_beliefs)
        
        # Compute target distribution statistics
        target_mean = new_params.center
        target_std = new_params.spread
        
        # Apply gradual belief evolution
        evolution_rate = 0.3  # Controls how quickly beliefs adapt
        
        for agent in self.agents:
            # Normalize current belief relative to current distribution
            if current_std > 1e-6:
                normalized_belief = (agent.belief_strength - current_mean) / current_std
            else:
                normalized_belief = 0.0
            
            # Compute target belief in new distribution
            target_belief = target_mean + normalized_belief * target_std
            
            # Apply gradual evolution
            new_belief = agent.belief_strength + evolution_rate * (target_belief - agent.belief_strength)
            
            # Add noise for realistic dynamics
            noise = np.random.normal(0, new_params.noise_level)
            new_belief += noise
            
            # Clamp to valid range
            agent.belief_strength = np.clip(new_belief, new_params.min_belief, new_params.max_belief)
    
    def _record_parameter_state(self, params):
        """Record current parameter values for evolution tracking"""
        
        self.parameter_history['polarization_strength'].append(params.polarization_strength)
        self.parameter_history['polarization_asymmetry'].append(params.polarization_asymmetry)
        self.parameter_history['gap_size'].append(params.gap_size)
        self.parameter_history['center'].append(params.center)
        self.parameter_history['spread'].append(params.spread)
    
    def _record_belief_trajectories(self):
        """Record current belief values for all agents"""
        
        for agent in self.agents:
            if agent.id in self.results.belief_trajectories:
                self.results.belief_trajectories[agent.id].append(agent.belief_strength)
            else:
                self.results.belief_trajectories[agent.id] = [agent.belief_strength]
    
    def _run_interaction_round(self):
        """Run one round of agent interactions"""
        
        if not self.network or not self.agents:
            return
        
        # Simple interaction approach: select random pairs of agents
        import random
        
        interactions_performed = 0
        max_interactions = self.config.interactions_per_round
        
        for _ in range(max_interactions):
            # Select two random agents
            if len(self.agents) < 2:
                break
                
            agent1, agent2 = random.sample(self.agents, 2)
            
            # Check if they should interact (based on network connection or random chance)
            connected_agents = self.network.get_connected_agents(agent1.id)
            connected_ids = [a.id for a in connected_agents]
            
            # Interact if connected or with small random probability
            if agent2.id in connected_ids or random.random() < 0.1:
                self._agent_interaction(agent1, agent2)
                interactions_performed += 1
        
        # Ensure minimum interactions happen
        if interactions_performed < max_interactions // 4:
            # Force some random interactions
            for _ in range(max_interactions // 4):
                if len(self.agents) >= 2:
                    agent1, agent2 = random.sample(self.agents, 2)
                    self._agent_interaction(agent1, agent2)
    
    def _agent_interaction(self, agent1: Agent, agent2: Agent):
        """
        Single agent interaction with belief influence.
        
        Mathematical Model:
        - Influence strength based on similarity and confidence
        - Belief update: b_new = b_old + α * influence * (b_other - b_old)
        - Where α is learning rate, influence depends on agent properties
        """
        
        # Compute influence strength
        belief_similarity = 1.0 - abs(agent1.belief_strength - agent2.belief_strength) / 2.0
        influence_strength = 0.1 * belief_similarity * (agent1.openness + agent2.openness) / 2.0
        
        # Mutual influence with asymmetric strength
        agent1_influence = influence_strength * agent2.confidence
        agent2_influence = influence_strength * agent1.confidence
        
        # Update beliefs
        agent1.belief_strength += agent1_influence * (agent2.belief_strength - agent1.belief_strength)
        agent2.belief_strength += agent2_influence * (agent1.belief_strength - agent2.belief_strength)
        
        # Clamp to valid range
        agent1.belief_strength = np.clip(agent1.belief_strength, -1.0, 1.0)
        agent2.belief_strength = np.clip(agent2.belief_strength, -1.0, 1.0)
    
    def _record_round_state(self):
        """Record current experiment state (delegates to existing framework)"""
        
        if not self.agents:
            return
        
        # Compute polarization
        beliefs = [agent.belief_strength for agent in self.agents]
        polarization = self._compute_polarization(beliefs)
        self.results.polarization_over_time.append(polarization)
        
        # Detect echo chambers (simplified)
        echo_chambers = self._detect_echo_chambers()
        self.results.echo_chambers_history.append(echo_chambers)
        
        # Record agent state
        agent_states = []
        for agent in self.agents:
            agent_states.append({
                'id': agent.id,
                'belief_strength': agent.belief_strength,
                'confidence': agent.confidence,
                'openness': agent.openness,
                'personality_type': agent.personality_type.value
            })
        self.results.agents_history.append(agent_states)
    
    def _compute_polarization(self, beliefs: List[float]) -> float:
        """Compute population polarization index"""
        if len(beliefs) < 2:
            return 0.0
        
        # Compute variance-based polarization measure
        variance = np.var(beliefs)
        # Normalize to [0, 1] range (max variance is 1.0 for beliefs in [-1, 1])
        return min(variance / 1.0, 1.0)
    
    def _detect_echo_chambers(self) -> List[List[int]]:
        """Detect echo chambers based on belief similarity and network connections"""
        
        if not self.network or not self.agents:
            return []
        
        # Simplified echo chamber detection
        chambers = []
        processed_agents = set()
        
        for agent in self.agents:
            if agent.id in processed_agents:
                continue
            
            # Find similar connected agents
            chamber = [agent.id]
            connected_agents = self.network.get_connected_agents(agent.id)
            
            for connected_agent in connected_agents:
                belief_diff = abs(agent.belief_strength - connected_agent.belief_strength)
                if belief_diff < 0.3:  # Similarity threshold
                    chamber.append(connected_agent.id)
                    processed_agents.add(connected_agent.id)
            
            if len(chamber) > 2:  # Minimum chamber size
                chambers.append(chamber)
                processed_agents.update(chamber)
        
        return chambers
    
    def _apply_intervention(self):
        """Apply intervention (fact-checking, diverse exposure, etc.)"""
        
        intervention_type = self.config.intervention_type
        print(f"   🛠️ Applying intervention: {intervention_type}")
        
        if intervention_type == "fact_check":
            # Reduce extreme beliefs slightly
            for agent in self.agents:
                if abs(agent.belief_strength) > 0.7:
                    agent.belief_strength *= 0.9
        
        elif intervention_type == "diverse_exposure":
            # Add random connections between different belief groups
            if self.network:
                self.network.add_bridge_connections(num_bridges=max(1, len(self.agents) // 10))
        
        elif intervention_type == "bridge_building":
            # Increase openness of moderate agents
            for agent in self.agents:
                if abs(agent.belief_strength) < 0.3:
                    agent.openness = min(1.0, agent.openness + 0.1)
    
    def _perform_trajectory_analysis(self):
        """Perform comprehensive mathematical analysis of belief trajectories"""
        
        print("\n📊 Analyzing belief trajectories...")
        
        if not self.results.belief_trajectories:
            return
        
        # Compute belief velocities (rate of change)
        self.results.belief_velocities = self._compute_population_velocities()
        
        # Compute belief accelerations
        if len(self.results.belief_velocities) > 1:
            self.results.belief_accelerations = list(np.diff(self.results.belief_velocities))
        
        # Store parameter evolution in results
        self.results.parameter_evolution = self.parameter_history.copy()
        
        # Compute trajectory statistics
        trajectory_stats = self.results.compute_trajectory_statistics()
        print(f"   Mean trajectory variance: {trajectory_stats.get('individual_trajectories', {}).get('mean_variance', 0):.4f}")
        
        if 'cross_agent_correlation' in trajectory_stats:
            mean_corr = trajectory_stats['cross_agent_correlation']['mean_correlation']
            print(f"   Mean cross-agent correlation: {mean_corr:.4f}")
    
    def _compute_population_velocities(self) -> List[float]:
        """Compute population-level belief velocities"""
        
        if not self.results.belief_trajectories:
            return []
        
        # Compute population mean at each time point
        num_rounds = len(list(self.results.belief_trajectories.values())[0])
        population_means = []
        
        for round_idx in range(num_rounds):
            round_beliefs = [
                trajectory[round_idx] for trajectory in self.results.belief_trajectories.values()
                if round_idx < len(trajectory)
            ]
            if round_beliefs:
                population_means.append(np.mean(round_beliefs))
        
        # Compute velocities as differences
        if len(population_means) > 1:
            return list(np.diff(population_means))
        return []
    
    def _detect_phase_transitions(self):
        """Detect rapid changes in belief parameters (phase transitions)"""
        
        print("\n🔍 Detecting phase transitions...")
        
        for param_name in self.parameter_history:
            values = self.parameter_history[param_name]
            
            if len(values) < 3:
                continue
            
            # Compute parameter velocities
            velocities = np.diff(values)
            rapid_changes = []
            
            # Find points with rapid change
            for i, velocity in enumerate(velocities):
                if abs(velocity) > self.config.phase_detection_threshold:
                    rapid_changes.append((i + 1, param_name, velocity))  # Round number, param, magnitude
            
            # Store rapid changes
            self.results.phase_transitions.extend(rapid_changes)
            
            # Detect periods of sustained rapid change
            rapid_periods = self.config.dynamic_parameters.detect_rapid_changes(
                param_name, self.config.phase_detection_threshold
            )
            
            if rapid_periods:
                self.results.rapid_change_periods[param_name] = [
                    (int(start), int(end)) for start, end in rapid_periods
                ]
                print(f"   {param_name}: {len(rapid_periods)} rapid change periods")
    
    def _analyze_crisis_impact(self):
        """Analyze the impact of the crisis scenario on belief dynamics"""
        
        print("\n🌊 Analyzing crisis impact...")
        
        if not self.results.polarization_over_time:
            return
        
        # Compute crisis impact metrics
        initial_polarization = self.results.polarization_over_time[0]
        peak_polarization = max(self.results.polarization_over_time)
        final_polarization = self.results.polarization_over_time[-1]
        
        self.results.crisis_impact_metrics = {
            'initial_polarization': initial_polarization,
            'peak_polarization': peak_polarization,
            'final_polarization': final_polarization,
            'polarization_increase': peak_polarization - initial_polarization,
            'recovery_ratio': (peak_polarization - final_polarization) / max(peak_polarization - initial_polarization, 0.001),
            'volatility': np.std(self.results.polarization_over_time)
        }
        
        print(f"   Peak polarization: {peak_polarization:.3f}")
        print(f"   Polarization increase: {self.results.crisis_impact_metrics['polarization_increase']:.3f}")
        print(f"   Recovery ratio: {self.results.crisis_impact_metrics['recovery_ratio']:.3f}")
        
        # Detect recovery timeline
        self._compute_recovery_timeline()
    
    def _compute_recovery_timeline(self):
        """Compute when parameters return to baseline levels"""
        
        recovery_timeline = {}
        
        for param_name, values in self.parameter_history.items():
            if len(values) < 2:
                continue
            
            initial_value = values[0]
            
            # Find when parameter returns to within 10% of initial value
            recovery_threshold = 0.1 * abs(initial_value) if initial_value != 0 else 0.05
            
            for i, value in enumerate(values):
                if i > len(values) // 4:  # Don't consider early rounds
                    if abs(value - initial_value) <= recovery_threshold:
                        recovery_timeline[param_name] = i
                        break
        
        if recovery_timeline:
            self.results.recovery_timeline = recovery_timeline
    
    def _optimize_intervention_timing(self):
        """
        Find optimal intervention timing through systematic evaluation.
        
        Mathematical Approach:
        - Test intervention at multiple candidate time points
        - Measure effectiveness as reduction in final polarization
        - Account for intervention cost (earlier = more expensive)
        - Return optimal timing with effectiveness scores
        """
        
        print("\n⚡ Optimizing intervention timing...")
        
        if not self.config.intervention_candidates:
            return
        
        baseline_results = self.results.polarization_over_time.copy()
        baseline_final_pol = baseline_results[-1] if baseline_results else 0
        
        effectiveness_scores = {}
        
        for candidate_round in self.config.intervention_candidates:
            if candidate_round >= self.config.num_rounds:
                continue
            
            print(f"   Testing intervention at round {candidate_round}...")
            
            # Run experiment with intervention at this time
            test_config = replace(self.config, intervention_round=candidate_round)
            test_experiment = DynamicEvolutionExperiment(test_config)
            test_results = test_experiment.run_full_experiment()
            
            # Compute effectiveness
            test_final_pol = test_results.polarization_over_time[-1] if test_results.polarization_over_time else 0
            effectiveness = baseline_final_pol - test_final_pol
            
            # Apply timing penalty (earlier interventions are more costly)
            timing_penalty = 0.01 * (self.config.num_rounds - candidate_round) / self.config.num_rounds
            adjusted_effectiveness = effectiveness - timing_penalty
            
            effectiveness_scores[candidate_round] = adjusted_effectiveness
        
        # Find optimal timing
        if effectiveness_scores:
            optimal_round = max(effectiveness_scores.keys(), key=lambda r: effectiveness_scores[r])
            self.results.optimal_intervention_round = optimal_round
            self.results.intervention_effectiveness_by_timing = effectiveness_scores
            
            print(f"   Optimal intervention round: {optimal_round}")
            print(f"   Effectiveness: {effectiveness_scores[optimal_round]:.3f}")


# Convenience functions for common experiment scenarios

def run_pandemic_experiment(num_agents: int = 100, duration: int = 25, 
                           severity: float = 0.8, **kwargs) -> DynamicEvolutionResults:
    """Run a pandemic scenario experiment with default parameters"""
    
    config = DynamicEvolutionConfig(
        name=f"Pandemic Crisis (severity={severity})",
        description="Dynamic belief evolution during pandemic crisis",
        num_agents=num_agents,
        num_rounds=duration,
        crisis_scenario=CrisisType.PANDEMIC,
        crisis_severity=severity,
        **kwargs
    )
    
    experiment = DynamicEvolutionExperiment(config)
    return experiment.run_full_experiment()


def run_election_experiment(num_agents: int = 80, duration: int = 20,
                           peak_polarization: float = 0.9, **kwargs) -> DynamicEvolutionResults:
    """Run an election cycle experiment with default parameters"""
    
    config = DynamicEvolutionConfig(
        name=f"Election Cycle (peak_pol={peak_polarization})",
        description="Dynamic belief evolution during election cycle",
        num_agents=num_agents,
        num_rounds=duration,
        crisis_scenario=CrisisType.ELECTION,
        crisis_severity=(peak_polarization - 0.5) / 0.4,  # Convert to severity scale
        **kwargs
    )
    
    experiment = DynamicEvolutionExperiment(config)
    return experiment.run_full_experiment()


def run_economic_shock_experiment(num_agents: int = 120, duration: int = 30,
                                 shock_severity: float = 0.7, **kwargs) -> DynamicEvolutionResults:
    """Run an economic shock experiment with default parameters"""
    
    config = DynamicEvolutionConfig(
        name=f"Economic Shock (severity={shock_severity})",
        description="Dynamic belief evolution during economic crisis",
        num_agents=num_agents,
        num_rounds=duration,
        crisis_scenario=CrisisType.ECONOMIC_SHOCK,
        crisis_severity=shock_severity,
        **kwargs
    )
    
    experiment = DynamicEvolutionExperiment(config)
    return experiment.run_full_experiment()


def compare_crisis_scenarios(num_agents: int = 100, duration: int = 25) -> Dict[str, DynamicEvolutionResults]:
    """
    Compare different crisis scenarios under identical conditions.
    
    Returns:
        Dictionary mapping scenario names to experiment results
    """
    
    scenarios = {
        'pandemic': run_pandemic_experiment(num_agents, duration, severity=0.7),
        'election': run_election_experiment(num_agents, duration, peak_polarization=0.8),
        'economic_shock': run_economic_shock_experiment(num_agents, duration, shock_severity=0.7)
    }
    
    return scenarios