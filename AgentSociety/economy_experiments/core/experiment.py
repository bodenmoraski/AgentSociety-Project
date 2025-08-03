"""
Main Experiment Framework for Echo Chamber Studies
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path

from .agent import Agent, TopicType, PersonalityType, create_diverse_agent_population
from .network import SocialNetwork, NetworkConfig


@dataclass
class ExperimentConfig:
    """Configuration for echo chamber experiments"""
    
    # Basic parameters
    name: str = "default_experiment"
    description: str = "Basic echo chamber experiment"
    
    # Population parameters
    num_agents: int = 50
    topic: TopicType = TopicType.GUN_CONTROL
    belief_distribution: str = "polarized"  # "polarized", "normal", "uniform"
    
    # Network parameters
    network_config: NetworkConfig = None
    
    # Simulation parameters
    num_rounds: int = 10
    interactions_per_round: int = 100
    intervention_round: Optional[int] = None  # Round to introduce intervention
    intervention_type: Optional[str] = None  # "fact_check", "diverse_exposure", "bridge_building"
    
    # Output parameters
    save_detailed_history: bool = True
    save_network_snapshots: bool = True
    output_directory: str = "experiment_results"
    
    # Randomization
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.network_config is None:
            self.network_config = NetworkConfig()


class ExperimentResults:
    """Container for experiment results and analysis"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.agents_history: List[List[Dict]] = []  # Agent states per round
        self.network_metrics_history: List[Dict] = []
        self.interaction_history: List[Dict] = []
        self.echo_chambers_history: List[List[List[int]]] = []
        self.bridge_agents_history: List[List[int]] = []
        
        # Summary statistics
        self.polarization_over_time: List[float] = []
        self.belief_variance_over_time: List[float] = []
        self.echo_chamber_count_over_time: List[int] = []
        self.network_fragmentation_over_time: List[float] = []
        
        # Final analysis
        self.final_echo_chambers: List[List[int]] = []
        self.most_influential_agents: List[int] = []
        self.most_polarized_agents: List[int] = []
        self.bridge_agents: List[int] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization"""
        return {
            'config': asdict(self.config),
            'summary_stats': {
                'polarization_over_time': self.polarization_over_time,
                'belief_variance_over_time': self.belief_variance_over_time,
                'echo_chamber_count_over_time': self.echo_chamber_count_over_time,
                'network_fragmentation_over_time': self.network_fragmentation_over_time
            },
            'final_analysis': {
                'final_echo_chambers': self.final_echo_chambers,
                'most_influential_agents': self.most_influential_agents,
                'most_polarized_agents': self.most_polarized_agents,
                'bridge_agents': self.bridge_agents
            },
            'detailed_history': {
                'agents_history': self.agents_history if self.config.save_detailed_history else [],
                'network_metrics_history': self.network_metrics_history,
                'interaction_history': self.interaction_history[:1000],  # Limit size
                'echo_chambers_history': self.echo_chambers_history,
                'bridge_agents_history': self.bridge_agents_history
            }
        }
    
    def save_to_file(self, filepath: str):
        """Save results to JSON file"""
        data = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert agent history to pandas DataFrame for analysis"""
        if not self.agents_history:
            return pd.DataFrame()
        
        rows = []
        for round_num, agent_states in enumerate(self.agents_history):
            for agent_state in agent_states:
                row = agent_state.copy()
                row['round'] = round_num
                rows.append(row)
        
        return pd.DataFrame(rows)


class EchoChamberExperiment:
    """Main experiment class for running echo chamber simulations"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = ExperimentResults(config)
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        # Initialize agents and network
        self.agents = create_diverse_agent_population(
            config.num_agents, 
            config.topic, 
            config.belief_distribution
        )
        
        self.network = SocialNetwork(self.agents, config.network_config)
        self.round_number = 0
        
        # Track initial state
        self._record_round_state()
    
    def run_full_experiment(self) -> ExperimentResults:
        """Run the complete experiment"""
        print(f"🚀 Starting experiment: {self.config.name}")
        print(f"   {self.config.num_agents} agents, {self.config.num_rounds} rounds")
        print(f"   Topic: {self.config.topic.value}")
        print(f"   Network: {self.config.network_config.network_type}")
        
        start_time = time.time()
        
        for round_num in range(1, self.config.num_rounds + 1):
            self.round_number = round_num
            print(f"\n🔄 Round {round_num}/{self.config.num_rounds}")
            
            # Check for intervention
            if round_num == self.config.intervention_round and self.config.intervention_type:
                self._apply_intervention()
            
            # Run interaction round
            self._run_interaction_round()
            
            # Update network dynamics
            self.network.update_network_dynamics(round_num)
            
            # Record state
            self._record_round_state()
            
            # Print progress
            current_polarization = self.results.polarization_over_time[-1]
            echo_chambers = len(self.results.echo_chambers_history[-1])
            print(f"   Polarization: {current_polarization:.3f}, Echo chambers: {echo_chambers}")
        
        # Final analysis
        self._perform_final_analysis()
        
        duration = time.time() - start_time
        print(f"\n✅ Experiment completed in {duration:.2f} seconds")
        
        return self.results
    
    def _run_interaction_round(self):
        """Run one round of agent interactions"""
        interaction_count = 0
        
        for _ in range(self.config.interactions_per_round):
            # Select random active agent
            active_agent = random.choice(self.agents)
            
            # Get connected agents
            connected_agents = self.network.get_connected_agents(active_agent.id)
            
            if not connected_agents:
                continue
            
            # Select target based on agent's sociability and recent activity
            if random.random() < active_agent.sociability:
                # More social agents interact more frequently
                target_agent = random.choice(connected_agents)
                
                # Generate message
                message = active_agent.generate_message()
                active_agent.messages_sent.append(f"[{self.round_number}] {message}")
                
                # Process influence
                influence_magnitude = target_agent.receive_influence(
                    active_agent, message, self.round_number
                )
                
                # Record interaction
                interaction_record = {
                    'round': self.round_number,
                    'sender_id': active_agent.id,
                    'receiver_id': target_agent.id,
                    'message': message,
                    'influence_magnitude': influence_magnitude,
                    'sender_belief': active_agent.belief_strength,
                    'receiver_belief_before': target_agent.belief_history[-2] if len(target_agent.belief_history) > 1 else target_agent.belief_history[-1],
                    'receiver_belief_after': target_agent.belief_strength
                }
                self.results.interaction_history.append(interaction_record)
                interaction_count += 1
    
    def _apply_intervention(self):
        """Apply experimental intervention"""
        print(f"🛠️  Applying intervention: {self.config.intervention_type}")
        
        if self.config.intervention_type == "fact_check":
            self._apply_fact_checking_intervention()
        elif self.config.intervention_type == "diverse_exposure":
            self._apply_diverse_exposure_intervention()
        elif self.config.intervention_type == "bridge_building":
            self._apply_bridge_building_intervention()
    
    def _apply_fact_checking_intervention(self):
        """Introduce fact-checking to reduce extreme beliefs"""
        for agent in self.agents:
            if abs(agent.belief_strength) > 0.8:  # Extreme agents
                # Moderate beliefs slightly toward center
                moderation_factor = 0.1 * (1 - agent.confirmation_bias)
                if agent.belief_strength > 0:
                    agent.belief_strength = max(0.1, agent.belief_strength - moderation_factor)
                else:
                    agent.belief_strength = min(-0.1, agent.belief_strength + moderation_factor)
                
                agent.belief_history.append(agent.belief_strength)
    
    def _apply_diverse_exposure_intervention(self):
        """Expose agents to diverse viewpoints"""
        for agent in self.agents:
            # Find agents with opposite beliefs
            opposite_agents = [a for a in self.agents 
                             if a.id != agent.id and 
                             (a.belief_strength * agent.belief_strength) < 0]  # Opposite signs
            
            if opposite_agents:
                # Select diverse agent
                diverse_agent = random.choice(opposite_agents)
                message = diverse_agent.generate_message()
                
                # Process with reduced resistance due to intervention
                original_openness = agent.openness
                agent.openness = min(1.0, agent.openness * 1.5)  # Temporarily increase openness
                
                agent.receive_influence(diverse_agent, f"[INTERVENTION] {message}", self.round_number)
                
                agent.openness = original_openness  # Restore original openness
    
    def _apply_bridge_building_intervention(self):
        """Strengthen connections between different belief groups"""
        # Identify agents from different belief groups
        pro_agents = [a for a in self.agents if a.belief_strength > 0.2]
        anti_agents = [a for a in self.agents if a.belief_strength < -0.2]
        
        # Create new cross-group connections
        num_bridges = min(5, len(pro_agents), len(anti_agents))
        for _ in range(num_bridges):
            if pro_agents and anti_agents:
                pro_agent = random.choice(pro_agents)
                anti_agent = random.choice(anti_agents)
                
                # Force connection if not already connected
                if anti_agent.id not in self.network.connections[pro_agent.id]:
                    self.network._add_connection(pro_agent.id, anti_agent.id)
    
    def _record_round_state(self):
        """Record current state of all agents and network"""
        
        # Record agent states
        if self.config.save_detailed_history:
            agent_states = []
            for agent in self.agents:
                state = {
                    'id': agent.id,
                    'name': agent.name,
                    'belief_strength': agent.belief_strength,
                    'openness': agent.openness,
                    'confidence': agent.confidence,
                    'sociability': agent.sociability,
                    'confirmation_bias': agent.confirmation_bias,
                    'personality_type': agent.personality_type.value,
                    'influence_power': agent.influence_power,
                    'network_centrality': agent.network_centrality,
                    'interaction_count': agent.interaction_count,
                    'polarization_score': agent.get_polarization_score(),
                    'num_connections': len(self.network.connections[agent.id])
                }
                agent_states.append(state)
            self.results.agents_history.append(agent_states)
        
        # Record network metrics
        network_stats = self.network.get_network_statistics()
        self.results.network_metrics_history.append(network_stats)
        
        # Record echo chambers
        echo_chambers = self.network.detect_echo_chambers()
        self.results.echo_chambers_history.append(echo_chambers)
        
        # Record bridge agents
        bridge_agents = self.network.get_bridge_agents()
        self.results.bridge_agents_history.append(bridge_agents)
        
        # Calculate summary statistics
        beliefs = [agent.belief_strength for agent in self.agents]
        
        # Polarization (average distance from center)
        polarization = np.mean([abs(belief) for belief in beliefs])
        self.results.polarization_over_time.append(polarization)
        
        # Belief variance
        belief_variance = np.var(beliefs)
        self.results.belief_variance_over_time.append(belief_variance)
        
        # Echo chamber count
        self.results.echo_chamber_count_over_time.append(len(echo_chambers))
        
        # Network fragmentation
        fragmentation = network_stats['num_components'] / len(self.agents)
        self.results.network_fragmentation_over_time.append(fragmentation)
    
    def _perform_final_analysis(self):
        """Perform final analysis of experiment results"""
        
        # Final echo chambers
        self.results.final_echo_chambers = self.network.detect_echo_chambers()
        
        # Most influential agents (based on total influence given)
        influence_scores = {}
        for agent in self.agents:
            network_position = agent.get_influence_network_position(self.agents)
            influence_scores[agent.id] = network_position['total_influence_given']
        
        sorted_by_influence = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
        self.results.most_influential_agents = [agent_id for agent_id, _ in sorted_by_influence[:5]]
        
        # Most polarized agents (largest change in belief extremeness)
        polarization_scores = [(agent.id, agent.get_polarization_score()) for agent in self.agents]
        polarization_scores.sort(key=lambda x: x[1], reverse=True)
        self.results.most_polarized_agents = [agent_id for agent_id, _ in polarization_scores[:5]]
        
        # Bridge agents
        self.results.bridge_agents = self.network.get_bridge_agents()
        
        print(f"\n📊 Final Analysis:")
        print(f"   Echo chambers formed: {len(self.results.final_echo_chambers)}")
        print(f"   Most influential agents: {self.results.most_influential_agents}")
        print(f"   Bridge agents: {len(self.results.bridge_agents)}")
        print(f"   Final polarization: {self.results.polarization_over_time[-1]:.3f}")


def run_predefined_experiment(experiment_name: str, **kwargs) -> ExperimentResults:
    """Run a predefined experiment configuration"""
    
    configs = {
        "basic_polarization": ExperimentConfig(
            name="Basic Polarization",
            description="Simple echo chamber formation with polarized initial beliefs",
            num_agents=50,
            topic=TopicType.GUN_CONTROL,
            belief_distribution="polarized",
            network_config=NetworkConfig(network_type="preferential_attachment", homophily_strength=0.8),
            num_rounds=15,
            interactions_per_round=200
        ),
        
        "intervention_study": ExperimentConfig(
            name="Intervention Study",
            description="Test effect of fact-checking intervention on polarization",
            num_agents=60,
            topic=TopicType.CLIMATE_CHANGE,
            belief_distribution="polarized",
            network_config=NetworkConfig(network_type="small_world", homophily_strength=0.7),
            num_rounds=20,
            interactions_per_round=250,
            intervention_round=10,
            intervention_type="fact_check"
        ),
        
        "bridge_building": ExperimentConfig(
            name="Bridge Building",
            description="Test effect of connecting opposing groups",
            num_agents=40,
            topic=TopicType.HEALTHCARE,
            belief_distribution="polarized",
            network_config=NetworkConfig(network_type="scale_free", homophily_strength=0.9),
            num_rounds=25,
            interactions_per_round=180,
            intervention_round=12,
            intervention_type="bridge_building"
        ),
        
        "large_scale": ExperimentConfig(
            name="Large Scale Dynamics",
            description="Large population dynamics over extended time",
            num_agents=200,
            topic=TopicType.IMMIGRATION,
            belief_distribution="normal",
            network_config=NetworkConfig(network_type="preferential_attachment", homophily_strength=0.6),
            num_rounds=30,
            interactions_per_round=500
        )
    }
    
    if experiment_name not in configs:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(configs.keys())}")
    
    config = configs[experiment_name]
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    experiment = EchoChamberExperiment(config)
    return experiment.run_full_experiment()