"""
Core Agent Classes for Echo Chamber Experiments
"""

import random
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class PersonalityType(Enum):
    """Different personality types affecting social behavior"""
    CONFORMIST = "conformist"          # Easily influenced by majority
    CONTRARIAN = "contrarian"          # Tends to oppose popular opinions  
    INDEPENDENT = "independent"        # Less influenced by others
    AMPLIFIER = "amplifier"           # Spreads beliefs with high intensity


class TopicType(Enum):
    """Different controversial topics for experiments"""
    GUN_CONTROL = "gun_control"
    CLIMATE_CHANGE = "climate_change"
    HEALTHCARE = "healthcare"
    TAXATION = "taxation"
    IMMIGRATION = "immigration"
    TECHNOLOGY_REGULATION = "tech_regulation"


@dataclass
class Agent:
    """Advanced agent with personality, social traits, and learning"""
    
    # Identity
    id: int
    name: str
    
    # Core beliefs (-1 = strongly oppose, +1 = strongly support)
    belief_strength: float
    topic: TopicType
    
    # Personality traits (0-1 scales)
    openness: float              # How easily influenced
    confidence: float            # How strongly they express beliefs
    sociability: float          # How much they interact
    confirmation_bias: float    # Tendency to seek confirming info
    
    # Social traits
    personality_type: PersonalityType
    influence_power: float      # How persuasive they are
    network_centrality: float  # How connected they are
    
    # Dynamic state
    messages_sent: List[str] = field(default_factory=list)
    messages_received: List[str] = field(default_factory=list)
    belief_history: List[float] = field(default_factory=list)
    interaction_count: int = 0
    last_interaction_time: int = 0
    
    # Learning and adaptation
    recent_influences: List[Tuple[int, float]] = field(default_factory=list)  # (agent_id, influence_amount)
    
    def __post_init__(self):
        """Initialize dynamic tracking"""
        self.belief_history.append(self.belief_strength)
        
        # Adjust traits based on personality type
        if self.personality_type == PersonalityType.CONFORMIST:
            self.openness = min(1.0, self.openness * 1.5)
            self.confidence *= 0.8
        elif self.personality_type == PersonalityType.CONTRARIAN:
            self.confirmation_bias *= 0.5
            self.confidence = min(1.0, self.confidence * 1.3)
        elif self.personality_type == PersonalityType.INDEPENDENT:
            self.openness *= 0.6
            self.confirmation_bias *= 0.7
        elif self.personality_type == PersonalityType.AMPLIFIER:
            self.influence_power = min(1.0, self.influence_power * 1.4)
            self.sociability = min(1.0, self.sociability * 1.2)
    
    def generate_message(self, context: Optional[str] = None) -> str:
        """Generate a message based on current beliefs and personality"""
        
        # Base messages by topic and belief
        topic_messages = {
            TopicType.GUN_CONTROL: {
                'positive': [
                    "We need comprehensive background checks for all gun purchases",
                    "Common-sense gun regulations save lives and protect communities", 
                    "Other countries with strict gun laws have much lower violence rates",
                    "The right to safety should outweigh unrestricted gun access"
                ],
                'negative': [
                    "The Second Amendment protects our fundamental right to bear arms",
                    "More gun laws won't stop criminals who ignore existing laws",
                    "Law-abiding citizens shouldn't be punished for others' crimes",
                    "Self-defense is a basic human right that requires access to firearms"
                ]
            },
            TopicType.CLIMATE_CHANGE: {
                'positive': [
                    "Climate science shows urgent action is needed to prevent catastrophe",
                    "Renewable energy creates jobs while protecting our planet",
                    "We have a moral obligation to future generations",
                    "The economic costs of inaction far exceed the costs of transition"
                ],
                'negative': [
                    "Climate has always changed naturally throughout Earth's history",
                    "Economic disruption from rapid changes will hurt working families",
                    "Technology and innovation will solve problems better than regulation",
                    "Many climate predictions have been wrong in the past"
                ]
            },
            TopicType.HEALTHCARE: {
                'positive': [
                    "Healthcare is a human right, not a privilege for the wealthy",
                    "Universal systems in other countries provide better outcomes for less cost",
                    "Medical bankruptcy shouldn't exist in a civilized society",
                    "Preventive care saves money and lives in the long run"
                ],
                'negative': [
                    "Government-run healthcare reduces quality and increases wait times",
                    "Competition and choice drive innovation in medical care",
                    "People should have control over their own healthcare decisions",
                    "Free market solutions are more efficient than bureaucratic systems"
                ]
            }
        }
        
        # Select message set
        message_set = topic_messages.get(self.topic, topic_messages[TopicType.GUN_CONTROL])
        
        # Choose positive or negative based on belief
        if self.belief_strength > 0:
            base_messages = message_set['positive']
        else:
            base_messages = message_set['negative']
        
        base_message = random.choice(base_messages)
        
        # Add intensity markers based on belief strength and confidence
        intensity = abs(self.belief_strength) * self.confidence
        
        if intensity > 0.9:
            prefix = "I'm absolutely convinced that"
        elif intensity > 0.7:
            prefix = "I strongly believe that"
        elif intensity > 0.5:
            prefix = "I think that"
        else:
            prefix = "It seems to me that"
        
        # Personality-based modifications
        if self.personality_type == PersonalityType.AMPLIFIER:
            if intensity > 0.7:
                prefix = "EVERYONE needs to understand that"
            base_message = base_message.upper() if intensity > 0.8 else base_message
        elif self.personality_type == PersonalityType.CONTRARIAN:
            prefix = "Unlike what most people think,"
        elif self.personality_type == PersonalityType.CONFORMIST:
            prefix = "I agree with many others that"
        
        return f"{prefix} {base_message.lower()}"
    
    def calculate_influence_susceptibility(self, sender: 'Agent') -> float:
        """Calculate how susceptible this agent is to influence from sender"""
        
        # Base susceptibility from openness
        base_susceptibility = self.openness
        
        # Personality modifications
        if self.personality_type == PersonalityType.CONFORMIST:
            # More influenced by confident agents
            base_susceptibility *= (1.0 + sender.confidence * 0.5)
        elif self.personality_type == PersonalityType.CONTRARIAN:
            # Less influenced, especially by similar beliefs
            base_susceptibility *= 0.7
            if abs(self.belief_strength - sender.belief_strength) < 0.3:
                base_susceptibility *= 0.5  # Resist similar views
        elif self.personality_type == PersonalityType.INDEPENDENT:
            # Consistently less influenced
            base_susceptibility *= 0.6
        
        # Confirmation bias effect
        belief_difference = abs(self.belief_strength - sender.belief_strength)
        if belief_difference > 1.0:  # Very different beliefs
            bias_resistance = self.confirmation_bias * 0.8
            base_susceptibility *= (1.0 - bias_resistance)
        
        # Sender's influence power
        base_susceptibility *= sender.influence_power
        
        return max(0.0, min(1.0, base_susceptibility))
    
    def receive_influence(self, sender: 'Agent', message: str, time_step: int) -> float:
        """Process influence from another agent and update beliefs"""
        
        susceptibility = self.calculate_influence_susceptibility(sender)
        
        # Calculate influence direction and magnitude
        belief_difference = sender.belief_strength - self.belief_strength
        influence_direction = 1 if belief_difference > 0 else -1
        
        # Base influence strength
        base_influence = 0.1 * susceptibility * abs(sender.belief_strength)
        
        # Distance-based decay (closer beliefs have more influence)
        distance_factor = 1.0 - min(1.0, abs(belief_difference) / 2.0)
        base_influence *= distance_factor
        
        # Apply influence
        old_belief = self.belief_strength
        influence_amount = base_influence * influence_direction
        
        self.belief_strength += influence_amount
        self.belief_strength = max(-1.0, min(1.0, self.belief_strength))  # Clamp to bounds
        
        # Track the interaction
        self.messages_received.append(f"[{time_step}] {sender.name}: {message}")
        self.recent_influences.append((sender.id, influence_amount))
        self.interaction_count += 1
        self.last_interaction_time = time_step
        self.belief_history.append(self.belief_strength)
        
        # Keep recent influences limited
        if len(self.recent_influences) > 10:
            self.recent_influences = self.recent_influences[-10:]
        
        return abs(self.belief_strength - old_belief)  # Return magnitude of change
    
    def get_polarization_score(self) -> float:
        """Calculate how polarized this agent has become"""
        if len(self.belief_history) < 2:
            return 0.0
        
        initial_belief = self.belief_history[0]
        current_belief = self.belief_strength
        
        return abs(current_belief) - abs(initial_belief)
    
    def get_influence_network_position(self, all_agents: List['Agent']) -> Dict[str, float]:
        """Calculate this agent's position in the influence network"""
        
        # Count how many agents this one has influenced
        influenced_count = 0
        total_influence_given = 0.0
        
        for other_agent in all_agents:
            if other_agent.id != self.id:
                for agent_id, influence in other_agent.recent_influences:
                    if agent_id == self.id:
                        influenced_count += 1
                        total_influence_given += abs(influence)
        
        # Count how much this agent was influenced
        total_influence_received = sum(abs(inf) for _, inf in self.recent_influences)
        
        return {
            'influenced_others_count': influenced_count,
            'total_influence_given': total_influence_given,
            'total_influence_received': total_influence_received,
            'influence_ratio': total_influence_given / max(0.001, total_influence_received)
        }


def create_diverse_agent_population(
    num_agents: int,
    topic: TopicType,
    belief_distribution: str = "polarized"  # "polarized", "normal", "uniform"
) -> List[Agent]:
    """Create a diverse population of agents with realistic trait distributions"""
    
    agents = []
    
    for i in range(num_agents):
        # Generate belief based on distribution type
        if belief_distribution == "polarized":
            # Create two peaks at extremes
            if random.random() < 0.5:
                belief = random.uniform(0.4, 1.0)
            else:
                belief = random.uniform(-1.0, -0.4)
        elif belief_distribution == "normal":
            # Normal distribution around center
            belief = random.gauss(0, 0.4)
            belief = max(-1.0, min(1.0, belief))
        else:  # uniform
            belief = random.uniform(-1.0, 1.0)
        
        # Generate personality traits with realistic correlations
        openness = np.random.beta(2, 2)  # Tends toward middle values
        confidence = np.random.beta(2, 3)  # Slightly lower average
        sociability = np.random.beta(2, 2)
        confirmation_bias = np.random.beta(3, 2)  # Slightly higher average
        
        # Select personality type
        personality_weights = [0.3, 0.15, 0.4, 0.15]  # conformist, contrarian, independent, amplifier
        personality_type = random.choices(list(PersonalityType), weights=personality_weights)[0]
        
        # Generate influence and network traits
        influence_power = np.random.beta(2, 3)
        network_centrality = np.random.beta(2, 2)
        
        agent = Agent(
            id=i,
            name=f"Agent_{i:03d}",
            belief_strength=belief,
            topic=topic,
            openness=openness,
            confidence=confidence,
            sociability=sociability,
            confirmation_bias=confirmation_bias,
            personality_type=personality_type,
            influence_power=influence_power,
            network_centrality=network_centrality
        )
        
        agents.append(agent)
    
    return agents