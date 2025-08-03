"""
Dynamic Agent with AI-Generated Topic Support

This module provides an enhanced agent class that can work with dynamically
generated topic systems while maintaining full compatibility with the existing
agent framework.
"""

import random
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum

from .agent import Agent, PersonalityType, create_diverse_agent_population
from .topic_generator import TopicGenerator, TopicBeliefSystem, TopicComplexity, generate_topic_system


class DynamicTopicType(Enum):
    """Dynamic topic type that can be created from string inputs"""
    # This will be populated dynamically
    pass


@dataclass
class DynamicAgent(Agent):
    """
    Enhanced agent that supports dynamic topic systems
    
    This class extends the base Agent class to work with AI-generated
    topic systems while maintaining full backward compatibility.
    """
    
    # Additional fields for dynamic topic support
    topic_system: Optional[TopicBeliefSystem] = None
    dynamic_topic_name: Optional[str] = None
    
    def __post_init__(self):
        """Initialize with dynamic topic support"""
        super().__post_init__()
        
        # If we have a topic system, use it for message generation
        if self.topic_system is not None:
            self.dynamic_topic_name = self.topic_system.topic_name
    
    def generate_message(self, context: Optional[str] = None) -> str:
        """Generate a message based on current beliefs and personality"""
        
        # If we have a dynamic topic system, use it
        if self.topic_system is not None:
            return self._generate_dynamic_message(context)
        else:
            # Fall back to original implementation
            return super().generate_message(context)
    
    def _generate_dynamic_message(self, context: Optional[str] = None) -> str:
        """Generate message using dynamic topic system"""
        
        # Choose positive or negative based on belief
        if self.belief_strength > 0:
            base_messages = self.topic_system.positive_messages
        else:
            base_messages = self.topic_system.negative_messages
        
        # If no messages available, use neutral ones or fallback
        if not base_messages:
            base_messages = self.topic_system.neutral_messages or [
                f"I have concerns about {self.topic_system.topic_name}",
                f"I support {self.topic_system.topic_name}",
                f"The issue of {self.topic_system.topic_name} is complex"
            ]
        
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
    
    def get_topic_context(self) -> Dict[str, Any]:
        """Get context information about the current topic"""
        if self.topic_system is None:
            return {"topic": "unknown", "dimensions": [], "controversy_level": 0.5}
        
        return {
            "topic": self.topic_system.topic_name,
            "description": self.topic_system.topic_description,
            "dimensions": self.topic_system.belief_dimensions,
            "key_terms": self.topic_system.key_terms,
            "controversy_level": self.topic_system.controversy_level,
            "complexity": self.topic_system.complexity.value
        }


def create_dynamic_agent_population(
    num_agents: int,
    topic_input: str,
    belief_distribution: str = "polarized",
    complexity: TopicComplexity = TopicComplexity.MODERATE,
    use_ai: bool = False,
    api_key: Optional[str] = None
) -> List[DynamicAgent]:
    """
    Create a diverse population of dynamic agents for a given topic
    
    Args:
        num_agents: Number of agents to create
        topic_input: String input for topic (e.g., "climate change")
        belief_distribution: Distribution type ("polarized", "normal", "uniform")
        complexity: Topic complexity level
        use_ai: Whether to use AI for topic generation
        api_key: OpenAI API key for AI generation
        
    Returns:
        List of DynamicAgent instances
    """
    
    # Generate topic system
    generator = TopicGenerator(use_ai=use_ai, api_key=api_key)
    topic_system = generator.generate_topic_system(topic_input, complexity)
    
    # Create dynamic topic enum
    DynamicTopicType = generator.create_topic_enum_class([topic_input])
    
    # Create agents using the existing function but with dynamic topic
    base_agents = create_diverse_agent_population(
        num_agents=num_agents,
        topic=DynamicTopicType(topic_input.lower()),
        belief_distribution=belief_distribution
    )
    
    # Convert to DynamicAgent instances
    dynamic_agents = []
    for agent in base_agents:
        dynamic_agent = DynamicAgent(
            id=agent.id,
            name=agent.name,
            belief_strength=agent.belief_strength,
            topic=agent.topic,
            openness=agent.openness,
            confidence=agent.confidence,
            sociability=agent.sociability,
            confirmation_bias=agent.confirmation_bias,
            personality_type=agent.personality_type,
            influence_power=agent.influence_power,
            network_centrality=agent.network_centrality,
            messages_sent=agent.messages_sent,
            messages_received=agent.messages_received,
            belief_history=agent.belief_history,
            interaction_count=agent.interaction_count,
            last_interaction_time=agent.last_interaction_time,
            recent_influences=agent.recent_influences,
            topic_system=topic_system,
            dynamic_topic_name=topic_system.topic_name
        )
        dynamic_agents.append(dynamic_agent)
    
    return dynamic_agents


def create_multi_topic_agent_population(
    num_agents: int,
    topics: List[str],
    agents_per_topic: Optional[List[int]] = None,
    belief_distribution: str = "polarized",
    complexity: TopicComplexity = TopicComplexity.MODERATE,
    use_ai: bool = False,
    api_key: Optional[str] = None
) -> List[DynamicAgent]:
    """
    Create agents for multiple topics
    
    Args:
        num_agents: Total number of agents
        topics: List of topic strings
        agents_per_topic: Optional list specifying agents per topic
        belief_distribution: Distribution type
        complexity: Topic complexity level
        use_ai: Whether to use AI for topic generation
        api_key: OpenAI API key for AI generation
        
    Returns:
        List of DynamicAgent instances across multiple topics
    """
    
    if agents_per_topic is None:
        # Distribute agents evenly across topics
        agents_per_topic = [num_agents // len(topics)] * len(topics)
        # Distribute remainder
        remainder = num_agents % len(topics)
        for i in range(remainder):
            agents_per_topic[i] += 1
    
    all_agents = []
    
    for i, topic in enumerate(topics):
        topic_agents = create_dynamic_agent_population(
            num_agents=agents_per_topic[i],
            topic_input=topic,
            belief_distribution=belief_distribution,
            complexity=complexity,
            use_ai=use_ai,
            api_key=api_key
        )
        all_agents.extend(topic_agents)
    
    return all_agents


# Enhanced experiment configuration for dynamic topics
@dataclass
class DynamicExperimentConfig:
    """Configuration for dynamic topic experiments"""
    
    # Basic parameters
    name: str = "dynamic_topic_experiment"
    description: str = "Experiment with dynamically generated topics"
    
    # Topic parameters
    topic_input: str = "climate change"
    topic_complexity: TopicComplexity = TopicComplexity.MODERATE
    
    # Population parameters
    num_agents: int = 50
    belief_distribution: str = "polarized"
    
    # AI parameters
    use_ai: bool = False
    api_key: Optional[str] = None
    
    # Simulation parameters
    num_rounds: int = 10
    interactions_per_round: int = 100
    
    # Output parameters
    save_detailed_history: bool = True
    output_directory: str = "dynamic_experiment_results"
    
    # Randomization
    random_seed: Optional[int] = None


# Utility functions for easy integration
def quick_dynamic_experiment(
    topic_input: str,
    num_agents: int = 30,
    num_rounds: int = 5,
    complexity: TopicComplexity = TopicComplexity.MODERATE
) -> Tuple[List[DynamicAgent], TopicBeliefSystem]:
    """
    Quick setup for a dynamic topic experiment
    
    Args:
        topic_input: Topic string (e.g., "vaccination", "AI regulation")
        num_agents: Number of agents
        num_rounds: Number of simulation rounds
        complexity: Topic complexity level
        
    Returns:
        Tuple of (agents, topic_system)
    """
    
    # Generate topic system
    topic_system = generate_topic_system(topic_input, complexity)
    
    # Create agents
    agents = create_dynamic_agent_population(
        num_agents=num_agents,
        topic_input=topic_input,
        complexity=complexity
    )
    
    return agents, topic_system


# Example usage and testing
if __name__ == "__main__":
    # Test dynamic agent creation
    print("Testing dynamic agent creation...")
    
    # Create agents for climate change
    agents = create_dynamic_agent_population(
        num_agents=10,
        topic_input="climate change",
        complexity=TopicComplexity.MODERATE
    )
    
    print(f"Created {len(agents)} agents for climate change")
    
    # Test message generation
    for i, agent in enumerate(agents[:3]):
        message = agent.generate_message()
        print(f"Agent {i}: {message}")
    
    # Test multi-topic creation
    print("\nTesting multi-topic creation...")
    multi_agents = create_multi_topic_agent_population(
        num_agents=15,
        topics=["vaccination", "AI regulation", "gun control"]
    )
    
    print(f"Created {len(multi_agents)} agents across 3 topics")
    
    # Test topic context
    for agent in multi_agents[:2]:
        context = agent.get_topic_context()
        print(f"Agent topic: {context['topic']}")
        print(f"Key terms: {context['key_terms']}") 