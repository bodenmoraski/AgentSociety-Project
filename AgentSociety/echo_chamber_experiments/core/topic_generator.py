"""
Dynamic Topic System Generator

This module provides an AI-powered topic generator that can dynamically construct
belief systems from string inputs (e.g., "climate change", "vaccination", "AI regulation").
Instead of relying on predefined topic sets, this generator uses AI to create
topic-specific content while maintaining the same structure as the existing hardcoded setup.
"""

import json
import random
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# For AI integration - we'll use a simple template-based approach that can be extended
# with actual AI APIs later
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class TopicComplexity(Enum):
    """Complexity levels for topic generation"""
    SIMPLE = "simple"      # Basic binary positions
    MODERATE = "moderate"  # Multiple nuanced positions
    COMPLEX = "complex"    # Multi-dimensional belief space


@dataclass
class TopicBeliefSystem:
    """Represents a complete belief system for a topic"""
    
    topic_name: str
    topic_description: str
    complexity: TopicComplexity
    
    # Core belief dimensions
    belief_dimensions: List[str] = field(default_factory=list)
    dimension_descriptions: Dict[str, str] = field(default_factory=dict)
    
    # Message templates for each position
    positive_messages: List[str] = field(default_factory=list)
    negative_messages: List[str] = field(default_factory=list)
    neutral_messages: List[str] = field(default_factory=list)
    
    # Additional context
    key_terms: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    controversy_level: float = 0.5  # 0-1 scale
    
    # Generated metadata
    generation_timestamp: Optional[str] = None
    generation_method: str = "template"


class TopicGenerator:
    """AI-powered topic system generator"""
    
    def __init__(self, use_ai: bool = False, api_key: Optional[str] = None):
        """
        Initialize the topic generator
        
        Args:
            use_ai: Whether to use actual AI APIs (requires OpenAI API key)
            api_key: OpenAI API key for AI-powered generation
        """
        self.use_ai = use_ai and OPENAI_AVAILABLE
        self.api_key = api_key
        
        if self.use_ai and api_key:
            openai.api_key = api_key
        
        # Template-based fallback system
        self.topic_templates = self._load_topic_templates()
    
    def _load_topic_templates(self) -> Dict[str, Dict]:
        """Load predefined templates for common topics"""
        return {
            "climate_change": {
                "description": "Environmental policy and climate action",
                "dimensions": ["scientific_consensus", "economic_impact", "policy_urgency"],
                "positive_messages": [
                    "Climate science shows urgent action is needed to prevent catastrophe",
                    "Renewable energy creates jobs while protecting our planet",
                    "We have a moral obligation to future generations",
                    "The economic costs of inaction far exceed the costs of transition"
                ],
                "negative_messages": [
                    "Climate has always changed naturally throughout Earth's history",
                    "Economic disruption from rapid changes will hurt working families",
                    "Technology and innovation will solve problems better than regulation",
                    "Many climate predictions have been wrong in the past"
                ],
                "key_terms": ["greenhouse gases", "renewable energy", "carbon tax", "Paris Agreement"],
                "controversy_level": 0.8
            },
            "vaccination": {
                "description": "Public health and vaccine policy",
                "dimensions": ["safety_concerns", "public_health", "personal_choice"],
                "positive_messages": [
                    "Vaccines are one of the greatest public health achievements",
                    "Herd immunity protects vulnerable populations",
                    "Scientific evidence overwhelmingly supports vaccine safety",
                    "Vaccination prevents serious diseases and saves lives"
                ],
                "negative_messages": [
                    "Individuals should have the right to choose what goes in their body",
                    "Vaccine safety studies are often industry-funded",
                    "Natural immunity is more effective than artificial immunity",
                    "Mandatory vaccination infringes on personal freedoms"
                ],
                "key_terms": ["herd immunity", "autism", "side effects", "mandatory vaccination"],
                "controversy_level": 0.7
            },
            "ai_regulation": {
                "description": "Artificial intelligence governance and safety",
                "dimensions": ["safety_concerns", "economic_impact", "innovation_pace"],
                "positive_messages": [
                    "AI regulation is essential to prevent harmful applications",
                    "We need safeguards before AI becomes too powerful",
                    "Ethical AI development requires government oversight",
                    "AI safety should be prioritized over rapid deployment"
                ],
                "negative_messages": [
                    "Over-regulation will stifle AI innovation and progress",
                    "AI companies are best positioned to self-regulate",
                    "Regulation will give other countries competitive advantages",
                    "AI development is moving too fast for government oversight"
                ],
                "key_terms": ["AI safety", "algorithmic bias", "job displacement", "existential risk"],
                "controversy_level": 0.6
            },
            "gun_control": {
                "description": "Firearm regulation and Second Amendment rights",
                "dimensions": ["public_safety", "constitutional_rights", "crime_prevention"],
                "positive_messages": [
                    "We need comprehensive background checks for all gun purchases",
                    "Common-sense gun regulations save lives and protect communities",
                    "Other countries with strict gun laws have much lower violence rates",
                    "The right to safety should outweigh unrestricted gun access"
                ],
                "negative_messages": [
                    "The Second Amendment protects our fundamental right to bear arms",
                    "More gun laws won't stop criminals who ignore existing laws",
                    "Law-abiding citizens shouldn't be punished for others' crimes",
                    "Self-defense is a basic human right that requires access to firearms"
                ],
                "key_terms": ["Second Amendment", "background checks", "assault weapons", "concealed carry"],
                "controversy_level": 0.9
            },
            "healthcare": {
                "description": "Healthcare system and universal coverage",
                "dimensions": ["access_to_care", "cost_effectiveness", "quality_of_care"],
                "positive_messages": [
                    "Healthcare is a human right, not a privilege for the wealthy",
                    "Universal systems in other countries provide better outcomes for less cost",
                    "Medical bankruptcy shouldn't exist in a civilized society",
                    "Preventive care saves money and lives in the long run"
                ],
                "negative_messages": [
                    "Government-run healthcare reduces quality and increases wait times",
                    "Competition and choice drive innovation in medical care",
                    "People should have control over their own healthcare decisions",
                    "Free market solutions are more efficient than bureaucratic systems"
                ],
                "key_terms": ["universal healthcare", "single-payer", "medical bankruptcy", "premiums"],
                "controversy_level": 0.8
            }
        }
    
    def generate_topic_system(self, topic_input: str, complexity: TopicComplexity = TopicComplexity.MODERATE) -> TopicBeliefSystem:
        """
        Generate a complete belief system for a given topic
        
        Args:
            topic_input: String input (e.g., "climate change", "vaccination")
            complexity: Desired complexity level
            
        Returns:
            TopicBeliefSystem with complete belief structure
        """
        # Normalize topic input
        normalized_topic = self._normalize_topic(topic_input)
        
        if self.use_ai:
            return self._generate_with_ai(normalized_topic, complexity)
        else:
            return self._generate_with_templates(normalized_topic, complexity)
    
    def _normalize_topic(self, topic_input: str) -> str:
        """Normalize topic input for consistent processing"""
        # Convert to lowercase and remove extra whitespace
        normalized = topic_input.lower().strip()
        
        # Replace spaces with underscores for template matching
        normalized = re.sub(r'\s+', '_', normalized)
        
        return normalized
    
    def _generate_with_templates(self, topic: str, complexity: TopicComplexity) -> TopicBeliefSystem:
        """Generate topic system using predefined templates"""
        
        # Check if we have a template for this topic
        if topic in self.topic_templates:
            template = self.topic_templates[topic]
            
            # Create belief system from template
            belief_system = TopicBeliefSystem(
                topic_name=topic.replace('_', ' ').title(),
                topic_description=template["description"],
                complexity=complexity,
                belief_dimensions=template["dimensions"],
                positive_messages=template["positive_messages"],
                negative_messages=template["negative_messages"],
                key_terms=template["key_terms"],
                controversy_level=template["controversy_level"],
                generation_method="template"
            )
            
            # Add dimension descriptions
            for dim in template["dimensions"]:
                belief_system.dimension_descriptions[dim] = f"Beliefs about {dim.replace('_', ' ')}"
            
            return belief_system
        
        # Generate a generic template for unknown topics
        return self._generate_generic_topic(topic, complexity)
    
    def _generate_generic_topic(self, topic: str, complexity: TopicComplexity) -> TopicBeliefSystem:
        """Generate a generic belief system for unknown topics"""
        
        topic_name = topic.replace('_', ' ').title()
        
        # Generic dimensions based on complexity
        if complexity == TopicComplexity.SIMPLE:
            dimensions = ["support", "opposition"]
        elif complexity == TopicComplexity.MODERATE:
            dimensions = ["policy_support", "economic_impact", "social_effects"]
        else:  # COMPLEX
            dimensions = ["policy_support", "economic_impact", "social_effects", "moral_considerations", "long_term_effects"]
        
        # Generate generic messages
        positive_messages = [
            f"Strong evidence supports {topic_name} as beneficial for society",
            f"Research shows that {topic_name} creates positive outcomes",
            f"Experts agree that {topic_name} should be supported",
            f"The benefits of {topic_name} outweigh any potential drawbacks"
        ]
        
        negative_messages = [
            f"There are serious concerns about the impact of {topic_name}",
            f"Evidence suggests that {topic_name} may cause more harm than good",
            f"Many experts oppose {topic_name} for valid reasons",
            f"The risks of {topic_name} are too great to ignore"
        ]
        
        neutral_messages = [
            f"The debate about {topic_name} involves complex trade-offs",
            f"Different perspectives on {topic_name} each have merit",
            f"The issue of {topic_name} requires careful consideration",
            f"Both sides of the {topic_name} debate raise important points"
        ]
        
        belief_system = TopicBeliefSystem(
            topic_name=topic_name,
            topic_description=f"Policy and social debate surrounding {topic_name}",
            complexity=complexity,
            belief_dimensions=dimensions,
            positive_messages=positive_messages,
            negative_messages=negative_messages,
            neutral_messages=neutral_messages,
            key_terms=[topic_name.lower()],
            controversy_level=0.6,
            generation_method="generic_template"
        )
        
        # Add dimension descriptions
        for dim in dimensions:
            belief_system.dimension_descriptions[dim] = f"Beliefs about {dim.replace('_', ' ')}"
        
        return belief_system
    
    def _generate_with_ai(self, topic: str, complexity: TopicComplexity) -> TopicBeliefSystem:
        """Generate topic system using AI (placeholder for OpenAI integration)"""
        
        # This is a placeholder for actual AI integration
        # In a real implementation, you would:
        # 1. Send a prompt to OpenAI API
        # 2. Parse the response to extract belief dimensions, messages, etc.
        # 3. Structure the response into a TopicBeliefSystem
        
        prompt = f"""
        Generate a belief system for the topic: {topic}
        
        Please provide:
        1. A brief description of the topic
        2. 3-5 key belief dimensions
        3. 4-6 positive messages (supporting the topic)
        4. 4-6 negative messages (opposing the topic)
        5. Key terms and concepts
        6. Controversy level (0-1)
        
        Format as JSON.
        """
        
        # For now, fall back to template-based generation
        return self._generate_with_templates(topic, complexity)
    
    def get_topic_enum_value(self, topic_input: str) -> str:
        """Convert topic input to enum-compatible string"""
        normalized = self._normalize_topic(topic_input)
        return normalized.upper()
    
    def create_topic_enum_class(self, topics: List[str]) -> type:
        """Dynamically create a TopicType enum class from a list of topics"""
        
        # Create enum values
        enum_values = {}
        for topic in topics:
            enum_key = self.get_topic_enum_value(topic)
            enum_values[enum_key] = topic.lower()
        
        # Create the enum class
        TopicType = Enum('TopicType', enum_values)
        return TopicType


# Utility functions for easy integration
def generate_topic_system(topic_input: str, complexity: TopicComplexity = TopicComplexity.MODERATE) -> TopicBeliefSystem:
    """Convenience function to generate a topic system"""
    generator = TopicGenerator()
    return generator.generate_topic_system(topic_input, complexity)


def create_topic_enum(topics: List[str]) -> type:
    """Convenience function to create a TopicType enum from topics"""
    generator = TopicGenerator()
    return generator.create_topic_enum_class(topics)


# Example usage and testing
if __name__ == "__main__":
    # Test the topic generator
    generator = TopicGenerator()
    
    # Test with known topic
    climate_system = generator.generate_topic_system("climate change")
    print(f"Generated topic system for: {climate_system.topic_name}")
    print(f"Positive messages: {len(climate_system.positive_messages)}")
    print(f"Negative messages: {len(climate_system.negative_messages)}")
    print(f"Belief dimensions: {climate_system.belief_dimensions}")
    
    # Test with unknown topic
    ai_system = generator.generate_topic_system("artificial intelligence regulation")
    print(f"\nGenerated topic system for: {ai_system.topic_name}")
    print(f"Positive messages: {len(ai_system.positive_messages)}")
    print(f"Negative messages: {len(ai_system.negative_messages)}")
    
    # Test enum creation
    topics = ["climate change", "vaccination", "gun control"]
    TopicType = create_topic_enum(topics)
    print(f"\nCreated enum with topics: {[t.value for t in TopicType]}") 