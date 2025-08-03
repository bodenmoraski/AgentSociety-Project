"""
Dynamic Topic System Demo

This demo showcases the new AI-powered topic system generator that can
dynamically construct belief systems from string inputs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.topic_generator import TopicGenerator, TopicComplexity, generate_topic_system
from core.dynamic_agent import create_dynamic_agent_population, quick_dynamic_experiment
from core.agent import TopicType, create_diverse_agent_population
import random


def demo_basic_topic_generation():
    """Demo basic topic generation functionality"""
    print("=" * 60)
    print("DEMO: Basic Topic Generation")
    print("=" * 60)
    
    generator = TopicGenerator()
    
    # Test with known topics
    topics = ["climate change", "vaccination", "AI regulation", "gun control"]
    
    for topic in topics:
        print(f"\n📋 Topic: {topic}")
        system = generator.generate_topic_system(topic)
        
        print(f"   Name: {system.topic_name}")
        print(f"   Description: {system.topic_description}")
        print(f"   Complexity: {system.complexity.value}")
        print(f"   Controversy Level: {system.controversy_level:.2f}")
        print(f"   Belief Dimensions: {system.belief_dimensions}")
        print(f"   Key Terms: {', '.join(system.key_terms)}")
        print(f"   Positive Messages: {len(system.positive_messages)}")
        print(f"   Negative Messages: {len(system.negative_messages)}")
        
        # Show sample messages
        if system.positive_messages:
            print(f"   Sample Positive: '{system.positive_messages[0]}'")
        if system.negative_messages:
            print(f"   Sample Negative: '{system.negative_messages[0]}'")


def demo_unknown_topic_generation():
    """Demo generation for unknown topics"""
    print("\n" + "=" * 60)
    print("DEMO: Unknown Topic Generation")
    print("=" * 60)
    
    generator = TopicGenerator()
    
    # Test with unknown topics
    unknown_topics = [
        "space exploration",
        "cryptocurrency regulation", 
        "universal basic income",
        "genetic engineering",
        "social media regulation"
    ]
    
    for topic in unknown_topics:
        print(f"\n🔍 Unknown Topic: {topic}")
        system = generator.generate_topic_system(topic, TopicComplexity.MODERATE)
        
        print(f"   Generated Name: {system.topic_name}")
        print(f"   Description: {system.topic_description}")
        print(f"   Dimensions: {system.belief_dimensions}")
        print(f"   Sample Positive: '{system.positive_messages[0]}'")
        print(f"   Sample Negative: '{system.negative_messages[0]}'")


def demo_complexity_levels():
    """Demo different complexity levels"""
    print("\n" + "=" * 60)
    print("DEMO: Complexity Levels")
    print("=" * 60)
    
    generator = TopicGenerator()
    topic = "artificial intelligence"
    
    for complexity in TopicComplexity:
        print(f"\n🧠 Complexity: {complexity.value.upper()}")
        system = generator.generate_topic_system(topic, complexity)
        
        print(f"   Dimensions: {len(system.belief_dimensions)}")
        print(f"   Dimension List: {system.belief_dimensions}")
        print(f"   Sample Positive: '{system.positive_messages[0]}'")
        print(f"   Sample Negative: '{system.negative_messages[0]}'")


def demo_dynamic_agents():
    """Demo dynamic agent creation and message generation"""
    print("\n" + "=" * 60)
    print("DEMO: Dynamic Agent Creation")
    print("=" * 60)
    
    # Create agents for a dynamic topic
    topic = "vaccination"
    agents = create_dynamic_agent_population(
        num_agents=5,
        topic_input=topic,
        belief_distribution="polarized"
    )
    
    print(f"Created {len(agents)} agents for topic: {topic}")
    
    # Show agent messages
    for i, agent in enumerate(agents):
        message = agent.generate_message()
        belief = agent.belief_strength
        personality = agent.personality_type.value
        
        print(f"\n🤖 Agent {i+1}:")
        print(f"   Belief: {belief:.2f}")
        print(f"   Personality: {personality}")
        print(f"   Message: '{message}'")
        
        # Show topic context
        context = agent.get_topic_context()
        print(f"   Topic: {context['topic']}")
        print(f"   Controversy: {context['controversy_level']:.2f}")


def demo_comparison_with_hardcoded():
    """Compare dynamic topic system with hardcoded system"""
    print("\n" + "=" * 60)
    print("DEMO: Comparison with Hardcoded System")
    print("=" * 60)
    
    # Create agents using hardcoded system
    print("\n📋 HARDCODED SYSTEM:")
    hardcoded_agents = create_diverse_agent_population(
        num_agents=3,
        topic=TopicType.CLIMATE_CHANGE,
        belief_distribution="polarized"
    )
    
    for i, agent in enumerate(hardcoded_agents):
        message = agent.generate_message()
        print(f"   Agent {i+1}: '{message}'")
    
    # Create agents using dynamic system
    print("\n🔧 DYNAMIC SYSTEM:")
    dynamic_agents = create_dynamic_agent_population(
        num_agents=3,
        topic_input="climate change",
        belief_distribution="polarized"
    )
    
    for i, agent in enumerate(dynamic_agents):
        message = agent.generate_message()
        print(f"   Agent {i+1}: '{message}'")
    
    # Show topic system details
    topic_system = dynamic_agents[0].topic_system
    print(f"\n📊 Dynamic Topic System Details:")
    print(f"   Name: {topic_system.topic_name}")
    print(f"   Description: {topic_system.topic_description}")
    print(f"   Dimensions: {topic_system.belief_dimensions}")
    print(f"   Key Terms: {', '.join(topic_system.key_terms)}")


def demo_multi_topic_experiment():
    """Demo multi-topic experiment setup"""
    print("\n" + "=" * 60)
    print("DEMO: Multi-Topic Experiment")
    print("=" * 60)
    
    topics = ["vaccination", "AI regulation", "climate change"]
    
    # Quick setup for each topic
    for topic in topics:
        print(f"\n🔬 Setting up experiment for: {topic}")
        agents, topic_system = quick_dynamic_experiment(
            topic_input=topic,
            num_agents=10,
            complexity=TopicComplexity.MODERATE
        )
        
        print(f"   Created {len(agents)} agents")
        print(f"   Topic: {topic_system.topic_name}")
        print(f"   Controversy Level: {topic_system.controversy_level:.2f}")
        print(f"   Dimensions: {len(topic_system.belief_dimensions)}")
        
        # Show sample messages from different agents
        positive_agent = next((a for a in agents if a.belief_strength > 0), agents[0])
        negative_agent = next((a for a in agents if a.belief_strength < 0), agents[-1])
        
        print(f"   Sample Positive: '{positive_agent.generate_message()}'")
        print(f"   Sample Negative: '{negative_agent.generate_message()}'")


def demo_ai_integration_preview():
    """Preview of AI integration capabilities"""
    print("\n" + "=" * 60)
    print("DEMO: AI Integration Preview")
    print("=" * 60)
    
    print("🤖 AI Integration Features:")
    print("   • OpenAI API integration for dynamic topic generation")
    print("   • Custom prompts for specific topic types")
    print("   • Real-time belief system generation")
    print("   • Context-aware message generation")
    print("   • Multi-language support")
    print("   • Controversy level prediction")
    print("   • Related topics discovery")
    
    print("\n📝 Example AI Prompt:")
    print("""
    Generate a belief system for the topic: [TOPIC]
    
    Please provide:
    1. A brief description of the topic
    2. 3-5 key belief dimensions
    3. 4-6 positive messages (supporting the topic)
    4. 4-6 negative messages (opposing the topic)
    5. Key terms and concepts
    6. Controversy level (0-1)
    7. Related topics
    
    Format as JSON.
    """)
    
    print("💡 To enable AI integration:")
    print("   1. Install openai: pip install openai")
    print("   2. Set your API key")
    print("   3. Use use_ai=True in TopicGenerator()")


def main():
    """Run all demos"""
    print("🚀 Dynamic Topic System Generator Demo")
    print("=" * 60)
    
    # Run all demos
    demo_basic_topic_generation()
    demo_unknown_topic_generation()
    demo_complexity_levels()
    demo_dynamic_agents()
    demo_comparison_with_hardcoded()
    demo_multi_topic_experiment()
    demo_ai_integration_preview()
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print("\n🎯 Key Benefits:")
    print("   • Dynamic topic generation from string inputs")
    print("   • No need to hardcode new topics")
    print("   • Maintains same structure as existing system")
    print("   • Extensible with AI integration")
    print("   • Backward compatible with existing code")
    print("   • Support for multiple complexity levels")
    print("   • Multi-topic experiments")
    
    print("\n🔧 Usage Examples:")
    print("""
    # Basic usage
    agents = create_dynamic_agent_population(50, "vaccination")
    
    # With complexity control
    agents = create_dynamic_agent_population(50, "AI regulation", 
                                           complexity=TopicComplexity.COMPLEX)
    
    # Multi-topic setup
    agents = create_multi_topic_agent_population(100, 
                                                ["climate change", "vaccination"])
    
    # Quick experiment
    agents, topic_system = quick_dynamic_experiment("gun control", 30)
    """)


if __name__ == "__main__":
    main() 