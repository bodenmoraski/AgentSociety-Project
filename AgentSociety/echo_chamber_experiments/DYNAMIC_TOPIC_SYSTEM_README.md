# 🌟 Dynamic Topic System Generator

A powerful AI-powered topic system generator that can dynamically construct belief systems from string inputs (e.g., "climate change", "vaccination", "AI regulation"). Instead of relying on predefined topic sets, this generator uses AI to create topic-specific content while maintaining the same structure as the existing hardcoded setup.

## 🎯 Key Features

### ✨ Dynamic Topic Generation
- **String Input**: Generate complete belief systems from simple string inputs
- **AI-Powered**: Uses OpenAI API for intelligent topic analysis (optional)
- **Template Fallback**: Comprehensive template system for common topics
- **Generic Generation**: Handles unknown topics with intelligent defaults

### 🔧 Flexible Complexity Levels
- **Simple**: Basic binary positions (support/opposition)
- **Moderate**: Multiple nuanced positions (policy, economic, social)
- **Complex**: Multi-dimensional belief space (5+ dimensions)

### 🎭 Rich Content Generation
- **Belief Dimensions**: Dynamic generation of topic-specific belief axes
- **Message Templates**: Contextual positive, negative, and neutral messages
- **Key Terms**: Topic-specific vocabulary and concepts
- **Controversy Levels**: Automatic controversy assessment

### 🔄 Seamless Integration
- **Backward Compatible**: Works with existing agent and experiment framework
- **Drop-in Replacement**: Can replace hardcoded topic systems
- **Multi-Topic Support**: Handle multiple topics in single experiments
- **Extensible**: Easy to add new templates or AI integration

## 🚀 Quick Start

### Basic Usage

```python
from core.topic_generator import generate_topic_system, TopicComplexity
from core.dynamic_agent import create_dynamic_agent_population

# Generate a topic system
topic_system = generate_topic_system("vaccination", TopicComplexity.MODERATE)

# Create agents with the dynamic topic
agents = create_dynamic_agent_population(
    num_agents=50,
    topic_input="vaccination",
    complexity=TopicComplexity.MODERATE
)

# Agents now generate contextual messages
for agent in agents[:3]:
    print(agent.generate_message())
```

### Advanced Usage

```python
from core.topic_generator import TopicGenerator
from core.dynamic_agent import create_multi_topic_agent_population

# Create generator with AI support
generator = TopicGenerator(use_ai=True, api_key="your-openai-key")

# Generate complex topic system
topic_system = generator.generate_topic_system(
    "artificial intelligence regulation",
    complexity=TopicComplexity.COMPLEX
)

# Multi-topic experiment
agents = create_multi_topic_agent_population(
    num_agents=100,
    topics=["climate change", "vaccination", "AI regulation"],
    complexity=TopicComplexity.MODERATE
)
```

## 📋 Supported Topics

### Pre-built Templates
- **Climate Change**: Environmental policy and climate action
- **Vaccination**: Public health and vaccine policy
- **AI Regulation**: Artificial intelligence governance
- **Gun Control**: Firearm regulation and rights
- **Healthcare**: Universal healthcare systems

### Dynamic Generation
Any topic can be generated dynamically:
- "space exploration"
- "cryptocurrency regulation"
- "universal basic income"
- "genetic engineering"
- "social media regulation"

## 🧠 Complexity Levels

### Simple (2 dimensions)
```python
dimensions = ["support", "opposition"]
```

### Moderate (3 dimensions)
```python
dimensions = ["policy_support", "economic_impact", "social_effects"]
```

### Complex (5+ dimensions)
```python
dimensions = ["policy_support", "economic_impact", "social_effects", 
              "moral_considerations", "long_term_effects"]
```

## 🤖 AI Integration

### OpenAI Integration
```python
from core.topic_generator import TopicGenerator

# Initialize with AI support
generator = TopicGenerator(use_ai=True, api_key="your-openai-key")

# Generate topic system using AI
topic_system = generator.generate_topic_system("quantum computing")
```

### AI Features
- **Intelligent Analysis**: AI analyzes topic context and controversy
- **Dynamic Dimensions**: AI generates topic-specific belief dimensions
- **Contextual Messages**: AI creates relevant positive/negative messages
- **Controversy Prediction**: AI assesses topic controversy level
- **Related Topics**: AI identifies related topics and concepts

### AI Prompt Example
```
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
```

## 🔧 Integration with Existing Framework

### Replace Hardcoded Topics
```python
# Before (hardcoded)
from core.agent import TopicType
agents = create_diverse_agent_population(50, TopicType.CLIMATE_CHANGE)

# After (dynamic)
from core.dynamic_agent import create_dynamic_agent_population
agents = create_dynamic_agent_population(50, "climate change")
```

### Use in Experiments
```python
from core.experiment import EchoChamberExperiment, ExperimentConfig
from core.dynamic_agent import create_dynamic_agent_population

# Create agents with dynamic topic
agents = create_dynamic_agent_population(50, "vaccination")

# Use in existing experiment framework
experiment = EchoChamberExperiment(config)
experiment.agents = agents
results = experiment.run_full_experiment()
```

## 📊 Demo and Examples

### Run the Demo
```bash
cd AgentSociety/echo_chamber_experiments
python examples/dynamic_topic_demo.py
```

### Run Experiments
```bash
python run_dynamic_topic_experiment.py
```

### Demo Features
- **Basic Topic Generation**: Test with known and unknown topics
- **Complexity Levels**: Compare different complexity settings
- **Dynamic Agents**: See agents generate contextual messages
- **Multi-Topic Setup**: Handle multiple topics simultaneously
- **AI Integration Preview**: See AI capabilities

## 🏗️ Architecture

### Core Components

#### TopicGenerator
- **Purpose**: Main interface for topic system generation
- **Features**: Template-based and AI-powered generation
- **Methods**: `generate_topic_system()`, `create_topic_enum_class()`

#### TopicBeliefSystem
- **Purpose**: Container for complete topic belief structure
- **Fields**: Dimensions, messages, key terms, controversy level
- **Metadata**: Generation method, timestamp, complexity

#### DynamicAgent
- **Purpose**: Enhanced agent with dynamic topic support
- **Features**: Contextual message generation, topic awareness
- **Compatibility**: Full backward compatibility with base Agent

### File Structure
```
core/
├── topic_generator.py      # Main topic generation logic
├── dynamic_agent.py        # Enhanced agent with dynamic topics
└── agent.py               # Original agent (unchanged)

examples/
└── dynamic_topic_demo.py  # Comprehensive demo

run_dynamic_topic_experiment.py  # Experiment runner
```

## 🔄 Migration Guide

### From Hardcoded to Dynamic

1. **Replace TopicType imports**:
```python
# Before
from core.agent import TopicType
topic = TopicType.CLIMATE_CHANGE

# After
from core.dynamic_agent import create_dynamic_agent_population
topic_input = "climate change"
```

2. **Update agent creation**:
```python
# Before
agents = create_diverse_agent_population(50, TopicType.CLIMATE_CHANGE)

# After
agents = create_dynamic_agent_population(50, "climate change")
```

3. **Add topic system access**:
```python
# Get topic context
context = agent.get_topic_context()
print(f"Topic: {context['topic']}")
print(f"Controversy: {context['controversy_level']}")
```

## 🎯 Benefits

### For Researchers
- **Flexibility**: Test any topic without hardcoding
- **Scalability**: Handle multiple topics simultaneously
- **Reproducibility**: Consistent topic generation
- **Extensibility**: Easy to add new topics or AI features

### For Developers
- **Backward Compatibility**: No breaking changes to existing code
- **Modular Design**: Clean separation of concerns
- **Extensible**: Easy to add new templates or AI providers
- **Well-Documented**: Comprehensive examples and documentation

### For Users
- **Simple Interface**: Just provide a topic string
- **Rich Content**: Contextual messages and dimensions
- **AI Enhancement**: Optional AI-powered generation
- **Multiple Options**: Different complexity levels

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Generate topics in different languages
- **Real-time Updates**: Dynamic topic evolution during experiments
- **Custom AI Providers**: Support for different AI services
- **Topic Clustering**: Automatic grouping of related topics
- **Controversy Prediction**: ML-based controversy assessment

### Extensibility Points
- **Custom Templates**: Add domain-specific templates
- **AI Providers**: Integrate with different AI services
- **Message Styles**: Customize message generation styles
- **Dimension Types**: Add new belief dimension types

## 📝 API Reference

### TopicGenerator
```python
class TopicGenerator:
    def __init__(self, use_ai: bool = False, api_key: Optional[str] = None)
    def generate_topic_system(self, topic_input: str, complexity: TopicComplexity) -> TopicBeliefSystem
    def create_topic_enum_class(self, topics: List[str]) -> type
```

### TopicBeliefSystem
```python
@dataclass
class TopicBeliefSystem:
    topic_name: str
    topic_description: str
    complexity: TopicComplexity
    belief_dimensions: List[str]
    positive_messages: List[str]
    negative_messages: List[str]
    key_terms: List[str]
    controversy_level: float
```

### DynamicAgent
```python
@dataclass
class DynamicAgent(Agent):
    topic_system: Optional[TopicBeliefSystem] = None
    dynamic_topic_name: Optional[str] = None
    
    def generate_message(self, context: Optional[str] = None) -> str
    def get_topic_context(self) -> Dict[str, Any]
```

## 🤝 Contributing

### Adding New Templates
1. Add topic to `_load_topic_templates()` in `TopicGenerator`
2. Include description, dimensions, messages, and key terms
3. Set appropriate controversy level

### Adding AI Providers
1. Implement AI interface in `_generate_with_ai()`
2. Add provider-specific prompt templates
3. Parse AI responses into `TopicBeliefSystem` format

### Testing
```bash
# Run demo
python examples/dynamic_topic_demo.py

# Run experiments
python run_dynamic_topic_experiment.py

# Test specific topic
python -c "from core.topic_generator import generate_topic_system; print(generate_topic_system('test topic'))"
```

## 📄 License

This project is part of the AgentSociety framework and follows the same licensing terms.

---

**🎉 The Dynamic Topic System Generator transforms the way you work with topics in agent-based simulations, providing unprecedented flexibility and power while maintaining full compatibility with existing code!** 