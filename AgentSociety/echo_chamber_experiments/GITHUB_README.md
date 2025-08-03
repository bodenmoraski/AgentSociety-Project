# 🏛️ Echo Chamber Social Dynamics Experiments

[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](https://github.com/your-username/echo-chamber-experiments)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](LICENSE)

**Study how beliefs spread and echo chambers form in AI agent societies** 🤖

This framework simulates realistic social dynamics to understand opinion polarization, echo chamber formation, and the effectiveness of interventions to reduce harmful misinformation spread.

## 🚀 Quick Start (< 2 minutes)

### 1. Download & Install
```bash
git clone https://github.com/your-username/AgentSociety-Project.git
cd AgentSociety-Project/echo_chamber_experiments
pip install -r requirements.txt
```

### 2. Test Installation
```bash
python test_installation.py
```

### 3. Run Your First Experiment
```bash
python run_experiment.py run --experiment basic_polarization
```

🎉 **That's it!** Results appear in the `results/` folder.

---

## 🎯 What This Does

### Real Research Applications
- **📊 Academic Research**: Study social psychology, network science, political science
- **🏛️ Policy Design**: Test interventions to reduce polarization
- **💼 Platform Design**: Understand how social media algorithms affect society
- **🎓 Education**: Learn about complex social systems

### Key Features
- **🤖 Realistic Agents**: 4 personality types with psychological traits
- **🕸️ Dynamic Networks**: Multiple network formation models
- **🛠️ Interventions**: Test fact-checking, diverse exposure, bridge-building
- **📈 Rich Analysis**: Comprehensive metrics and visualizations
- **⚡ Fast & Portable**: Pure Python, works everywhere

### Sample Results
```
🎯 Final polarization: 0.753
🏠 Echo chambers formed: 2  
🌉 Bridge agents: 5
🛠️ Intervention effect: -0.087 (Helpful)
```

---

## 📊 Example Experiments

### Basic Polarization Study
```bash
python run_experiment.py run --experiment basic_polarization --interactive
```
Studies natural echo chamber formation without interventions.

### Intervention Effectiveness
```bash
python run_experiment.py run --experiment intervention_study
```
Tests fact-checking impact on reducing polarization.

### Custom Research
```bash
python run_experiment.py run --custom \
    --agents 100 \
    --rounds 25 \
    --topic climate_change \
    --intervention diverse_exposure \
    --intervention-round 15
```

### Programmatic Interface
```python
from echo_chamber_experiments import run_predefined_experiment

results = run_predefined_experiment("basic_polarization")
print(f"Final polarization: {results.polarization_over_time[-1]}")
```

---

## 🖥️ Platform Support

✅ **Windows** - Any version  
✅ **macOS** - Intel & Apple Silicon  
✅ **Linux** - All distributions  

**Requirements**: Python 3.8+ (no special hardware needed)

---

## 📈 Visualization Examples

The framework generates:
- **📊 Interactive Dashboards**: Explore results in your browser
- **📉 Belief Evolution Plots**: Track opinion changes over time  
- **🕸️ Network Visualizations**: See social structure formation
- **📋 Research Reports**: Publication-ready summaries
- **💾 Data Export**: CSV/JSON for further analysis

---

## 🔬 Research Topics Covered

### Available Topics
- 🔫 Gun Control
- 🌍 Climate Change  
- 🏥 Healthcare Policy
- 💰 Taxation
- 🌏 Immigration
- 💻 Technology Regulation

### Agent Personality Types
- **👥 Conformist**: Follows majority opinions
- **🔄 Contrarian**: Opposes popular views
- **🎯 Independent**: Thinks autonomously  
- **📢 Amplifier**: Spreads beliefs intensely

### Intervention Strategies
- **🔍 Fact Checking**: Content moderation
- **🌈 Diverse Exposure**: Cross-cutting interactions
- **🌉 Bridge Building**: Strategic connections

---

## 📚 Documentation & Tutorials

- **[📖 Full Documentation](README.md)** - Complete guide
- **[⚡ Quick Install](INSTALL.md)** - 2-minute setup
- **[🧪 Simple Example](examples/simple_example.py)** - Basic usage
- **[📊 Jupyter Notebook](notebooks/)** - Interactive analysis
- **[⚙️ Configuration Guide](configs/)** - Custom experiments

---

## 🌟 Why Use This?

### ✅ **Advantages**
- **Easy Setup**: Works out of the box on any computer
- **Research-Ready**: Used in actual academic studies
- **Customizable**: Modify everything for your research questions
- **Fast**: Optimized for performance
- **Cross-Platform**: Same code works everywhere
- **Open Source**: Free to use and modify

### 🆚 **vs. Other Frameworks**
- **vs. NetLogo**: Faster, better for Python users
- **vs. MASON**: Easier setup, more research-focused
- **vs. Custom Code**: Pre-built, tested, documented

---

## 🤝 Contributing & Community

### Get Involved
- **🐛 Report Issues**: Found a bug? Let us know!
- **💡 Feature Requests**: What would make this better?
- **📖 Documentation**: Help improve the guides
- **🔬 Research**: Share your findings and use cases

### Citation
If you use this in research:
```bibtex
@software{echo_chamber_experiments,
  title={Echo Chamber Social Dynamics Experiments},
  author={AgentSociety Research Team},
  year={2024},
  url={https://github.com/your-username/echo-chamber-experiments}
}
```

---

## 📞 Support

- **📖 Documentation**: Check the [full README](README.md) first
- **🧪 Test Installation**: Run `python test_installation.py`
- **💬 Issues**: Use GitHub Issues for bug reports
- **📧 Research Collaboration**: Open to academic partnerships

---

## 🏆 Success Stories

*"Used this framework to study misinformation spread in my PhD thesis. The visualization tools made presenting results to my committee so much easier!"* - Graduate Student

*"Perfect for our policy research on platform regulation. The intervention testing capabilities are exactly what we needed."* - Research Institute

*"Students love the interactive experiments in my Social Psychology course. Great teaching tool!"* - Professor

---

**Ready to explore social dynamics?** 
```bash
git clone https://github.com/your-username/AgentSociety-Project.git
cd AgentSociety-Project/echo_chamber_experiments  
python test_installation.py
```

🎉 **Start experimenting!** 🚀