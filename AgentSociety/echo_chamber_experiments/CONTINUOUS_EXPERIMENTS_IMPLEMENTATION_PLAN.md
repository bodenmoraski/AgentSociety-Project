# 📋 Continuous Belief Experiments: Comprehensive Implementation Plan

Based on thorough codebase analysis, this plan outlines implementing **three high-impact experiments** that leverage the new continuous belief system while following established patterns and ensuring cross-platform compatibility.

---

## 🎯 **Selected Experiments for Implementation**

After analyzing the codebase architecture, I recommend focusing on these three experiments for maximum impact:

### **1. A1: Gradual vs Sudden Polarization Dynamics** ⭐⭐⭐
- **Impact**: High - Fundamental to understanding polarization processes
- **Complexity**: Low - Uses existing infrastructure 
- **Implementation Time**: 2-3 days
- **Mathematical Interest**: ★★★★☆ (Time-series analysis, change point detection)

### **2. A2: Asymmetric Bias Effects on Social Dynamics** ⭐⭐⭐
- **Impact**: High - Critical for real-world applications
- **Complexity**: Low - Leverages new asymmetry parameter
- **Implementation Time**: 1-2 days  
- **Mathematical Interest**: ★★★★☆ (Network asymmetry analysis, influence flow)

### **3. B1: Dynamic Belief Evolution** ⭐⭐⭐⭐
- **Impact**: Very High - Models real-world belief change during crises
- **Complexity**: Medium - Requires time-varying parameter system
- **Implementation Time**: 4-5 days
- **Mathematical Interest**: ★★★★★ (Interpolation, trajectory modeling, crisis dynamics)

---

## 🏗️ **Architecture Integration Analysis**

### **Current Infrastructure Strengths**
- ✅ **Modular Design**: Clean separation between core, visualization, and CLI
- ✅ **Configuration System**: JSON-based with dataclass validation
- ✅ **CLI Integration**: Click-based with predefined + custom experiments
- ✅ **Visualization Pipeline**: Both static (matplotlib) and interactive (plotly)
- ✅ **Testing Framework**: Function-based tests with installation validation
- ✅ **Cross-Platform**: Pure Python with pip dependencies

### **Integration Points**
```python
# Existing pattern for new experiments:
AgentSociety/echo_chamber_experiments/
├── core/
│   ├── continuous_beliefs.py         # ✅ Already implemented
│   ├── continuous_integration.py     # ✅ Already implemented  
│   ├── continuous_experiments.py     # 🆕 New experiment implementations
│   └── dynamic_parameters.py         # 🆕 For B1: time-varying parameters
├── configs/
│   ├── gradual_polarization.json     # 🆕 Experiment configs
│   ├── asymmetric_bias.json          # 🆕
│   └── dynamic_evolution.json        # 🆕
├── experiments/
│   ├── gradual_polarization/          # 🆕 Experiment-specific code
│   ├── asymmetric_bias/               # 🆕
│   └── dynamic_evolution/             # 🆕
└── tests/
    └── test_continuous_experiments.py # 🆕 Comprehensive test suite
```

---

## 📊 **Detailed Implementation Plan**

## **Phase 1: Foundation & A1 (Days 1-3)**

### **Day 1: Foundation Setup**

#### **1.1 Create Continuous Experiments Framework**
```python
# File: core/continuous_experiments.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from .continuous_integration import ContinuousAgentConfig, create_continuous_agent_population
from .experiment import ExperimentResults, EchoChamberExperiment

@dataclass
class ContinuousExperimentConfig(ExperimentConfig):
    """Extended config for continuous parameter experiments"""
    
    # New fields for continuous experiments
    experiment_type: str = "continuous_basic"
    parameter_sweep: Optional[Dict[str, List[float]]] = None
    comparison_baselines: Optional[List[str]] = None
    analysis_focus: List[str] = field(default_factory=lambda: ['polarization', 'echo_chambers'])
    
    # Continuous belief parameters (integrated with existing system)
    continuous_belief_config: Optional[ContinuousAgentConfig] = None

class ContinuousExperimentRunner:
    """Enhanced experiment runner for continuous parameter studies"""
    
    def __init__(self, config: ContinuousExperimentConfig):
        self.config = config
        self.results_collection = []
    
    def run_parameter_sweep(self) -> List[ExperimentResults]:
        """Run experiment across parameter space"""
        pass
    
    def run_comparative_analysis(self) -> Dict[str, ExperimentResults]:
        """Compare different parameter configurations"""
        pass
```

#### **1.2 Extend CLI Interface**
```python
# Add to run_experiment.py - following existing patterns
@click.option('--continuous-experiment', type=click.Choice([
    'gradual_polarization', 'asymmetric_bias', 'dynamic_evolution'
]), help='Run continuous parameter experiment')

@click.option('--parameter-sweep', is_flag=True, 
              help='Run parameter sweep analysis')

@click.option('--comparison-mode', is_flag=True,
              help='Compare with baseline discrete parameters')
```

#### **1.3 Testing Infrastructure**
```python
# File: tests/test_continuous_experiments.py
# Following existing test patterns from test_continuous_beliefs.py

def test_continuous_experiment_runner():
    """Test basic continuous experiment functionality"""
    
def test_parameter_sweep_execution():
    """Test parameter sweep runs correctly"""
    
def test_comparative_analysis():
    """Test baseline comparison functionality"""
    
def test_cli_integration():
    """Test CLI interface for continuous experiments"""
    
def test_config_validation():
    """Test configuration validation and error handling"""
```

### **Day 2-3: A1 Implementation - Gradual vs Sudden Polarization**

#### **2.1 Experiment Design**
```python
# File: experiments/gradual_polarization/experiment.py
class GradualPolarizationExperiment:
    """
    Tests how polarization speed affects echo chamber formation.
    
    Mathematical Foundation:
    - Models polarization as P(t) = P₀ + (P_final - P₀) * f(t/T)
    - Where f(x) varies: step function (sudden), linear (gradual), sigmoid (realistic)
    - Measures: polarization rate dP/dt, echo chamber stability, intervention resistance
    """
    
    def __init__(self, config: GradualPolarizationConfig):
        self.config = config
        self.polarization_functions = {
            'sudden': lambda t, T: 1.0 if t > T/2 else 0.0,
            'linear': lambda t, T: t/T, 
            'sigmoid': lambda t, T: 1/(1 + np.exp(-10*(t/T - 0.5))),
            'exponential': lambda t, T: 1 - np.exp(-5*t/T)
        }
    
    def run_polarization_timeline(self, timeline_type: str) -> ExperimentResults:
        """Run experiment with specific polarization timeline"""
        
        results_per_round = []
        
        for round_num in range(self.config.num_rounds):
            # Calculate polarization strength for this round
            progress = round_num / self.config.num_rounds
            pol_strength = self.polarization_functions[timeline_type](
                round_num, self.config.num_rounds
            )
            
            # Create agents with this polarization level
            belief_params = create_polarized_params(
                polarization_strength=0.2 + 0.7 * pol_strength  # 0.2 to 0.9 range
            )
            
            agents = create_continuous_agent_population(
                ContinuousAgentConfig(
                    num_agents=self.config.num_agents,
                    topic=self.config.topic,
                    belief_params=belief_params,
                    random_seed=self.config.random_seed
                )
            )
            
            # Run interaction round
            # ... (follows existing EchoChamberExperiment pattern)
            
        return self.analyze_polarization_effects(results_per_round)
```

#### **2.2 Mathematical Analysis**
```python
# File: experiments/gradual_polarization/analysis.py
class PolarizationDynamicsAnalyzer:
    """
    Advanced mathematical analysis of polarization dynamics.
    
    Metrics computed:
    - Polarization velocity: dP/dt at each time point
    - Echo chamber formation rate: d(EC_count)/dt  
    - Stability index: resistance to belief change
    - Intervention susceptibility: response magnitude to interventions
    """
    
    def compute_polarization_velocity(self, polarization_series: List[float]) -> np.ndarray:
        """Compute rate of polarization change"""
        return np.gradient(polarization_series)
    
    def measure_echo_chamber_stability(self, ec_history: List[List[List[int]]]) -> List[float]:
        """Measure how stable echo chambers are over time"""
        stability_scores = []
        for t in range(1, len(ec_history)):
            # Jaccard similarity between consecutive echo chamber sets
            prev_chambers = set(tuple(sorted(chamber)) for chamber in ec_history[t-1])
            curr_chambers = set(tuple(sorted(chamber)) for chamber in ec_history[t])
            
            intersection = len(prev_chambers.intersection(curr_chambers))
            union = len(prev_chambers.union(curr_chambers))
            stability = intersection / union if union > 0 else 0
            stability_scores.append(stability)
        
        return stability_scores
    
    def detect_polarization_phase_transitions(self, polarization_series: List[float]) -> Dict[str, Any]:
        """Detect sudden changes in polarization dynamics using change point detection"""
        # Implement CUSUM or other change point detection algorithm
        pass
```

#### **2.3 Visualization Extensions**
```python
# File: experiments/gradual_polarization/visualizations.py
class GradualPolarizationVisualizer(EchoChamberVisualizer):
    """Specialized visualizations for polarization dynamics"""
    
    def plot_polarization_timeline_comparison(self, results_dict: Dict[str, ExperimentResults]) -> plt.Figure:
        """Compare different polarization timelines"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Polarization over time for each timeline
        for timeline, results in results_dict.items():
            axes[0,0].plot(results.polarization_over_time, label=timeline, linewidth=2)
        axes[0,0].set_title('Polarization Evolution by Timeline Type')
        axes[0,0].set_xlabel('Round')
        axes[0,0].set_ylabel('Polarization Index')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Echo chamber formation rate
        for timeline, results in results_dict.items():
            ec_counts = [len(chambers) for chambers in results.echo_chambers_history]
            axes[0,1].plot(ec_counts, label=timeline, linewidth=2)
        axes[0,1].set_title('Echo Chamber Formation')
        axes[0,1].set_xlabel('Round')
        axes[0,1].set_ylabel('Number of Echo Chambers')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Polarization velocity (rate of change)
        for timeline, results in results_dict.items():
            velocity = np.gradient(results.polarization_over_time)
            axes[1,0].plot(velocity, label=timeline, linewidth=2)
        axes[1,0].set_title('Polarization Rate of Change')
        axes[1,0].set_xlabel('Round')
        axes[1,0].set_ylabel('dP/dt')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Final outcome comparison
        outcomes = {}
        for timeline, results in results_dict.items():
            outcomes[timeline] = {
                'final_polarization': results.polarization_over_time[-1],
                'max_echo_chambers': max(len(chambers) for chambers in results.echo_chambers_history),
                'bridge_agents': len(results.bridge_agents)
            }
        
        timeline_names = list(outcomes.keys())
        final_pols = [outcomes[name]['final_polarization'] for name in timeline_names]
        
        bars = axes[1,1].bar(timeline_names, final_pols, alpha=0.7)
        axes[1,1].set_title('Final Polarization by Timeline Type')
        axes[1,1].set_ylabel('Final Polarization')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_pols):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_polarization_dynamics_dashboard(self, results_dict: Dict[str, ExperimentResults]) -> go.Figure:
        """Interactive dashboard for polarization dynamics analysis"""
        # Create comprehensive Plotly dashboard following existing patterns
        pass
```

#### **2.4 Configuration and CLI Integration**
```python
# File: configs/gradual_polarization.json
{
  "name": "Gradual vs Sudden Polarization Study",
  "description": "Comparing polarization timeline effects on echo chamber formation",
  "experiment_type": "gradual_polarization",
  "num_agents": 100,
  "topic": "climate_change",
  "num_rounds": 20,
  "interactions_per_round": 300,
  "timeline_types": ["sudden", "linear", "sigmoid", "exponential"],
  "parameter_sweep": {
    "polarization_speed": [0.1, 0.3, 0.5, 0.7, 0.9]
  },
  "analysis_focus": ["polarization_velocity", "echo_chamber_stability", "intervention_resistance"],
  "random_seed": 42,
  "comparison_baselines": ["discrete_polarized", "discrete_normal"]
}
```

---

## **Phase 2: A2 Implementation (Days 4-5)**

### **A2: Asymmetric Bias Effects**
```python
# File: experiments/asymmetric_bias/experiment.py
class AsymmetricBiasExperiment:
    """
    Studies how population bias affects network formation and dynamics.
    
    Mathematical Foundation:
    - Population bias B ∈ [-1, 1] where B < 0 is left-leaning, B > 0 is right-leaning
    - Network asymmetry metrics: directional clustering, influence flow analysis
    - Minority influence amplification: measures when small groups have outsized impact
    """
    
    def run_bias_spectrum_analysis(self) -> Dict[float, ExperimentResults]:
        """Run experiments across bias spectrum"""
        
        bias_levels = [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8]
        results = {}
        
        for bias in bias_levels:
            belief_params = create_polarized_params(
                polarization_strength=0.7,
                asymmetry=bias
            )
            
            # Run experiment with this bias level
            results[bias] = self.run_single_bias_experiment(belief_params)
        
        return results
    
    def analyze_network_asymmetry(self, agents: List[Agent], network: SocialNetwork) -> Dict[str, float]:
        """Compute network asymmetry metrics"""
        
        # Separate agents by belief side
        left_agents = [a for a in agents if a.belief_strength < -0.2]
        right_agents = [a for a in agents if a.belief_strength > 0.2]
        center_agents = [a for a in agents if -0.2 <= a.belief_strength <= 0.2]
        
        # Compute cross-group connection rates
        left_to_right_connections = self.count_cross_connections(left_agents, right_agents, network)
        right_to_left_connections = self.count_cross_connections(right_agents, left_agents, network)
        
        # Influence flow analysis
        influence_asymmetry = self.compute_influence_asymmetry(agents, network)
        
        return {
            'population_bias': np.mean([a.belief_strength for a in agents]),
            'network_asymmetry': (right_to_left_connections - left_to_right_connections) / (right_to_left_connections + left_to_right_connections + 1e-6),
            'influence_asymmetry': influence_asymmetry,
            'minority_amplification': self.compute_minority_amplification(left_agents, right_agents, center_agents, network)
        }
```

### **Mathematical Insights**
- **Network Asymmetry Index**: Measures directional bias in connections
- **Influence Flow Analysis**: Tracks how beliefs spread across bias boundaries  
- **Minority Amplification**: Quantifies when small groups have disproportionate influence

---

## **Phase 3: B1 Implementation (Days 6-10)**

### **B1: Dynamic Belief Evolution** 
```python
# File: core/dynamic_parameters.py
class DynamicBeliefParameters:
    """
    Time-varying belief distribution parameters.
    
    Mathematical Foundation:
    - Parameter interpolation: P(t) = P₀ + (P₁ - P₀) * φ(t) where φ is interpolation function
    - Support for: linear, cubic spline, sigmoid transitions
    - Crisis modeling: sudden parameter shifts with recovery patterns
    """
    
    def __init__(self, timeline: Dict[int, BeliefDistributionParams]):
        self.timeline = timeline
        self.sorted_timepoints = sorted(timeline.keys())
    
    def get_parameters_at_round(self, round_num: int) -> BeliefDistributionParams:
        """Get interpolated parameters for specific round"""
        
        # Find surrounding keyframes
        before_key = max([t for t in self.sorted_timepoints if t <= round_num], default=0)
        after_key = min([t for t in self.sorted_timepoints if t > round_num], default=before_key)
        
        if before_key == after_key:
            return self.timeline[before_key]
        
        # Linear interpolation between keyframes
        alpha = (round_num - before_key) / (after_key - before_key)
        return self.interpolate_parameters(
            self.timeline[before_key], 
            self.timeline[after_key], 
            alpha
        )
    
    def interpolate_parameters(self, p1: BeliefDistributionParams, 
                             p2: BeliefDistributionParams, 
                             alpha: float) -> BeliefDistributionParams:
        """Smoothly interpolate between parameter sets"""
        
        return BeliefDistributionParams(
            distribution_type=p1.distribution_type,  # Don't interpolate discrete values
            polarization_strength=p1.polarization_strength * (1-alpha) + p2.polarization_strength * alpha,
            polarization_asymmetry=p1.polarization_asymmetry * (1-alpha) + p2.polarization_asymmetry * alpha,
            gap_size=p1.gap_size * (1-alpha) + p2.gap_size * alpha,
            # ... interpolate other continuous parameters
        )

# File: experiments/dynamic_evolution/crisis_scenarios.py
class CrisisScenarioGenerator:
    """Generate realistic crisis scenarios for belief evolution"""
    
    @staticmethod
    def pandemic_scenario() -> Dict[int, BeliefDistributionParams]:
        """Model belief evolution during pandemic"""
        return {
            0: create_moderate_params(spread=0.4),                    # Pre-crisis
            3: create_polarized_params(polarization_strength=0.3),    # Initial uncertainty
            8: create_polarized_params(polarization_strength=0.8),    # Peak polarization  
            15: create_polarized_params(polarization_strength=0.6),   # Adaptation
            25: create_moderate_params(spread=0.5)                    # New normal
        }
    
    @staticmethod
    def election_scenario() -> Dict[int, BeliefDistributionParams]:
        """Model belief evolution during election cycle"""
        return {
            0: create_moderate_params(spread=0.3),                    # Early campaign
            5: create_polarized_params(polarization_strength=0.5),    # Campaign heats up
            10: create_polarized_params(polarization_strength=0.9),   # Pre-election peak
            12: create_polarized_params(polarization_strength=0.7),   # Post-election
            20: create_moderate_params(spread=0.4)                    # Return to baseline
        }
```

### **Advanced Mathematical Analysis**
- **Trajectory Modeling**: Fit mathematical models to belief evolution curves
- **Phase Transition Detection**: Identify critical points in belief dynamics
- **Intervention Timing Optimization**: Find optimal moments for interventions

---

## 🧪 **Testing Strategy**

### **Comprehensive Test Suite**
```python
# File: tests/test_continuous_experiments.py
class TestContinuousExperiments:
    """Comprehensive testing following existing patterns"""
    
    def test_gradual_polarization_timeline_generation(self):
        """Test polarization timeline functions work correctly"""
        
    def test_asymmetric_bias_network_effects(self):
        """Test network asymmetry calculations"""
        
    def test_dynamic_parameter_interpolation(self):
        """Test parameter interpolation is smooth and mathematically sound"""
        
    def test_experiment_reproducibility(self):
        """Test experiments produce consistent results with same seeds"""
        
    def test_cli_integration(self):
        """Test command-line interface works for all new experiments"""
        
    def test_visualization_generation(self):
        """Test all visualizations generate without errors"""
        
    def test_cross_platform_compatibility(self):
        """Test experiments work on different operating systems"""
```

### **Performance Testing**
```python
def test_experiment_performance():
    """Ensure experiments complete in reasonable time"""
    
    start_time = time.time()
    
    # Run scaled-down version of each experiment
    gradual_pol_results = run_gradual_polarization_experiment(num_agents=20, num_rounds=5)
    asymmetric_results = run_asymmetric_bias_experiment(num_agents=20, num_rounds=5)
    dynamic_results = run_dynamic_evolution_experiment(num_agents=20, num_rounds=10)
    
    duration = time.time() - start_time
    
    assert duration < 30.0, f"Experiments too slow: {duration:.2f}s > 30s"
    assert all(results is not None for results in [gradual_pol_results, asymmetric_results, dynamic_results])
```

---

## 📈 **Visualization Strategy**

### **Static Visualizations (matplotlib/seaborn)**
Following existing patterns in `visualizations/plots.py`:

```python
class ContinuousExperimentVisualizer(EchoChamberVisualizer):
    """Enhanced visualizer for continuous experiments"""
    
    def create_experiment_comparison_report(self, experiment_results: Dict[str, Any]) -> List[plt.Figure]:
        """Generate publication-ready comparison plots"""
        
        figures = []
        
        # Figure 1: Multi-experiment overview
        fig1 = self.plot_multi_experiment_overview(experiment_results)
        figures.append(fig1)
        
        # Figure 2: Mathematical analysis plots
        fig2 = self.plot_mathematical_analysis(experiment_results)
        figures.append(fig2)
        
        # Figure 3: Network dynamics comparison
        fig3 = self.plot_network_dynamics_comparison(experiment_results)
        figures.append(fig3)
        
        return figures
```

### **Interactive Dashboards (plotly)**
```python
def create_continuous_experiments_dashboard(experiment_results: Dict[str, Any]) -> go.Figure:
    """
    Interactive dashboard for exploring continuous experiment results.
    
    Features:
    - Parameter sliders for real-time exploration
    - Linked brushing between plots
    - Drill-down capabilities for detailed analysis
    - Export functionality for static plots
    """
    
    # Create multi-tab dashboard with:
    # Tab 1: Gradual Polarization Analysis
    # Tab 2: Asymmetric Bias Network Effects  
    # Tab 3: Dynamic Evolution Trajectories
    # Tab 4: Cross-Experiment Comparison
```

---

## 🖥️ **Cross-Platform Compatibility**

### **Dependency Management**
```python
# Updated requirements.txt (following existing pattern)
# Core Dependencies for Echo Chamber Experiments
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
networkx>=2.6.0
plotly>=5.0.0
click>=8.0.0
pyyaml>=6.0.0
tqdm>=4.62.0
scipy>=1.7.0

# New dependencies for continuous experiments
scikit-learn>=1.0.0    # For mathematical analysis
statsmodels>=0.13.0    # For time series analysis

# Optional: Enhanced functionality  
jupyter>=1.0.0
ipywidgets>=7.6.0
bokeh>=2.4.0
dash>=2.0.0
```

### **Installation Validation**
```python
# Extended test_installation.py
def test_continuous_experiments():
    """Test new continuous experiments functionality"""
    
    print("\n🧪 Testing continuous experiments...")
    
    try:
        # Test basic continuous belief generation
        from core.continuous_beliefs import ContinuousBeliefGenerator, create_polarized_params
        
        params = create_polarized_params(polarization_strength=0.8)
        generator = ContinuousBeliefGenerator(params)
        beliefs = generator.generate_beliefs(10)
        
        assert len(beliefs) == 10
        assert all(-1 <= b <= 1 for b in beliefs)
        
        # Test experiment imports
        from experiments.gradual_polarization.experiment import GradualPolarizationExperiment
        from experiments.asymmetric_bias.experiment import AsymmetricBiasExperiment
        
        print("✅ Continuous experiments working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Continuous experiments error: {e}")
        return False
```

---

## 📊 **Expected Research Outcomes**

### **Publication Potential**

#### **Paper 1: "Gradual vs Sudden Polarization" (Conference)**
- **Venue**: ICWSM, ASONAM, or similar computational social science venue
- **Key Finding**: Gradual polarization creates more stable, intervention-resistant echo chambers
- **Mathematical Contribution**: Polarization velocity analysis and phase transition detection

#### **Paper 2: "Asymmetric Population Bias Effects" (Journal)**  
- **Venue**: Journal of Computational Social Science, PLoS One
- **Key Finding**: Population bias creates predictable network asymmetries with implications for intervention targeting
- **Mathematical Contribution**: Network asymmetry indices and minority amplification metrics

#### **Paper 3: "Dynamic Belief Evolution in Crisis Scenarios" (Top-tier Journal)**
- **Venue**: Nature Human Behaviour, PNAS, Science Advances
- **Key Finding**: Crisis-driven belief evolution follows predictable mathematical patterns
- **Mathematical Contribution**: Time-varying parameter models and intervention timing optimization

### **Practical Applications**

#### **Platform Design**
- **Content Algorithm Tuning**: Use asymmetric bias findings to balance recommendation systems
- **Intervention Timing**: Use dynamic evolution research to optimize fact-checking deployment
- **Community Detection**: Use gradual polarization insights to identify at-risk communities

#### **Policy Research**
- **Crisis Communication**: Use dynamic evolution models to plan public health messaging
- **Political Campaign Strategy**: Use polarization timeline research to optimize messaging timing
- **Social Media Regulation**: Use comprehensive findings to inform platform governance

---

## 🚀 **Implementation Timeline**

### **Week 1: Foundation + A1**
- **Day 1**: Set up continuous experiments framework, CLI integration
- **Day 2**: Implement gradual polarization experiment core logic
- **Day 3**: Add mathematical analysis and visualization for A1

### **Week 2: A2 + B1 Foundation**
- **Day 4**: Implement asymmetric bias experiment
- **Day 5**: Complete A2 visualization and analysis
- **Day 6**: Begin dynamic parameters system for B1

### **Week 3: B1 + Testing**
- **Day 7-8**: Complete dynamic belief evolution experiment
- **Day 9**: Comprehensive testing and bug fixes
- **Day 10**: Documentation, examples, and final integration

---

## ✅ **Success Criteria**

### **Technical Criteria**
- ✅ All experiments run successfully on Windows/Mac/Linux
- ✅ Complete test suite passes (>95% coverage)
- ✅ Visualizations generate without errors
- ✅ CLI interface works for all experiments
- ✅ Configuration system handles all parameter combinations
- ✅ Performance: experiments complete in <5 minutes for standard configs

### **Research Criteria**
- ✅ Statistically significant differences between experimental conditions
- ✅ Mathematically interpretable results with clear patterns  
- ✅ Novel insights not achievable with discrete parameter system
- ✅ Reproducible results across multiple runs with same seeds
- ✅ Publication-ready visualizations and analysis

### **Usability Criteria**
- ✅ Documentation explains how to run each experiment
- ✅ Example configurations provided for common use cases
- ✅ Error messages are clear and actionable
- ✅ Results are saved in analyzable formats (JSON, CSV)
- ✅ Integration with existing AgentSociety workflow

---

This plan leverages the existing robust infrastructure while adding cutting-edge continuous parameter capabilities. The experiments are designed to be mathematically rigorous, computationally efficient, and scientifically impactful.

**Ready to begin implementation?** 🚀