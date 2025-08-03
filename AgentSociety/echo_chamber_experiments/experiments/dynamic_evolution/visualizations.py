"""
Specialized Visualizations for Dynamic Belief Evolution

This module provides comprehensive visualization tools for analyzing
dynamic belief evolution experiments, including trajectory plots,
phase transition diagrams, crisis impact analysis, and interactive dashboards.

Visualization Categories:
- Trajectory Analysis: Multi-dimensional time series plots
- Phase Transitions: Change point detection and crisis timeline visualization  
- Mathematical Models: Fitted curves and model comparison
- Cross-Agent Analysis: Correlation networks and synchronization plots
- Interactive Dashboards: Real-time exploration tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

# Plotly imports for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available, interactive visualizations disabled")

# Network visualization
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Import experiment components
try:
    from .experiment import DynamicEvolutionResults
    from .analysis import TrajectoryAnalyzer, CrisisAnalyzer, TrajectoryModel, ChangePoint
    from ...visualizations.plots import EchoChamberVisualizer
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from experiments.dynamic_evolution.experiment import DynamicEvolutionResults
    from experiments.dynamic_evolution.analysis import TrajectoryAnalyzer, CrisisAnalyzer, TrajectoryModel, ChangePoint
    from visualizations.plots import EchoChamberVisualizer


class DynamicEvolutionVisualizer(EchoChamberVisualizer):
    """
    Specialized visualizer for dynamic belief evolution experiments.
    
    Extends the base EchoChamberVisualizer with advanced capabilities for:
    - Time-varying parameter visualization
    - Trajectory modeling and analysis
    - Phase transition detection
    - Crisis scenario comparison
    - Interactive exploration tools
    """
    
    def __init__(self, results: DynamicEvolutionResults):
        super().__init__(results)
        self.dynamic_results = results
        
        # Initialize analyzers
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.crisis_analyzer = CrisisAnalyzer()
        
        # Set up enhanced plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        # Figure size standards
        self.single_fig_size = (12, 8)
        self.multi_fig_size = (16, 12)
        self.dashboard_fig_size = (20, 15)
    
    def plot_dynamic_evolution_overview(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive overview of dynamic belief evolution.
        
        Mathematical Visualization:
        - Parameter evolution timeline (top panel)
        - Belief distribution evolution (middle panel) 
        - Phase transition detection (bottom panel)
        - Crisis impact annotations
        
        Returns:
            Matplotlib figure with multi-panel overview
        """
        
        fig = plt.figure(figsize=self.dashboard_fig_size)
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.25)
        
        # Panel 1: Parameter Evolution
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_parameter_evolution(ax1)
        
        # Panel 2: Belief Trajectories
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_belief_trajectories(ax2)
        
        # Panel 3: Polarization and Echo Chambers
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_polarization_evolution(ax3)
        
        # Panel 4: Phase Analysis
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_phase_transitions(ax4)
        
        # Panel 5: Crisis Impact Summary
        ax5 = fig.add_subplot(gs[3, :])
        self._plot_crisis_impact_summary(ax5)
        
        # Overall title and styling
        fig.suptitle(f'Dynamic Belief Evolution: {self.dynamic_results.config.name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add experiment metadata
        metadata_text = (f"Agents: {self.dynamic_results.config.num_agents} | "
                        f"Rounds: {self.dynamic_results.config.num_rounds} | "
                        f"Crisis: {self.dynamic_results.config.crisis_scenario.value if self.dynamic_results.config.crisis_scenario else 'Custom'}")
        fig.text(0.5, 0.01, metadata_text, ha='center', fontsize=10, style='italic')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def _plot_parameter_evolution(self, ax: plt.Axes):
        """Plot evolution of dynamic parameters over time"""
        
        if not self.dynamic_results.parameter_evolution:
            ax.text(0.5, 0.5, 'No parameter evolution data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        rounds = range(len(list(self.dynamic_results.parameter_evolution.values())[0]))
        
        # Plot key parameters
        param_styles = {
            'polarization_strength': {'color': '#e74c3c', 'linestyle': '-', 'linewidth': 3},
            'polarization_asymmetry': {'color': '#3498db', 'linestyle': '--', 'linewidth': 2},
            'gap_size': {'color': '#2ecc71', 'linestyle': '-.', 'linewidth': 2},
            'center': {'color': '#f39c12', 'linestyle': ':', 'linewidth': 2}
        }
        
        for param, values in self.dynamic_results.parameter_evolution.items():
            if param in param_styles:
                style = param_styles[param]
                ax.plot(rounds, values, label=param.replace('_', ' ').title(), **style)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Parameter Value', fontsize=12)
        ax.set_title('Dynamic Parameter Evolution', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Highlight intervention point if applicable
        if self.dynamic_results.config.intervention_round:
            ax.axvline(x=self.dynamic_results.config.intervention_round, 
                      color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(self.dynamic_results.config.intervention_round, ax.get_ylim()[1] * 0.9, 
                   'Intervention', rotation=90, fontsize=10, color='red')
    
    def _plot_belief_trajectories(self, ax: plt.Axes):
        """Plot individual agent belief trajectories"""
        
        if not self.dynamic_results.belief_trajectories:
            ax.text(0.5, 0.5, 'No trajectory data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # Sample trajectories to avoid overcrowding
        max_trajectories = 15
        agent_ids = list(self.dynamic_results.belief_trajectories.keys())
        
        if len(agent_ids) > max_trajectories:
            # Sample diverse trajectories
            sampled_ids = np.random.choice(agent_ids, max_trajectories, replace=False)
        else:
            sampled_ids = agent_ids
        
        # Plot trajectories
        for i, agent_id in enumerate(sampled_ids):
            trajectory = self.dynamic_results.belief_trajectories[agent_id]
            rounds = range(len(trajectory))
            
            # Color based on final belief
            final_belief = trajectory[-1]
            color = plt.cm.RdBu_r((final_belief + 1) / 2)  # Map [-1,1] to [0,1]
            
            ax.plot(rounds, trajectory, color=color, alpha=0.6, linewidth=1.5)
        
        # Population mean trajectory
        if self.dynamic_results.belief_trajectories:
            rounds = range(len(list(self.dynamic_results.belief_trajectories.values())[0]))
            population_means = []
            
            for round_idx in rounds:
                round_beliefs = [
                    traj[round_idx] for traj in self.dynamic_results.belief_trajectories.values()
                    if round_idx < len(traj)
                ]
                if round_beliefs:
                    population_means.append(np.mean(round_beliefs))
            
            ax.plot(rounds[:len(population_means)], population_means, 
                   color='black', linewidth=3, label='Population Mean', alpha=0.8)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Belief Strength', fontsize=12)
        ax.set_title('Agent Belief Trajectories', fontsize=14, fontweight='bold')
        ax.set_ylim(-1.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for belief interpretation
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Final Belief', fontsize=10)
    
    def _plot_polarization_evolution(self, ax: plt.Axes):
        """Plot polarization and echo chamber evolution"""
        
        if not self.dynamic_results.polarization_over_time:
            ax.text(0.5, 0.5, 'No polarization data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        rounds = range(len(self.dynamic_results.polarization_over_time))
        
        # Primary axis: Polarization
        ax.plot(rounds, self.dynamic_results.polarization_over_time, 
               color='#e74c3c', linewidth=3, label='Polarization', marker='o', markersize=4)
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Polarization Index', fontsize=12, color='#e74c3c')
        ax.tick_params(axis='y', labelcolor='#e74c3c')
        ax.set_title('Polarization & Echo Chamber Evolution', fontsize=14, fontweight='bold')
        
        # Secondary axis: Echo Chambers
        if self.dynamic_results.echo_chambers_history:
            ax2 = ax.twinx()
            echo_chamber_counts = [len(chambers) for chambers in self.dynamic_results.echo_chambers_history]
            
            ax2.plot(rounds[:len(echo_chamber_counts)], echo_chamber_counts, 
                    color='#3498db', linewidth=2, label='Echo Chambers', 
                    marker='s', markersize=4, alpha=0.7)
            ax2.set_ylabel('Number of Echo Chambers', fontsize=12, color='#3498db')
            ax2.tick_params(axis='y', labelcolor='#3498db')
        
        ax.grid(True, alpha=0.3)
        
        # Add velocity indicators
        if self.dynamic_results.belief_velocities:
            # Show periods of rapid change
            velocity_threshold = np.std(self.dynamic_results.belief_velocities) * 1.5
            rapid_periods = np.where(np.abs(self.dynamic_results.belief_velocities) > velocity_threshold)[0]
            
            for period in rapid_periods:
                ax.axvspan(period, period + 1, alpha=0.2, color='orange', label='Rapid Change' if period == rapid_periods[0] else "")
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        if 'ax2' in locals():
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax.legend()
    
    def _plot_phase_transitions(self, ax: plt.Axes):
        """Plot detected phase transitions and crisis timeline"""
        
        if not self.dynamic_results.phase_transitions:
            ax.text(0.5, 0.5, 'No phase transitions detected', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # Create timeline
        rounds = range(self.dynamic_results.config.num_rounds)
        ax.plot(rounds, [0] * len(rounds), 'k-', alpha=0.3, linewidth=1)
        
        # Plot phase transitions
        transition_types = set(pt[1] for pt in self.dynamic_results.phase_transitions)
        colors = plt.cm.Set1(np.linspace(0, 1, len(transition_types)))
        type_colors = dict(zip(transition_types, colors))
        
        for round_num, param_name, magnitude in self.dynamic_results.phase_transitions:
            color = type_colors[param_name]
            
            # Plot transition marker
            ax.scatter(round_num, magnitude, s=100 * abs(magnitude), 
                      c=[color], alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add annotation for significant transitions
            if abs(magnitude) > 0.1:
                ax.annotate(f'{param_name}\n({magnitude:.2f})', 
                           xy=(round_num, magnitude), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Transition Magnitude', fontsize=12)
        ax.set_title('Phase Transitions & Change Points', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend for parameter types
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=8, label=param)
                          for param, color in type_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Highlight crisis periods
        if hasattr(self.dynamic_results, 'rapid_change_periods'):
            for param, periods in self.dynamic_results.rapid_change_periods.items():
                for start, end in periods:
                    ax.axvspan(start, end, alpha=0.2, color='red', 
                              label='Crisis Period' if param == list(self.dynamic_results.rapid_change_periods.keys())[0] else "")
    
    def _plot_crisis_impact_summary(self, ax: plt.Axes):
        """Plot crisis impact summary metrics"""
        
        if not self.dynamic_results.crisis_impact_metrics:
            ax.text(0.5, 0.5, 'No crisis impact data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        metrics = self.dynamic_results.crisis_impact_metrics
        
        # Create bar chart of key metrics
        metric_names = ['Polarization\nIncrease', 'Recovery\nRatio', 'Volatility']
        metric_values = [
            metrics.get('polarization_increase', 0),
            metrics.get('recovery_ratio', 0),
            metrics.get('volatility', 0)
        ]
        
        bars = ax.bar(metric_names, metric_values, 
                     color=['#e74c3c', '#2ecc71', '#f39c12'], alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Crisis Impact Summary', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation text
        interpretation = self._interpret_crisis_metrics(metrics)
        ax.text(0.02, 0.98, interpretation, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    def _interpret_crisis_metrics(self, metrics: Dict[str, float]) -> str:
        """Generate text interpretation of crisis metrics"""
        
        pol_increase = metrics.get('polarization_increase', 0)
        recovery = metrics.get('recovery_ratio', 0)
        volatility = metrics.get('volatility', 0)
        
        interpretation = []
        
        if pol_increase > 0.3:
            interpretation.append("• High polarization impact")
        elif pol_increase > 0.1:
            interpretation.append("• Moderate polarization impact")
        else:
            interpretation.append("• Low polarization impact")
        
        if recovery > 0.7:
            interpretation.append("• Strong recovery")
        elif recovery > 0.3:
            interpretation.append("• Partial recovery")
        else:
            interpretation.append("• Weak recovery")
        
        if volatility > 0.2:
            interpretation.append("• High instability")
        elif volatility > 0.1:
            interpretation.append("• Moderate instability")
        else:
            interpretation.append("• Low instability")
        
        return '\n'.join(interpretation)
    
    def plot_trajectory_model_comparison(self, agent_id: int, 
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot fitted mathematical models for a specific agent's trajectory.
        
        Mathematical Visualization:
        - Original trajectory data points
        - Fitted model curves (linear, quadratic, exponential, etc.)
        - Model comparison metrics (R², AIC, BIC)
        - Turning points and mathematical features
        
        Args:
            agent_id: ID of agent to analyze
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure with model comparison
        """
        
        if agent_id not in self.dynamic_results.belief_trajectories:
            raise ValueError(f"Agent {agent_id} not found in trajectory data")
        
        trajectory = np.array(self.dynamic_results.belief_trajectories[agent_id])
        time_points = np.arange(len(trajectory))
        
        # Fit multiple models
        fitted_models = self.trajectory_analyzer.fit_trajectory_models(time_points, trajectory)
        
        fig, axes = plt.subplots(2, 2, figsize=self.multi_fig_size)
        axes = axes.flatten()
        
        # Plot top 4 models
        model_types = sorted(fitted_models.keys(), key=lambda m: fitted_models[m].aic)[:4]
        
        for i, model_type in enumerate(model_types):
            ax = axes[i]
            model = fitted_models[model_type]
            
            # Plot original data
            ax.scatter(time_points, trajectory, alpha=0.7, color='blue', s=30, label='Data')
            
            # Plot fitted model
            t_fine = np.linspace(time_points[0], time_points[-1], 200)
            try:
                y_pred = model.predict(t_fine)
                ax.plot(t_fine, y_pred, 'r-', linewidth=2, label=f'{model_type.value.title()} Model')
            except:
                y_pred = model.predict(time_points)
                ax.plot(time_points, y_pred, 'r-', linewidth=2, label=f'{model_type.value.title()} Model')
            
            # Plot turning points
            for tp_time, tp_value in model.turning_points:
                if time_points[0] <= tp_time <= time_points[-1]:
                    ax.scatter(tp_time, tp_value, s=100, marker='*', 
                              color='gold', edgecolor='black', label='Turning Point')
            
            # Plot inflection points
            for ip_time, ip_value in model.inflection_points:
                if time_points[0] <= ip_time <= time_points[-1]:
                    ax.scatter(ip_time, ip_value, s=100, marker='^', 
                              color='green', edgecolor='black', label='Inflection Point')
            
            ax.set_xlabel('Round')
            ax.set_ylabel('Belief Strength')
            ax.set_title(f'{model_type.value.title()} Model\nR²={model.r_squared:.3f}, AIC={model.aic:.1f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Trajectory Model Comparison: Agent {agent_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cross_agent_correlation_network(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot network visualization of agent belief correlations.
        
        Network Visualization:
        - Nodes represent agents (sized by belief volatility)
        - Edges represent correlation strength
        - Colors indicate final belief positions
        - Layout optimized for correlation structure
        
        Returns:
            Matplotlib figure with correlation network
        """
        
        if not NETWORKX_AVAILABLE:
            warnings.warn("NetworkX not available, skipping network visualization")
            return plt.figure()
        
        if not self.dynamic_results.belief_trajectories:
            warnings.warn("No trajectory data available for correlation analysis")
            return plt.figure()
        
        # Compute correlation matrix
        agent_ids = sorted(self.dynamic_results.belief_trajectories.keys())
        trajectory_matrix = np.array([
            self.dynamic_results.belief_trajectories[aid] for aid in agent_ids
        ])
        
        correlation_matrix = np.corrcoef(trajectory_matrix)
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for i, agent_id in enumerate(agent_ids):
            trajectory = self.dynamic_results.belief_trajectories[agent_id]
            final_belief = trajectory[-1]
            volatility = np.std(trajectory)
            
            G.add_node(agent_id, 
                      final_belief=final_belief,
                      volatility=volatility,
                      trajectory=trajectory)
        
        # Add edges for strong correlations
        correlation_threshold = 0.5
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                correlation = correlation_matrix[i, j]
                if abs(correlation) > correlation_threshold:
                    G.add_edge(agent_ids[i], agent_ids[j], 
                              correlation=correlation,
                              weight=abs(correlation))
        
        # Create visualization
        fig, ax = plt.subplots(figsize=self.single_fig_size)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        node_colors = [G.nodes[node]['final_belief'] for node in G.nodes()]
        node_sizes = [300 + 1000 * G.nodes[node]['volatility'] for node in G.nodes()]
        
        nodes = nx.draw_networkx_nodes(G, pos, 
                                      node_color=node_colors,
                                      node_size=node_sizes,
                                      cmap=plt.cm.RdBu_r,
                                      vmin=-1, vmax=1,
                                      alpha=0.8,
                                      ax=ax)
        
        # Draw edges
        edge_weights = [G[u][v]['correlation'] for u, v in G.edges()]
        edge_colors = ['red' if w > 0 else 'blue' for w in edge_weights]
        edge_widths = [3 * abs(w) for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.6,
                              ax=ax)
        
        # Draw labels for high-volatility nodes
        high_volatility_nodes = {node: pos[node] for node in G.nodes() 
                               if G.nodes[node]['volatility'] > np.mean([G.nodes[n]['volatility'] for n in G.nodes()])}
        nx.draw_networkx_labels(G, high_volatility_nodes, font_size=8, ax=ax)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Final Belief')
        
        ax.set_title('Agent Belief Correlation Network', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        legend_text = (f"Nodes: {len(G.nodes())} agents\n"
                      f"Edges: {len(G.edges())} correlations > {correlation_threshold}\n"
                      f"Node size ∝ belief volatility\n"
                      f"Edge width ∝ |correlation|\n"
                      f"Red edges: positive correlation\n"
                      f"Blue edges: negative correlation")
        
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Create interactive Plotly dashboard for dynamic evolution exploration.
        
        Interactive Features:
        - Parameter timeline with hover details
        - Belief trajectory selection and highlighting
        - Phase transition annotations
        - Crisis impact metrics
        - Model comparison tools
        
        Returns:
            Plotly figure object (if Plotly available)
        """
        
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available, cannot create interactive dashboard")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Parameter Evolution', 'Belief Trajectories',
                'Polarization & Echo Chambers', 'Phase Transitions',
                'Agent Correlation Heatmap', 'Crisis Impact Metrics'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"type": "heatmap"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08
        )
        
        # 1. Parameter Evolution
        if self.dynamic_results.parameter_evolution:
            rounds = list(range(len(list(self.dynamic_results.parameter_evolution.values())[0])))
            
            for param, values in self.dynamic_results.parameter_evolution.items():
                fig.add_trace(
                    go.Scatter(
                        x=rounds, y=values,
                        mode='lines+markers',
                        name=param.replace('_', ' ').title(),
                        line=dict(width=3),
                        hovertemplate=f'<b>{param}</b><br>Round: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # 2. Belief Trajectories (sample)
        if self.dynamic_results.belief_trajectories:
            agent_ids = list(self.dynamic_results.belief_trajectories.keys())
            sample_size = min(10, len(agent_ids))
            sampled_agents = np.random.choice(agent_ids, sample_size, replace=False)
            
            for agent_id in sampled_agents:
                trajectory = self.dynamic_results.belief_trajectories[agent_id]
                rounds = list(range(len(trajectory)))
                
                fig.add_trace(
                    go.Scatter(
                        x=rounds, y=trajectory,
                        mode='lines',
                        name=f'Agent {agent_id}',
                        opacity=0.7,
                        hovertemplate=f'<b>Agent {agent_id}</b><br>Round: %{{x}}<br>Belief: %{{y:.3f}}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # 3. Polarization and Echo Chambers
        if self.dynamic_results.polarization_over_time:
            rounds = list(range(len(self.dynamic_results.polarization_over_time)))
            
            fig.add_trace(
                go.Scatter(
                    x=rounds, y=self.dynamic_results.polarization_over_time,
                    mode='lines+markers',
                    name='Polarization',
                    line=dict(color='red', width=3),
                    hovertemplate='<b>Polarization</b><br>Round: %{x}<br>Value: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            if self.dynamic_results.echo_chambers_history:
                echo_counts = [len(chambers) for chambers in self.dynamic_results.echo_chambers_history]
                fig.add_trace(
                    go.Scatter(
                        x=rounds[:len(echo_counts)], y=echo_counts,
                        mode='lines+markers',
                        name='Echo Chambers',
                        line=dict(color='blue', width=2),
                        yaxis='y2',
                        hovertemplate='<b>Echo Chambers</b><br>Round: %{x}<br>Count: %{y}<extra></extra>'
                    ),
                    row=2, col=1, secondary_y=True
                )
        
        # 4. Phase Transitions
        if self.dynamic_results.phase_transitions:
            transition_rounds = [pt[0] for pt in self.dynamic_results.phase_transitions]
            transition_magnitudes = [pt[2] for pt in self.dynamic_results.phase_transitions]
            transition_params = [pt[1] for pt in self.dynamic_results.phase_transitions]
            
            fig.add_trace(
                go.Scatter(
                    x=transition_rounds, y=transition_magnitudes,
                    mode='markers',
                    name='Phase Transitions',
                    marker=dict(
                        size=[50 * abs(mag) for mag in transition_magnitudes],
                        color=transition_magnitudes,
                        colorscale='RdBu',
                        showscale=True
                    ),
                    text=transition_params,
                    hovertemplate='<b>%{text}</b><br>Round: %{x}<br>Magnitude: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # 5. Agent Correlation Heatmap
        if self.dynamic_results.belief_trajectories and len(self.dynamic_results.belief_trajectories) > 1:
            agent_ids = sorted(list(self.dynamic_results.belief_trajectories.keys())[:20])  # Limit for readability
            correlation_data = []
            
            for aid1 in agent_ids:
                row = []
                for aid2 in agent_ids:
                    traj1 = self.dynamic_results.belief_trajectories[aid1]
                    traj2 = self.dynamic_results.belief_trajectories[aid2]
                    correlation = np.corrcoef(traj1, traj2)[0, 1]
                    row.append(correlation)
                correlation_data.append(row)
            
            fig.add_trace(
                go.Heatmap(
                    z=correlation_data,
                    x=[f'Agent {aid}' for aid in agent_ids],
                    y=[f'Agent {aid}' for aid in agent_ids],
                    colorscale='RdBu',
                    zmid=0,
                    hovertemplate='Correlation: %{z:.3f}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # 6. Crisis Impact Metrics
        if self.dynamic_results.crisis_impact_metrics:
            metrics = self.dynamic_results.crisis_impact_metrics
            metric_names = ['Polarization Increase', 'Recovery Ratio', 'Volatility']
            metric_values = [
                metrics.get('polarization_increase', 0),
                metrics.get('recovery_ratio', 0),
                metrics.get('volatility', 0)
            ]
            
            fig.add_trace(
                go.Bar(
                    x=metric_names, y=metric_values,
                    name='Crisis Metrics',
                    marker_color=['red', 'green', 'orange'],
                    hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Dynamic Belief Evolution Dashboard: {self.dynamic_results.config.name}",
            title_x=0.5,
            showlegend=True
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Parameter Value", row=1, col=1)
        fig.update_yaxes(title_text="Belief Strength", row=1, col=2)
        fig.update_yaxes(title_text="Polarization", row=2, col=1)
        fig.update_yaxes(title_text="Echo Chambers", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Magnitude", row=2, col=2)
        fig.update_yaxes(title_text="Metric Value", row=3, col=2)
        
        # Update x-axis labels
        for row in range(1, 4):
            for col in range(1, 3):
                if row < 3 or col == 2:  # Skip heatmap x-axis
                    fig.update_xaxes(title_text="Round", row=row, col=col)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_comprehensive_report(self, output_dir: str) -> Dict[str, str]:
        """
        Generate a comprehensive visual report of the dynamic evolution experiment.
        
        Creates multiple visualization files:
        - Overview dashboard (PNG)
        - Individual trajectory models (PNG)
        - Correlation network (PNG)
        - Interactive dashboard (HTML)
        - Summary statistics (JSON)
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Dictionary mapping report components to file paths
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_files = {}
        
        # 1. Overview dashboard
        try:
            overview_fig = self.plot_dynamic_evolution_overview()
            overview_path = output_path / "dynamic_evolution_overview.png"
            overview_fig.savefig(overview_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(overview_fig)
            report_files['overview'] = str(overview_path)
        except Exception as e:
            warnings.warn(f"Failed to generate overview: {e}")
        
        # 2. Trajectory models for sample agents
        if self.dynamic_results.belief_trajectories:
            agent_ids = list(self.dynamic_results.belief_trajectories.keys())
            sample_agents = agent_ids[:min(3, len(agent_ids))]  # Up to 3 sample agents
            
            for agent_id in sample_agents:
                try:
                    model_fig = self.plot_trajectory_model_comparison(agent_id)
                    model_path = output_path / f"trajectory_models_agent_{agent_id}.png"
                    model_fig.savefig(model_path, dpi=300, bbox_inches='tight')
                    plt.close(model_fig)
                    report_files[f'trajectory_models_agent_{agent_id}'] = str(model_path)
                except Exception as e:
                    warnings.warn(f"Failed to generate trajectory models for agent {agent_id}: {e}")
        
        # 3. Correlation network
        try:
            network_fig = self.plot_cross_agent_correlation_network()
            network_path = output_path / "correlation_network.png"
            network_fig.savefig(network_path, dpi=300, bbox_inches='tight')
            plt.close(network_fig)
            report_files['correlation_network'] = str(network_path)
        except Exception as e:
            warnings.warn(f"Failed to generate correlation network: {e}")
        
        # 4. Interactive dashboard
        if PLOTLY_AVAILABLE:
            try:
                dashboard_fig = self.create_interactive_dashboard()
                if dashboard_fig:
                    dashboard_path = output_path / "interactive_dashboard.html"
                    dashboard_fig.write_html(str(dashboard_path))
                    report_files['interactive_dashboard'] = str(dashboard_path)
            except Exception as e:
                warnings.warn(f"Failed to generate interactive dashboard: {e}")
        
        # 5. Summary statistics
        try:
            stats = self._generate_summary_statistics()
            stats_path = output_path / "summary_statistics.json"
            import json
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            report_files['summary_statistics'] = str(stats_path)
        except Exception as e:
            warnings.warn(f"Failed to generate summary statistics: {e}")
        
        return report_files
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        
        stats = {
            'experiment_config': {
                'name': self.dynamic_results.config.name,
                'num_agents': self.dynamic_results.config.num_agents,
                'num_rounds': self.dynamic_results.config.num_rounds,
                'crisis_scenario': self.dynamic_results.config.crisis_scenario.value if self.dynamic_results.config.crisis_scenario else None,
                'intervention_round': self.dynamic_results.config.intervention_round
            },
            'trajectory_statistics': {},
            'crisis_impact': self.dynamic_results.crisis_impact_metrics,
            'phase_analysis': {
                'num_phase_transitions': len(self.dynamic_results.phase_transitions),
                'transition_parameters': list(set(pt[1] for pt in self.dynamic_results.phase_transitions))
            }
        }
        
        # Trajectory statistics
        if self.dynamic_results.belief_trajectories:
            trajectory_stats = self.dynamic_results.compute_trajectory_statistics()
            stats['trajectory_statistics'] = trajectory_stats
        
        # Cross-agent analysis
        if self.dynamic_results.belief_trajectories:
            cross_agent_metrics = self.trajectory_analyzer.compute_cross_trajectory_metrics(
                {aid: np.array(traj) for aid, traj in self.dynamic_results.belief_trajectories.items()}
            )
            stats['cross_agent_analysis'] = cross_agent_metrics
        
        return stats