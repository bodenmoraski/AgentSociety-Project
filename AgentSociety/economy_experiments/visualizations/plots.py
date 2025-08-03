"""
Visualization tools for echo chamber experiment results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from ..core.experiment import ExperimentResults
    from ..core.agent import Agent, PersonalityType
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from core.experiment import ExperimentResults
    from core.agent import Agent, PersonalityType


class EchoChamberVisualizer:
    """Comprehensive visualization suite for echo chamber experiments"""
    
    def __init__(self, results: ExperimentResults):
        self.results = results
        self.config = results.config
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_comprehensive_dashboard(self, save_path: Optional[str] = None) -> go.Figure:
        """Create an interactive dashboard with all key visualizations"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Belief Evolution Over Time',
                'Network Polarization Metrics',
                'Echo Chamber Formation',
                'Agent Personality Distribution',
                'Influence Network',
                'Intervention Effects'
            ],
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"colspan": 2}, None]
            ]
        )
        
        # 1. Belief evolution over time
        if self.results.agents_history:
            df = self.results.to_dataframe()
            
            # Plot belief trajectories for sample of agents
            sample_agents = df['id'].unique()[:10]  # Show first 10 agents
            
            for agent_id in sample_agents:
                agent_data = df[df['id'] == agent_id]
                fig.add_trace(
                    go.Scatter(
                        x=agent_data['round'],
                        y=agent_data['belief_strength'],
                        mode='lines',
                        name=f'Agent {agent_id}',
                        line=dict(width=1, opacity=0.7),
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # 2. Polarization metrics
        rounds = list(range(len(self.results.polarization_over_time)))
        
        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=self.results.polarization_over_time,
                mode='lines+markers',
                name='Polarization',
                line=dict(color='red', width=3)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=self.results.echo_chamber_count_over_time,
                mode='lines+markers',
                name='Echo Chambers',
                line=dict(color='blue', width=3),
                yaxis='y2'
            ),
            row=1, col=2, secondary_y=True
        )
        
        # 3. Echo chamber size distribution
        if self.results.final_echo_chambers:
            chamber_sizes = [len(chamber) for chamber in self.results.final_echo_chambers]
            fig.add_trace(
                go.Bar(
                    x=list(range(1, len(chamber_sizes) + 1)),
                    y=chamber_sizes,
                    name='Echo Chamber Sizes',
                    marker_color='orange'
                ),
                row=2, col=1
            )
        
        # 4. Personality distribution
        if self.results.agents_history:
            final_round = self.results.agents_history[-1]
            personality_counts = {}
            for agent in final_round:
                ptype = agent['personality_type']
                personality_counts[ptype] = personality_counts.get(ptype, 0) + 1
            
            fig.add_trace(
                go.Pie(
                    labels=list(personality_counts.keys()),
                    values=list(personality_counts.values()),
                    name="Personality Types"
                ),
                row=2, col=2
            )
        
        # 5. Network visualization (simplified)
        fig.add_trace(
            go.Scatter(
                x=rounds,
                y=self.results.network_fragmentation_over_time,
                mode='lines+markers',
                name='Network Fragmentation',
                line=dict(color='green', width=3)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Echo Chamber Experiment Dashboard: {self.config.name}",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Round", row=1, col=1)
        fig.update_yaxes(title_text="Belief Strength", row=1, col=1)
        
        fig.update_xaxes(title_text="Round", row=1, col=2)
        fig.update_yaxes(title_text="Polarization", row=1, col=2)
        fig.update_yaxes(title_text="Echo Chambers", secondary_y=True, row=1, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_belief_evolution(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot belief evolution over time with distribution"""
        
        if not self.results.agents_history:
            print("No detailed agent history available for plotting")
            return None
        
        df = self.results.to_dataframe()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Individual trajectories (sample)
        sample_agents = df['id'].unique()[:20]  # Show 20 agents
        
        for agent_id in sample_agents:
            agent_data = df[df['id'] == agent_id]
            personality = agent_data['personality_type'].iloc[0]
            
            # Color by personality type
            color_map = {
                'conformist': 'blue',
                'contrarian': 'red', 
                'independent': 'green',
                'amplifier': 'orange'
            }
            
            ax1.plot(
                agent_data['round'], 
                agent_data['belief_strength'],
                alpha=0.6,
                linewidth=1,
                color=color_map.get(personality, 'gray'),
                label=personality if agent_id == sample_agents[0] else ""
            )
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Belief Strength')
        ax1.set_title('Individual Belief Trajectories')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add intervention line if applicable
        if self.config.intervention_round:
            ax1.axvline(x=self.config.intervention_round, color='red', linestyle='--', 
                       label=f'Intervention ({self.config.intervention_type})')
        
        # 2. Belief distribution evolution
        rounds_to_show = [0, len(self.results.agents_history)//2, len(self.results.agents_history)-1]
        
        for i, round_num in enumerate(rounds_to_show):
            if round_num < len(self.results.agents_history):
                round_data = self.results.agents_history[round_num]
                beliefs = [agent['belief_strength'] for agent in round_data]
                
                ax2.hist(beliefs, bins=20, alpha=0.6, 
                        label=f'Round {round_num}',
                        density=True)
        
        ax2.set_xlabel('Belief Strength')
        ax2.set_ylabel('Density')
        ax2.set_title('Belief Distribution Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_network_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot network structure and evolution"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Network metrics over time
        rounds = list(range(len(self.results.network_metrics_history)))
        
        metrics_to_plot = ['density', 'clustering_coefficient', 'assortativity']
        for metric in metrics_to_plot:
            values = [m.get(metric, 0) for m in self.results.network_metrics_history]
            ax1.plot(rounds, values, marker='o', label=metric.replace('_', ' ').title(), linewidth=2)
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Network Structure Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if self.config.intervention_round:
            ax1.axvline(x=self.config.intervention_round, color='red', linestyle='--', alpha=0.7)
        
        # 2. Echo chamber formation
        ax2.plot(rounds, self.results.echo_chamber_count_over_time, 
                marker='s', color='orange', linewidth=3, markersize=6)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Number of Echo Chambers')
        ax2.set_title('Echo Chamber Formation Over Time')
        ax2.grid(True, alpha=0.3)
        
        if self.config.intervention_round:
            ax2.axvline(x=self.config.intervention_round, color='red', linestyle='--', alpha=0.7)
        
        # 3. Polarization vs Echo Chambers
        ax3.scatter(self.results.polarization_over_time, 
                   self.results.echo_chamber_count_over_time,
                   c=rounds, cmap='viridis', s=50, alpha=0.7)
        ax3.set_xlabel('Polarization Level')
        ax3.set_ylabel('Number of Echo Chambers')
        ax3.set_title('Polarization vs Echo Chamber Formation')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(rounds), vmax=max(rounds)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax3)
        cbar.set_label('Round')
        
        # 4. Bridge agents over time
        bridge_counts = [len(bridge_list) for bridge_list in self.results.bridge_agents_history]
        ax4.plot(rounds, bridge_counts, marker='^', color='purple', linewidth=3, markersize=6)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Number of Bridge Agents')
        ax4.set_title('Bridge Agents Over Time')
        ax4.grid(True, alpha=0.3)
        
        if self.config.intervention_round:
            ax4.axvline(x=self.config.intervention_round, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_agent_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """Analyze individual agent characteristics and outcomes"""
        
        if not self.results.agents_history:
            print("No detailed agent history available for plotting")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get final round data
        final_round = self.results.agents_history[-1]
        df_final = pd.DataFrame(final_round)
        
        # 1. Personality vs Polarization
        personality_polarization = {}
        for ptype in PersonalityType:
            agents_of_type = df_final[df_final['personality_type'] == ptype.value]
            if len(agents_of_type) > 0:
                personality_polarization[ptype.value] = agents_of_type['polarization_score'].mean()
        
        if personality_polarization:
            ax1.bar(personality_polarization.keys(), personality_polarization.values(), 
                   color=['blue', 'red', 'green', 'orange'])
            ax1.set_xlabel('Personality Type')
            ax1.set_ylabel('Average Polarization Score')
            ax1.set_title('Polarization by Personality Type')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Openness vs Belief Change
        df = self.results.to_dataframe()
        if len(df) > 0:
            # Calculate belief change for each agent
            belief_changes = []
            openness_values = []
            
            for agent_id in df['id'].unique():
                agent_data = df[df['id'] == agent_id].sort_values('round')
                if len(agent_data) > 1:
                    initial_belief = agent_data['belief_strength'].iloc[0]
                    final_belief = agent_data['belief_strength'].iloc[-1]
                    belief_change = abs(final_belief - initial_belief)
                    
                    belief_changes.append(belief_change)
                    openness_values.append(agent_data['openness'].iloc[0])
            
            ax2.scatter(openness_values, belief_changes, alpha=0.6, s=50)
            ax2.set_xlabel('Openness')
            ax2.set_ylabel('Total Belief Change')
            ax2.set_title('Openness vs Belief Flexibility')
            ax2.grid(True, alpha=0.3)
            
            # Add trend line
            if len(openness_values) > 1:
                z = np.polyfit(openness_values, belief_changes, 1)
                p = np.poly1d(z)
                ax2.plot(sorted(openness_values), p(sorted(openness_values)), "r--", alpha=0.8)
        
        # 3. Connection count vs Influence
        if len(df_final) > 0:
            # Note: This is simplified - in full implementation, you'd calculate actual influence
            ax3.scatter(df_final['num_connections'], df_final['interaction_count'], 
                       c=df_final['confidence'], cmap='viridis', s=60, alpha=0.7)
            ax3.set_xlabel('Number of Connections')
            ax3.set_ylabel('Interaction Count')
            ax3.set_title('Network Position vs Activity')
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis')
            sm.set_array(df_final['confidence'])
            cbar = plt.colorbar(sm, ax=ax3)
            cbar.set_label('Confidence Level')
        
        # 4. Belief distribution by personality
        personality_types = df_final['personality_type'].unique()
        for i, ptype in enumerate(personality_types):
            agents_of_type = df_final[df_final['personality_type'] == ptype]
            beliefs = agents_of_type['belief_strength']
            
            ax4.hist(beliefs, bins=15, alpha=0.6, label=ptype, density=True)
        
        ax4.set_xlabel('Final Belief Strength')
        ax4.set_ylabel('Density')
        ax4.set_title('Final Belief Distribution by Personality')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_network_graph(self, round_num: int = -1, save_path: Optional[str] = None) -> plt.Figure:
        """Create a network graph visualization"""
        
        # This is a simplified version - full implementation would require access to network object
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        if self.results.agents_history and round_num < len(self.results.agents_history):
            round_data = self.results.agents_history[round_num]
            
            # Create a simple network layout
            num_agents = len(round_data)
            
            # Arrange agents in a circle for visualization
            angles = np.linspace(0, 2*np.pi, num_agents, endpoint=False)
            x_positions = np.cos(angles)
            y_positions = np.sin(angles)
            
            # Plot agents colored by belief
            beliefs = [agent['belief_strength'] for agent in round_data]
            personalities = [agent['personality_type'] for agent in round_data]
            
            # Color map for beliefs (red for negative, blue for positive)
            colors = ['red' if b < 0 else 'blue' for b in beliefs]
            sizes = [100 + abs(b) * 200 for b in beliefs]  # Size by belief strength
            
            # Plot nodes
            scatter = ax.scatter(x_positions, y_positions, c=beliefs, cmap='RdBu', 
                               s=sizes, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add agent labels
            for i, agent in enumerate(round_data):
                ax.annotate(f"{agent['id']}", (x_positions[i], y_positions[i]), 
                           ha='center', va='center', fontsize=8)
            
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.set_title(f'Network Structure - Round {round_num if round_num >= 0 else "Final"}')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Belief Strength')
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_summary_report(self) -> str:
        """Generate a text summary of experiment results"""
        
        report = f"""
ECHO CHAMBER EXPERIMENT SUMMARY REPORT
=====================================

Experiment: {self.config.name}
Description: {self.config.description}
Topic: {self.config.topic.value}

EXPERIMENTAL PARAMETERS:
- Number of agents: {self.config.num_agents}
- Number of rounds: {self.config.num_rounds}
- Network type: {self.config.network_config.network_type}
- Belief distribution: {self.config.belief_distribution}
- Homophily strength: {self.config.network_config.homophily_strength}

INTERVENTION:
- Applied: {'Yes' if self.config.intervention_type else 'No'}
- Type: {self.config.intervention_type or 'None'}
- Round: {self.config.intervention_round or 'N/A'}

KEY FINDINGS:
=============

POLARIZATION DYNAMICS:
- Initial polarization: {self.results.polarization_over_time[0]:.3f}
- Final polarization: {self.results.polarization_over_time[-1]:.3f}
- Change: {self.results.polarization_over_time[-1] - self.results.polarization_over_time[0]:+.3f}
- Peak polarization: {max(self.results.polarization_over_time):.3f}

ECHO CHAMBER FORMATION:
- Final echo chambers: {len(self.results.final_echo_chambers)}
- Largest chamber size: {max([len(c) for c in self.results.final_echo_chambers]) if self.results.final_echo_chambers else 0}
- Agents in echo chambers: {sum(len(c) for c in self.results.final_echo_chambers)}
- Echo chamber participation: {sum(len(c) for c in self.results.final_echo_chambers) / self.config.num_agents * 100:.1f}%

NETWORK STRUCTURE:
- Final network density: {self.results.network_metrics_history[-1].get('density', 0):.3f}
- Final clustering coefficient: {self.results.network_metrics_history[-1].get('clustering_coefficient', 0):.3f}
- Network modularity: {self.results.network_metrics_history[-1].get('modularity', 0):.3f}
- Bridge agents identified: {len(self.results.bridge_agents)}

INFLUENCE PATTERNS:
- Most influential agents: {self.results.most_influential_agents[:3]}
- Most polarized agents: {self.results.most_polarized_agents[:3]}
- Total interactions recorded: {len(self.results.interaction_history)}

CONCLUSIONS:
- {'Strong' if self.results.polarization_over_time[-1] > 0.7 else 'Moderate' if self.results.polarization_over_time[-1] > 0.4 else 'Weak'} polarization observed
- {'High' if len(self.results.final_echo_chambers) > self.config.num_agents // 8 else 'Moderate' if len(self.results.final_echo_chambers) > 2 else 'Low'} echo chamber formation
- Network showed {'high' if self.results.network_metrics_history[-1].get('modularity', 0) > 0.3 else 'low'} community structure
"""

        if self.config.intervention_type:
            pre_intervention_polarization = self.results.polarization_over_time[self.config.intervention_round - 1] if self.config.intervention_round > 0 else 0
            post_intervention_polarization = self.results.polarization_over_time[-1]
            intervention_effect = post_intervention_polarization - pre_intervention_polarization
            
            report += f"""
INTERVENTION ANALYSIS:
- Pre-intervention polarization: {pre_intervention_polarization:.3f}
- Post-intervention polarization: {post_intervention_polarization:.3f}
- Intervention effect: {intervention_effect:+.3f}
- Effectiveness: {'Positive' if intervention_effect < -0.05 else 'Negative' if intervention_effect > 0.05 else 'Neutral'}
"""

        return report