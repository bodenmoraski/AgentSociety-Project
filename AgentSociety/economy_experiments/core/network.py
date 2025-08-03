"""
Social Network Formation and Management for Echo Chamber Experiments
"""

import random
import math
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from .agent import Agent, PersonalityType


@dataclass
class NetworkConfig:
    """Configuration for social network formation"""
    network_type: str = "preferential_attachment"  # "random", "small_world", "scale_free", "preferential_attachment"
    homophily_strength: float = 0.7  # 0-1, how much agents prefer similar others
    average_connections: int = 5
    rewiring_probability: float = 0.1  # For small-world networks
    bridge_probability: float = 0.05  # Probability of cross-group connections
    dynamic_rewiring: bool = True  # Whether connections change over time


class SocialNetwork:
    """Manages the social network structure and dynamics"""
    
    def __init__(self, agents: List[Agent], config: NetworkConfig):
        self.agents = {agent.id: agent for agent in agents}
        self.config = config
        self.connections: Dict[int, Set[int]] = {agent.id: set() for agent in agents}
        self.connection_history: List[Dict[int, Set[int]]] = []
        self.time_step = 0
        
        # NetworkX graph for analysis
        self.graph = nx.Graph()
        self.graph.add_nodes_from([agent.id for agent in agents])
        
        # Initialize network
        self._create_initial_network()
        self._update_graph()
    
    def _create_initial_network(self):
        """Create the initial social network based on configuration"""
        
        if self.config.network_type == "random":
            self._create_random_network()
        elif self.config.network_type == "small_world":
            self._create_small_world_network()
        elif self.config.network_type == "scale_free":
            self._create_scale_free_network()
        else:  # preferential_attachment (default)
            self._create_preferential_attachment_network()
    
    def _create_random_network(self):
        """Create random connections"""
        agents = list(self.agents.values())
        
        for agent in agents:
            num_connections = random.randint(2, self.config.average_connections * 2)
            potential_connections = [a for a in agents if a.id != agent.id]
            
            connections_made = 0
            while connections_made < num_connections and potential_connections:
                target = random.choice(potential_connections)
                if self._should_connect(agent, target):
                    self._add_connection(agent.id, target.id)
                    connections_made += 1
                potential_connections.remove(target)
    
    def _create_small_world_network(self):
        """Create small-world network with belief-based clustering"""
        agents = list(self.agents.values())
        n = len(agents)
        
        # Sort agents by belief for initial ring structure
        agents.sort(key=lambda a: a.belief_strength)
        
        # Create ring with local connections
        k = self.config.average_connections
        for i, agent in enumerate(agents):
            for j in range(1, k//2 + 1):
                neighbor_idx = (i + j) % n
                neighbor = agents[neighbor_idx]
                if self._should_connect(agent, neighbor):
                    self._add_connection(agent.id, neighbor.id)
        
        # Rewire some connections
        for agent in agents:
            connected_ids = list(self.connections[agent.id])
            for connected_id in connected_ids:
                if random.random() < self.config.rewiring_probability:
                    self._remove_connection(agent.id, connected_id)
                    # Find new random connection
                    potential_targets = [a for a in agents 
                                       if a.id != agent.id and a.id not in self.connections[agent.id]]
                    if potential_targets:
                        new_target = random.choice(potential_targets)
                        self._add_connection(agent.id, new_target.id)
    
    def _create_scale_free_network(self):
        """Create scale-free network using preferential attachment"""
        agents = list(self.agents.values())
        random.shuffle(agents)
        
        # Start with small complete graph
        initial_size = min(3, len(agents))
        for i in range(initial_size):
            for j in range(i + 1, initial_size):
                self._add_connection(agents[i].id, agents[j].id)
        
        # Add remaining agents with preferential attachment
        for i in range(initial_size, len(agents)):
            new_agent = agents[i]
            
            # Calculate connection probabilities based on degree and belief similarity
            connection_probs = []
            for existing_agent in agents[:i]:
                degree = len(self.connections[existing_agent.id])
                belief_similarity = 1 - abs(new_agent.belief_strength - existing_agent.belief_strength) / 2
                
                # Combine preferential attachment with homophily
                prob = (degree + 1) * (1 + self.config.homophily_strength * belief_similarity)
                connection_probs.append(prob)
            
            # Normalize probabilities
            total_prob = sum(connection_probs)
            if total_prob > 0:
                connection_probs = [p / total_prob for p in connection_probs]
                
                # Make connections
                num_connections = min(self.config.average_connections, i)
                for _ in range(num_connections):
                    if connection_probs:
                        target_idx = np.random.choice(i, p=connection_probs)
                        target_agent = agents[target_idx]
                        
                        if target_agent.id not in self.connections[new_agent.id]:
                            self._add_connection(new_agent.id, target_agent.id)
                            connection_probs[target_idx] = 0  # Avoid duplicate connections
                            connection_probs = [p / sum(connection_probs) if sum(connection_probs) > 0 else 0 
                                              for p in connection_probs]
    
    def _create_preferential_attachment_network(self):
        """Create network with homophily-based preferential attachment"""
        agents = list(self.agents.values())
        
        for agent in agents:
            # Calculate connection preferences for all other agents
            preferences = []
            for other_agent in agents:
                if other_agent.id != agent.id:
                    preference = self._calculate_connection_preference(agent, other_agent)
                    preferences.append((other_agent, preference))
            
            # Sort by preference and connect to top candidates
            preferences.sort(key=lambda x: x[1], reverse=True)
            
            # Number of connections varies by sociability
            base_connections = self.config.average_connections
            sociability_modifier = agent.sociability * 0.5  # Up to 50% more connections
            num_connections = int(base_connections * (1 + sociability_modifier))
            
            connections_made = 0
            for other_agent, preference in preferences:
                if connections_made >= num_connections:
                    break
                    
                # Probabilistic connection based on preference
                if random.random() < preference * 0.8:  # Scale down for realism
                    self._add_connection(agent.id, other_agent.id)
                    connections_made += 1
    
    def _calculate_connection_preference(self, agent1: Agent, agent2: Agent) -> float:
        """Calculate preference for connection between two agents"""
        
        # Belief similarity (homophily)
        belief_similarity = 1 - abs(agent1.belief_strength - agent2.belief_strength) / 2
        
        # Personality compatibility
        personality_bonus = 0.0
        if agent1.personality_type == PersonalityType.CONFORMIST:
            # Conformists prefer confident agents
            personality_bonus = agent2.confidence * 0.3
        elif agent1.personality_type == PersonalityType.CONTRARIAN:
            # Contrarians prefer independent types
            if agent2.personality_type == PersonalityType.INDEPENDENT:
                personality_bonus = 0.2
        elif agent1.personality_type == PersonalityType.AMPLIFIER:
            # Amplifiers prefer other social agents
            personality_bonus = agent2.sociability * 0.2
        
        # Sociability factor
        sociability_factor = (agent1.sociability + agent2.sociability) / 2
        
        # Network centrality attraction (popular agents attract more connections)
        centrality_bonus = agent2.network_centrality * 0.2
        
        # Combine factors
        base_preference = (
            self.config.homophily_strength * belief_similarity +
            (1 - self.config.homophily_strength) * sociability_factor
        )
        
        total_preference = base_preference + personality_bonus + centrality_bonus
        return min(1.0, max(0.0, total_preference))
    
    def _should_connect(self, agent1: Agent, agent2: Agent) -> bool:
        """Determine if two agents should be connected"""
        preference = self._calculate_connection_preference(agent1, agent2)
        return random.random() < preference
    
    def _add_connection(self, agent1_id: int, agent2_id: int):
        """Add bidirectional connection between two agents"""
        self.connections[agent1_id].add(agent2_id)
        self.connections[agent2_id].add(agent1_id)
    
    def _remove_connection(self, agent1_id: int, agent2_id: int):
        """Remove bidirectional connection between two agents"""
        self.connections[agent1_id].discard(agent2_id)
        self.connections[agent2_id].discard(agent1_id)
    
    def _update_graph(self):
        """Update NetworkX graph for analysis"""
        self.graph.clear_edges()
        for agent_id, connected_ids in self.connections.items():
            for connected_id in connected_ids:
                if agent_id < connected_id:  # Avoid duplicate edges
                    self.graph.add_edge(agent_id, connected_id)
    
    def update_network_dynamics(self, time_step: int):
        """Update network structure based on recent interactions"""
        if not self.config.dynamic_rewiring:
            return
            
        self.time_step = time_step
        
        # Save current state
        self.connection_history.append({
            agent_id: connected_ids.copy() 
            for agent_id, connected_ids in self.connections.items()
        })
        
        # Dynamic rewiring based on recent interactions
        for agent_id, agent in self.agents.items():
            if len(agent.recent_influences) > 0:
                
                # Strengthen connections with agents who influenced us positively
                for influencer_id, influence_amount in agent.recent_influences:
                    if abs(influence_amount) > 0.05:  # Significant influence
                        influencer = self.agents[influencer_id]
                        
                        # If influence was positive (moved beliefs closer), strengthen connection
                        belief_difference_before = abs(agent.belief_history[-2] - influencer.belief_strength) if len(agent.belief_history) > 1 else 1.0
                        belief_difference_after = abs(agent.belief_strength - influencer.belief_strength)
                        
                        if belief_difference_after < belief_difference_before:
                            # Positive influence - maintain/strengthen connection
                            if influencer_id not in self.connections[agent_id]:
                                if random.random() < 0.1:  # Small chance to form new connection
                                    self._add_connection(agent_id, influencer_id)
                        
                        # If influence was negative (created more distance), possibly break connection
                        elif belief_difference_after > belief_difference_before + 0.1:
                            if influencer_id in self.connections[agent_id]:
                                if random.random() < 0.05:  # Small chance to break connection
                                    self._remove_connection(agent_id, influencer_id)
        
        # Occasionally form new connections based on current preferences
        if random.random() < 0.1:  # 10% chance per time step
            self._form_new_connections()
        
        self._update_graph()
    
    def _form_new_connections(self):
        """Occasionally form new connections based on current belief similarity"""
        agents = list(self.agents.values())
        
        for _ in range(max(1, len(agents) // 20)):  # Try a few new connections
            agent1 = random.choice(agents)
            
            # Find agents not currently connected
            unconnected = [a for a in agents 
                          if a.id != agent1.id and a.id not in self.connections[agent1.id]]
            
            if unconnected:
                # Prefer agents with similar beliefs
                similarities = []
                for agent2 in unconnected:
                    similarity = 1 - abs(agent1.belief_strength - agent2.belief_strength) / 2
                    similarities.append(similarity)
                
                # Weighted random selection
                if similarities:
                    total_similarity = sum(similarities)
                    if total_similarity > 0:
                        probs = [s / total_similarity for s in similarities]
                        selected_idx = np.random.choice(len(unconnected), p=probs)
                        selected_agent = unconnected[selected_idx]
                        
                        if random.random() < 0.3:  # 30% chance to actually connect
                            self._add_connection(agent1.id, selected_agent.id)
    
    def get_connected_agents(self, agent_id: int) -> List[Agent]:
        """Get list of agents connected to the given agent"""
        connected_ids = self.connections[agent_id]
        return [self.agents[aid] for aid in connected_ids if aid in self.agents]
    
    def get_network_statistics(self) -> Dict[str, float]:
        """Calculate network statistics"""
        self._update_graph()
        
        if self.graph.number_of_edges() == 0:
            return {
                'density': 0.0,
                'clustering_coefficient': 0.0,
                'average_path_length': 0.0,
                'assortativity': 0.0,
                'modularity': 0.0,
                'num_components': len(self.agents)
            }
        
        # Basic network metrics
        density = nx.density(self.graph)
        clustering = nx.average_clustering(self.graph)
        
        # Path length (only for connected components)
        if nx.is_connected(self.graph):
            avg_path_length = nx.average_shortest_path_length(self.graph)
        else:
            # Average over connected components
            components = list(nx.connected_components(self.graph))
            path_lengths = []
            for component in components:
                if len(component) > 1:
                    subgraph = self.graph.subgraph(component)
                    path_lengths.append(nx.average_shortest_path_length(subgraph))
            avg_path_length = np.mean(path_lengths) if path_lengths else 0.0
        
        # Belief-based assortativity
        belief_values = {agent.id: agent.belief_strength for agent in self.agents.values()}
        nx.set_node_attributes(self.graph, belief_values, 'belief')
        
        try:
            assortativity = nx.attribute_assortativity_coefficient(self.graph, 'belief')
        except:
            assortativity = 0.0
        
        # Modularity (community structure)
        try:
            communities = nx.community.greedy_modularity_communities(self.graph)
            modularity = nx.community.modularity(self.graph, communities)
        except:
            modularity = 0.0
        
        return {
            'density': density,
            'clustering_coefficient': clustering,
            'average_path_length': avg_path_length,
            'assortativity': assortativity,
            'modularity': modularity,
            'num_components': nx.number_connected_components(self.graph)
        }
    
    def detect_echo_chambers(self, similarity_threshold: float = 0.3) -> List[List[int]]:
        """Detect echo chambers based on belief similarity and network structure"""
        self._update_graph()
        
        echo_chambers = []
        
        # Find connected components
        components = list(nx.connected_components(self.graph))
        
        for component in components:
            if len(component) < 3:  # Too small to be meaningful echo chamber
                continue
                
            # Check if component has high belief similarity
            component_agents = [self.agents[aid] for aid in component]
            beliefs = [agent.belief_strength for agent in component_agents]
            
            belief_std = np.std(beliefs)
            if belief_std < similarity_threshold:
                # This is an echo chamber
                echo_chambers.append(list(component))
        
        return echo_chambers
    
    def get_bridge_agents(self) -> List[int]:
        """Identify agents who bridge between different belief groups"""
        self._update_graph()
        
        bridge_agents = []
        
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(self.graph)
        
        # Agents with high betweenness and diverse connections
        for agent_id, centrality in betweenness.items():
            if centrality > 0.1:  # High betweenness
                agent = self.agents[agent_id]
                connected_agents = self.get_connected_agents(agent_id)
                
                if len(connected_agents) > 0:
                    # Check belief diversity of connections
                    connected_beliefs = [ca.belief_strength for ca in connected_agents]
                    belief_diversity = np.std(connected_beliefs)
                    
                    if belief_diversity > 0.4:  # High diversity
                        bridge_agents.append(agent_id)
        
        return bridge_agents