import numpy as np
import networkx as nx

class GraphBandit:
    
    def __init__(self, G, init_weights = None, init_node = (0, 0), target_node = (7, 7)):
        
        # Graph
        self.G = G.copy()
        
        # Size
        self.nV = len(self.V)
        self.nE = len(self.E)
        
        # Weights & UCB Params
        if init_weights == None:
            init_weights = np.random.uniform(1, 10, size = self.nE)
        else:
            assert len(init_weights) == self.nE, f"Weights don't match the number of edges. Provide {self.nE} weights"
        
        for i, e in enumerate(self.G.edges):
            self.G.edges[e]["weight"] = init_weights[i]
            self.G.edges[e]["max_traffic"] = np.random.uniform(0, 10)
            self.G.edges[e]["rewards"] = [-init_weights[i]]
            self.G.edges[e]["ucb"] = -init_weights[i]
        
        # Current & target position
        self.pos = init_node
        self.source = init_node
        self.target = target_node
        
        # Gym State Tracking
        self.init_weights = {e: self.G.edges[e]["weight"] for e in self.G.edges}
        self.expected_rewards = {e: self.G.edges[e]["weight"] + self.G.edges[e]["max_traffic"]/2 for e in self.G.edges}
        self.visited_nodes = [init_node]
        self.visited_edges = []
        self.visited_expected_rewards = []
        
    def step(self, node):
        # Denote edge to be taken
        e = (self.pos, node)
        self.visited_nodes.append(node)
        self.visited_edges.append(e)
        self.visited_expected_rewards.append(self.init_weights[e])
        
        # Move along it
        self.pos = node
        
        # Sample reward
        reward = self.G.edges[e]["weight"] + np.random.uniform(0, self.G.edges[e]["max_traffic"])
        self.G.edges[e]["rewards"].append(-reward)  
        
        # Reset position if target is reached
        if self.pos == self.target:
            self.pos = self.source 

