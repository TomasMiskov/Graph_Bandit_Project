# GYM
import numpy as np
import networkx as nx

class GraphBandit:
    
    def __init__(self, G, init_weights = None, init_node = (0, 0), target_node = (7, 7), seed = 101010):
        
        # Graph
        self.G = G.copy()
        
        # Size
        self.nV = len(self.G.nodes)
        self.nE = len(self.G.edges)
        self.gridSize = int(self.nV**.5)
        
        # Weights & UCB Params
        if init_weights == None:
            np.random.seed(seed)
            init_weights = np.random.uniform(1, 10, size = self.nE)
        else:
            assert len(init_weights) == self.nE, f"Weights don't match the number of edges. Provide {self.nE} weights"
        
        for i, e in enumerate(self.G.edges):
            self.G.edges[e]["weight"] = init_weights[i]
            self.G.edges[e]["max_traffic"] = np.random.uniform(5, 20)
            self.G.edges[e]["rewards"] = [-init_weights[i]]
            self.G.edges[e]["ucb"] = -init_weights[i]
            self.G.edges[e]["expected_reward"] = -(init_weights[i] + self.G.edges[e]["max_traffic"]/2)
        
        # Current & target position
        self.pos = init_node
        self.source = init_node
        self.target = target_node
        self.current_path = []    # holds all edges taken from source to target
        
        # Gym State Tracking
        self.optimal_path_nodes = nx.dijkstra_path(self.G, self.source, self.target, 
                                                   lambda i, j, edge_dict: -edge_dict["expected_reward"])
        self.optimal_path_edges = [(self.optimal_path_nodes[i], self.optimal_path_nodes[i + 1]) 
                                   for i in range(len(self.optimal_path_nodes) - 1)]
        self.init_optimal_path = nx.dijkstra_path(self.G, self.source, self.target, "weight")
        self.visited_nodes = [init_node]
        self.visited_edges = []
        self.paths = []
        self.visited_expected_rewards = []
        
    def step(self, node):
        """
        Function that moves an angent in the environment along its chosen edge.
        It updates the state of the agent, collects the reward, and stores the updated parameters.
        """
        # Denote edge to be taken
        e = (self.pos, node)
        self.visited_nodes.append(node)
        self.visited_edges.append(e)
        self.current_path.append(e)
        self.visited_expected_rewards.append(self.G.edges[e]["expected_reward"])
        
        # Move along it
        self.pos = node
        
        # Sample reward
        reward = self.G.edges[e]["weight"] + np.random.uniform(0, self.G.edges[e]["max_traffic"])
        self.G.edges[e]["rewards"].append(-reward)  
        
        # Reset position if target is reached
        if self.pos == self.target:
            self.pos = self.source
            self.paths.append(self.current_path)
            self.current_path = []
            
    def pathReward(self, path_edges):
        """
        Calculate the expected reward of an arbitrary path in the graph G
        """
        return sum([self.G.edges[e]["expected_reward"] for e in path_edges])
    
    def expectedRegret(self):
        """
        Calculate the expected regret of all taken paths by the agent at the end of the simulation
        """
        optimal_reward = self.pathReward(self.optimal_path_edges)
        realized_expected_rewards = np.array([self.pathReward(path) for path in self.paths])
        
        return optimal_reward - realized_expected_rewards