# AGENTS
import networkx as nx
import numpy as np
import helperFunctions as helpers

def everyEdgeAgent(env):
    """
    Agent that updates UCB indices and recomputes the most optimal path 
    at EVERY traveresed edge 
    """
    helpers.updateUCB(env)
    shortest_path = nx.dijkstra_path(env.G, env.pos, env.target, lambda i, j, edge_dict: -edge_dict["ucb"])
    env.step(shortest_path[1])
    
def everyPathAgent(env, c = 2):
    """
    Agent that updates UCB indices and recomputes the most optimal path 
    at EVERY traveresed path from source to target
    """
    helpers.updateUCB(env, c = c)
    shortest_path = nx.dijkstra_path(env.G, env.pos, env.target, lambda i, j, edge_dict: -edge_dict["ucb"])
    for node in shortest_path[1:]:
        env.step(node)
        
def samplingAgentSample(env, k_samples, verbose = False):
    """
    Sampling part of the agent that first observers at least k reward samples of every edge 
    and then exploits the most optimal path based on the sample average weights
    """
    paths = helpers.samplingPaths(grid_size = env.gridSize)
    
    if verbose:
        print(f"To get {k_samples} samples of every edge you will need to send {k_samples * len(paths)} messages")
        print(f"Make sure the number of bits/messages you are planning to send is lower than the {k_samples * len(paths)} messages needed for sampling")
    
    for i in range(k_samples):
        for path in paths:
            for edge in path:                
                env.step(edge[1])
        
    for e in env.G.edges:
        env.G.edges[e]["k_avg_reward"] = np.array(env.G.edges[e]["rewards"]).mean()
        
    env.k_sample_optimal_path = nx.dijkstra_path(env.G, env.source, env.target, 
                                                 lambda i, j, edge_dict: -edge_dict["k_avg_reward"])

def samplingAgentExecute(env):
    """
    Execution part of the agent that first observers at least k reward samples of every edge 
    and then exploits the most optimal path based on the sample average weights
    """
    for node in env.k_sample_optimal_path[1:]:
        env.step(node)
    
def locallyDirectedAgent(env):
    """
    Agent that knows the general direction from source to target.
    At each node it chooses the next edge based on the UCB values, conditional on the general direction.
    I.e. if going from lower-left node to the upper-right node, the agent only considers the edges going up or right
    at each crossroad
    """
    helpers.updateUCB(env)
    
    directed_neighboring_nodes = []
    if env.pos[0] < (env.gridSize - 1) and env.pos[1] < (env.gridSize - 1):
        directed_neighboring_nodes.append((env.pos[0] + 1, env.pos[1]))
        directed_neighboring_nodes.append((env.pos[0], env.pos[1] + 1))
    elif env.pos[0] < (env.gridSize - 1) and env.pos[1] == (env.gridSize - 1):
        directed_neighboring_nodes.append((env.pos[0] + 1, env.pos[1]))
    elif env.pos[0] == (env.gridSize - 1) and env.pos[1] < (env.gridSize - 1):
        directed_neighboring_nodes.append((env.pos[0], env.pos[1] + 1))
            
    edges_to_choose_from = [(env.pos, node) for node in directed_neighboring_nodes]
    ucb_values = np.array([env.G.edges[e]["ucb"] for e in edges_to_choose_from])
    
    chosen_edge = edges_to_choose_from[np.argmax(ucb_values)]
    
    env.step(chosen_edge[1])

def staticWorldAgent(env):
    """
    Agent that ignores the fact that edge weights are stochastic and just exploits the most optimal path
    given the initialized weights.
    """
    for node in env.init_optimal_path[1:]:
        env.step(node)