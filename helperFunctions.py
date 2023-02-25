import numpy as np
import pandas as pd

def cornerPathCol(col, grid_size):
    """
    Function to generate structured paths from lower-left corner 
    to the upper-right corner of a rectangular grid graph following columns
    
    Inputs:
    col: column in which the path takes its first corner
    grid_size: the size of the rectangular grid graph
    """
    columns = grid_size - 1
    start = [((i, 0), (i + 1, 0)) for i in range(col)]
    up = [((col, i), (col, i + 1)) for i in range(columns - col)]
    right = [((i, columns - col), (i + 1, columns - col)) for i in range(col, columns)]
    end = [((columns, i), (columns, i + 1)) for i in range(columns - col, columns)]
    
    return start + up + right + end

def cornerPathRow(row, grid_size):
    """
    Function to generate structured paths from lower-left corner 
    to the upper-right corner of a rectangular grid graph following rows
    
    Inputs:
    col: column in which the path takes its first corner
    grid_size: the size of the rectangular grid graph
    """
    rows = grid_size - 1
    start = [((0, i), (0, i + 1)) for i in range(row)]
    right = [((i, row), (i + 1, row)) for i in range(rows - row)]
    up = [((rows - row, i), (rows - row, i + 1)) for i in range(row, rows)]
    end = [((i, rows), (i + 1, rows)) for i in range(rows - row, rows)]
    
    return start + right + up + end

def samplingPaths(grid_size):
    """
    Function that generates a list of paths that span all the edges
    in a rectangular grid graph of size *grid_size*
    
    Inputs:
    grid_size: the size of the rectangular grid graph
    """
    paths = []
    for i in range(grid_size):
        paths.append(cornerPathCol(i, grid_size))
        paths.append(cornerPathRow(i, grid_size))
        
    return paths

def updateUCB(env, edges = None, c = 2):
    """
    Function to update the current UCB indices
    
    Input:
    edges: edge list to update the UCB indices for. If none, all edges in the graph are updated
    """
    if edges is None:
        edges = env.G.edges
        
    edges_visits = np.array([len(env.G.edges[e]["rewards"]) for e in edges])
    edges_total_rew = np.array([sum(env.G.edges[e]["rewards"]) for e in edges])
    edges_avg_rew = edges_total_rew/edges_visits
        
    T = len(env.visited_edges) + 1
    
    ucb = edges_avg_rew - np.sqrt(c * np.log(T) / edges_visits)
        
    for i, e in enumerate(edges):
        env.G.edges[e]["ucb"] = ucb[i]  
        
def encodePath(path):
    """
    Function that takes in a path made up of edge tuples and encodes it as a string of nodes
    """
    path_string = "00"
    for edge in path:
        path_string += str(edge[1][0]) + str(edge[1][1])
    return path_string

def decodePath(path_string):
    """
    Function that takes in a tring of nodes and decodes it into a path of edge tuples
    """
    ps = path_string
    path = [((int(ps[i]), int(ps[i+1])), (int(ps[i+2]), int(ps[i+3]))) for i in range(0, len(ps) - 2, 2)]
    return path

def countUniquePaths(env):
    """
    Function that counts the number of unique paths taken by the agent
    """
    encoded_paths = []
    for path in env.paths:
        encoded_paths.append(encodePath(path))

    arr_paths = np.array(encoded_paths)
    df_paths = pd.DataFrame(np.unique(arr_paths, return_counts = True), index = ["path", "counter"]).T
    df_paths["path_edges"] = df_paths.path.map(decodePath)
    df_paths.sort_values("counter", ignore_index = True, inplace = True)
        
    return df_paths  