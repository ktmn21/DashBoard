import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque
import time

def calculate_distance_matrix(cities):
    """Calculate distance matrix for cities"""
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = np.sqrt((cities[i][0] - cities[j][0])**2 + 
                                           (cities[i][1] - cities[j][1])**2)
    return dist_matrix

def reduce_matrix(matrix):
    """Reduce matrix and return reduction cost"""
    n = len(matrix)
    cost = 0
    reduced = matrix.copy()
    
    # Reduce rows
    for i in range(n):
        row = reduced[i]
        valid_values = row[row != np.inf]
        if len(valid_values) > 0:
            row_min = np.min(valid_values)
            if row_min != np.inf and row_min > 0:
                cost += row_min
                reduced[i] = np.where(reduced[i] != np.inf, reduced[i] - row_min, np.inf)
    
    # Reduce columns
    for j in range(n):
        col = reduced[:, j]
        valid_values = col[col != np.inf]
        if len(valid_values) > 0:
            col_min = np.min(valid_values)
            if col_min != np.inf and col_min > 0:
                cost += col_min
                reduced[:, j] = np.where(reduced[:, j] != np.inf, 
                                        reduced[:, j] - col_min, np.inf)
    
    return reduced, cost

def calculate_bound(matrix, path):
    """Calculate lower bound for current path"""
    n = len(matrix)
    temp_matrix = matrix.copy()
    
    # Set visited rows and columns to infinity
    if len(path) > 0:
        for i in range(len(path) - 1):
            temp_matrix[path[i], :] = np.inf
            temp_matrix[:, path[i+1]] = np.inf
        temp_matrix[path[-1], path[0]] = np.inf
    
    reduced, cost = reduce_matrix(temp_matrix)
    return cost

class TSPNode:
    def __init__(self, level, path, bound, cost_matrix):
        self.level = level
        self.path = path
        self.bound = bound
        self.cost_matrix = cost_matrix
        self.pruned = False

def tsp_branch_bound(cities, cost_matrix):
    """Solve TSP using Branch and Bound with step visualization"""
    n = len(cities)
    nodes_explored = []
    best_cost = float('inf')
    best_path = None
    step = 1
    
    # Initialize root node
    initial_matrix = cost_matrix.copy()
    st.write("# Branch and Bound Steps")
    
    # Root node reduction
    st.write("## Root Node")
    initial_bound = show_node_reductions(initial_matrix, [], step)
    root = TSPNode(0, [0], initial_bound, initial_matrix)
    
    queue = deque([root])
    nodes_explored.append(root)
    
    # Show initial tree
    st.write(f"### Tree after Step {step}")
    tree_fig = visualize_tsp_tree(nodes_explored)
    st.pyplot(tree_fig)
    
    while queue:
        current = queue.popleft()
        step += 1
        
        if current.bound >= best_cost:
            current.pruned = True
            st.write(f"## Step {step}: Pruning node with path {current.path}")
            st.write(f"Bound ({current.bound:.2f}) ≥ Best cost ({best_cost:.2f})")
            continue
        
        if current.level == n - 1:
            # Complete path found
            total_cost = sum(cost_matrix[current.path[i]][current.path[i+1]] 
                           for i in range(len(current.path) - 1))
            total_cost += cost_matrix[current.path[-1]][current.path[0]]
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = current.path + [current.path[0]]
                st.write(f"## Step {step}: New best solution found!")
                st.write(f"Path: {best_path}")
                st.write(f"Cost: {best_cost:.2f}")
            continue
        
        # Generate children
        st.write(f"## Step {step}: Expanding node with path {current.path}")
        for next_city in range(n):
            if next_city not in current.path:
                new_path = current.path + [next_city]
                new_matrix = current.cost_matrix.copy()
                
                # Show matrix reductions for this node
                new_bound = current.bound + show_node_reductions(new_matrix, new_path, step)
                
                if new_bound < best_cost:
                    child = TSPNode(current.level + 1, new_path, new_bound, new_matrix)
                    queue.append(child)
                    nodes_explored.append(child)
                    st.write(f"Added node with path {new_path} and bound {new_bound:.2f}")
                else:
                    st.write(f"Pruned node with path {new_path} (bound {new_bound:.2f} ≥ {best_cost:.2f})")
        
        # Show updated tree
        st.write(f"### Tree after Step {step}")
        tree_fig = visualize_tsp_tree(nodes_explored)
        st.pyplot(tree_fig)
    
    return best_path, best_cost, nodes_explored

def visualize_tsp_tree(nodes_explored, max_nodes=30):
    """Visualize branch and bound tree"""
    G = nx.DiGraph()
    
    nodes_to_show = nodes_explored[:max_nodes]
    
    for i, node in enumerate(nodes_to_show):
        label = f"L{node.level}\nP:{node.path}\nB:{node.bound:.1f}"
        if node.pruned:
            label += "\n[PRUNED]"
        G.add_node(i, label=label, level=node.level, bound=node.bound, 
                   path=str(node.path), pruned=node.pruned)
        
        # Connect to parent
        if i > 0:
            parent_idx = max([j for j in range(i) if nodes_to_show[j].level < node.level], 
                           default=0)
            G.add_edge(parent_idx, i)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Color nodes
    node_colors = ['red' if G.nodes[i]['pruned'] else 'lightblue' for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=15, ax=ax)
    
    # Labels
    labels = {i: G.nodes[i]['label'] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    
    ax.set_title("Branch and Bound Exploration Tree\n(L=Level, P=Path, B=Bound)", 
                fontsize=14, weight='bold')
    ax.axis('off')
    
    return fig

def visualize_tsp_path(cities, path, cost, title="TSP Solution"):
    """Visualize TSP path"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_coords = [c[0] for c in cities]
    y_coords = [c[1] for c in cities]
    ax.scatter(x_coords, y_coords, c='red', s=200, zorder=3)
    
    for i, (x, y) in enumerate(cities):
        ax.annotate(f'City {i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=10, weight='bold')
    
    path_x = [cities[i][0] for i in path]
    path_y = [cities[i][1] for i in path]
    ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, zorder=1)
    ax.plot(path_x, path_y, 'bo', markersize=10, zorder=2)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'{title}\nTotal Distance: {cost:.2f}', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig

def parse_edge_input(input_str):
    """Parse edge input string into list of (destination, weight) tuples"""
    if not input_str.strip():
        return []
    edges = []
    pairs = input_str.strip().split()
    for pair in pairs:
        if not pair.startswith('(') or not pair.endswith(')'):
            continue
        try:
            dest, weight = pair[1:-1].split(',')
            edges.append((int(dest), float(weight)))
        except:
            continue
    return edges

def create_cost_matrix(n, edge_inputs, directed=False):
    """Create cost matrix from edge inputs"""
    matrix = np.full((n, n), np.inf)
    np.fill_diagonal(matrix, 0)
    
    for i in range(n):
        edges = edge_inputs[i]
        for dest, weight in edges:
            if dest-1 != i:  # Skip self-loops
                matrix[i][dest-1] = weight
                if not directed:  # For undirected graphs
                    matrix[dest-1][i] = weight
    return matrix

def show_reduction_step(matrix, step_type, reduction_values):
    """Show matrix reduction step"""
    df = pd.DataFrame(matrix)
    df = df.replace(np.inf, '∞')
    
    if step_type == "row":
        st.write("Row Reduction:")
        for i, val in enumerate(reduction_values):
            if val > 0:
                st.write(f"Row {i+1}: subtract {val:.2f}")
    else:
        st.write("Column Reduction:")
        for i, val in enumerate(reduction_values):
            if val > 0:
                st.write(f"Column {i+1}: subtract {val:.2f}")
    
    st.dataframe(df)
    return df

# Add new function to show matrix reductions for each node
def show_node_reductions(matrix, current_path, step_number):
    """Show row and column reductions for current node"""
    st.write(f"### Step {step_number}: Node with Path {current_path}")
    
    # Copy matrix and set visited paths to infinity
    temp_matrix = matrix.copy()
    if len(current_path) > 0:
        for i in range(len(current_path) - 1):
            temp_matrix[current_path[i], :] = np.inf
            temp_matrix[:, current_path[i+1]] = np.inf
        if len(current_path) > 1:
            temp_matrix[current_path[-1], current_path[0]] = np.inf
    
    # Show initial matrix
    st.write("Initial Matrix:")
    initial_df = pd.DataFrame(temp_matrix)
    initial_df = initial_df.replace(np.inf, '∞')
    st.dataframe(initial_df)
    
    # Row reduction
    row_reductions = []
    for i in range(len(matrix)):
        row = temp_matrix[i]
        valid_values = row[row != np.inf]
        if len(valid_values) > 0:
            row_min = np.min(valid_values)
            if row_min != np.inf and row_min > 0:
                temp_matrix[i] = np.where(temp_matrix[i] != np.inf, 
                                        temp_matrix[i] - row_min, np.inf)
                row_reductions.append((i, row_min))
    
    if row_reductions:
        st.write("After Row Reduction:")
        for i, red in row_reductions:
            st.write(f"Row {i+1}: subtract {red:.2f}")
        row_df = pd.DataFrame(temp_matrix)
        row_df = row_df.replace(np.inf, '∞')
        st.dataframe(row_df)
    
    # Column reduction
    col_reductions = []
    for j in range(len(matrix)):
        col = temp_matrix[:, j]
        valid_values = col[col != np.inf]
        if len(valid_values) > 0:
            col_min = np.min(valid_values)
            if col_min != np.inf and col_min > 0:
                temp_matrix[:, j] = np.where(temp_matrix[:, j] != np.inf, 
                                           temp_matrix[:, j] - col_min, np.inf)
                col_reductions.append((j, col_min))
    
    if col_reductions:
        st.write("After Column Reduction:")
        for j, red in col_reductions:
            st.write(f"Column {j+1}: subtract {red:.2f}")
        final_df = pd.DataFrame(temp_matrix)
        final_df = final_df.replace(np.inf, '∞')
        st.dataframe(final_df)
    
    total_reduction = sum(r for _, r in row_reductions) + sum(r for _, r in col_reductions)
    st.write(f"Total Reduction Cost: {total_reduction:.2f}")
    return total_reduction

def show_tsp_bb():
    st.header("Traveling Salesman Problem - Branch and Bound")
    st.markdown("""
    Create your custom graph by specifying edges and weights for each city.
    Format: (destination,weight) (destination,weight) ...
    Example for City 2: (3,32) (5,7) (1,14) means:
    - Edge from City 2 to City 3 with cost 32
    - Edge from City 2 to City 5 with cost 7
    - Edge from City 2 to City 1 with cost 14
    """)
    
    # Input for number of cities
    n_cities = st.number_input("Number of Cities", min_value=2, max_value=10, value=4)
    
    # Directed/Undirected toggle
    is_directed = st.toggle("Directed Graph", value=False)
    
    # Edge inputs for each city
    st.subheader("Edge Inputs")
    edge_inputs = []
    cols = st.columns(2)
    for i in range(n_cities):
        with cols[i % 2]:
            edge_input = st.text_input(
                f"City {i+1} edges", 
                help=f"Enter edges from city {i+1} as (dest,weight) pairs",
                key=f"city_{i}"
            )
            edge_inputs.append(parse_edge_input(edge_input))
    
    if st.button("Process and Solve", type="primary"):
        # Create initial cost matrix
        cost_matrix = create_cost_matrix(n_cities, edge_inputs, is_directed)
        
        # Show initial matrix
        st.subheader("Initial Cost Matrix")
        initial_df = pd.DataFrame(
            cost_matrix,
            index=[f"City {i+1}" for i in range(n_cities)],
            columns=[f"City {i+1}" for i in range(n_cities)]
        )
        initial_df = initial_df.replace(np.inf, '∞')
        st.dataframe(initial_df)
        
        # Row Reduction Step
        st.subheader("Row Reduction")
        reduced_matrix = cost_matrix.copy()
        row_reductions = []
        
        for i in range(n_cities):
            row = reduced_matrix[i]
            valid_values = row[row != np.inf]
            if len(valid_values) > 0:
                row_min = np.min(valid_values)
                if row_min != np.inf and row_min > 0:
                    reduced_matrix[i] = np.where(reduced_matrix[i] != np.inf, 
                                              reduced_matrix[i] - row_min, np.inf)
                    row_reductions.append((i, row_min))
        
        # Show row reduction details
        for i, reduction in row_reductions:
            st.write(f"Row {i+1}: subtract {reduction:.2f}")
        
        row_reduced_df = pd.DataFrame(
            reduced_matrix,
            index=[f"City {i+1}" for i in range(n_cities)],
            columns=[f"City {i+1}" for i in range(n_cities)]
        )
        row_reduced_df = row_reduced_df.replace(np.inf, '∞')
        st.dataframe(row_reduced_df)
        
        # Column Reduction Step
        st.subheader("Column Reduction")
        col_reductions = []
        
        for j in range(n_cities):
            col = reduced_matrix[:, j]
            valid_values = col[col != np.inf]
            if len(valid_values) > 0:
                col_min = np.min(valid_values)
                if col_min != np.inf and col_min > 0:
                    reduced_matrix[:, j] = np.where(reduced_matrix[:, j] != np.inf, 
                                                 reduced_matrix[:, j] - col_min, np.inf)
                    col_reductions.append((j, col_min))
        
        # Show column reduction details
        for j, reduction in col_reductions:
            st.write(f"Column {j+1}: subtract {reduction:.2f}")
        
        final_reduced_df = pd.DataFrame(
            reduced_matrix,
            index=[f"City {i+1}" for i in range(n_cities)],
            columns=[f"City {i+1}" for i in range(n_cities)]
        )
        final_reduced_df = final_reduced_df.replace(np.inf, '∞')
        st.dataframe(final_reduced_df)
        
        # Calculate initial lower bound
        initial_bound = sum(red for _, red in row_reductions) + sum(red for _, red in col_reductions)
        st.success(f"Initial Lower Bound = {initial_bound:.2f}")
        
        # Solve TSP using Branch and Bound
        best_path, best_cost, nodes_explored = tsp_branch_bound(range(n_cities), cost_matrix)
        
        if best_path:
            st.success(f"Optimal Solution Found!")
            path_cities = [i+1 for i in best_path]
            st.write(f"Path: {' → '.join(map(str, path_cities))}")
            st.write(f"Total Cost: {best_cost:.2f}")
            
            # Visualize the solution
            st.subheader("Solution Visualization")
            G = nx.DiGraph() if is_directed else nx.Graph()
            
            # Add nodes and edges
            for i in range(n_cities):
                G.add_node(i+1)
                for dest, weight in edge_inputs[i]:
                    G.add_edge(i+1, dest, weight=weight)
            
            # Draw solution
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G)
            
            # Draw original graph in light gray
            nx.draw(G, pos, with_labels=True, node_color='lightgray',
                   edge_color='lightgray', node_size=500)
            
            # Draw solution path in red
            path_edges = list(zip(path_cities[:-1], path_cities[1:]))
            solution_graph = nx.DiGraph() if is_directed else nx.Graph()
            solution_graph.add_edges_from(path_edges)
            nx.draw_networkx_edges(solution_graph, pos, edge_color='red',
                                 width=2)
            
            st.pyplot(fig)
            
            # Show Branch and Bound tree
            st.subheader("Branch and Bound Tree")
            tree_fig = visualize_tsp_tree(nodes_explored)
            st.pyplot(tree_fig)
        else:
            st.error("No valid solution found! Graph may be disconnected.")

# Add to requirements.txt:
# networkx
# matplotlib
# numpy
# pandas

