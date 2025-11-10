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
    """Solve TSP using Branch and Bound"""
    n = len(cities)
    nodes_explored = []
    best_cost = float('inf')
    best_path = None
    
    # Initialize root node
    initial_matrix = cost_matrix.copy()
    initial_bound = calculate_bound(initial_matrix, [])
    root = TSPNode(0, [0], initial_bound, initial_matrix)
    
    queue = deque([root])
    step = 0
    
    while queue:
        current = queue.popleft()
        nodes_explored.append(current)
        
        if current.bound >= best_cost:
            current.pruned = True
            continue
        
        if current.level == n - 1:
            # Complete path
            total_cost = sum(cost_matrix[current.path[i]][current.path[i+1]] 
                           for i in range(len(current.path) - 1))
            total_cost += cost_matrix[current.path[-1]][current.path[0]]
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = current.path + [current.path[0]]
            continue
        
        # Generate children
        for next_city in range(n):
            if next_city not in current.path:
                new_path = current.path + [next_city]
                new_matrix = current.cost_matrix.copy()
                
                # Set appropriate entries to infinity
                if len(current.path) > 0:
                    new_matrix[current.path[-1], :] = np.inf
                    new_matrix[:, next_city] = np.inf
                    new_matrix[next_city, current.path[0]] = np.inf
                
                new_bound = current.bound + calculate_bound(new_matrix, new_path)
                
                if new_bound < best_cost:
                    child = TSPNode(current.level + 1, new_path, new_bound, new_matrix)
                    queue.append(child)
    
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

def show_tsp_bb():
    st.header("TSP (Traveling Salesman Problem) - Branch and Bound")
    st.markdown("""
    Solve TSP using Branch and Bound algorithm. The algorithm explores the solution 
    space systematically, using cost matrix reduction to calculate lower bounds and 
    prune suboptimal paths.
    """)
    
    # Initialize session state for step-by-step
    if 'tsp_step' not in st.session_state:
        st.session_state.tsp_step = 0
    if 'tsp_auto_run' not in st.session_state:
        st.session_state.tsp_auto_run = False
    if 'tsp_nodes' not in st.session_state:
        st.session_state.tsp_nodes = []
    if 'tsp_best_path' not in st.session_state:
        st.session_state.tsp_best_path = None
    if 'tsp_best_cost' not in st.session_state:
        st.session_state.tsp_best_cost = float('inf')
    
    st.subheader("Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        num_cities = st.number_input("Number of Cities", min_value=3, max_value=8, value=5)
        random_seed = st.number_input("Random Seed", value=42)
    
    with col2:
        st.write("**City Coordinates**")
        use_random = st.checkbox("Use Random Cities", value=True)
    
    if use_random:
        np.random.seed(int(random_seed))
        cities = [(np.random.rand() * 100, np.random.rand() * 100) for _ in range(num_cities)]
    else:
        cities = []
        st.write("Enter coordinates manually:")
        for i in range(num_cities):
            col_x, col_y = st.columns(2)
            with col_x:
                x = st.number_input(f"City {i} X", value=i*20, key=f"x{i}")
            with col_y:
                y = st.number_input(f"City {i} Y", value=i*15, key=f"y{i}")
            cities.append((x, y))
    
    # Calculate distance matrix
    cost_matrix = calculate_distance_matrix(cities)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Run Algorithm", type="primary"):
            st.session_state.tsp_step = 0
            st.session_state.tsp_auto_run = False
            best_path, best_cost, nodes = tsp_branch_bound(cities, cost_matrix)
            st.session_state.tsp_nodes = nodes
            st.session_state.tsp_best_path = best_path
            st.session_state.tsp_best_cost = best_cost
            st.session_state.tsp_step = len(nodes)
            st.rerun()
    
    with col2:
        if st.button("Next Step") and st.session_state.tsp_nodes:
            if st.session_state.tsp_step < len(st.session_state.tsp_nodes):
                st.session_state.tsp_step += 1
                st.rerun()
    
    with col3:
        if st.button("Auto Run") and st.session_state.tsp_nodes:
            st.session_state.tsp_auto_run = True
            st.rerun()
    
    if st.session_state.tsp_nodes:
        nodes_to_show = st.session_state.tsp_nodes[:st.session_state.tsp_step]
        
        if st.session_state.tsp_step > 0:
            current_node = st.session_state.tsp_nodes[st.session_state.tsp_step - 1]
            
            st.subheader("Current Step Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Level", current_node.level)
            with col2:
                st.metric("Path", str(current_node.path))
            with col3:
                st.metric("Bound", f"{current_node.bound:.2f}")
            
            if st.session_state.tsp_best_path:
                st.success(f"Best Path Found: {st.session_state.tsp_best_path}")
                st.success(f"Best Cost: {st.session_state.tsp_best_cost:.2f}")
        
        # Visualization: Tree
        st.subheader("Branch and Bound Tree")
        if len(nodes_to_show) > 0:
            fig = visualize_tsp_tree(nodes_to_show)
            st.pyplot(fig)
        
        # Visualization: Cost Matrix Reduction
        if st.session_state.tsp_step > 0:
            st.subheader("Cost Matrix (Current Node)")
            current_node = st.session_state.tsp_nodes[st.session_state.tsp_step - 1]
            
            # Display cost matrix
            matrix_df = pd.DataFrame(current_node.cost_matrix)
            matrix_df.index = [f"City {i}" for i in range(len(cities))]
            matrix_df.columns = [f"City {i}" for i in range(len(cities))]
            st.dataframe(matrix_df, use_container_width=True)
        
        # Final path visualization
        if st.session_state.tsp_best_path:
            st.subheader("Optimal Path")
            fig = visualize_tsp_path(cities, st.session_state.tsp_best_path, 
                                    st.session_state.tsp_best_cost, 
                                    "Optimal TSP Solution")
            st.pyplot(fig)
        
        # Show pruning statistics
        if len(nodes_to_show) > 0:
            pruned_count = sum(1 for n in nodes_to_show if n.pruned)
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes Explored", len(nodes_to_show))
            with col2:
                st.metric("Nodes Pruned", pruned_count)
            with col3:
                st.metric("Nodes Remaining", len(nodes_to_show) - pruned_count)

