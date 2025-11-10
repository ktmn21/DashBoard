import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
import numpy as np
import pandas as pd
import random

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

def create_distance_matrix(n, edge_inputs, directed=False):
    """Create distance matrix from edge inputs"""
    matrix = np.full((n, n), float('inf'))
    np.fill_diagonal(matrix, 0)
    
    for i in range(n):
        edges = edge_inputs[i]
        for dest, weight in edges:
            if dest-1 != i:  # Skip self-loops
                matrix[i][dest-1] = weight
                if not directed:  # For undirected graphs
                    matrix[dest-1][i] = weight
    
    return matrix

def show_tsp():
    st.header("Traveling Salesman Problem")
    st.markdown("""
    Create your custom graph by specifying edges and weights for each city.
    Format: (destination,weight) (destination,weight) ...
    Example: (2,10) (3,15) means edges to city 2 with weight 10 and city 3 with weight 15
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
    
    # Create and solve button
    if st.button("Create and Solve TSP", type="primary"):
        # Create distance matrix
        dist_matrix = create_distance_matrix(n_cities, edge_inputs, is_directed)
        
        # Visualize graph
        st.subheader("Graph Visualization")
        G = nx.DiGraph() if is_directed else nx.Graph()
        
        # Add nodes and edges
        for i in range(n_cities):
            G.add_node(i+1)
            for dest, weight in edge_inputs[i]:
                G.add_edge(i+1, dest, weight=weight)
        
        # Draw graph
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=16, font_weight='bold')
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        
        st.pyplot(fig)
        
        # Show distance matrix
        st.subheader("Distance Matrix")
        st.dataframe(pd.DataFrame(
            dist_matrix,
            index=[f"City {i+1}" for i in range(n_cities)],
            columns=[f"City {i+1}" for i in range(n_cities)]
        ))
        
        # Solve TSP
        if not is_directed:
            # For undirected graphs
            cities = list(range(n_cities))
            best_path = None
            min_cost = float('inf')
            
            for path in permutations(cities[1:]):
                path = (0,) + path + (0,)
                cost = sum(dist_matrix[path[i]][path[i+1]] 
                          for i in range(len(path)-1))
                if cost < min_cost:
                    min_cost = cost
                    best_path = path
            
            if min_cost != float('inf'):
                st.success("Solution Found!")
                path_cities = [i+1 for i in best_path]
                st.write(f"Optimal Path: {' → '.join(map(str, path_cities))}")
                st.write(f"Total Cost: {min_cost:.2f}")
                
                # Visualize solution path
                st.subheader("Solution Path Visualization")
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Draw original graph in light gray
                nx.draw(G, pos, with_labels=True, node_color='lightgray',
                       edge_color='lightgray', node_size=500)
                
                # Draw solution path in bold red
                path_edges = list(zip(path_cities[:-1], path_cities[1:]))
                solution_graph = nx.Graph()
                solution_graph.add_edges_from(path_edges)
                nx.draw_networkx_edges(solution_graph, pos, edge_color='red',
                                     width=2)
                
                st.pyplot(fig)
            else:
                st.error("No valid solution found! Graph may be disconnected.")
        else:
            st.warning("TSP solution visualization for directed graphs coming soon!")

# Add to requirements.txt:
# networkx
# matplotlib

