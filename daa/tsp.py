import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from itertools import permutations

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def tsp_brute_force(cities):
    """Brute force TSP solution (for small n)"""
    n = len(cities)
    if n > 8:
        return None, None  # Too slow for large n
    
    min_distance = float('inf')
    best_path = None
    
    for perm in permutations(range(1, n)):  # Start from city 0
        path = [0] + list(perm) + [0]
        distance = sum(calculate_distance(cities[path[i]], cities[path[i+1]]) 
                      for i in range(len(path) - 1))
        
        if distance < min_distance:
            min_distance = distance
            best_path = path
    
    return best_path, min_distance

def tsp_nearest_neighbor(cities):
    """Nearest Neighbor heuristic for TSP"""
    n = len(cities)
    unvisited = set(range(1, n))
    path = [0]
    current = 0
    total_distance = 0
    steps = []
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: calculate_distance(cities[current], cities[x]))
        distance = calculate_distance(cities[current], cities[nearest])
        total_distance += distance
        path.append(nearest)
        unvisited.remove(nearest)
        steps.append({
            'from': current,
            'to': nearest,
            'distance': distance,
            'total': total_distance
        })
        current = nearest
    
    # Return to start
    distance = calculate_distance(cities[current], cities[0])
    total_distance += distance
    path.append(0)
    steps.append({
        'from': current,
        'to': 0,
        'distance': distance,
        'total': total_distance
    })
    
    return path, total_distance, steps

def visualize_tsp(cities, path, title="TSP Solution"):
    """Visualize TSP path"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot cities
    x_coords = [c[0] for c in cities]
    y_coords = [c[1] for c in cities]
    ax.scatter(x_coords, y_coords, c='red', s=100, zorder=3)
    
    # Label cities
    for i, (x, y) in enumerate(cities):
        ax.annotate(f'City {i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot path
    path_x = [cities[i][0] for i in path]
    path_y = [cities[i][1] for i in path]
    ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.6, zorder=1)
    ax.plot(path_x, path_y, 'bo', markersize=8, zorder=2)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig

def show_tsp():
    st.header("Travelling Salesman Problem (TSP)")
    st.markdown("""
    The TSP finds the shortest route that visits each city exactly once and returns 
    to the starting city. This is an NP-hard problem.
    """)
    
    st.subheader("Input Parameters")
    method = st.radio("Select Method:", ["Nearest Neighbor (Heuristic)", "Brute Force (Exact, n≤8)"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_cities = st.number_input("Number of Cities", min_value=3, max_value=15, value=5)
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
    
    if st.button("Run Algorithm", type="primary"):
        if method == "Nearest Neighbor (Heuristic)":
            path, distance, steps = tsp_nearest_neighbor(cities)
            st.success(f"Total Distance: {distance:.2f}")
            st.write(f"Path: {' → '.join([f'City {i}' for i in path])}")
            
            # Visualization
            fig = visualize_tsp(cities, path, f"TSP Solution (Nearest Neighbor) - Distance: {distance:.2f}")
            st.pyplot(fig)
            
            # Show steps
            with st.expander("View Algorithm Steps"):
                steps_df = pd.DataFrame(steps)
                st.dataframe(steps_df, use_container_width=True)
        
        else:  # Brute Force
            if num_cities > 8:
                st.warning("Brute force is too slow for more than 8 cities. Using Nearest Neighbor instead.")
                path, distance, steps = tsp_nearest_neighbor(cities)
            else:
                path, distance = tsp_brute_force(cities)
                if path is None:
                    st.error("Error computing solution!")
                    return
            
            st.success(f"Optimal Distance: {distance:.2f}")
            st.write(f"Path: {' → '.join([f'City {i}' for i in path])}")
            
            # Visualization
            fig = visualize_tsp(cities, path, f"TSP Optimal Solution - Distance: {distance:.2f}")
            st.pyplot(fig)

