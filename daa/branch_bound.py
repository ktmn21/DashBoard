import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class BranchBoundNode:
    def __init__(self, level, profit, weight, bound, items):
        self.level = level
        self.profit = profit
        self.weight = weight
        self.bound = bound
        self.items = items  # List of item indices included
    
    def __lt__(self, other):
        return self.bound > other.bound  # Max heap behavior

def calculate_bound(node, items, capacity):
    """Calculate upper bound for branch and bound"""
    if node.weight >= capacity:
        return 0
    
    bound = node.profit
    j = node.level + 1
    total_weight = node.weight
    
    # Greedily add items
    while j < len(items) and total_weight + items[j][1] <= capacity:
        total_weight += items[j][1]
        bound += items[j][0]
        j += 1
    
    # Add fraction of next item if space remains
    if j < len(items):
        bound += (capacity - total_weight) * (items[j][0] / items[j][1])
    
    return bound

def branch_and_bound_knapsack(items, capacity):
    """Branch and Bound for 0/1 Knapsack"""
    # Sort by value/weight ratio
    items_sorted = sorted(items, key=lambda x: x[0]/x[1], reverse=True)
    
    # Priority queue (using list as min-heap with negative bounds)
    import heapq
    queue = []
    nodes_explored = []
    
    # Root node
    root = BranchBoundNode(-1, 0, 0, 0, [])
    root.bound = calculate_bound(root, items_sorted, capacity)
    heapq.heappush(queue, (-root.bound, root))
    
    best_profit = 0
    best_node = None
    
    while queue:
        _, node = heapq.heappop(queue)
        nodes_explored.append(node)
        
        if node.bound > best_profit and node.level < len(items_sorted) - 1:
            level = node.level + 1
            value, weight = items_sorted[level]
            
            # Include item
            include_node = BranchBoundNode(
                level,
                node.profit + value,
                node.weight + weight,
                0,
                node.items + [level]
            )
            include_node.bound = calculate_bound(include_node, items_sorted, capacity)
            
            if include_node.weight <= capacity and include_node.profit > best_profit:
                best_profit = include_node.profit
                best_node = include_node
            
            if include_node.bound > best_profit:
                heapq.heappush(queue, (-include_node.bound, include_node))
            
            # Exclude item
            exclude_node = BranchBoundNode(
                level,
                node.profit,
                node.weight,
                0,
                node.items.copy()
            )
            exclude_node.bound = calculate_bound(exclude_node, items_sorted, capacity)
            
            if exclude_node.bound > best_profit:
                heapq.heappush(queue, (-exclude_node.bound, exclude_node))
    
    # Map back to original indices
    if best_node:
        original_indices = [items_sorted[i][2] for i in best_node.items]
        return best_profit, original_indices, nodes_explored
    
    return 0, [], nodes_explored

def visualize_exploration_tree(nodes_explored, max_nodes=50):
    """Visualize branch and bound exploration tree"""
    G = nx.DiGraph()
    
    # Limit nodes for visualization
    nodes_to_show = nodes_explored[:max_nodes]
    
    for i, node in enumerate(nodes_to_show):
        G.add_node(i, 
                   profit=node.profit,
                   weight=node.weight,
                   bound=node.bound,
                   level=node.level)
        
        # Connect to parent (simplified - find closest parent)
        if i > 0:
            parent_idx = max([j for j in range(i) if nodes_to_show[j].level < node.level], 
                           default=0, key=lambda j: nodes_to_show[j].level)
            G.add_edge(parent_idx, i)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Color nodes by bound
    bounds = [G.nodes[i]['bound'] for i in G.nodes()]
    colors = plt.cm.viridis([(b - min(bounds)) / (max(bounds) - min(bounds) + 1) 
                            for b in bounds])
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10, ax=ax)
    
    # Labels
    labels = {i: f"L{node.level}\nP:{node.profit}\nB:{node.bound:.1f}" 
              for i, node in enumerate(nodes_to_show)}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    
    ax.set_title("Branch and Bound Exploration Tree\n(L=Level, P=Profit, B=Bound)", 
                fontsize=12, weight='bold')
    ax.axis('off')
    
    return fig

def show_branch_bound():
    st.header("Branch and Bound")
    st.markdown("""
    Branch and Bound is an optimization technique that systematically explores 
    the solution space by branching on decisions and bounding to prune suboptimal paths.
    This example solves the 0/1 Knapsack problem.
    """)
    
    st.subheader("Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        num_items = st.number_input("Number of Items", min_value=1, max_value=10, value=5)
        capacity = st.number_input("Knapsack Capacity", min_value=1, value=15)
    
    with col2:
        st.write("**Item Values and Weights**")
        items = []
        for i in range(num_items):
            col_v, col_w = st.columns(2)
            with col_v:
                v = st.number_input(f"Value {i+1}", min_value=1, value=i+5, key=f"v{i}")
            with col_w:
                w = st.number_input(f"Weight {i+1}", min_value=1, value=i+3, key=f"w{i}")
            items.append((int(v), int(w), i))  # (value, weight, original_index)
    
    if st.button("Run Algorithm", type="primary"):
        max_profit, selected, nodes_explored = branch_and_bound_knapsack(items, capacity)
        
        st.success(f"Maximum Profit: {max_profit}")
        st.write(f"Selected Items (indices): {[i+1 for i in selected]}")
        st.write(f"Nodes Explored: {len(nodes_explored)}")
        
        # Visualization
        st.subheader("Exploration Tree")
        if len(nodes_explored) > 0:
            fig = visualize_exploration_tree(nodes_explored)
            st.pyplot(fig)
        else:
            st.info("No nodes to visualize")
        
        # Show selected items
        st.subheader("Solution Details")
        if selected:
            selected_items_data = [items[i] for i in selected]
            total_weight = sum(w for _, w, _ in selected_items_data)
            st.write(f"Total Weight: {total_weight}/{capacity}")
            st.write(f"Total Value: {max_profit}")
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            item_indices = [i+1 for i in selected]
            values = [v for v, _, _ in selected_items_data]
            weights = [w for _, w, _ in selected_items_data]
            
            x = np.arange(len(item_indices))
            width = 0.35
            
            ax.bar(x - width/2, values, width, label='Value', color='steelblue', alpha=0.7)
            ax.bar(x + width/2, weights, width, label='Weight', color='coral', alpha=0.7)
            
            ax.set_xlabel('Item')
            ax.set_ylabel('Value/Weight')
            ax.set_title('Selected Items')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Item {i}' for i in item_indices])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)

