import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def knapsack_01_dp(weights, values, capacity):
    """0/1 Knapsack using Dynamic Programming"""
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    steps = []
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
                steps.append({
                    'item': i-1,
                    'weight': w,
                    'decision': 'include' if dp[i][w] == dp[i-1][w-weights[i-1]] + values[i-1] else 'exclude',
                    'value': dp[i][w]
                })
            else:
                dp[i][w] = dp[i-1][w]
                steps.append({
                    'item': i-1,
                    'weight': w,
                    'decision': 'exclude',
                    'value': dp[i][w]
                })
    
    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i-1)
            w -= weights[i-1]
    
    return dp[n][capacity], selected, steps

def knapsack_fractional(weights, values, capacity):
    """Fractional Knapsack using Greedy"""
    items = [(values[i]/weights[i], weights[i], values[i], i) for i in range(len(weights))]
    items.sort(reverse=True)
    
    total_value = 0
    selected = []
    remaining = capacity
    
    for ratio, weight, value, idx in items:
        if remaining >= weight:
            selected.append((idx, 1.0))
            total_value += value
            remaining -= weight
        else:
            fraction = remaining / weight
            selected.append((idx, fraction))
            total_value += value * fraction
            break
    
    return total_value, selected

def show_knapsack():
    st.header("Knapsack Problem")
    st.markdown("""
    The Knapsack Problem is a classic optimization problem where you need to maximize 
    the value of items in a knapsack without exceeding its weight capacity.
    """)
    
    knapsack_type = st.radio("Select Knapsack Type:", ["0/1 Knapsack", "Fractional Knapsack"])
    
    st.subheader("Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        num_items = st.number_input("Number of Items", min_value=1, max_value=10, value=5)
        capacity = st.number_input("Knapsack Capacity", min_value=1, value=10)
    
    with col2:
        st.write("**Item Weights and Values**")
        weights = []
        values = []
        for i in range(num_items):
            col_w, col_v = st.columns(2)
            with col_w:
                w = st.number_input(f"W{i+1}", min_value=1, value=i+2, key=f"w{i}")
                weights.append(int(w))
            with col_v:
                v = st.number_input(f"V{i+1}", min_value=1, value=i+3, key=f"v{i}")
                values.append(int(v))
    
    if st.button("Run Algorithm", type="primary"):
        if knapsack_type == "0/1 Knapsack":
            max_value, selected, steps = knapsack_01_dp(weights, values, capacity)
            
            st.success(f"Maximum Value: {max_value}")
            st.write(f"Selected Items: {[i+1 for i in selected]}")
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Items visualization
            items_list = list(range(1, num_items + 1))
            colors = ['green' if i in selected else 'red' for i in range(num_items)]
            ax1.barh(items_list, values, color=colors, alpha=0.7)
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Item')
            ax1.set_title('Items (Green=Selected, Red=Not Selected)')
            ax1.set_yticks(items_list)
            
            # Weight vs Value scatter
            ax2.scatter(weights, values, s=100, c=colors, alpha=0.7)
            for i, (w, v) in enumerate(zip(weights, values)):
                ax2.annotate(f'Item {i+1}', (w, v))
            ax2.set_xlabel('Weight')
            ax2.set_ylabel('Value')
            ax2.set_title('Weight vs Value')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show DP table
            st.subheader("DP Table (Last Row)")
            dp_table = [[0 for _ in range(capacity + 1)]]
            for i in range(1, num_items + 1):
                row = [0]
                for w in range(1, capacity + 1):
                    if weights[i-1] <= w:
                        row.append(max(dp_table[i-1][w], dp_table[i-1][w-weights[i-1]] + values[i-1]))
                    else:
                        row.append(dp_table[i-1][w])
                dp_table.append(row)
            
            st.dataframe(dp_table[-1], use_container_width=True)
        
        else:  # Fractional
            max_value, selected = knapsack_fractional(weights, values, capacity)
            
            st.success(f"Maximum Value: {max_value:.2f}")
            st.write("Selected Items (with fractions):")
            for idx, fraction in selected:
                st.write(f"  Item {idx+1}: {fraction*100:.1f}%")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ratios = [v/w for v, w in zip(values, weights)]
            items_list = list(range(1, num_items + 1))
            
            # Color based on selection
            colors = []
            for i in range(num_items):
                found = False
                for idx, frac in selected:
                    if idx == i:
                        colors.append('green' if frac == 1.0 else 'orange')
                        found = True
                        break
                if not found:
                    colors.append('red')
            
            bars = ax.barh(items_list, ratios, color=colors, alpha=0.7)
            ax.set_xlabel('Value/Weight Ratio')
            ax.set_ylabel('Item')
            ax.set_title('Greedy Selection (Green=Full, Orange=Partial, Red=Not Selected)')
            ax.set_yticks(items_list)
            plt.tight_layout()
            st.pyplot(fig)

