import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def knapsack_01_dp_exact(weights, values, capacity):
    """
    0/1 Knapsack using DP tabulation
    Algorithm:
    for w = 0 to W:
        V[0, w] = 0
    for i = 1 to n:
        V[i,0] = 0
    for i = 1 to n:
        for k = 0 to W:
            if w[i] ≤ k:
                V[i, k] = max(b[i] + V[i-1,k-w[i]], V[i-1,k])
            else:
                V[i, k] = V[i-1, k]
    """
    n = len(weights)
    W = capacity
    V = np.zeros((n + 1, W + 1))
    steps = []
    
    # for w = 0 to W: V[0, w] = 0
    for w in range(W + 1):
        V[0, w] = 0
        steps.append({
            'step': f'Initialize V[0, {w}] = 0',
            'i': 0,
            'k': w,
            'value': 0,
            'action': 'Initialize'
        })
    
    # for i = 1 to n: V[i,0] = 0
    for i in range(1, n + 1):
        V[i, 0] = 0
        steps.append({
            'step': f'Initialize V[{i}, 0] = 0',
            'i': i,
            'k': 0,
            'value': 0,
            'action': 'Initialize'
        })
    
    # for i = 1 to n:
    for i in range(1, n + 1):
        # for k = 0 to W:
        for k in range(W + 1):
            if weights[i-1] <= k:  # if w[i] ≤ k:
                # V[i, k] = max(b[i] + V[i-1,k-w[i]], V[i-1,k])
                include = values[i-1] + V[i-1, k - weights[i-1]]
                exclude = V[i-1, k]
                V[i, k] = max(include, exclude)
                decision = 'include' if include > exclude else 'exclude'
                steps.append({
                    'step': f'V[{i}, {k}] = max(b[{i}] + V[{i-1}, {k-weights[i-1]}], V[{i-1}, {k}]) = max({values[i-1]} + {V[i-1, k-weights[i-1]]:.0f}, {V[i-1, k]:.0f}) = {V[i, k]:.0f}',
                    'i': i,
                    'k': k,
                    'value': V[i, k],
                    'action': f'Max calculation ({decision})'
                })
            else:
                # V[i, k] = V[i-1, k]
                V[i, k] = V[i-1, k]
                steps.append({
                    'step': f'V[{i}, {k}] = V[{i-1}, {k}] = {V[i, k]:.0f} (w[{i}] > {k})',
                    'i': i,
                    'k': k,
                    'value': V[i, k],
                    'action': 'Copy from above'
                })
    
    # Reconstruct solution
    # Let i = n, k = W
    selected = []
    i = n
    k = W
    reconstruction_steps = []
    
    while i > 0 and k > 0:
        if V[i, k] != V[i-1, k]:  # if V[i,k] ≠ V[i-1,k]:
            # mark item i as in knapsack
            selected.append(i-1)
            reconstruction_steps.append(f'Item {i} selected (V[{i}, {k}] ≠ V[{i-1}, {k}])')
            i = i - 1
            k = k - weights[i]  # k = k - w[i]
        else:
            reconstruction_steps.append(f'Item {i} not selected (V[{i}, {k}] = V[{i-1}, {k}])')
            i = i - 1
    
    selected.reverse()
    return V[n, W], selected, V, steps, reconstruction_steps

def knapsack_fractional_greedy(weights, values, capacity):
    """Fractional Knapsack using Greedy approach"""
    n = len(weights)
    items = [(values[i]/weights[i], weights[i], values[i], i) for i in range(n)]
    items.sort(reverse=True)  # Sort by ratio descending
    
    total_value = 0
    selected = []
    remaining = capacity
    steps = []
    
    for ratio, weight, value, idx in items:
        if remaining >= weight:
            selected.append((idx, 1.0))
            total_value += value
            remaining -= weight
            steps.append({
                'item': idx + 1,
                'ratio': ratio,
                'fraction': 1.0,
                'value_added': value,
                'remaining': remaining
            })
        else:
            fraction = remaining / weight
            selected.append((idx, fraction))
            total_value += value * fraction
            steps.append({
                'item': idx + 1,
                'ratio': ratio,
                'fraction': fraction,
                'value_added': value * fraction,
                'remaining': 0
            })
            break
    
    return total_value, selected, steps

def show_knapsack():
    st.header("Knapsack Problem")
    st.markdown("""
    The Knapsack Problem is a classic optimization problem where you need to maximize 
    the value of items in a knapsack without exceeding its weight capacity.
    """)
    
    knapsack_type = st.radio("Select Knapsack Type:", ["0/1 Knapsack", "Fractional Knapsack"])
    
    # Initialize session state
    if 'knap_step' not in st.session_state:
        st.session_state.knap_step = 0
    if 'knap_auto_run' not in st.session_state:
        st.session_state.knap_auto_run = False
    if 'knap_table' not in st.session_state:
        st.session_state.knap_table = None
    if 'knap_steps' not in st.session_state:
        st.session_state.knap_steps = []
    
    st.subheader("Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        num_items = st.number_input("Number of Items (n)", min_value=1, max_value=10, value=4)
        capacity = st.number_input("Knapsack Capacity (W)", min_value=1, value=10)
    
    with col2:
        st.write("**Item Weights (w) and Values (b)**")
        weights = []
        values = []
        for i in range(num_items):
            col_w, col_v = st.columns(2)
            with col_w:
                w = st.number_input(f"w[{i+1}]", min_value=1, value=i+2, key=f"w{i}")
                weights.append(int(w))
            with col_v:
                v = st.number_input(f"b[{i+1}]", min_value=1, value=i+3, key=f"v{i}")
                values.append(int(v))
    
    if knapsack_type == "0/1 Knapsack":
        st.write("**Algorithm:**")
        st.code("""
        for w = 0 to W: V[0, w] = 0
        for i = 1 to n: V[i,0] = 0
        for i = 1 to n:
            for k = 0 to W:
                if w[i] ≤ k:
                    V[i, k] = max(b[i] + V[i-1,k-w[i]], V[i-1,k])
                else:
                    V[i, k] = V[i-1, k]
        
        Reconstruction:
        Let i = n, k = W
        if V[i,k] ≠ V[i-1,k]:
            mark item i as in knapsack
            i = i-1, k = k-w[i]
        else:
            i = i-1
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Run Algorithm", type="primary"):
                max_value, selected, table, steps, recon_steps = knapsack_01_dp_exact(weights, values, capacity)
                st.session_state.knap_table = table
                st.session_state.knap_steps = steps
                st.session_state.knap_recon_steps = recon_steps
                st.session_state.knap_step = len(steps)
                st.session_state.knap_selected = selected
                st.session_state.knap_max_value = max_value
                st.session_state.knap_auto_run = False
                st.rerun()
        
        with col2:
            if st.button("Next Step") and st.session_state.knap_table is not None:
                if st.session_state.knap_step < len(st.session_state.knap_steps):
                    st.session_state.knap_step += 1
                    st.rerun()
        
        with col3:
            if st.button("Auto Run") and st.session_state.knap_table is not None:
                st.session_state.knap_auto_run = True
                st.rerun()
        
        if st.session_state.knap_table is not None:
            table = st.session_state.knap_table
            n, W = table.shape
            n -= 1
            W -= 1
            
            st.success(f"Maximum Value: {int(st.session_state.knap_max_value)}")
            st.write(f"Selected Items: {[i+1 for i in st.session_state.knap_selected]}")
            
            # Show DP table with highlighting
            st.subheader("DP Table V[i, k] (Step-by-Step)")
            
            steps_to_show = st.session_state.knap_steps[:st.session_state.knap_step]
            current_cell = None
            if steps_to_show:
                last_step = steps_to_show[-1]
                current_cell = (last_step['i'], last_step['k'])
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(14, 8))
            
            table_data = []
            for i in range(n + 1):
                row = []
                for k in range(W + 1):
                    row.append(f'{int(table[i, k])}')
                table_data.append(row)
            
            table_plot = ax.table(cellText=table_data,
                                 rowLabels=[f'Item {i} (w={weights[i-1]}, b={values[i-1]})' if i > 0 else 'Item 0' 
                                           for i in range(n + 1)],
                                 colLabels=[f'k={k}' for k in range(W + 1)],
                                 cellLoc='center',
                                 loc='center')
            
            # Highlight current cell
            if current_cell:
                cell_row = current_cell[0] + 1
                cell_col = current_cell[1] + 1
                # Check if the cell exists in the table (bounds checking)
                if (cell_row, cell_col) in table_plot._cells:
                    table_plot[(cell_row, cell_col)].set_facecolor('#ffff00')
                    table_plot[(cell_row, cell_col)].set_text_props(weight='bold')
            
            # Color updated cells
            for step in steps_to_show:
                if step['action'] != 'Initialize':
                    i_idx = step['i']
                    k_idx = step['k']
                    cell_row = i_idx + 1
                    cell_col = k_idx + 1
                    # Check if the cell exists in the table (bounds checking)
                    if (cell_row, cell_col) in table_plot._cells:
                        table_plot[(cell_row, cell_col)].set_facecolor('#90EE90')
            
            ax.axis('off')
            ax.set_title('DP Table: V[i, k] = maximum value with first i items and capacity k', 
                        fontsize=12, weight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show current step
            if steps_to_show:
                st.subheader("Current Step")
                last_step = steps_to_show[-1]
                st.info(f"**Step {st.session_state.knap_step}**: {last_step['step']}")
            
            # Show reconstruction
            if hasattr(st.session_state, 'knap_recon_steps'):
                st.subheader("Solution Reconstruction")
                for recon_step in st.session_state.knap_recon_steps:
                    st.write(f"- {recon_step}")
    
    else:  # Fractional Knapsack
        if st.button("Run Algorithm", type="primary"):
            max_value, selected, steps = knapsack_fractional_greedy(weights, values, capacity)
            
            st.success(f"Maximum Value: {max_value:.2f}")
            st.write("Selected Items (with fractions):")
            for idx, fraction in selected:
                st.write(f"  Item {idx+1}: {fraction*100:.1f}% (Value: {values[idx]*fraction:.2f})")
            
            # Visualization: Ratio and cumulative value
            st.subheader("Greedy Selection Process")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Ratio chart
            ratios = [v/w for v, w in zip(values, weights)]
            items_list = list(range(1, num_items + 1))
            
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
            
            bars = ax1.barh(items_list, ratios, color=colors, alpha=0.7)
            ax1.set_xlabel('Value/Weight Ratio', fontsize=12)
            ax1.set_ylabel('Item', fontsize=12)
            ax1.set_title('Items Sorted by Ratio\n(Green=Full, Orange=Partial, Red=Not Selected)', 
                         fontsize=12, weight='bold')
            ax1.set_yticks(items_list)
            for i, (idx, frac) in enumerate(selected):
                ax1.text(ratios[idx], idx+1, f'{ratios[idx]:.2f}', 
                        ha='left', va='center', fontweight='bold')
            
            # Cumulative value graph
            cumulative_value = 0
            cumulative_values = [0]
            cumulative_weights = [0]
            for idx, frac in selected:
                cumulative_value += values[idx] * frac
                cumulative_values.append(cumulative_value)
                cumulative_weights.append(cumulative_weights[-1] + weights[idx] * frac)
            
            ax2.plot(cumulative_weights, cumulative_values, 'o-', linewidth=2, markersize=8)
            ax2.axvline(capacity, color='red', linestyle='--', linewidth=2, label=f'Capacity: {capacity}')
            ax2.fill_between(cumulative_weights, cumulative_values, alpha=0.3)
            ax2.set_xlabel('Cumulative Weight', fontsize=12)
            ax2.set_ylabel('Cumulative Value', fontsize=12)
            ax2.set_title('Cumulative Value vs Weight', fontsize=12, weight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show steps
            with st.expander("View Selection Steps"):
                steps_df = pd.DataFrame(steps)
                st.dataframe(steps_df, use_container_width=True, hide_index=True)
