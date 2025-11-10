import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

def coin_change_dp_exact(d, N):
    """
    Function coins(d, N)
    Input: Array d[1..n] specifies the coinage;
           N is the number of units for which to make change
    Output: Minimum number of coins needed to make change for N units using coins from d
    """
    n = len(d)
    # Array c[1..n,0..N] - using 0-indexed, so c[0..n-1, 0..N]
    m = np.full((n, N + 1), float('inf'))
    steps = []
    
    # for i=1 to n do: m[i,0] = 0
    for i in range(n):
        m[i, 0] = 0
        steps.append({
            'step': f'Initialize m[{i+1}, 0] = 0',
            'i': i+1,
            'j': 0,
            'value': 0,
            'action': 'Initialize'
        })
    
    # for i=1 to n do:
    for i in range(n):
        # for j=1 to N do:
        for j in range(1, N + 1):
            if i == 0 and j < d[i]:  # if i=1 and j < d[i] then
                m[i, j] = float('infinity')
                steps.append({
                    'step': f'm[{i+1}, {j}] = +infinity (j < d[{i+1}])',
                    'i': i+1,
                    'j': j,
                    'value': float('inf'),
                    'action': 'Set to infinity'
                })
            elif i == 0:  # else if i=1 then
                m[i, j] = 1 + m[0, j - d[0]]
                steps.append({
                    'step': f'm[{i+1}, {j}] = 1 + m[1, {j} - {d[0]}] = 1 + m[1, {j-d[0]}] = {m[i, j]:.0f}',
                    'i': i+1,
                    'j': j,
                    'value': m[i, j],
                    'action': 'First row calculation'
                })
            elif j < d[i]:  # else if j < d[i] then
                m[i, j] = m[i-1, j]
                steps.append({
                    'step': f'm[{i+1}, {j}] = m[{i}, {j}] = {m[i, j]:.0f} (j < d[{i+1}])',
                    'i': i+1,
                    'j': j,
                    'value': m[i, j],
                    'action': 'Copy from above'
                })
            else:  # else
                m[i, j] = min(m[i-1, j], 1 + m[i, j - d[i]])
                steps.append({
                    'step': f'm[{i+1}, {j}] = min(m[{i}, {j}], 1 + m[{i+1}, {j} - {d[i]}]) = min({m[i-1, j]:.0f}, {1 + m[i, j - d[i]]:.0f}) = {m[i, j]:.0f}',
                    'i': i+1,
                    'j': j,
                    'value': m[i, j],
                    'action': 'Take minimum'
                })
    
    return m[n-1, N], m, steps

def show_coin_change():
    st.header("Coin Changing (Dynamic Programming)")
    st.markdown("""
    The Coin Change problem finds the minimum number of coins needed to make a given amount.
    This uses Dynamic Programming with the exact algorithm provided.
    """)
    
    # Initialize session state for step-by-step
    if 'coin_step' not in st.session_state:
        st.session_state.coin_step = 0
    if 'coin_auto_run' not in st.session_state:
        st.session_state.coin_auto_run = False
    if 'coin_table' not in st.session_state:
        st.session_state.coin_table = None
    if 'coin_steps' not in st.session_state:
        st.session_state.coin_steps = []
    
    st.subheader("Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        coins_input = st.text_input("Coins d[1..n] (comma-separated)", value="1, 3, 4")
        amount = st.number_input("Target Amount N", min_value=1, value=6)
    
    with col2:
        st.write("**Algorithm**")
        st.code("""
        Function coins(d, N)
        for i=1 to n do:
            m[i,0] = 0
        for i=1 to n do:
            for j=1 to N do:
                if i=1 and j < d[i]:
                    m[i,j] = +infinity
                else if i=1:
                    m[i,j] = 1+m[1,j-d[1]]
                else if j < d[i]:
                    m[i,j] = m[i-1,j]
                else:
                    m[i,j] = min(m[i-1,j], 1+m[i,j-d[i]])
        return m[n,N]
        """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Run Algorithm", type="primary"):
            try:
                d = [int(x.strip()) for x in coins_input.split(',')]
                d.sort()  # Ensure sorted
                min_coins, table, steps = coin_change_dp_exact(d, int(amount))
                st.session_state.coin_table = table
                st.session_state.coin_steps = steps
                st.session_state.coin_step = 0  # Start from step 0
                st.session_state.coin_auto_run = False
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("Next Step") and st.session_state.coin_table is not None:
            if st.session_state.coin_step < len(st.session_state.coin_steps):
                st.session_state.coin_step += 1
                st.rerun()
            else:
                st.info("All steps completed!")
    
    with col3:
        if st.button("Auto Run") and st.session_state.coin_table is not None:
            st.session_state.coin_auto_run = True
            st.rerun()
    
    if st.session_state.coin_table is not None:
        table = st.session_state.coin_table
        n, N = table.shape
        n -= 1  # Adjust for 0-indexing
        N -= 1
        
        # Get coins
        try:
            d = [int(x.strip()) for x in coins_input.split(',')]
            d.sort()
        except:
            d = []
        
        # Get step information
        total_steps = len(st.session_state.coin_steps)
        current_step = st.session_state.coin_step
        
        # Show result only when all steps are complete
        if current_step >= total_steps and total_steps > 0:
            min_coins = table[n, N]
            if min_coins == float('inf'):
                st.error("Cannot make the amount with given coins!")
            else:
                st.success(f"✅ Algorithm Complete! Minimum coins needed: {int(min_coins)}")
        
        # Show table with step-by-step updates
        st.subheader(f"DP Table (Step {current_step}/{total_steps})")
        
        if current_step == 0:
            st.info("Click 'Next Step' to start the algorithm visualization")
        
        # Get steps shown so far
        steps_to_show = st.session_state.coin_steps[:current_step]
        
        # Build table state up to current step
        # Initialize table with infinity
        current_table = np.full((n + 1, N + 1), float('inf'))
        
        # Apply all steps up to current step
        for step in steps_to_show:
            i_idx = step['i'] - 1  # Convert to 0-indexed (i is 1-indexed in steps)
            j_idx = step['j']  # j is 1-indexed in steps (1 to N), but 0 for initialization
            
            if step['action'] == 'Initialize':
                # Initialize: m[i, 0] = 0
                if j_idx == 0:  # j=0 means column 0
                    current_table[i_idx, 0] = 0
            else:
                # Update the cell value (j is 1-indexed, so j_idx is 1 to N)
                # But we need to map to 0-indexed table: j_idx 1->0, 2->1, ..., N->N-1
                # Actually wait, let me check: in the table, columns are 0 to N (0-indexed)
                # In steps, j goes from 1 to N (1-indexed)
                # So j_idx=1 maps to table column 1, j_idx=2 maps to table column 2, etc.
                # But wait, the algorithm uses j from 1 to N, and the table has columns 0 to N
                # Column 0 is for amount 0, column 1 is for amount 1, etc.
                # So j_idx=1 should map to table column 1, j_idx=2 to column 2, etc.
                if 1 <= j_idx <= N:  # j is 1-indexed (1 to N)
                    table_col = j_idx  # Map directly (j=1 -> col 1, j=2 -> col 2, etc.)
                    val = step['value']
                    if val == float('inf'):
                        current_table[i_idx, table_col] = float('inf')
                    else:
                        current_table[i_idx, table_col] = val
        
        # Get current step info
        current_cell = None
        current_step_info = None
        if steps_to_show:
            last_step = steps_to_show[-1]
            current_cell = (last_step['i'] - 1, last_step['j'])
            current_step_info = last_step
        
        # Display table with enhanced visualization
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create table data from current state
        table_data = []
        for i in range(n + 1):
            row = []
            for j in range(N + 1):
                val = current_table[i, j]
                if val == float('inf'):
                    row.append('∞')
                else:
                    row.append(f'{int(val)}')
            table_data.append(row)
        
        # Create table with colors
        table_plot = ax.table(cellText=table_data,
                              rowLabels=[f'Coin {i+1}\n(d={d[i]})' if i < len(d) else f'Row {i+1}' 
                                        for i in range(n + 1)],
                              colLabels=[f'Amount\n{j}' for j in range(N + 1)],
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0, 1, 1])
        
        # Get cells used in current calculation (dependencies)
        dependency_cells = []
        if current_step_info and current_step_info['action'] != 'Initialize':
            step_i = current_step_info['i'] - 1  # 0-indexed
            step_j = current_step_info['j']  # 1-indexed
            
            # Determine which cells are used based on the algorithm
            if step_i == 0:  # First row
                if step_j >= d[0]:
                    # m[1, j] = 1 + m[1, j-d[1]]
                    dep_j = step_j - d[0]
                    if dep_j >= 0:
                        dependency_cells.append((step_i, dep_j))
            else:  # Other rows
                # m[i, j] = min(m[i-1, j], 1 + m[i, j-d[i]])
                # Always uses m[i-1, j]
                dependency_cells.append((step_i - 1, step_j))
                # If j >= d[i], also uses m[i, j-d[i]]
                if step_j >= d[step_i]:
                    dep_j = step_j - d[step_i]
                    if dep_j >= 0:
                        dependency_cells.append((step_i, dep_j))
        
        # Color cells based on their state
        for i in range(n + 1):
            for j in range(N + 1):
                row_idx = i + 1  # Matplotlib table row (1 to n+1)
                col_idx = j + 1  # Matplotlib table col (1 to N+1)
                
                if (row_idx, col_idx) in table_plot._cells:
                    # Check if this cell has been computed
                    cell_computed = False
                    for step in steps_to_show:
                        if step['action'] != 'Initialize':
                            step_i = step['i'] - 1
                            step_j = step['j']
                            if step_i == i and step_j == j:
                                cell_computed = True
                                break
                        elif step['action'] == 'Initialize' and step['j'] == 0:
                            step_i = step['i'] - 1
                            if step_i == i and j == 0:
                                cell_computed = True
                                break
                    
                    # Check if this is a dependency cell (used in current calculation)
                    is_dependency = (i, j) in dependency_cells
                    
                    # Check if this is the current cell being computed
                    if current_cell:
                        curr_i = current_cell[0]
                        curr_j = current_cell[1]
                        if curr_i == i and curr_j == j:
                            table_plot[(row_idx, col_idx)].set_facecolor('#ffff00')  # Yellow for current
                            table_plot[(row_idx, col_idx)].set_text_props(weight='bold', fontsize=11)
                            table_plot[(row_idx, col_idx)].set_edgecolor('black')
                            table_plot[(row_idx, col_idx)].set_linewidth(3)
                        elif is_dependency:
                            table_plot[(row_idx, col_idx)].set_facecolor('#87CEEB')  # Sky blue for dependencies
                            table_plot[(row_idx, col_idx)].set_text_props(weight='bold', fontsize=10)
                            table_plot[(row_idx, col_idx)].set_edgecolor('blue')
                            table_plot[(row_idx, col_idx)].set_linewidth(2)
                        elif cell_computed:
                            table_plot[(row_idx, col_idx)].set_facecolor('#90EE90')  # Green for computed
                        elif current_table[i, j] == 0:
                            table_plot[(row_idx, col_idx)].set_facecolor('#E0E0E0')  # Gray for initialized
                        else:
                            table_plot[(row_idx, col_idx)].set_facecolor('#FFE0E0')  # Light red for not computed
                    elif cell_computed:
                        table_plot[(row_idx, col_idx)].set_facecolor('#90EE90')
                    elif current_table[i, j] == 0:
                        table_plot[(row_idx, col_idx)].set_facecolor('#E0E0E0')
                    else:
                        table_plot[(row_idx, col_idx)].set_facecolor('#FFE0E0')
        
        # Draw arrows showing dependencies (if current step exists)
        if current_step_info and current_step_info['action'] != 'Initialize' and dependency_cells:
            curr_i = current_cell[0]
            curr_j = current_cell[1]
            
            # Convert table coordinates to plot coordinates
            # Table is centered, so we need to estimate positions
            cell_width = 1.0 / (N + 2)  # Approximate
            cell_height = 1.0 / (n + 2)
            
            for dep_i, dep_j in dependency_cells:
                # Calculate positions (approximate, since table layout is complex)
                # We'll use annotation arrows instead
                try:
                    # Get cell positions from table
                    curr_cell_pos = table_plot[(curr_i + 1, curr_j + 1)].get_xy()
                    dep_cell_pos = table_plot[(dep_i + 1, dep_j + 1)].get_xy()
                    
                    # Draw arrow from dependency to current
                    ax.annotate('', 
                               xy=(curr_cell_pos[0], curr_cell_pos[1]),
                               xytext=(dep_cell_pos[0], dep_cell_pos[1]),
                               arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.6))
                except:
                    pass  # Skip if positions can't be determined
        
        ax.axis('off')
        ax.set_title('DP Table: m[i, j] = minimum coins for amount j using first i coins\n' +
                     f'Step {current_step}/{total_steps}', 
                    fontsize=12, weight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show current step info with calculation details
        if current_step_info and current_step > 0:
            st.subheader("Current Step Information")
            
            # Show the step description
            st.info(f"**Step {current_step}**: {current_step_info['step']}")
            
            # Show calculation details
            step_i = current_step_info['i'] - 1  # 0-indexed
            step_j = current_step_info['j']  # 1-indexed
            
            if current_step_info['action'] != 'Initialize':
                st.markdown("**Calculation Details:**")
                
                if step_i == 0:  # First row
                    if step_j >= d[0]:
                        dep_j = step_j - d[0]
                        dep_val = current_table[step_i, dep_j] if dep_j >= 0 else float('inf')
                        dep_val_str = '∞' if dep_val == float('inf') else f'{int(dep_val)}'
                        formula = f"m[{current_step_info['i']}, {step_j}] = 1 + m[{current_step_info['i']}, {step_j} - {d[0]}] = 1 + m[{current_step_info['i']}, {step_j - d[0]}] = 1 + {dep_val_str}"
                        st.code(formula, language='text')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Using", f"m[{current_step_info['i']}, {step_j - d[0]}] = {dep_val_str}")
                        with col2:
                            st.metric("Result", f"1 + {dep_val_str} = {int(current_step_info['value']) if current_step_info['value'] != float('inf') else '∞'}")
                        with col3:
                            st.metric("Cell", f"m[{current_step_info['i']}, {step_j}]")
                else:  # Other rows
                    above_val = current_table[step_i - 1, step_j] if step_i > 0 else float('inf')
                    above_val_str = '∞' if above_val == float('inf') else f'{int(above_val)}'
                    
                    if step_j >= d[step_i]:
                        left_val = current_table[step_i, step_j - d[step_i]]
                        left_val_str = '∞' if left_val == float('inf') else f'{int(left_val)}'
                        formula = f"m[{current_step_info['i']}, {step_j}] = min(m[{current_step_info['i']-1}, {step_j}], 1 + m[{current_step_info['i']}, {step_j} - {d[step_i]}])"
                        st.code(formula, language='text')
                        st.code(f"  = min({above_val_str}, 1 + {left_val_str}) = min({above_val_str}, {1 + int(left_val) if left_val != float('inf') else '∞'}) = {int(current_step_info['value']) if current_step_info['value'] != float('inf') else '∞'}", language='text')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Option 1", f"m[{current_step_info['i']-1}, {step_j}] = {above_val_str}")
                        with col2:
                            st.metric("Option 2", f"1 + m[{current_step_info['i']}, {step_j - d[step_i]}] = 1 + {left_val_str} = {1 + int(left_val) if left_val != float('inf') else '∞'}")
                        with col3:
                            st.metric("Result", f"min = {int(current_step_info['value']) if current_step_info['value'] != float('inf') else '∞'}")
                    else:
                        formula = f"m[{current_step_info['i']}, {step_j}] = m[{current_step_info['i']-1}, {step_j}] = {above_val_str}"
                        st.code(formula, language='text')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Copy from above", f"m[{current_step_info['i']-1}, {step_j}] = {above_val_str}")
                        with col2:
                            st.metric("Result", f"m[{current_step_info['i']}, {step_j}] = {above_val_str}")
            
            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Action", current_step_info['action'])
            with col2:
                st.metric("Cell", f"m[{current_step_info['i']}, {current_step_info['j']}]")
            with col3:
                val = current_step_info['value']
                if val == float('inf'):
                    st.metric("Value", "∞")
                else:
                    st.metric("Value", f"{int(val)}")
        elif current_step == 0:
            st.info("Ready to start! Click 'Next Step' to begin the algorithm.")
        
        # Show legend
        st.caption("""
        **Legend:**
        - 🟡 **Yellow (Bold Border)**: Current cell being computed
        - 🔵 **Sky Blue (Blue Border)**: Cells used in current calculation (dependencies)
        - 🟢 **Green**: Already computed cells
        - ⚪ **Gray**: Initialized cells (value = 0)
        - 🔴 **Light Red**: Not yet computed cells
        """)
        
        # Show all steps
        with st.expander("View All Steps"):
            steps_df = pd.DataFrame(steps_to_show)
            if len(steps_df) > 0:
                st.dataframe(steps_df[['step', 'i', 'j', 'value', 'action']], 
                            use_container_width=True, hide_index=True)
