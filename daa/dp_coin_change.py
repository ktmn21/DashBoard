import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def coin_change_dp(coins, amount):
    """Coin Change using Dynamic Programming"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    parent = [-1] * (amount + 1)
    steps = []
    
    for coin in coins:
        for i in range(coin, amount + 1):
            if dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                parent[i] = i - coin
                steps.append({
                    'amount': i,
                    'coin': coin,
                    'min_coins': dp[i],
                    'from': i - coin
                })
    
    # Reconstruct solution
    solution = []
    current = amount
    coin_count = {}
    while current > 0:
        prev = parent[current]
        coin_used = current - prev
        solution.append(coin_used)
        coin_count[coin_used] = coin_count.get(coin_used, 0) + 1
        current = prev
    
    return dp[amount] if dp[amount] != float('inf') else -1, solution, coin_count, dp, steps

def show_coin_change():
    st.header("Coin Change Problem (Dynamic Programming)")
    st.markdown("""
    The Coin Change problem finds the minimum number of coins needed to make a given amount.
    This uses Dynamic Programming to build up solutions from smaller subproblems.
    """)
    
    st.subheader("Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        coins_input = st.text_input("Coins (comma-separated)", value="1, 3, 4")
        amount = st.number_input("Target Amount", min_value=1, value=6)
    
    with col2:
        st.write("**Algorithm Explanation**")
        st.write("""
        - DP[i] = minimum coins needed for amount i
        - For each coin, update DP[i] = min(DP[i], DP[i-coin] + 1)
        - Build solution bottom-up
        """)
    
    if st.button("Run Algorithm", type="primary"):
        try:
            coins = [int(x.strip()) for x in coins_input.split(',')]
            coins.sort()
            
            min_coins, solution, coin_count, dp_table, steps = coin_change_dp(coins, amount)
            
            if min_coins == -1:
                st.error("Cannot make the amount with given coins!")
            else:
                st.success(f"Minimum coins needed: {min_coins}")
                st.write(f"Solution: {solution}")
                st.write(f"Coin breakdown: {coin_count}")
                
                # Visualization 1: DP Table
                st.subheader("DP Table")
                dp_df = pd.DataFrame({
                    'Amount': list(range(amount + 1)),
                    'Min Coins': dp_table
                })
                dp_df['Min Coins'] = dp_df['Min Coins'].replace(float('inf'), '∞')
                st.dataframe(dp_df, use_container_width=True, hide_index=True)
                
                # Visualization 2: Bar chart of coin usage
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Coin usage
                if coin_count:
                    coin_list = list(coin_count.keys())
                    counts = list(coin_count.values())
                    ax1.bar(coin_list, counts, color='steelblue', alpha=0.7)
                    ax1.set_xlabel('Coin Value')
                    ax1.set_ylabel('Count')
                    ax1.set_title('Coins Used in Solution')
                    ax1.set_xticks(coin_list)
                    for i, (coin, count) in enumerate(zip(coin_list, counts)):
                        ax1.text(coin, count, str(count), ha='center', va='bottom')
                
                # DP progression
                valid_dp = [x if x != float('inf') else 0 for x in dp_table]
                ax2.plot(range(amount + 1), valid_dp, marker='o', linewidth=2, markersize=6)
                ax2.set_xlabel('Amount')
                ax2.set_ylabel('Minimum Coins')
                ax2.set_title('DP Table Progression')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show steps
                with st.expander("View Algorithm Steps"):
                    if steps:
                        steps_df = pd.DataFrame(steps[:20])  # Show first 20 steps
                        st.dataframe(steps_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

