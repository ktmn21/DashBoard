import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from collections import Counter
import heapq

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree_stepwise(text):
    """Build Huffman tree step-by-step"""
    freq = Counter(text)
    heap = [HuffmanNode(char=char, freq=freq) for char, freq in freq.items()]
    heapq.heapify(heap)
    steps = []
    step_num = 0
    
    while len(heap) > 1:
        step_num += 1
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = HuffmanNode(
            freq=left.freq + right.freq,
            left=left,
            right=right
        )
        heapq.heappush(heap, merged)
        
        left_label = left.char if left.char else f"Node({left.freq})"
        right_label = right.char if right.char else f"Node({right.freq})"
        
        steps.append({
            'step': step_num,
            'left': left_label,
            'left_freq': left.freq,
            'right': right_label,
            'right_freq': right.freq,
            'merged_freq': merged.freq,
            'heap_size': len(heap)
        })
    
    return heap[0] if heap else None, steps, freq

def generate_codes(node, code="", codes=None):
    """Generate Huffman codes from tree"""
    if codes is None:
        codes = {}
    
    if node.char is not None:
        codes[node.char] = code if code else "0"
    else:
        if node.left:
            generate_codes(node.left, code + "0", codes)
        if node.right:
            generate_codes(node.right, code + "1", codes)
    
    return codes

def draw_tree_stepwise(node, ax, x=0, y=0, dx=1, dy=1, level=0, current_step=None, step_num=0):
    """Recursively draw Huffman tree with step highlighting"""
    if node is None:
        return
    
    label = f"{node.char}\n({node.freq})" if node.char else f"{node.freq}"
    
    # Highlight if this is the current step
    color = 'yellow' if current_step == step_num else 'lightblue'
    if node.char:
        color = 'lightgreen' if current_step == step_num else 'lightcoral'
    
    if node.left:
        ax.plot([x, x - dx], [y, y - dy], 'b-', linewidth=2)
        draw_tree_stepwise(node.left, ax, x - dx, y - dy, dx * 0.6, dy, level + 1, 
                          current_step, step_num + 1)
        ax.text(x - dx, y - dy - 0.1, "0", ha='center', fontsize=8, weight='bold')
    
    if node.right:
        ax.plot([x, x + dx], [y, y - dy], 'b-', linewidth=2)
        draw_tree_stepwise(node.right, ax, x + dx, y - dy, dx * 0.6, dy, level + 1, 
                          current_step, step_num + 1)
        ax.text(x + dx, y - dy - 0.1, "1", ha='center', fontsize=8, weight='bold')
    
    circle = plt.Circle((x, y), 0.15, color=color, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, weight='bold', zorder=4)

def show_huffman():
    st.header("Huffman Coding")
    st.markdown("""
    Huffman Coding is a lossless data compression algorithm that assigns variable-length 
    binary codes to characters based on their frequency. More frequent characters get 
    shorter codes. The algorithm builds a binary tree by repeatedly merging the two 
    smallest frequency nodes.
    """)
    
    # Initialize session state
    if 'huff_step' not in st.session_state:
        st.session_state.huff_step = 0
    if 'huff_tree' not in st.session_state:
        st.session_state.huff_tree = None
    if 'huff_steps' not in st.session_state:
        st.session_state.huff_steps = []
    if 'huff_codes' not in st.session_state:
        st.session_state.huff_codes = {}
    if 'huff_freq' not in st.session_state:
        st.session_state.huff_freq = {}
    
    st.subheader("Input")
    text_input = st.text_input("Enter text to encode", value="AABBCDDD")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Build Tree", type="primary"):
            if not text_input:
                st.warning("Please enter some text!")
            else:
                root, steps, freq = build_huffman_tree_stepwise(text_input)
                st.session_state.huff_tree = root
                st.session_state.huff_steps = steps
                st.session_state.huff_freq = freq
                st.session_state.huff_step = len(steps)
                if root:
                    st.session_state.huff_codes = generate_codes(root)
                st.rerun()
    
    with col2:
        if st.button("Next Step") and st.session_state.huff_tree:
            if st.session_state.huff_step > 0:
                st.session_state.huff_step -= 1
                st.rerun()
    
    with col3:
        if st.button("Show Codes") and st.session_state.huff_codes:
            st.rerun()
    
    if st.session_state.huff_tree:
        root = st.session_state.huff_tree
        steps = st.session_state.huff_steps
        freq = st.session_state.huff_freq
        
        # Show frequencies
        st.subheader("Character Frequencies")
        freq_df = pd.DataFrame({
            'Symbol': list(freq.keys()),
            'Frequency': list(freq.values())
        })
        st.dataframe(freq_df, use_container_width=True, hide_index=True)
        
        # Show current step
        steps_to_show = steps[:st.session_state.huff_step]
        if steps_to_show:
            st.subheader(f"Tree Building Steps (Step {st.session_state.huff_step}/{len(steps)})")
            current_step = steps_to_show[-1] if steps_to_show else None
            
            if current_step:
                st.info(f"""
                **Step {current_step['step']}**: Merge nodes with smallest frequencies
                - Left: {current_step['left']} (freq: {current_step['left_freq']})
                - Right: {current_step['right']} (freq: {current_step['right_freq']})
                - Merged: Node with frequency {current_step['merged_freq']}
                - Remaining nodes in heap: {current_step['heap_size']}
                """)
        
        # Visualization: Tree
        st.subheader("Huffman Binary Tree Construction")
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 2)
        ax.axis('off')
        
        current_step_num = st.session_state.huff_step if steps_to_show else None
        draw_tree_stepwise(root, ax, 0, 0, 2.5, 1.2, 0, current_step_num, 0)
        ax.set_title("Huffman Binary Tree (0=left, 1=right)\nYellow/Green = Current Step", 
                    fontsize=14, weight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show all steps
        with st.expander("View All Tree Building Steps"):
            if steps:
                steps_df = pd.DataFrame(steps)
                st.dataframe(steps_df, use_container_width=True, hide_index=True)
        
        # Show codes
        if st.session_state.huff_codes:
            st.subheader("Huffman Codes")
            codes = st.session_state.huff_codes
            codes_df = pd.DataFrame({
                'Symbol': list(codes.keys()),
                'Code': list(codes.values()),
                'Code Length': [len(codes[char]) for char in codes.keys()],
                'Frequency': [freq[char] for char in codes.keys()]
            })
            st.dataframe(codes_df, use_container_width=True, hide_index=True)
            
            # Statistics
            original_bits = len(text_input) * 8
            encoded_bits = sum(len(codes[char]) * freq[char] for char in text_input)
            compression_ratio = (1 - encoded_bits / original_bits) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Size", f"{original_bits} bits")
            with col2:
                st.metric("Encoded Size", f"{encoded_bits} bits")
            with col3:
                st.metric("Compression Ratio", f"{compression_ratio:.2f}%")
            
            # Show encoded text
            encoded_text = ''.join(codes[char] for char in text_input)
            st.subheader("Encoded Text")
            st.code(encoded_text, language='text')
