import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from collections import Counter, defaultdict
import heapq

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    """Build Huffman tree from text"""
    freq = Counter(text)
    heap = [HuffmanNode(char=char, freq=freq) for char, freq in freq.items()]
    heapq.heapify(heap)
    steps = []
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = HuffmanNode(
            freq=left.freq + right.freq,
            left=left,
            right=right
        )
        heapq.heappush(heap, merged)
        steps.append({
            'left': left.char if left.char else f"Node({left.freq})",
            'right': right.char if right.char else f"Node({right.freq})",
            'merged_freq': merged.freq
        })
    
    return heap[0] if heap else None, steps

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

def draw_tree(node, ax, x=0, y=0, dx=1, dy=1, level=0):
    """Recursively draw Huffman tree"""
    if node is None:
        return
    
    label = f"{node.char}\n({node.freq})" if node.char else f"{node.freq}"
    
    if node.left:
        ax.plot([x, x - dx], [y, y - dy], 'b-', linewidth=2)
        draw_tree(node.left, ax, x - dx, y - dy, dx * 0.6, dy, level + 1)
        ax.text(x - dx, y - dy - 0.1, "0", ha='center', fontsize=8)
    
    if node.right:
        ax.plot([x, x + dx], [y, y - dy], 'b-', linewidth=2)
        draw_tree(node.right, ax, x + dx, y - dy, dx * 0.6, dy, level + 1)
        ax.text(x + dx, y - dy - 0.1, "1", ha='center', fontsize=8)
    
    ax.plot(x, y, 'ro', markersize=20)
    ax.text(x, y, label, ha='center', va='center', fontsize=9, weight='bold')

def show_huffman():
    st.header("Huffman Coding")
    st.markdown("""
    Huffman Coding is a lossless data compression algorithm that assigns variable-length 
    binary codes to characters based on their frequency. More frequent characters get 
    shorter codes.
    """)
    
    st.subheader("Input")
    text_input = st.text_input("Enter text to encode", value="AABBCDDD")
    
    if st.button("Run Algorithm", type="primary"):
        if not text_input:
            st.warning("Please enter some text!")
            return
        
        # Build tree
        root, steps = build_huffman_tree(text_input)
        
        if root is None:
            st.error("Cannot build tree for empty input!")
            return
        
        # Generate codes
        codes = generate_codes(root)
        
        # Calculate statistics
        freq = Counter(text_input)
        original_bits = len(text_input) * 8
        encoded_bits = sum(len(codes[char]) * freq[char] for char in text_input)
        compression_ratio = (1 - encoded_bits / original_bits) * 100
        
        st.success("Huffman Tree Built Successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Character Frequencies")
            freq_df = pd.DataFrame({
                'Character': list(freq.keys()),
                'Frequency': list(freq.values())
            })
            st.dataframe(freq_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Huffman Codes")
            codes_df = pd.DataFrame({
                'Character': list(codes.keys()),
                'Code': list(codes.values()),
                'Length': [len(codes[char]) for char in codes.keys()]
            })
            st.dataframe(codes_df, use_container_width=True, hide_index=True)
        
        st.subheader("Statistics")
        st.write(f"Original size: {original_bits} bits")
        st.write(f"Encoded size: {encoded_bits} bits")
        st.write(f"Compression ratio: {compression_ratio:.2f}%")
        
        # Visualization: Tree
        st.subheader("Huffman Tree Visualization")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 1)
        ax.axis('off')
        
        draw_tree(root, ax, 0, 0, 2, 1)
        ax.set_title("Huffman Binary Tree (0=left, 1=right)", fontsize=14, weight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show steps
        with st.expander("View Tree Building Steps"):
            if steps:
                steps_df = pd.DataFrame(steps)
                st.dataframe(steps_df, use_container_width=True)
        
        # Show encoded text
        encoded_text = ''.join(codes[char] for char in text_input)
        st.subheader("Encoded Text")
        st.code(encoded_text)

