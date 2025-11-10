import streamlit as st
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Algorithm & Image Processing Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-card {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .section-card:hover {
        transform: scale(1.05);
    }
    .daa-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .dip-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Home page
    if st.session_state.page == 'home':
        st.markdown('<h1 class="main-header">🎯 Algorithm & Image Processing Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="section-card daa-card">
                    <h2>🧩 DAA</h2>
                    <h3>Design and Analysis of Algorithms</h3>
                    <p>Explore various algorithms with interactive visualizations</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Enter DAA Section", key="daa_btn", use_container_width=True):
                st.session_state.page = 'daa'
                st.rerun()
        
        with col2:
            st.markdown("""
                <div class="section-card dip-card">
                    <h2>🖼 DIP</h2>
                    <h3>Digital Image Processing</h3>
                    <p>Process and visualize images with various techniques</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Enter DIP Section", key="dip_btn", use_container_width=True):
                st.session_state.page = 'dip'
                st.rerun()
    
    # DAA Page
    elif st.session_state.page == 'daa':
        st.title("🧩 Design and Analysis of Algorithms")
        
        if st.button("← Back to Home", key="back_daa"):
            st.session_state.page = 'home'
            st.rerun()
        
        st.sidebar.title("DAA Algorithms")
        main_section = st.sidebar.radio(
            "Select Section:",
            ["DP (Dynamic Programming)", "Huffman Coding", "Genetic Algorithm"]
        )
        
        if main_section == "DP (Dynamic Programming)":
            dp_algorithm = st.sidebar.radio(
                "Select DP Algorithm:",
                ["TSP (Traveling Salesman Problem)", "Coin Changing", "Knapsack"]
            )
            
            if dp_algorithm == "TSP (Traveling Salesman Problem)":
                from daa.tsp_bb import show_tsp_bb
                show_tsp_bb()
            elif dp_algorithm == "Coin Changing":
                from daa.dp_coin_change import show_coin_change
                show_coin_change()
            elif dp_algorithm == "Knapsack":
                from daa.knapsack import show_knapsack
                show_knapsack()
        elif main_section == "Huffman Coding":
            from daa.huffman import show_huffman
            show_huffman()
        elif main_section == "Genetic Algorithm":
            from daa.genetic import show_genetic
            show_genetic()
    
    # DIP Page
    elif st.session_state.page == 'dip':
        st.title("🖼 Digital Image Processing")
        
        if st.button("← Back to Home", key="back_dip"):
            st.session_state.page = 'home'
            st.rerun()
        
        st.sidebar.title("DIP Operations")
        operation = st.sidebar.radio(
            "Select Operation:",
            ["Grayscale", "Negative", "Histogram Equalization", 
             "Brightness & Contrast", "Blur", "Edge Detection",
             "Thresholding", "Sharpen", "Convolution", 
             "RGB Channels", "ANN & CNN Visualization"]
        )
        
        if operation == "Grayscale":
            from dip.grayscale import show_grayscale
            show_grayscale()
        elif operation == "Negative":
            from dip.negative import show_negative
            show_negative()
        elif operation == "Histogram Equalization":
            from dip.histogram_eq import show_histogram_eq
            show_histogram_eq()
        elif operation == "Brightness & Contrast":
            from dip.brightness_contrast import show_brightness_contrast
            show_brightness_contrast()
        elif operation == "Blur":
            from dip.blur import show_blur
            show_blur()
        elif operation == "Edge Detection":
            from dip.edge_detection import show_edge_detection
            show_edge_detection()
        elif operation == "Thresholding":
            from dip.threshold import show_threshold
            show_threshold()
        elif operation == "Sharpen":
            from dip.sharpen import show_sharpen
            show_sharpen()
        elif operation == "Convolution":
            from dip.convolution import show_convolution
            show_convolution()
        elif operation == "RGB Channels":
            from dip.channels import show_channels
            show_channels()
        elif operation == "ANN & CNN Visualization":
            from dip.ann_cnn import show_ann_cnn
            show_ann_cnn()

if __name__ == "__main__":
    main()

