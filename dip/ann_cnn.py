import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def create_ann_diagram():
    """Create a simple ANN architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Input layer
    input_x = 1
    input_y = 4
    ax.scatter([input_x] * 4, [input_y + i*0.5 for i in range(4)], 
              s=500, c='blue', zorder=3)
    ax.text(input_x, input_y + 1.5, 'Input Layer\n(4 neurons)', 
           ha='center', fontsize=10, weight='bold')
    
    # Hidden layer 1
    hidden1_x = 3
    hidden1_y = 3.5
    ax.scatter([hidden1_x] * 5, [hidden1_y + i*0.4 for i in range(5)], 
              s=500, c='green', zorder=3)
    ax.text(hidden1_x, hidden1_y + 1, 'Hidden Layer 1\n(5 neurons)', 
           ha='center', fontsize=10, weight='bold')
    
    # Hidden layer 2
    hidden2_x = 5
    hidden2_y = 3.5
    ax.scatter([hidden2_x] * 3, [hidden2_y + i*0.4 for i in range(3)], 
              s=500, c='orange', zorder=3)
    ax.text(hidden2_x, hidden2_y + 0.6, 'Hidden Layer 2\n(3 neurons)', 
           ha='center', fontsize=10, weight='bold')
    
    # Output layer
    output_x = 7
    output_y = 4
    ax.scatter([output_x] * 2, [output_y + i*0.5 for i in range(2)], 
              s=500, c='red', zorder=3)
    ax.text(output_x, output_y + 0.5, 'Output Layer\n(2 neurons)', 
           ha='center', fontsize=10, weight='bold')
    
    # Draw connections
    for i in range(4):
        for j in range(5):
            ax.plot([input_x, hidden1_x], 
                   [input_y + i*0.5, hidden1_y + j*0.4], 
                   'gray', alpha=0.2, linewidth=0.5)
    
    for i in range(5):
        for j in range(3):
            ax.plot([hidden1_x, hidden2_x], 
                   [hidden1_y + i*0.4, hidden2_y + j*0.4], 
                   'gray', alpha=0.2, linewidth=0.5)
    
    for i in range(3):
        for j in range(2):
            ax.plot([hidden2_x, output_x], 
                   [hidden2_y + i*0.4, output_y + j*0.5], 
                   'gray', alpha=0.2, linewidth=0.5)
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.set_title('Artificial Neural Network (ANN) Architecture', fontsize=14, weight='bold')
    ax.axis('off')
    
    return fig

def create_cnn_diagram():
    """Create a simple CNN architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Input image
    input_rect = plt.Rectangle((0.5, 3), 1, 1, fill=True, color='lightblue', edgecolor='black')
    ax.add_patch(input_rect)
    ax.text(1, 3.5, 'Input\nImage\n28x28', ha='center', va='center', fontsize=9, weight='bold')
    
    # Conv Layer 1
    conv1_rect = plt.Rectangle((2, 2.5), 1.5, 2, fill=True, color='lightgreen', edgecolor='black')
    ax.add_patch(conv1_rect)
    ax.text(2.75, 3.5, 'Conv1\n32 filters\n26x26', ha='center', va='center', fontsize=9, weight='bold')
    
    # Pooling 1
    pool1_rect = plt.Rectangle((4, 2.5), 1.5, 2, fill=True, color='lightyellow', edgecolor='black')
    ax.add_patch(pool1_rect)
    ax.text(4.75, 3.5, 'MaxPool1\n13x13', ha='center', va='center', fontsize=9, weight='bold')
    
    # Conv Layer 2
    conv2_rect = plt.Rectangle((6, 2.5), 1.5, 2, fill=True, color='lightgreen', edgecolor='black')
    ax.add_patch(conv2_rect)
    ax.text(6.75, 3.5, 'Conv2\n64 filters\n11x11', ha='center', va='center', fontsize=9, weight='bold')
    
    # Pooling 2
    pool2_rect = plt.Rectangle((8, 2.5), 1.5, 2, fill=True, color='lightyellow', edgecolor='black')
    ax.add_patch(pool2_rect)
    ax.text(8.75, 3.5, 'MaxPool2\n5x5', ha='center', va='center', fontsize=9, weight='bold')
    
    # Flatten
    flatten_rect = plt.Rectangle((10, 3), 1, 1, fill=True, color='lightcoral', edgecolor='black')
    ax.add_patch(flatten_rect)
    ax.text(10.5, 3.5, 'Flatten', ha='center', va='center', fontsize=9, weight='bold')
    
    # Dense layers
    dense1_rect = plt.Rectangle((12, 2.5), 1, 2, fill=True, color='lightpink', edgecolor='black')
    ax.add_patch(dense1_rect)
    ax.text(12.5, 3.5, 'Dense\n128', ha='center', va='center', fontsize=9, weight='bold')
    
    output_rect = plt.Rectangle((13.5, 3), 1, 1, fill=True, color='lightsteelblue', edgecolor='black')
    ax.add_patch(output_rect)
    ax.text(14, 3.5, 'Output\n10', ha='center', va='center', fontsize=9, weight='bold')
    
    # Arrows
    arrows_x = [1.5, 3.5, 5.5, 7.5, 9.5, 11, 13]
    for x in arrows_x:
        ax.arrow(x, 3.5, 0.5, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.set_title('Convolutional Neural Network (CNN) Architecture', fontsize=14, weight='bold')
    ax.axis('off')
    
    return fig

def visualize_feature_maps(image, num_filters=8):
    """Visualize CNN-like feature maps using simple filters"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create different filters to simulate feature maps
    filters = [
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),  # Horizontal edge
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),  # Vertical edge
        np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),  # Diagonal
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),  # Sharpen
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,  # Blur
        np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),  # Emboss
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),     # Laplacian
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),  # Sobel
    ]
    
    feature_maps = []
    for i, kernel in enumerate(filters[:num_filters]):
        feature_map = cv2.filter2D(gray, -1, kernel)
        feature_maps.append(feature_map)
    
    return feature_maps

def show_ann_cnn():
    st.header("ANN & CNN Visualization")
    st.markdown("""
    Visualize the architecture and behavior of Artificial Neural Networks (ANN) 
    and Convolutional Neural Networks (CNN).
    """)
    
    model_type = st.radio("Select Model Type:", ["ANN (Artificial Neural Network)", "CNN (Convolutional Neural Network)"])
    
    if model_type == "ANN (Artificial Neural Network)":
        st.subheader("ANN Architecture")
        st.markdown("""
        An Artificial Neural Network consists of:
        - **Input Layer**: Receives input features
        - **Hidden Layers**: Process information through weighted connections
        - **Output Layer**: Produces final predictions
        """)
        
        fig = create_ann_diagram()
        st.pyplot(fig)
        
        st.subheader("ANN Forward Pass Simulation")
        st.write("Simulate how data flows through the network:")
        
        # Simple forward pass visualization
        input_data = st.text_input("Input values (comma-separated)", value="0.5, 0.3, 0.8, 0.2")
        
        if st.button("Run Forward Pass"):
            try:
                inputs = [float(x.strip()) for x in input_data.split(',')]
                if len(inputs) == 4:
                    # Simulate simple forward pass
                    # Hidden layer 1 (5 neurons)
                    hidden1 = [sum(inputs) / len(inputs) + np.random.normal(0, 0.1) 
                              for _ in range(5)]
                    hidden1 = [max(0, h) for h in hidden1]  # ReLU
                    
                    # Hidden layer 2 (3 neurons)
                    hidden2 = [sum(hidden1) / len(hidden1) + np.random.normal(0, 0.1) 
                              for _ in range(3)]
                    hidden2 = [max(0, h) for h in hidden2]  # ReLU
                    
                    # Output layer (2 neurons)
                    outputs = [sum(hidden2) / len(hidden2) + np.random.normal(0, 0.1) 
                              for _ in range(2)]
                    outputs = [1 / (1 + np.exp(-o)) for o in outputs]  # Sigmoid
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write("**Input**")
                        st.write(inputs)
                    with col2:
                        st.write("**Hidden 1**")
                        st.write([f"{h:.3f}" for h in hidden1])
                    with col3:
                        st.write("**Hidden 2**")
                        st.write([f"{h:.3f}" for h in hidden2])
                    with col4:
                        st.write("**Output**")
                        st.write([f"{o:.3f}" for o in outputs])
                else:
                    st.warning("Please enter exactly 4 input values")
            except:
                st.error("Invalid input format")
    
    else:  # CNN
        st.subheader("CNN Architecture")
        st.markdown("""
        A Convolutional Neural Network consists of:
        - **Convolutional Layers**: Extract features using filters
        - **Pooling Layers**: Reduce spatial dimensions
        - **Fully Connected Layers**: Make final predictions
        """)
        
        fig = create_cnn_diagram()
        st.pyplot(fig)
        
        st.subheader("Feature Map Visualization")
        st.write("Upload an image to see how CNN filters extract features:")
        
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'], key="cnn_upload")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            num_filters = st.slider("Number of Feature Maps", 4, 8, 8)
            
            feature_maps = visualize_feature_maps(image_array, num_filters)
            
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
            st.subheader("Feature Maps (Simulated CNN Filters)")
            # Display feature maps in a grid
            cols = 4
            rows = (num_filters + cols - 1) // cols
            
            for row in range(rows):
                cols_list = st.columns(cols)
                for col in range(cols):
                    idx = row * cols + col
                    if idx < len(feature_maps):
                        with cols_list[col]:
                            st.image(feature_maps[idx], use_container_width=True, 
                                   channels='GRAY', caption=f'Filter {idx+1}')

