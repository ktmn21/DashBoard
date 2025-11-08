import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_convolution(image, kernel):
    """Apply custom convolution kernel"""
    if len(image.shape) == 3:
        result = cv2.filter2D(image, -1, kernel)
    else:
        result = cv2.filter2D(image, -1, kernel)
    
    return result

def show_convolution():
    st.header("Custom Convolution")
    st.markdown("""
    Apply custom convolution kernels to images. Convolution is a fundamental operation 
    in image processing used for filtering, edge detection, and feature extraction.
    """)
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    st.sidebar.subheader("Kernel Selection")
    kernel_type = st.sidebar.selectbox(
        "Predefined Kernel",
        ["Custom", "Identity", "Edge Detection", "Emboss", "Box Blur", "Gaussian Blur"]
    )
    
    # Define kernels
    kernels = {
        "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        "Box Blur": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
        "Gaussian Blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    }
    
    if kernel_type == "Custom":
        st.sidebar.write("**Enter Custom 3x3 Kernel**")
        kernel_values = []
        for i in range(3):
            row = []
            for j in range(3):
                val = st.sidebar.number_input(
                    f"K[{i}][{j}]", value=0.0, key=f"k{i}{j}", format="%.2f"
                )
                row.append(val)
            kernel_values.append(row)
        kernel = np.array(kernel_values)
    else:
        kernel = kernels[kernel_type]
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Apply convolution
        convolved = apply_convolution(image_array, kernel)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Convolved Image")
            st.image(convolved, use_container_width=True)
        
        # Show kernel
        st.subheader("Convolution Kernel")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(kernel, cmap='coolwarm', interpolation='nearest')
        ax.set_title('Kernel Matrix')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{kernel[i, j]:.2f}', 
                       ha='center', va='center', color='black', fontweight='bold')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Side by side comparison
        st.subheader("Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(image_array)
        ax1.set_title('Original')
        ax1.axis('off')
        ax2.imshow(convolved)
        ax2.set_title(f'After {kernel_type} Convolution')
        ax2.axis('off')
        plt.tight_layout()
        st.pyplot(fig)

