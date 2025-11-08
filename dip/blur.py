import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_gaussian_blur(image, kernel_size):
    """Apply Gaussian blur"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_median_blur(image, kernel_size):
    """Apply median blur"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)

def show_blur():
    st.header("Image Blurring")
    st.markdown("""
    Apply blurring filters to reduce noise or create artistic effects.
    - **Gaussian Blur**: Smooths image using Gaussian distribution
    - **Median Blur**: Reduces salt-and-pepper noise by replacing pixels with median
    """)
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        st.sidebar.subheader("Blur Parameters")
        blur_type = st.sidebar.radio("Blur Type", ["Gaussian", "Median"])
        kernel_size = st.sidebar.slider("Kernel Size", 3, 31, 5, 2)
        
        # Apply blur
        if blur_type == "Gaussian":
            blurred_image = apply_gaussian_blur(image_array, kernel_size)
        else:
            blurred_image = apply_median_blur(image_array, kernel_size)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader(f"{blur_type} Blurred Image")
            st.image(blurred_image, use_container_width=True)
            st.write(f"Kernel Size: {kernel_size}x{kernel_size}")
        
        # Side by side comparison
        st.subheader("Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(image_array)
        ax1.set_title('Original')
        ax1.axis('off')
        ax2.imshow(blurred_image)
        ax2.set_title(f'{blur_type} Blur (Kernel: {kernel_size}x{kernel_size})')
        ax2.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show difference
        st.subheader("Difference (Original - Blurred)")
        difference = cv2.absdiff(image_array, blurred_image)
        st.image(difference, use_container_width=True)

