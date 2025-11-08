import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_sharpen(image, strength=1.0):
    """Apply sharpening filter"""
    # Sharpen kernel
    kernel = np.array([
        [0, -strength, 0],
        [-strength, 1 + 4*strength, -strength],
        [0, -strength, 0]
    ])
    
    if len(image.shape) == 3:
        sharpened = cv2.filter2D(image, -1, kernel)
    else:
        sharpened = cv2.filter2D(image, -1, kernel)
    
    # Clip values
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def show_sharpen():
    st.header("Image Sharpening")
    st.markdown("""
    Sharpen images by enhancing edges and fine details. Uses a convolution kernel 
    that emphasizes high-frequency components.
    """)
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        st.sidebar.subheader("Sharpening Parameters")
        strength = st.sidebar.slider("Sharpening Strength", 0.0, 2.0, 1.0, 0.1)
        
        # Apply sharpening
        sharpened = apply_sharpen(image_array, strength)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Sharpened Image")
            st.image(sharpened, use_container_width=True)
            st.write(f"Strength: {strength}")
        
        # Side by side comparison
        st.subheader("Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(image_array)
        ax1.set_title('Original')
        ax1.axis('off')
        ax2.imshow(sharpened)
        ax2.set_title(f'Sharpened (Strength: {strength})')
        ax2.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show difference
        st.subheader("Difference (Sharpened - Original)")
        difference = cv2.absdiff(image_array, sharpened)
        st.image(difference, use_container_width=True)

