import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rgb_to_grayscale(image):
    """Convert RGB image to grayscale"""
    if len(image.shape) == 3:
        # Using weighted method (luminosity)
        gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return gray.astype(np.uint8)
    return image

def show_grayscale():
    st.header("Grayscale Conversion")
    st.markdown("""
    Convert a color image to grayscale. Grayscale images contain only intensity 
    information, removing color but preserving luminance.
    """)
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            st.write(f"Shape: {image_array.shape}")
        
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray_image = rgb_to_grayscale(image_array)
        else:
            gray_image = image_array
        
        with col2:
            st.subheader("Grayscale Image")
            st.image(gray_image, use_container_width=True, channels='GRAY')
            st.write(f"Shape: {gray_image.shape}")
        
        # Show histogram
        st.subheader("Intensity Histogram")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        if len(image_array.shape) == 3:
            # Original RGB histogram
            for i, color in enumerate(['Red', 'Green', 'Blue']):
                ax1.hist(image_array[:,:,i].ravel(), bins=256, alpha=0.5, 
                        label=color, color=color.lower())
            ax1.set_title('Original RGB Histogram')
            ax1.set_xlabel('Pixel Intensity')
            ax1.set_ylabel('Frequency')
            ax1.legend()
        else:
            ax1.hist(image_array.ravel(), bins=256, color='gray', alpha=0.7)
            ax1.set_title('Original Grayscale Histogram')
            ax1.set_xlabel('Pixel Intensity')
            ax1.set_ylabel('Frequency')
        
        # Grayscale histogram
        ax2.hist(gray_image.ravel(), bins=256, color='black', alpha=0.7)
        ax2.set_title('Grayscale Histogram')
        ax2.set_xlabel('Pixel Intensity')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

