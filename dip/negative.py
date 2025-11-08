import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def image_negative(image):
    """Create negative of an image"""
    if len(image.shape) == 3:
        return 255 - image
    else:
        return 255 - image

def show_negative():
    st.header("Image Negative")
    st.markdown("""
    Create the negative (inverse) of an image by subtracting each pixel value from 255.
    This inverts the brightness levels of the image.
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
        
        # Create negative
        negative_image = image_negative(image_array)
        
        with col2:
            st.subheader("Negative Image")
            st.image(negative_image, use_container_width=True)
        
        # Side by side comparison
        st.subheader("Side by Side Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(image_array)
        ax1.set_title('Original')
        ax1.axis('off')
        ax2.imshow(negative_image)
        ax2.set_title('Negative')
        ax2.axis('off')
        plt.tight_layout()
        st.pyplot(fig)

