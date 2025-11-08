import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_threshold(image, threshold_value, threshold_type='binary'):
    """Apply thresholding to image"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    if threshold_type == 'binary':
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    elif threshold_type == 'binary_inv':
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    elif threshold_type == 'otsu':
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_type == 'adaptive_mean':
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    elif threshold_type == 'adaptive_gaussian':
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    
    return thresh

def show_threshold():
    st.header("Image Thresholding")
    st.markdown("""
    Thresholding converts grayscale images to binary images by setting pixels 
    above/below a threshold value. Useful for segmentation and object detection.
    """)
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        st.sidebar.subheader("Threshold Parameters")
        threshold_type = st.sidebar.selectbox(
            "Threshold Type",
            ["binary", "binary_inv", "otsu", "adaptive_mean", "adaptive_gaussian"]
        )
        
        if threshold_type in ['binary', 'binary_inv']:
            threshold_value = st.sidebar.slider("Threshold Value", 0, 255, 127)
        else:
            threshold_value = 127  # Not used for adaptive/otsu
        
        # Apply threshold
        thresholded = apply_threshold(image_array, threshold_value, threshold_type)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Thresholded Image")
            st.image(thresholded, use_container_width=True, channels='GRAY')
            st.write(f"Type: {threshold_type}")
            if threshold_type in ['binary', 'binary_inv']:
                st.write(f"Threshold: {threshold_value}")
        
        # Show histogram with threshold line
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        st.subheader("Histogram with Threshold")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(gray.ravel(), bins=256, color='blue', alpha=0.7)
        if threshold_type in ['binary', 'binary_inv']:
            ax.axvline(threshold_value, color='red', linestyle='--', 
                      linewidth=2, label=f'Threshold: {threshold_value}')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('Grayscale Histogram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

