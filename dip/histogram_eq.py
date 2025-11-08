import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def histogram_equalization(image):
    """Apply histogram equalization to grayscale image"""
    if len(image.shape) == 3:
        # Convert to grayscale first
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate histogram
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    
    # Calculate cumulative distribution
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    
    # Apply equalization
    equalized = cdf_normalized[gray].astype(np.uint8)
    
    return equalized, hist, cdf

def show_histogram_eq():
    st.header("Histogram Equalization")
    st.markdown("""
    Histogram equalization improves image contrast by redistributing pixel intensities 
    to use the full dynamic range. This enhances details in both dark and bright regions.
    """)
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray_original = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_original = image_array
        
        # Apply histogram equalization
        equalized, hist_orig, cdf = histogram_equalization(image_array)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(gray_original, use_container_width=True, channels='GRAY')
        
        with col2:
            st.subheader("Equalized Image")
            st.image(equalized, use_container_width=True, channels='GRAY')
        
        # Histogram comparison
        st.subheader("Histogram Comparison")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Original image
        ax1.imshow(gray_original, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Original histogram
        ax2.hist(gray_original.ravel(), bins=256, color='blue', alpha=0.7)
        ax2.set_title('Original Histogram')
        ax2.set_xlabel('Pixel Intensity')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Equalized image
        ax3.imshow(equalized, cmap='gray')
        ax3.set_title('Equalized Image')
        ax3.axis('off')
        
        # Equalized histogram
        ax4.hist(equalized.ravel(), bins=256, color='green', alpha=0.7)
        ax4.set_title('Equalized Histogram')
        ax4.set_xlabel('Pixel Intensity')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # CDF plot
        st.subheader("Cumulative Distribution Function (CDF)")
        fig2, ax = plt.subplots(figsize=(10, 5))
        hist_eq, _ = np.histogram(equalized.flatten(), 256, [0, 256])
        cdf_eq = hist_eq.cumsum()
        cdf_eq_normalized = cdf_eq * 255 / cdf_eq[-1]
        
        ax.plot(cdf, label='Original CDF', linewidth=2)
        ax.plot(cdf_eq_normalized, label='Equalized CDF', linewidth=2)
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Cumulative Frequency')
        ax.set_title('CDF Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

