import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_sobel(image):
    """Apply Sobel edge detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Sobel X and Y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    sobel_combined = np.uint8(np.absolute(sobel_combined))
    
    return sobel_combined, sobelx, sobely

def apply_canny(image, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def show_edge_detection():
    st.header("Edge Detection")
    st.markdown("""
    Detect edges in images using gradient-based methods.
    - **Sobel**: Uses first-order derivatives to detect edges
    - **Canny**: Multi-stage algorithm with better edge localization
    """)
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        st.sidebar.subheader("Edge Detection Parameters")
        method = st.sidebar.radio("Method", ["Sobel", "Canny"])
        
        if method == "Canny":
            low_threshold = st.sidebar.slider("Low Threshold", 0, 200, 50)
            high_threshold = st.sidebar.slider("High Threshold", 0, 300, 150)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Apply edge detection
        if method == "Sobel":
            edges, sobelx, sobely = apply_sobel(image_array)
            
            with col2:
                st.subheader("Sobel Edge Detection")
                st.image(edges, use_container_width=True, channels='GRAY')
            
            # Show X and Y gradients
            st.subheader("Sobel Gradients")
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(np.absolute(sobelx), cmap='gray')
            ax1.set_title('Sobel X (Vertical Edges)')
            ax1.axis('off')
            ax2.imshow(np.absolute(sobely), cmap='gray')
            ax2.set_title('Sobel Y (Horizontal Edges)')
            ax2.axis('off')
            ax3.imshow(edges, cmap='gray')
            ax3.set_title('Combined Sobel')
            ax3.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
        
        else:  # Canny
            edges = apply_canny(image_array, low_threshold, high_threshold)
            
            with col2:
                st.subheader("Canny Edge Detection")
                st.image(edges, use_container_width=True, channels='GRAY')
                st.write(f"Thresholds: Low={low_threshold}, High={high_threshold}")
            
            # Overlay edges on original
            st.subheader("Edges Overlaid on Original")
            overlay = image_array.copy()
            if len(overlay.shape) == 3:
                overlay[edges > 0] = [255, 0, 0]  # Red edges
            else:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
                overlay[edges > 0] = [255, 0, 0]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(overlay)
            ax.set_title('Canny Edges Overlay')
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)

