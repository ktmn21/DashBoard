import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Adjust brightness and contrast of an image"""
    # Brightness: add/subtract value
    adjusted = image.astype(np.float32) + brightness
    
    # Contrast: multiply by factor
    adjusted = adjusted * contrast
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted

def show_brightness_contrast():
    st.header("Brightness & Contrast Adjustment")
    st.markdown("""
    Adjust the brightness (additive) and contrast (multiplicative) of an image.
    - **Brightness**: Adds or subtracts a constant value to all pixels
    - **Contrast**: Multiplies pixel values by a factor
    """)
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        st.sidebar.subheader("Adjustment Parameters")
        brightness = st.sidebar.slider("Brightness", -100, 100, 0)
        contrast = st.sidebar.slider("Contrast", 0.0, 3.0, 1.0, 0.1)
        
        # Apply adjustments
        adjusted_image = adjust_brightness_contrast(image_array, brightness, contrast)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            st.write(f"Mean intensity: {np.mean(image_array):.2f}")
        
        with col2:
            st.subheader("Adjusted Image")
            st.image(adjusted_image, use_container_width=True)
            st.write(f"Mean intensity: {np.mean(adjusted_image):.2f}")
        
        # Histogram comparison
        st.subheader("Histogram Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        if len(image_array.shape) == 3:
            # RGB histograms
            for i, color in enumerate(['Red', 'Green', 'Blue']):
                ax1.hist(image_array[:,:,i].ravel(), bins=256, alpha=0.5, 
                        label=color, color=color.lower())
                ax2.hist(adjusted_image[:,:,i].ravel(), bins=256, alpha=0.5, 
                        label=color, color=color.lower())
        else:
            ax1.hist(image_array.ravel(), bins=256, color='gray', alpha=0.7)
            ax2.hist(adjusted_image.ravel(), bins=256, color='gray', alpha=0.7)
        
        ax1.set_title('Original Histogram')
        ax1.set_xlabel('Pixel Intensity')
        ax1.set_ylabel('Frequency')
        ax1.legend() if len(image_array.shape) == 3 else None
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title(f'Adjusted Histogram (B={brightness}, C={contrast:.1f})')
        ax2.set_xlabel('Pixel Intensity')
        ax2.set_ylabel('Frequency')
        ax2.legend() if len(image_array.shape) == 3 else None
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

