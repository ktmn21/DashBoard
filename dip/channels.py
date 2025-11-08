import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_channels():
    st.header("RGB Channel Separation")
    st.markdown("""
    Separate and visualize individual color channels (Red, Green, Blue) of an RGB image.
    Each channel represents the intensity of that color component.
    """)
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        if len(image_array.shape) == 3:
            # Separate channels
            r_channel = image_array.copy()
            r_channel[:, :, [1, 2]] = 0  # Keep only red
            
            g_channel = image_array.copy()
            g_channel[:, :, [0, 2]] = 0  # Keep only green
            
            b_channel = image_array.copy()
            b_channel[:, :, [0, 1]] = 0  # Keep only blue
            
            # Grayscale versions of individual channels
            r_gray = image_array[:, :, 0]
            g_gray = image_array[:, :, 1]
            b_gray = image_array[:, :, 2]
            
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
            # Display channels
            st.subheader("Color Channels")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Red Channel**")
                st.image(r_channel, use_container_width=True)
                st.image(r_gray, use_container_width=True, channels='GRAY')
            
            with col2:
                st.write("**Green Channel**")
                st.image(g_channel, use_container_width=True)
                st.image(g_gray, use_container_width=True, channels='GRAY')
            
            with col3:
                st.write("**Blue Channel**")
                st.image(b_channel, use_container_width=True)
                st.image(b_gray, use_container_width=True, channels='GRAY')
            
            # Histograms
            st.subheader("Channel Histograms")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Individual channel histograms
            ax1.hist(r_gray.ravel(), bins=256, color='red', alpha=0.5, label='Red')
            ax1.hist(g_gray.ravel(), bins=256, color='green', alpha=0.5, label='Green')
            ax1.hist(b_gray.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
            ax1.set_xlabel('Pixel Intensity')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Individual Channel Histograms')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Overlaid channels
            ax2.imshow(image_array)
            ax2.set_title('Original RGB Image')
            ax2.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistics
            st.subheader("Channel Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Red Mean", f"{np.mean(r_gray):.2f}")
            with col2:
                st.metric("Green Mean", f"{np.mean(g_gray):.2f}")
            with col3:
                st.metric("Blue Mean", f"{np.mean(b_gray):.2f}")
            with col4:
                st.metric("Overall Mean", f"{np.mean(image_array):.2f}")
        else:
            st.warning("Image is already grayscale. Please upload a color image.")

