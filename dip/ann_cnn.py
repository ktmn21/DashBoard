import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def calculate_output_size(input_size, kernel_size, stride=1, padding=0):
    """Calculate output size after convolution"""
    return (input_size - kernel_size + 2 * padding) // stride + 1

def calculate_pool_output_size(input_size, pool_size, stride=None):
    """Calculate output size after pooling"""
    if stride is None:
        stride = pool_size
    return input_size // stride

def calculate_dense_input_size(input_size, conv_configs, pool_configs):
    """Calculate the number of input neurons that will go into the dense layers"""
    current_size = input_size
    last_num_filters = 1  # Start with 1 channel (grayscale input)
    
    # Process through all conv and pool layers
    for i in range(len(conv_configs)):
        # Apply convolution
        kernel_size = conv_configs[i]['kernel_size']
        current_size = calculate_output_size(current_size, kernel_size)
        last_num_filters = conv_configs[i]['filters']  # Update to number of filters from this layer
        
        # Apply pooling
        if i < len(pool_configs):
            pool_size = pool_configs[i]['pool_size']
            current_size = calculate_pool_output_size(current_size, pool_size)
    
    # After flattening: feature_map_size * feature_map_size * num_channels
    flattened_size = current_size * current_size * last_num_filters
    
    return flattened_size, current_size, last_num_filters

def resize_kernel(kernel, target_size):
    """Resize a kernel to target size using numpy interpolation"""
    if kernel.shape[0] == target_size:
        return kernel
    
    old_size = kernel.shape[0]
    
    # Validate inputs
    if old_size <= 0 or target_size <= 0:
        raise ValueError(f"Invalid kernel sizes: old_size={old_size}, target_size={target_size}")
    
    # Special case for very small kernels
    if target_size == 1:
        return np.array([[np.mean(kernel)]])
    
    # Use numpy's interpolation - create coordinate mapping
    # Map from new coordinates to old coordinates
    resized = np.zeros((target_size, target_size), dtype=kernel.dtype)
    
    for i in range(target_size):
        for j in range(target_size):
            # Map new coordinates to old coordinates
            y_old = (i / (target_size - 1)) * (old_size - 1) if target_size > 1 else 0
            x_old = (j / (target_size - 1)) * (old_size - 1) if target_size > 1 else 0
            
            # Get integer and fractional parts
            y0 = int(np.floor(y_old))
            y1 = min(int(np.ceil(y_old)), old_size - 1)
            x0 = int(np.floor(x_old))
            x1 = min(int(np.ceil(x_old)), old_size - 1)
            
            # Bilinear interpolation weights
            dy = y_old - y0
            dx = x_old - x0
            
            # Handle edge cases
            if y0 == y1 and x0 == x1:
                resized[i, j] = kernel[y0, x0]
            elif y0 == y1:
                resized[i, j] = kernel[y0, x0] * (1 - dx) + kernel[y0, x1] * dx
            elif x0 == x1:
                resized[i, j] = kernel[y0, x0] * (1 - dy) + kernel[y1, x0] * dy
            else:
                resized[i, j] = (kernel[y0, x0] * (1 - dy) * (1 - dx) +
                                kernel[y0, x1] * (1 - dy) * dx +
                                kernel[y1, x0] * dy * (1 - dx) +
                                kernel[y1, x1] * dy * dx)
    
    return resized

def create_cnn_diagram(layers_config, input_size):
    """Create a flexible CNN architecture diagram based on user configuration"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x_pos = 0.5
    y_center = 3.5
    layer_width = 1.2
    spacing = 0.3
    
    # Input image
    input_rect = plt.Rectangle((x_pos, y_center - 0.5), layer_width, 1, 
                              fill=True, color='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(x_pos + layer_width/2, y_center, f'Input\nImage\n{input_size}x{input_size}', 
           ha='center', va='center', fontsize=9, weight='bold')
    
    x_pos += layer_width + spacing
    current_size = input_size
    
    # Process each layer
    for i, layer in enumerate(layers_config):
        layer_type = layer['type']
        
        if layer_type == 'conv':
            filters = layer['filters']
            kernel_size = layer['kernel_size']
            output_size = calculate_output_size(current_size, kernel_size)
            
            rect = plt.Rectangle((x_pos, y_center - 0.75), layer_width, 1.5, 
                               fill=True, color='lightgreen', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_pos + layer_width/2, y_center, 
                   f'Conv{i+1}\n{filters} filters\n{output_size}x{output_size}', 
                   ha='center', va='center', fontsize=9, weight='bold')
            
            current_size = output_size
            x_pos += layer_width + spacing
            
            # Draw arrow
            ax.arrow(x_pos - spacing - 0.1, y_center, 0.1, 0, 
                    head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=1.5)
            
        elif layer_type == 'pool':
            pool_size = layer['pool_size']
            output_size = calculate_pool_output_size(current_size, pool_size)
            
            rect = plt.Rectangle((x_pos, y_center - 0.75), layer_width, 1.5, 
                               fill=True, color='lightyellow', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_pos + layer_width/2, y_center, 
                   f'MaxPool{i+1}\n{pool_size}x{pool_size}\n{output_size}x{output_size}', 
                   ha='center', va='center', fontsize=9, weight='bold')
            
            current_size = output_size
            x_pos += layer_width + spacing
            
            # Draw arrow
            ax.arrow(x_pos - spacing - 0.1, y_center, 0.1, 0, 
                    head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=1.5)
            
        elif layer_type == 'flatten':
            rect = plt.Rectangle((x_pos, y_center - 0.5), layer_width, 1, 
                               fill=True, color='lightcoral', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_pos + layer_width/2, y_center, 'Flatten', 
                   ha='center', va='center', fontsize=9, weight='bold')
            x_pos += layer_width + spacing
            
            # Draw arrow
            ax.arrow(x_pos - spacing - 0.1, y_center, 0.1, 0, 
                    head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=1.5)
            
        elif layer_type == 'dense':
            units = layer['units']
            rect = plt.Rectangle((x_pos, y_center - 0.75), layer_width, 1.5, 
                               fill=True, color='lightpink', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_pos + layer_width/2, y_center, f'Dense\n{units}', 
                   ha='center', va='center', fontsize=9, weight='bold')
            x_pos += layer_width + spacing
            
            # Draw arrow
            if i < len(layers_config) - 1:  # Don't draw arrow after last layer
                ax.arrow(x_pos - spacing - 0.1, y_center, 0.1, 0, 
                        head_width=0.15, head_length=0.08, fc='black', ec='black', linewidth=1.5)
    
    # Output layer (if not already added)
    if layers_config[-1]['type'] != 'dense' or 'output_units' in layers_config[-1]:
        output_units = layers_config[-1].get('output_units', 10) if layers_config[-1]['type'] == 'dense' else 10
        if layers_config[-1]['type'] != 'dense':
            output_rect = plt.Rectangle((x_pos, y_center - 0.5), layer_width, 1, 
                                       fill=True, color='lightsteelblue', edgecolor='black', linewidth=2)
            ax.add_patch(output_rect)
            ax.text(x_pos + layer_width/2, y_center, f'Output\n{output_units}', 
                   ha='center', va='center', fontsize=9, weight='bold')
    
    ax.set_xlim(0, x_pos + layer_width + 1)
    ax.set_ylim(0, 6)
    ax.set_title('Convolutional Neural Network (CNN) Architecture', fontsize=14, weight='bold')
    ax.axis('off')
    
    return fig

def visualize_feature_maps(image, layers_config):
    """Visualize feature maps based on CNN architecture configuration"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Normalize to 0-255
    if gray.max() > 1:
        gray = gray.astype(np.float32)
    else:
        gray = (gray * 255).astype(np.float32)
    
    # Resize image to square if needed
    h, w = gray.shape
    if h != w:
        size = min(h, w)
        if size > 0:
            gray = cv2.resize(gray, (size, size))
        else:
            raise ValueError(f"Invalid image dimensions: {h}x{w}")
    
    # Ensure image has valid dimensions
    if gray.shape[0] == 0 or gray.shape[1] == 0:
        raise ValueError(f"Image has invalid dimensions after processing: {gray.shape}")
    
    feature_maps = []
    current_image = gray.copy()
    layer_idx = 0
    
    # Process each layer in the architecture
    for layer in layers_config:
        layer_type = layer['type']
        
        if layer_type == 'conv':
            filters = layer['filters']
            kernel_size = layer['kernel_size']
            
            # Generate filters for this convolutional layer
            # Create a variety of edge detection and feature extraction filters
            base_filters = [
                np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),  # Horizontal edge
                np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),  # Vertical edge
                np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),  # Diagonal
                np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),  # Sharpen
                np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,  # Blur
                np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),  # Emboss
                np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),     # Laplacian
                np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),  # Sobel
            ]
            
            filter_names = ['Horizontal Edge', 'Vertical Edge', 'Diagonal', 'Sharpen', 
                           'Blur', 'Emboss', 'Laplacian', 'Sobel']
            
            # Apply each filter
            layer_feature_maps = []
            for i in range(filters):
                if i < len(base_filters):
                    # Use predefined filter, resize to kernel_size
                    kernel = base_filters[i]
                    filter_name = f"Conv{layer_idx+1} - {filter_names[i]}"
                else:
                    # Generate random filter
                    kernel = np.random.randn(kernel_size, kernel_size) * 0.5
                    filter_name = f"Conv{layer_idx+1} - Random Filter {i+1}"
                
                # Resize kernel if needed
                if kernel.shape[0] != kernel_size:
                    kernel = resize_kernel(kernel, kernel_size)
                
                # Ensure kernel has valid dimensions
                if kernel.shape[0] != kernel_size or kernel.shape[1] != kernel_size:
                    raise ValueError(f"Kernel size mismatch: expected {kernel_size}x{kernel_size}, got {kernel.shape}")
                
                # Ensure kernel is 2D and has valid values
                if len(kernel.shape) != 2 or kernel.shape[0] <= 0 or kernel.shape[1] <= 0:
                    raise ValueError(f"Invalid kernel shape: {kernel.shape}")
                
                # Convert kernel to float32 for cv2.filter2D
                kernel = kernel.astype(np.float32)
                
                # Apply convolution
                feature_map = cv2.filter2D(current_image, -1, kernel)
                
                # Normalize for display
                feature_map = np.clip(feature_map, 0, 255).astype(np.uint8)
                
                layer_feature_maps.append(feature_map)
                feature_maps.append((feature_map, filter_name))
            
            # Update current_image to the first feature map for next layer
            if layer_feature_maps:
                current_image = layer_feature_maps[0].astype(np.float32)
            
            layer_idx += 1
            
        elif layer_type == 'pool':
            pool_size = layer['pool_size']
            
            # Apply max pooling
            h, w = current_image.shape
            pooled = np.zeros((h // pool_size, w // pool_size), dtype=np.float32)
            
            for i in range(0, h - pool_size + 1, pool_size):
                for j in range(0, w - pool_size + 1, pool_size):
                    pooled[i // pool_size, j // pool_size] = np.max(
                        current_image[i:i+pool_size, j:j+pool_size]
                    )
            
            current_image = pooled
            feature_maps.append((current_image.astype(np.uint8), f"Pool{layer_idx+1} - {pool_size}x{pool_size}"))
            layer_idx += 1
    
    return feature_maps

def show_ann_cnn():
    st.header("CNN Visualization")
    st.markdown("""
    Visualize the architecture and behavior of Convolutional Neural Networks (CNN).
    Configure your own CNN architecture with customizable layers.
    """)
    
    # Image upload first to determine input size
    st.subheader("Step 1: Upload Image")
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'], key="cnn_upload")
    
    input_size = None
    image = None
    image_array = None
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Get image dimensions
        if len(image_array.shape) == 3:
            h, w = image_array.shape[:2]
        else:
            h, w = image_array.shape
        
        # Use minimum dimension as input size (square)
        input_size = min(h, w)
        
        st.success(f"Image uploaded! Input size: {input_size}x{input_size} pixels")
        st.image(image, use_container_width=True, caption="Uploaded Image")
    else:
        st.info("Please upload an image to proceed with CNN configuration.")
        return
    
    # CNN Architecture Configuration on main page
    st.subheader("Step 2: CNN Architecture Configuration")
    
    # Layer configuration
    col1, col2 = st.columns(2)
    with col1:
        num_conv_layers = st.number_input("Number of Convolutional Layers", min_value=1, max_value=5, value=2, step=1)
    with col2:
        use_flatten = st.checkbox("Use Flatten Layer", value=True)
        num_dense_layers = st.number_input("Number of Dense Layers", min_value=0, max_value=3, value=1, step=1)
    
    # Number of pooling layers automatically matches number of convolutional layers
    num_pool_layers = num_conv_layers
    st.info(f"Number of Pooling Layers: {num_pool_layers} (automatically matches number of convolutional layers)")
    
    # Convolutional layers configuration
    conv_configs = []
    if num_conv_layers > 0:
        st.markdown("### Convolutional Layers Configuration")
        for i in range(num_conv_layers):
            with st.expander(f"Conv Layer {i+1}", expanded=(i == 0)):
                col1, col2 = st.columns(2)
                with col1:
                    filters = st.number_input(f"Number of Filters", min_value=1, max_value=512, 
                                            value=32 if i == 0 else 64, step=1, key=f"conv_filters_{i}")
                with col2:
                    kernel_size = st.number_input(f"Kernel Size", min_value=3, max_value=11, 
                                                value=3, step=2, key=f"conv_kernel_{i}")
                conv_configs.append({'filters': filters, 'kernel_size': kernel_size})
    
    # Pooling layers configuration (automatically matches number of convolutional layers)
    pool_configs = []
    if num_conv_layers > 0:
        st.markdown("### Pooling Layers Configuration")
        for i in range(num_conv_layers):
            with st.expander(f"Pool Layer {i+1}", expanded=(i == 0)):
                pool_size = st.number_input(f"Pool Size", min_value=2, max_value=8, 
                                          value=2, step=1, key=f"pool_size_{i}")
                pool_configs.append({'pool_size': pool_size})
    
    # Dense layers configuration
    dense_configs = []
    if num_dense_layers > 0:
        st.markdown("### Dense Layers Configuration")
        for i in range(num_dense_layers):
            with st.expander(f"Dense Layer {i+1}", expanded=(i == 0)):
                units = st.number_input(f"Number of Units", min_value=16, max_value=1024, 
                                      value=128 if i == 0 else 64, step=16, key=f"dense_units_{i}")
                dense_configs.append({'units': units})
    
    # Output configuration
    st.markdown("### Output Layer Configuration")
    output_units = st.number_input("Output Units", min_value=1, max_value=1000, value=10, step=1)
    
    # Build layers configuration
    layers_config = []
    current_size = input_size
    
    # Add convolutional and pooling layers alternately (one pool layer per conv layer)
    for i in range(num_conv_layers):
        # Add convolutional layer
        layers_config.append({
            'type': 'conv',
            'filters': conv_configs[i]['filters'],
            'kernel_size': conv_configs[i]['kernel_size']
        })
        current_size = calculate_output_size(current_size, conv_configs[i]['kernel_size'])
        
        # Add corresponding pooling layer
        layers_config.append({
            'type': 'pool',
            'pool_size': pool_configs[i]['pool_size']
        })
        current_size = calculate_pool_output_size(current_size, pool_configs[i]['pool_size'])
    
    # Add flatten layer
    if use_flatten:
        layers_config.append({'type': 'flatten'})
    
    # Add dense layers
    for dense_config in dense_configs:
        layers_config.append({
            'type': 'dense',
            'units': dense_config['units']
        })
    
    # Add output layer info to last dense layer
    if layers_config and layers_config[-1]['type'] == 'dense':
        layers_config[-1]['output_units'] = output_units
    
    # Calculate and display input layer size for dense layers
    if num_conv_layers > 0 and use_flatten:
        try:
            flattened_size, feature_map_size, num_filters = calculate_dense_input_size(
                input_size, conv_configs, pool_configs
            )
            
            st.subheader("Step 3: Input Layer Size for Dense Layers")
            st.markdown("""
            *After all convolutional and pooling layers, the feature maps are flattened and fed into the dense layers.*
            """)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Feature Map Size", f"{feature_map_size}×{feature_map_size}")
            with col2:
                st.metric("Number of Channels", f"{num_filters}")
            with col3:
                st.metric("Total Input Neurons", f"{flattened_size:,}")
            with col4:
                st.metric("Calculation", f"{feature_map_size}×{feature_map_size}×{num_filters}")
            
            with st.expander("📊 Detailed Breakdown"):
                st.write(f"*Input Image Size:* {input_size}×{input_size} pixels")
                st.write(f"*After Convolutional & Pooling Layers:*")
                st.write(f"  - Feature Map Dimensions: {feature_map_size}×{feature_map_size}")
                st.write(f"  - Number of Feature Maps (from last Conv layer): {num_filters}")
                st.write(f"*After Flattening:*")
                st.write(f"  - Total Input Neurons to Dense Layers: *{flattened_size:,}*")
                st.write(f"  - Formula: {feature_map_size} × {feature_map_size} × {num_filters} = {flattened_size:,}")
                
                if num_dense_layers > 0:
                    st.write(f"\n*First Dense Layer:*")
                    st.write(f"  - Input: {flattened_size:,} neurons")
                    st.write(f"  - Output: {dense_configs[0]['units']} neurons")
                    st.write(f"  - Total Parameters: {flattened_size * dense_configs[0]['units']:,} (without bias)")
        except Exception as e:
            st.warning(f"Could not calculate input layer size: {str(e)}")
    elif not use_flatten:
        st.info("💡 Enable 'Flatten Layer' to see the input size calculation for dense layers.")
    
    # Display architecture diagram
    st.subheader("Step 4: CNN Architecture Visualization")
    st.markdown("""
    A Convolutional Neural Network consists of:
    - *Convolutional Layers*: Extract features using filters
    - *Pooling Layers*: Reduce spatial dimensions
    - *Fully Connected (Dense) Layers*: Make final predictions
    """)
    
    try:
        fig = create_cnn_diagram(layers_config, input_size)
        st.pyplot(fig)
        
        # Display architecture summary
        with st.expander("Architecture Summary"):
            st.write(f"*Input Size:* {input_size}x{input_size}")
            st.write(f"*Total Layers:* {len(layers_config)}")
            st.write("*Layer Details:*")
            for i, layer in enumerate(layers_config):
                if layer['type'] == 'conv':
                    st.write(f"  - Conv{i+1}: {layer['filters']} filters, kernel {layer['kernel_size']}x{layer['kernel_size']}")
                elif layer['type'] == 'pool':
                    st.write(f"  - Pool{i+1}: {layer['pool_size']}x{layer['pool_size']}")
                elif layer['type'] == 'flatten':
                    st.write(f"  - Flatten")
                elif layer['type'] == 'dense':
                    st.write(f"  - Dense: {layer['units']} units")
    except Exception as e:
        st.error(f"Error creating diagram: {str(e)}")
        st.info("Please check your layer configuration. Make sure the input size is compatible with your kernel and pool sizes.")
    
    # Feature Map Visualization based on CNN architecture
    st.subheader("Step 5: Feature Map Visualization")
    st.write("Feature maps generated based on your CNN architecture configuration:")
    
    if uploaded_file is not None and layers_config:
        # Get only conv and pool layers for feature map visualization
        vis_layers_config = [layer for layer in layers_config if layer['type'] in ['conv', 'pool']]
        
        if vis_layers_config:
            show_original = st.checkbox("Show Original Image", value=True, key="show_original_cnn")
            
            if show_original:
                st.image(image, use_container_width=True, caption="Original Image")
            
            try:
                feature_maps = visualize_feature_maps(image_array, vis_layers_config)
                
                st.subheader("Feature Maps (Based on CNN Architecture)")
                # Display feature maps in a grid
                cols = 4
                rows = (len(feature_maps) + cols - 1) // cols
                
                for row in range(rows):
                    cols_list = st.columns(cols)
                    for col in range(cols):
                        idx = row * cols + col
                        if idx < len(feature_maps):
                            with cols_list[col]:
                                feature_map, filter_name = feature_maps[idx]
                                st.image(feature_map, use_container_width=True, 
                                       channels='GRAY', caption=filter_name)
                
                # Show statistics
                with st.expander("Feature Map Statistics"):
                    st.write("*Feature Map Analysis:*")
                    for idx, (feature_map, filter_name) in enumerate(feature_maps):
                        mean_val = np.mean(feature_map)
                        std_val = np.std(feature_map)
                        min_val = np.min(feature_map)
                        max_val = np.max(feature_map)
                        st.write(f"{filter_name}:** Mean={mean_val:.2f}, Std={std_val:.2f}, Range=[{min_val:.0f}, {max_val:.0f}]")
            except Exception as e:
                st.error(f"Error generating feature maps: {str(e)}")
        else:
            st.warning("Please add at least one convolutional or pooling layer to visualize feature maps.")