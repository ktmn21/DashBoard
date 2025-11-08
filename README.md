# Algorithm & Image Processing Dashboard

An interactive dashboard built with Python and Streamlit that provides visualizations and demos for:

- **DAA (Design and Analysis of Algorithms)**: Knapsack, Coin Change, Huffman Coding, TSP, Genetic Algorithm, and Branch & Bound
- **DIP (Digital Image Processing)**: Grayscale, Negative, Histogram Equalization, Brightness/Contrast, Blur, Edge Detection, Thresholding, Sharpen, Convolution, RGB Channels, and ANN/CNN Visualization

## Features

### 🧩 DAA Section
- **Knapsack Problem**: 0/1 and Fractional knapsack with visualizations
- **Coin Change (DP)**: Dynamic programming solution with step-by-step table updates
- **Huffman Coding**: Binary tree visualization and compression statistics
- **Travelling Salesman Problem**: Nearest neighbor and brute force solutions
- **Genetic Algorithm**: Population evolution with fitness graphs
- **Branch and Bound**: Exploration tree visualization

### 🖼 DIP Section
- **Grayscale Conversion**: RGB to grayscale with histograms
- **Image Negative**: Inverted image visualization
- **Histogram Equalization**: Contrast enhancement with CDF plots
- **Brightness & Contrast**: Adjustable parameters with real-time preview
- **Blur**: Gaussian and Median blur filters
- **Edge Detection**: Sobel and Canny edge detection
- **Thresholding**: Multiple threshold methods (binary, Otsu, adaptive)
- **Sharpen**: Image sharpening with adjustable strength
- **Convolution**: Custom kernel convolution
- **RGB Channels**: Individual channel separation and visualization
- **ANN & CNN**: Architecture diagrams and feature map visualization

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Project Structure

```
project/
├── app.py                # Main dashboard
├── daa/
│   ├── __init__.py
│   ├── knapsack.py
│   ├── dp_coin_change.py
│   ├── huffman.py
│   ├── tsp.py
│   ├── genetic.py
│   └── branch_bound.py
├── dip/
│   ├── __init__.py
│   ├── grayscale.py
│   ├── negative.py
│   ├── histogram_eq.py
│   ├── brightness_contrast.py
│   ├── blur.py
│   ├── edge_detection.py
│   ├── threshold.py
│   ├── sharpen.py
│   ├── convolution.py
│   ├── channels.py
│   └── ann_cnn.py
├── assets/
│   └── sample_images/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.7+
- Streamlit
- Matplotlib
- NumPy
- OpenCV
- Pandas
- Pillow
- NetworkX

## Features Overview

- **Interactive Visualizations**: Step-by-step algorithm demonstrations
- **Real-time Processing**: Adjust parameters and see results instantly
- **Image Upload**: Upload your own images for DIP operations
- **Educational**: Learn algorithms and image processing techniques visually

## Notes

- For TSP brute force, use n ≤ 8 cities for reasonable performance
- Some algorithms may take a few seconds to compute for larger inputs
- Image processing operations support PNG, JPG, and JPEG formats

