# ASL Alphabet Recognition Project

## Overview

This project implements a real-time American Sign Language (ASL) alphabet recognition system. It uses a deep learning model built with PyTorch and includes a Streamlit dashboard for interactive use. The model, `EfficientASLNet`, is a custom Convolutional Neural Network (CNN) designed for efficient ASL alphabet recognition and optimized for NVIDIA RTX 4060 GPUs.

## Key Features

*   **Real-time ASL Alphabet Recognition:** Interprets ASL hand signs using a live webcam feed or uploaded images.
*   **Interactive Streamlit Dashboard:** Provides a user-friendly interface for real-time predictions, image analysis, and performance monitoring.
*   **Custom CNN Architecture:** Employs the `EfficientASLNet` model, optimized for performance and accuracy.
*   **RTX 4060 GPU Optimization:** Leverages Tensor Cores and memory management techniques for fast inference on NVIDIA RTX 4060 GPUs.
*   **Comprehensive Performance Metrics:** Tracks and visualizes key performance indicators, including accuracy, confidence levels, and prediction history.

## Tech Stack

*   **Deep Learning Framework:** PyTorch
*   **Computer Vision:** OpenCV
*   **Dashboard:** Streamlit
*   **Hardware Optimization:** CUDA (for NVIDIA GPUs), optimized for RTX 4060
*   **Languages:** Python

## Model Architecture

The core of the project is the `EfficientASLNet` model, a custom Convolutional Neural Network (CNN) designed for efficient ASL alphabet recognition.

*   **Key Architectural Features:**
    *   Multiple convolutional layers with batch normalization and ReLU activation.
    *   Max pooling for downsampling.
    *   Dropout for regularization.
    *   Adaptive average pooling before fully connected layers.
*   **RTX 4060 Optimizations:**
    *   Automatic Mixed Precision (AMP) for Tensor Core utilization.
    *   Channel sizes optimized for Tensor Core efficiency.

## Installation

### Prerequisites

1.  Python 3.7+
2.  CUDA-enabled NVIDIA GPU (recommended for training and optimized inference)

### Steps

1.  **Create a Virtual Environment (Recommended):**

    ```bash
    conda create -n asl_env python=3.8
    conda activate asl_env
    ```

    or

    ```bash
    python -m venv asl_env
    source asl_env/bin/activate  # Linux/Mac
    asl_env\Scripts\activate  # Windows
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r Code/requirements_frameworks.txt
    pip install -r Code/requirements_dashboard.txt
    ```

## Usage

### 1. Train the Model (Optional)

If you want to train the model yourself:

```bash
python Code/asl_training_pytorch.py
```

This script will:

*   Download the ASL alphabet dataset (if not already present).
*   Train the `EfficientASLNet` model.
*   Save the best model as `best_asl_model_rtx4060_optimized.pth`.

### 2. Run the Dashboard

```bash
cd Code
./run_dashboard.sh
```

or

```bash
cd Code
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Dashboard Features

*   **Real-time Camera Recognition:** Live webcam feed with instant ASL predictions.
*   **Image Upload Support:** Analyze static images of ASL hand signs.
*   **Prediction Smoothing:** Temporal smoothing for stable predictions.
*   **Confidence Metrics:** Real-time confidence scores and visualizations.
*   **Interactive Charts:** Plotly-powered confidence distribution charts.
*   **Prediction History:** Track and analyze prediction patterns over time.

## Performance

The model is optimized for the NVIDIA RTX 4060 GPU. Performance may vary depending on the hardware and input conditions.

*   **Validation Accuracy:** ~85% on the "TestNew" dataset.
*   **Inference Time:** 10-30ms per frame (GPU)
*   **FPS:** 30+ frames per second

## Project Structure

```
ASL_Alphabet_Recognition_Project/
├── .gitignore
├── best_asl_model_rtx4060_optimized.pth
├── README.md
├── Code/
│   ├── app.py
│   ├── asl_evaluate_pytorch.py
│   ├── asl_inference_engine.py
│   ├── asl_training_pytorch.py
│   ├── DASHBOARD_README.md
│   ├── LINUX_SETUP.md
│   ├── requirements_dashboard.txt
│   ├── requirements_frameworks.txt
│   ├── run_dashboard.sh
│   └── validationResults/
│       ├── classification_report_Test.txt
│       └── metrics_Test.txt
```

## Troubleshooting

Refer to `Code/DASHBOARD_README.md` and `Code/LINUX_SETUP.md` for detailed troubleshooting instructions.

## License

This project is for educational purposes.
