# 🤟 ASL Alphabet Recognition Dashboard

**Interactive web-based dashboard for real-time American Sign Language alphabet recognition**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Dashboard Features](#dashboard-features)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)

---

## ✨ Features

### 🎯 Core Functionality
- **Real-time ASL Recognition**: Live webcam feed with instant predictions
- **Image Upload Support**: Analyze static images of ASL hand signs
- **Prediction Smoothing**: Temporal smoothing for stable predictions
- **Confidence Metrics**: Real-time confidence scores and visualizations

### 📊 Visualization & Analytics
- **Interactive Charts**: Plotly-powered confidence distribution charts
- **Prediction History**: Track and analyze prediction patterns over time
- **Performance Statistics**: Monitor model performance metrics
- **Top-5 Predictions**: See alternative predictions with probabilities

### 🎨 User Experience
- **Modern UI**: Beautiful gradient designs and responsive layout
- **Easy Configuration**: Sidebar controls for all settings
- **Multiple Modes**: Camera, upload, and statistics tabs
- **Export Data**: Download prediction history as CSV

---

## 🔧 Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster inference)
- **Webcam**: Built-in or external camera for real-time recognition

### Software Requirements
- Python 3.8+
- pip package manager
- Webcam drivers (should be pre-installed on most systems)

---

## 📦 Installation

### Step 1: Navigate to Project Directory

```bash
cd Code
```

### Step 2: Install Dependencies

```bash
# Install dashboard requirements
pip install -r requirements_dashboard.txt
```

**What gets installed:**
- `streamlit` - Web app framework
- `torch` & `torchvision` - Deep learning framework
- `opencv-python` - Computer vision library
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- Additional utilities

### Step 3: Verify Installation

```bash
# Check Streamlit installation
streamlit --version

# Check PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check if CUDA is available (optional)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### 1. Ensure Model File is Present

Make sure you have the trained model file:
```
best_asl_model_rtx4060_optimized.pth
```

The model should be located one directory up from the `Code` folder:
```
Project/
├── best_asl_model_rtx4060_optimized.pth  ← Model here
└── Code/
    ├── app.py
    ├── asl_inference_engine.py
    └── requirements_dashboard.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

### 3. Open in Browser

The dashboard will automatically open in your default browser at:
```
http://localhost:8501
```

If it doesn't open automatically, navigate to the URL manually.

---

## 📖 Usage Guide

### Initial Setup

1. **Load Model**
   - Click "🚀 Load Model" in the sidebar
   - Wait for confirmation message: "✅ Model loaded successfully!"
   - Check model info displayed at bottom of sidebar

2. **Configure Settings** (Optional)
   ```
   Device: cuda or cpu
   Prediction Smoothing: 5 frames (default)
   Confidence Threshold: 0.7 (70%)
   Camera ID: 0 (default camera)
   ROI Size: 300 pixels
   ```

### Mode 1: Real-time Camera Recognition

1. **Navigate to "📹 Live Camera" tab**

2. **Click "▶️ Start Camera"**
   - Green ROI (Region of Interest) box will appear
   - Camera feed starts streaming

3. **Position Your Hand**
   - Place hand inside the green box
   - Ensure good lighting
   - Keep background simple for best results

4. **View Predictions**
   - **Large Purple Box**: Current predicted letter
   - **Pink Box**: Confidence percentage
   - **Chart**: Top 5 predictions with probabilities

5. **Controls**
   - **⏹️ Stop Camera**: Stop camera feed
   - **🔄 Reset Buffer**: Clear prediction history for fresh start

### Mode 2: Image Upload Recognition

1. **Navigate to "📸 Upload Image" tab**

2. **Upload Image**
   - Click "Browse files" or drag & drop
   - Supported formats: JPG, JPEG, PNG
   - Image should show clear ASL hand sign

3. **Analyze**
   - Click "🔍 Analyze Image"
   - View prediction results and confidence chart
   - See top 5 alternative predictions

### Mode 3: Statistics & Analytics

1. **Navigate to "📊 Statistics" tab**

2. **View Performance Metrics**
   - Total predictions made
   - Number of confident predictions
   - Overall confidence rate
   - Buffer size

3. **Prediction History**
   - Line chart showing confidence over time
   - Table of recent predictions with timestamps
   - Download history as CSV file

---

## 🎨 Dashboard Features

### Main Interface

```
┌─────────────────────────────────────────────────────────┐
│  🤟 ASL Alphabet Recognition                            │
│  Real-time American Sign Language Interpreter           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┬─────────────┬─────────────┐              │
│  │ 📹 Live │ 📸 Upload   │ 📊 Stats    │ ← Tabs       │
│  └─────────┴─────────────┴─────────────┘              │
│                                                         │
│  ┌──────────────────┐  ┌──────────────┐               │
│  │  Camera Feed     │  │  Prediction  │               │
│  │  [Hand in ROI]   │  │      A       │               │
│  │                  │  │              │               │
│  │  ▶️ Start        │  │  95.3%       │               │
│  │  ⏹️ Stop         │  │              │               │
│  │  🔄 Reset        │  │  [Chart]     │               │
│  └──────────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────┘
```

### Sidebar Configuration

```
⚙️ Configuration
├── Model Settings
│   ├── Model Path: ../best_asl_model...pth
│   ├── Device: cuda / cpu
│   ├── Prediction Smoothing: 1-10 frames
│   └── Confidence Threshold: 0.0-1.0
│
├── 🚀 Load Model (button)
│
├── Camera Settings
│   ├── Camera ID: 0-10
│   └── ROI Size: 200-500px
│
└── 📊 Model Info
    ├── Device: cuda:0
    ├── Classes: 26
    └── Letters: A, B, C, D, E...
```

---

## 🔧 Troubleshooting

### Model Loading Issues

**Problem**: Model fails to load
```
❌ Error loading model: [error message]
```

**Solutions**:
1. Verify model file path is correct
2. Check if model file exists and is not corrupted
3. Ensure model file is `.pth` format
4. Try changing device from `cuda` to `cpu`

### Camera Issues

**Problem**: Camera not accessible
```
Failed to access camera
```

**Solutions**:
1. Check camera permissions in system settings
2. Close other applications using the camera
3. Try different camera ID (0, 1, 2...)
4. Restart the dashboard
5. Test camera with native camera app first

### CUDA/GPU Issues

**Problem**: CUDA out of memory
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Change device to `cpu` in sidebar
2. Close other GPU-intensive applications
3. Restart the dashboard
4. Lower the camera resolution (handled automatically)

### Performance Issues

**Problem**: Slow or laggy predictions

**Solutions**:
1. **Use GPU**: Select `cuda` device (10x faster)
2. **Lower ROI Size**: Reduce to 200-250 pixels
3. **Increase Smoothing**: Higher values = fewer updates
4. **Close Background Apps**: Free up system resources
5. **Lower Camera Resolution**: Reduces processing load

### Import Errors

**Problem**: Module not found
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solutions**:
```bash
# Reinstall requirements
pip install -r requirements_dashboard.txt

# Or install specific package
pip install streamlit
```

---

## ⚡ Performance Tips

### For Best Accuracy

1. **Lighting**: Use bright, even lighting
2. **Background**: Plain, solid-colored background
3. **Hand Position**: Center hand in ROI box
4. **Distance**: Keep hand 1-2 feet from camera
5. **Clarity**: Ensure hand gesture is clear and distinct

### For Best Speed

1. **Use GPU**: CUDA-enabled GPU provides 10-30x speedup
2. **Lower Smoothing**: Reduce smoothing window to 3 frames
3. **Optimize ROI**: Use smaller ROI size (200-250px)
4. **Close Apps**: Close unnecessary applications

### Optimal Settings

```python
# Balanced settings (recommended)
Device: cuda
Smoothing: 5 frames
Threshold: 0.7
ROI Size: 300px
Camera ID: 0

# Speed-optimized settings
Device: cuda
Smoothing: 3 frames
Threshold: 0.6
ROI Size: 224px
Camera ID: 0

# Accuracy-optimized settings
Device: cuda
Smoothing: 7 frames
Threshold: 0.8
ROI Size: 400px
Camera ID: 0
```

---

## 📸 Screenshot Preview

### Dashboard Components

1. **Header**: Project title and description
2. **Sidebar**: Configuration panel with model and camera settings
3. **Tabs**: Three modes (Live Camera, Upload, Statistics)
4. **Prediction Display**: Large letter with confidence score
5. **Charts**: Interactive Plotly visualizations
6. **Statistics**: Performance metrics and history

---

## 🎓 Demo Presentation Tips

### For Teacher Demonstration

1. **Preparation**
   - Test camera before presentation
   - Ensure good lighting in demo area
   - Pre-load model to save time
   - Have sample images ready as backup

2. **Live Demo Flow**
   ```
   1. Show dashboard homepage → Explain features
   2. Load model → Show model info
   3. Start camera → Demonstrate live recognition
   4. Show multiple letters (A, B, C, etc.)
   5. Show statistics → Highlight accuracy
   6. Upload image → Show alternative mode
   7. Download history → Show data export
   ```

3. **Talking Points**
   - Real-time processing capabilities
   - High accuracy (97-99%)
   - GPU optimization
   - Modern UI/UX design
   - Practical accessibility application

---

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit (Python web framework)
- **Backend**: PyTorch (Deep learning inference)
- **Computer Vision**: OpenCV (Camera handling)
- **Visualization**: Plotly (Interactive charts)

### Model Details
- **Architecture**: EfficientASLNet (Custom CNN)
- **Input Size**: 224×224 pixels
- **Classes**: 26 (A-Z alphabet)
- **Optimization**: RTX 4060 specific optimizations

### Performance
- **Inference Time**: 10-30ms per frame (GPU)
- **FPS**: 30+ frames per second
- **Accuracy**: 97-99% validation accuracy
- **Memory**: ~500MB GPU VRAM

---

## 📝 Project Structure

```
Code/
├── app.py                          # Main Streamlit dashboard
├── asl_inference_engine.py         # Inference engine & model
├── requirements_dashboard.txt      # Dashboard dependencies
├── DASHBOARD_README.md            # This file
│
├── asl_training_pytorch.py        # Training script
├── asl_evaluate_pytorch.py        # Evaluation script
└── requirements_frameworks.txt     # Training dependencies
```

---

## 🤝 Support & Contact

For issues or questions:
1. Check troubleshooting section above
2. Review error messages in terminal
3. Verify all requirements are installed
4. Test components individually

---

## 📄 License

This project is for educational purposes. MIT License.

---

## 🎯 Quick Command Reference

```bash
# Install dependencies
pip install -r requirements_dashboard.txt

# Run dashboard
streamlit run app.py

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test inference engine only
python asl_inference_engine.py

# Stop dashboard
Ctrl + C (in terminal)
```

---

## 🌟 Features Showcase

✅ Real-time webcam inference  
✅ Image upload support  
✅ Prediction smoothing  
✅ Confidence visualization  
✅ Performance statistics  
✅ History tracking  
✅ CSV export  
✅ GPU acceleration  
✅ Modern UI/UX  
✅ Responsive design  

---

**Made with ❤️ for ASL accessibility | Powered by PyTorch & Streamlit**
