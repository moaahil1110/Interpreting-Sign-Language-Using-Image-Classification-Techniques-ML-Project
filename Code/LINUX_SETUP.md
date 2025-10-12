# 🐧 ASL Recognition Dashboard - Fedora 42 Setup Guide

## Prerequisites

### 1. Install System Dependencies
```bash
# Update system
sudo dnf update -y

# Install Python 3.11 and development tools
sudo dnf install python3.11 python3.11-devel python3-pip -y

# Install OpenCV dependencies for camera access
sudo dnf install mesa-libGL gtk3 libXtst libXScrnSaver -y

# Install git (if not already installed)
sudo dnf install git -y
```

### 2. Install Conda (Recommended) or Use venv

#### Option A: Install Miniconda (Recommended)
```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run installer
bash Miniconda3-latest-Linux-x86_64.sh

# Follow the prompts and restart terminal
# Initialize conda
source ~/.bashrc
```

#### Option B: Use Python venv (Alternative)
```bash
# Will be covered in step 4 if you prefer not to use conda
```

---

## Setup Instructions

### Step 1: Clone the Repository
```bash
cd ~
git clone https://github.com/moaahil1110/Interpreting-Sign-Language-Using-Image-Classification-Techniques-ML-Project.git
cd Interpreting-Sign-Language-Using-Image-Classification-Techniques-ML-Project/Code
```

### Step 2: Download the Model File
The model file (`best_asl_model_rtx4060_optimized.pth`) is too large for GitHub. Get it from your friend or download it from the shared location.

Place it in the root directory:
```bash
# Copy model file to project root
cp /path/to/best_asl_model_rtx4060_optimized.pth ../
```

### Step 3: Create Python Environment

#### Using Conda (Recommended):
```bash
# Create conda environment with Python 3.11
conda create -n asl-dashboard python=3.11 -y

# Activate environment
conda activate asl-dashboard
```

#### Using venv (Alternative):
```bash
# Create virtual environment
python3.11 -m venv ~/asl-env

# Activate environment
source ~/asl-env/bin/activate
```

### Step 4: Install Dependencies
```bash
# Make sure you're in the Code directory
cd ~/Interpreting-Sign-Language-Using-Image-Classification-Techniques-ML-Project/Code

# Install PyTorch (CPU version for compatibility, or CUDA version if you have NVIDIA GPU)
# For CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For NVIDIA GPU (if available):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dashboard requirements
pip install -r requirements_dashboard.txt
```

### Step 5: Grant Camera Permissions
```bash
# Check camera device
ls -la /dev/video*

# If permission denied, add user to video group
sudo usermod -a -G video $USER

# Log out and log back in for changes to take effect
```

---

## Running the Dashboard

### Method 1: Using the Shell Script
```bash
# Make script executable (first time only)
chmod +x run_dashboard.sh

# Run dashboard
./run_dashboard.sh
```

### Method 2: Manual Command
```bash
# Activate environment (if not already activated)
# For conda:
conda activate asl-dashboard
# For venv:
# source ~/asl-env/bin/activate

# Run Streamlit
streamlit run app.py
```

The dashboard will open automatically in your default browser at: **http://localhost:8501**

---

## Troubleshooting

### Camera Issues
If camera doesn't work:

1. **Check camera access:**
```bash
# Test camera with v4l2
sudo dnf install v4l-utils -y
v4l2-ctl --list-devices

# Test with simple capture
ffplay /dev/video0
```

2. **SELinux issues (Fedora specific):**
```bash
# Check SELinux status
getenforce

# If issues persist, temporarily set to permissive
sudo setenforce 0

# Or create proper SELinux policy for camera access
```

3. **Webcam permissions:**
```bash
# Check current permissions
ls -l /dev/video0

# If needed, change permissions (temporary)
sudo chmod 666 /dev/video0
```

### OpenCV Import Errors
```bash
# If you get "libGL.so.1: cannot open shared object file"
sudo dnf install mesa-libGL -y

# If you get GTK errors
sudo dnf install gtk3 -y
```

### Streamlit Port Already in Use
```bash
# Kill existing Streamlit process
pkill -9 streamlit

# Or run on different port
streamlit run app.py --server.port 8502
```

### Missing Model File Error
```bash
# Verify model file exists in parent directory
ls -lh ../best_asl_model_rtx4060_optimized.pth

# Check file path in asl_inference_engine.py matches the location
```

---

## Performance Tips for Fedora

### 1. GPU Acceleration (If NVIDIA GPU available)
```bash
# Install NVIDIA drivers
sudo dnf install akmod-nvidia -y
sudo dnf install xorg-x11-drv-nvidia-cuda -y

# Install CUDA PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Optimize for Better Camera Performance
```bash
# Install v4l2loopback for better camera control
sudo dnf install v4l2loopback -y
```

### 3. System Resources
- Close unnecessary applications for smoother camera feed
- Recommended: 4GB+ RAM, quad-core processor
- Camera resolution: 640x480 (default, adjustable in sidebar)

---

## Quick Start Commands (After Initial Setup)

```bash
# Navigate to project
cd ~/Interpreting-Sign-Language-Using-Image-Classification-Techniques-ML-Project/Code

# Activate environment
conda activate asl-dashboard
# OR: source ~/asl-env/bin/activate

# Run dashboard
streamlit run app.py
```

---

## Stopping the Dashboard

Press `Ctrl + C` in the terminal where Streamlit is running.

---

## Features Available on Fedora

✅ **Live Camera Recognition** - Real-time ASL alphabet detection  
✅ **Image Upload** - Test with saved images  
✅ **Statistics Dashboard** - View prediction history and analytics  
✅ **Interactive Charts** - Plotly-based confidence visualization  
✅ **Adjustable Settings** - ROI size, camera device selection  

---

## System Requirements

- **OS**: Fedora 42 (or compatible RHEL-based distros)
- **Python**: 3.11+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: ~2GB for dependencies + model file
- **Camera**: USB webcam or built-in camera
- **Browser**: Firefox, Chrome, or any modern browser

---

## Additional Resources

- **PyTorch Fedora Guide**: https://pytorch.org/get-started/locally/
- **Streamlit Docs**: https://docs.streamlit.io/
- **OpenCV Linux Setup**: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html

---

## Need Help?

If you encounter issues specific to Fedora 42:

1. Check Fedora forums: https://ask.fedoraproject.org/
2. Verify all system dependencies are installed
3. Check camera permissions and SELinux settings
4. Ensure Python 3.11 is properly installed

---

**Happy Sign Language Recognition! 🤟**
