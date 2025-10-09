# ASL Alphabet CNN Model Setup Instructions

## Prerequisites

1. Python 3.7+ installed on your system
2. GPU with CUDA support (optional but recommended for training)
3. Kaggle account and API credentials

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n asl_env python=3.8
conda activate asl_env

# Or using venv
python -m venv asl_env
source asl_env/bin/activate  # Linux/Mac
# or
asl_env\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Kaggle API

1. Go to Kaggle.com → Account → API → Create New API Token
2. Download kaggle.json file
3. Place it in the appropriate location:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
4. Set permissions (Linux/Mac only): `chmod 600 ~/.kaggle/kaggle.json`

## Quick Start

### 1. Download and Preprocess Dataset

```python
python asl_preprocessing.py
```

This will:
- Download the ASL alphabet dataset from Kaggle
- Organize it into train/validation/test splits
- Generate dataset analysis visualizations
- Preprocess images to standard format

### 2. Train the Model

```python
python asl_training.py
```

This will:
- Load the preprocessed dataset
- Train a CNN model (or transfer learning model)
- Save the best model as 'best_asl_model.h5'
- Generate training history plots and evaluation metrics

### 3. Run Inference

#### Real-time webcam inference:
```python
python asl_inference.py --model best_asl_model.h5 --camera 0
```

#### Single image inference:
```python
python asl_inference.py --model best_asl_model.h5 --image path/to/image.jpg
```

## Project Structure

```
asl_project/
├── asl_preprocessing.py    # Dataset download and preprocessing
├── asl_training.py         # Model training script
├── asl_inference.py        # Real-time inference script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Downloaded dataset (auto-created)
├── asl_organized/         # Organized dataset (auto-created)
├── asl_preprocessed/      # Preprocessed dataset (auto-created)
├── best_asl_model.h5     # Trained model (generated)
├── training_history.png   # Training plots (generated)
└── confusion_matrix.png   # Evaluation plots (generated)
```

## Model Architecture Options

### 1. CNN from Scratch (Default)
- Custom 4-layer CNN with batch normalization and dropout
- Optimized for ASL alphabet recognition
- Fast training and good performance

### 2. Transfer Learning
Available pre-trained models:
- ResNet50 (recommended)
- VGG16
- InceptionV3

To use transfer learning, modify `asl_training.py`:
```python
# Replace this line:
trainer.build_cnn_model()

# With this:
trainer.build_transfer_learning_model('ResNet50')
```

## Training Configuration

Key hyperparameters (modify in `asl_training.py`):

```python
DATA_PATH = "asl_organized"  # Path to dataset
IMG_SIZE = (200, 200)       # Input image size
BATCH_SIZE = 32             # Batch size
EPOCHS = 50                 # Training epochs
```

## Inference Configuration

Key parameters for `asl_inference.py`:

- `--model`: Path to trained model file
- `--camera`: Camera device ID (0 for default)
- `--confidence`: Confidence threshold (0.0-1.0)
- `--image`: Single image file path

## Real-time Inference Controls

When running real-time inference:
- **'q'**: Quit application
- **'r'**: Reset prediction history (clear smoothing)
- **'s'**: Save current frame as image

## Expected Performance

With the default CNN architecture:
- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~97-98%
- **Real-time FPS**: 15-30 (depending on hardware)

## Troubleshooting

### Common Issues:

1. **CUDA/GPU errors**: Install appropriate CUDA version for your TensorFlow
2. **Camera not found**: Check camera permissions and try different camera IDs
3. **Low accuracy**: Ensure proper lighting and hand positioning in ROI
4. **Slow inference**: Reduce image size or use CPU-optimized model

### Performance Tips:

1. **Use GPU acceleration** for training (10x faster)
2. **Proper lighting** for better real-time recognition
3. **Consistent hand positioning** in the green ROI box
4. **Clean background** helps improve accuracy

## Dataset Information

- **Source**: Kaggle ASL Alphabet Dataset by grassknoted
- **Total Images**: ~87,000 training images
- **Classes**: 29 (A-Z letters + SPACE, DELETE, NOTHING)
- **Image Size**: 200x200 pixels (RGB)
- **Format**: JPG/PNG files organized by class

## Model Files

After training, you'll have:
- `best_asl_model.h5`: Best model based on validation accuracy
- `training_history.png`: Training/validation curves
- `confusion_matrix.png`: Model performance visualization

## Advanced Usage

### Custom Dataset
To use your own images:
1. Organize them in folders by class name
2. Modify `DATA_PATH` in training script
3. Update class names in both training and inference scripts

### Model Optimization
For deployment:
1. Convert to TensorFlow Lite: Use `tf.lite.TFLiteConverter`
2. Quantization: Reduce model size for mobile deployment
3. Pruning: Remove unnecessary weights for faster inference

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review TensorFlow/OpenCV documentation
3. Ensure all dependencies are correctly installed

## License

This project is for educational purposes. Please respect the original dataset license terms.
