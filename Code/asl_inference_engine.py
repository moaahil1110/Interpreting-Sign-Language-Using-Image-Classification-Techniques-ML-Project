"""
ASL Alphabet Recognition - Inference Engine
Real-time inference module with prediction smoothing
Optimized for RTX 4060 GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from collections import deque
import string


class EfficientASLNet(nn.Module):
    """
    Efficient CNN for ASL Alphabet Recognition
    Must match the architecture used in training
    """
    def __init__(self, num_classes=26, dropout_rate=0.3):
        super(EfficientASLNet, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Pooling and dropout layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.adaptive_pool(x)
        x = self.dropout2d(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ASLInferenceEngine:
    """
    Real-time ASL inference engine with prediction smoothing
    """
    
    def __init__(self, model_path, device='cuda', smoothing_window=5, confidence_threshold=0.7):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            device: 'cuda' or 'cpu'
            smoothing_window: Number of frames for prediction smoothing
            confidence_threshold: Minimum confidence for predictions
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.prediction_buffer = deque(maxlen=smoothing_window)
        
        # Load model checkpoint
        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get metadata from checkpoint
        self.class_names = checkpoint.get('class_names', list(string.ascii_uppercase[:26]))
        img_size = checkpoint.get('img_size', 224)
        
        # Handle img_size if it's a tuple (e.g., (224, 224))
        if isinstance(img_size, (tuple, list)):
            img_size = img_size[0]  # Take first dimension
        
        # Initialize model
        self.model = EfficientASLNet(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Image size: {img_size}x{img_size}")
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Statistics tracking
        self.total_predictions = 0
        self.confident_predictions = 0
        self.prediction_history = []
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image or numpy array (BGR or RGB)
        
        Returns:
            Preprocessed tensor ready for model
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # OpenCV uses BGR, convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
        
        # Apply transforms
        img_tensor = self.transform(image)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        return img_tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, image, return_probabilities=True):
        """
        Predict ASL letter from image
        
        Args:
            image: PIL Image or numpy array
            return_probabilities: If True, returns all class probabilities
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Get model predictions
        outputs = self.model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
        predicted_letter = self.class_names[predicted_idx]
        
        # Update statistics
        self.total_predictions += 1
        if confidence >= self.confidence_threshold:
            self.confident_predictions += 1
        
        # Add to prediction buffer for smoothing
        self.prediction_buffer.append((predicted_letter, confidence))
        
        # Get smoothed prediction (most common in buffer)
        if len(self.prediction_buffer) >= self.smoothing_window:
            letters = [p[0] for p in self.prediction_buffer]
            smoothed_letter = max(set(letters), key=letters.count)
        else:
            smoothed_letter = predicted_letter
        
        # Prepare result
        result = {
            'predicted_letter': predicted_letter,
            'smoothed_letter': smoothed_letter,
            'confidence': confidence,
            'is_confident': confidence >= self.confidence_threshold,
        }
        
        # Add full probability distribution if requested
        if return_probabilities:
            all_probs = probabilities[0].cpu().numpy()
            result['probabilities'] = {
                self.class_names[i]: float(all_probs[i]) 
                for i in range(len(self.class_names))
            }
            # Get top 5 predictions
            top5_idx = np.argsort(all_probs)[-5:][::-1]
            result['top5'] = [
                (self.class_names[i], float(all_probs[i])) 
                for i in top5_idx
            ]
        
        return result
    
    def reset_buffer(self):
        """Reset prediction smoothing buffer"""
        self.prediction_buffer.clear()
    
    def get_statistics(self):
        """Get inference statistics"""
        accuracy_rate = (self.confident_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
        return {
            'total_predictions': self.total_predictions,
            'confident_predictions': self.confident_predictions,
            'confidence_rate': accuracy_rate,
            'buffer_size': len(self.prediction_buffer)
        }
    
    def __repr__(self):
        return f"ASLInferenceEngine(device={self.device}, classes={len(self.class_names)}, smoothing={self.smoothing_window})"


# Utility function for batch processing
def process_image_file(engine, image_path):
    """
    Process a single image file
    
    Args:
        engine: ASLInferenceEngine instance
        image_path: Path to image file
    
    Returns:
        Prediction results
    """
    image = Image.open(image_path).convert('RGB')
    return engine.predict(image)


if __name__ == "__main__":
    # Test inference engine
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "../best_asl_model_rtx4060_optimized.pth"
    
    # Initialize engine
    engine = ASLInferenceEngine(
        model_path=model_path,
        device='cuda',
        smoothing_window=5,
        confidence_threshold=0.7
    )
    
    print(f"\n{engine}")
    print(f"Class names: {engine.class_names}")
    print("\nInference engine ready!")
