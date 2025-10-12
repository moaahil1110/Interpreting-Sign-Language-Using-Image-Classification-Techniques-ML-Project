"""

ASL Alphabet CNN Model Training Script - RTX 4060 Optimized Version

Complete implementation optimized for NVIDIA RTX 4060 Mobile 8GB VRAM
- Utilizes Tensor Cores via Automatic Mixed Precision (AMP)
- Optimized memory management for 8GB VRAM
- Enhanced CUDA core utilization
- Dynamic batch size optimization

"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time
import gc

# RTX 4060 OPTIMIZATION IMPORTS
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# RTX 4060 SPECIFIC OPTIMIZATIONS
def optimize_cuda_settings():
    """Optimize CUDA settings for RTX 4060"""
    if torch.cuda.is_available():
        # Enable cuDNN optimizations
        cudnn.benchmark = True
        cudnn.enabled = True

        # Set memory fraction to avoid OOM - use 85% of available memory
        torch.cuda.set_per_process_memory_fraction(0.85)

        # Clear cache
        torch.cuda.empty_cache()

        # Print GPU info
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class ASLDataset(Dataset):
    """Custom Dataset for ASL alphabet images with memory optimization"""

    def __init__(self, data_path, transform=None, classes=None):
        """
        Initialize ASL Dataset with memory optimization

        Args:
            data_path: Path to dataset directory
            transform: Transformations to apply to images
            classes: List of class names (optional)
        """
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.classes = classes or []

        # Build dataset
        self._build_dataset()
        print(f"Dataset initialized with {len(self.samples)} samples")
        print(f"Classes: {self.classes}")

    def _build_dataset(self):
        """Build dataset from directory structure"""
        class_folders = [f for f in os.listdir(self.data_path) 
                        if os.path.isdir(os.path.join(self.data_path, f))]

        # Sort class folders for consistent ordering
        class_folders.sort()

        if not self.classes:
            self.classes = class_folders

        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Collect all samples
        for class_name in class_folders:
            if class_name not in self.class_to_idx:
                continue

            class_path = os.path.join(self.data_path, class_name)
            class_idx = self.class_to_idx[class_name]

            # Get all image files
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image with memory optimization
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

class EfficientASLNet(nn.Module):
    """Optimized CNN for ASL alphabet recognition - RTX 4060 optimized"""

    def __init__(self, num_classes=26, dropout_rate=0.3):
        super(EfficientASLNet, self).__init__()

        # Optimized architecture for RTX 4060 tensor cores
        # Use channel sizes that are multiples of 8 for tensor core efficiency

        # Convolutional layers with optimal channel sizes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        # Additional conv layer for better feature extraction
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Reduces to 4x4
        self.dropout2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers - optimized for tensor cores (multiples of 8)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Convolutional blocks with efficient activation patterns
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

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class OptimizedASLTransferLearningNet(nn.Module):
    """Transfer learning model optimized for RTX 4060"""

    def __init__(self, num_classes=26, backbone='efficientnet_b2', pretrained=True):
        super(OptimizedASLTransferLearningNet, self).__init__()

        if backbone == 'efficientnet_b2':
            import torchvision.models as models
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Optimized classifier with tensor core friendly dimensions
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class OptimizedASLTrainer:
    """RTX 4060 Optimized ASL Trainer with AMP and memory optimization"""

    def __init__(self, data_path, img_size=(224, 224), batch_size=None, device=None):
        """
        Initialize optimized ASL Trainer for RTX 4060

        Args:
            data_path: Path to ASL alphabet dataset
            img_size: Target image size (height, width)
            batch_size: Training batch size (auto-optimized if None)
            device: Device to use ('cuda' or 'cpu')
        """
        self.data_path = data_path
        self.img_size = img_size

        # Set device and optimize CUDA
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            optimize_cuda_settings()

        # Auto-optimize batch size for RTX 4060 if not specified
        if batch_size is None:
            self.batch_size = self._determine_optimal_batch_size()
        else:
            self.batch_size = batch_size

        print(f"Using batch size: {self.batch_size}")

        # Initialize AMP scaler for tensor core utilization
        self.scaler = GradScaler()

        # Initialize training components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        # Gradient accumulation for effective larger batch sizes
        self.accumulation_steps = 2  # Effective batch size = batch_size * accumulation_steps

        # Setup data transforms
        self.setup_transforms()

    def _determine_optimal_batch_size(self):
        """Determine optimal batch size for RTX 4060 8GB VRAM"""
        if not torch.cuda.is_available():
            return 16

        # RTX 4060 optimal batch sizes based on available VRAM
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        if gpu_memory_gb >= 7.5:  # RTX 4060 8GB
            return 24  # Optimal for 224x224 images with mixed precision
        elif gpu_memory_gb >= 5.5:
            return 16
        else:
            return 8

    def setup_transforms(self):
        """Setup optimized data augmentation and preprocessing transforms"""
        # Training transforms with aggressive augmentation for better generalization
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size[0] + 32, self.img_size[1] + 32)),
            transforms.RandomCrop(self.img_size),
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Validation transforms
        self.val_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup_data_loaders(self, val_split=0.2):
        """Setup optimized data loaders for RTX 4060"""
        # Determine optimal number of workers
        num_workers = min(8, os.cpu_count())  # RTX 4060 benefits from multiple workers

        # Build datasets
        base_train_dataset = ASLDataset(self.data_path, transform=self.train_transform)

        # Split dataset
        dataset_size = len(base_train_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size

        train_subset, val_subset_tmp = random_split(base_train_dataset, [train_size, val_size])

        # Create validation dataset with different transforms
        base_val_dataset = ASLDataset(self.data_path, transform=self.val_transform, 
                                    classes=base_train_dataset.classes)
        val_indices = val_subset_tmp.indices if hasattr(val_subset_tmp, 'indices') else list(range(dataset_size - val_size, dataset_size))
        val_subset = Subset(base_val_dataset, val_indices)

        # Create optimized data loaders
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch batches
        )

        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        self.num_classes = len(base_train_dataset.classes)
        self.class_names = base_train_dataset.classes

        print(f"Training samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Effective batch size (with accumulation): {self.batch_size * self.accumulation_steps}")

    def build_model(self, model_type='efficient', **kwargs):
        """
        Build optimized model architecture for RTX 4060

        Args:
            model_type: 'efficient', 'transfer', or 'custom'
            **kwargs: Additional arguments for model
        """
        if model_type == 'efficient':
            self.model = EfficientASLNet(num_classes=self.num_classes, **kwargs)
        elif model_type == 'transfer':
            backbone = kwargs.get('backbone', 'efficientnet_b2')
            self.model = OptimizedASLTransferLearningNet(
                num_classes=self.num_classes,
                backbone=backbone,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Move model to device and enable mixed precision
        self.model = self.model.to(self.device)

        # Use label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimized optimizer for RTX 4060
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler with warmup
        total_steps = len(self.train_loader) * 50  # Assume 50 epochs max
        warmup_steps = total_steps // 10

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.002,
            total_steps=total_steps,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos'
        )

        # Print model info
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model built: {model_type}")
        print(f"Total parameters: {param_count:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Estimate memory usage
        if self.device.type == 'cuda':
            model_memory = param_count * 4 / 1e9  # 4 bytes per parameter
            print(f"Estimated model memory: {model_memory:.2f} GB")

    def train_epoch(self):
        """Train for one epoch with AMP and gradient accumulation"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            # Forward pass with AMP (enables tensor cores)
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Statistics
            running_loss += loss.item() * self.accumulation_steps
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

            # Memory cleanup every 50 batches
            if batch_idx % 50 == 0 and batch_idx > 0:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validate for one epoch with AMP"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation', leave=False):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                # Forward pass with AMP
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train_model(self, epochs=50, save_path='best_asl_model_rtx4060.pth', early_stopping_patience=15, start_epoch=0, resume_checkpoint=None):
        """
        Train the model with RTX 4060 optimizations

        Args:
            epochs: Number of training epochs
            save_path: Path to save best model
            early_stopping_patience: Patience for early stopping
        """
        best_val_acc = 0.0
        patience_counter = 0
        current_epoch = start_epoch

        # Resume from checkpoint if provided
        if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
            print(f"Resuming training from checkpoint: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_acc = checkpoint.get('val_acc', 0.0)
            current_epoch = checkpoint.get('epoch', start_epoch) + 1
            print(f"Resumed at epoch {current_epoch}, best val acc: {best_val_acc:.2f}%")

        print(f"Starting optimized training for {epochs} epochs on RTX 4060...")
        print(f"Using AMP for tensor core utilization")
        print(f"Gradient accumulation steps: {self.accumulation_steps}")

        for epoch in range(current_epoch, epochs):
            start_time = time.time()

            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print epoch results
            epoch_time = time.time() - start_time
            current_lr = self.scheduler.get_last_lr()[0]

            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")

            # Memory info
            if self.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_cached = torch.cuda.memory_reserved() / 1e9
                print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")

            print("-" * 60)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.class_names,
                    'img_size': self.img_size
                }, save_path)
                print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping if accuracy >= 99%
            if val_acc >= 99.0:
                print(f"Early stopping: Validation accuracy reached {val_acc:.2f}% at epoch {epoch+1}")
                break

            # Early stopping by patience
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Memory cleanup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")

        # Load best model
        checkpoint = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        return self.history

    def plot_training_history(self):
        """Plot training history with enhanced visualization"""
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy plot
        epochs = range(1, len(self.history['train_acc']) + 1)
        ax1.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss plot
        ax2.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning rate plot (if available)
        if hasattr(self.scheduler, 'get_last_lr'):
            ax3.plot(epochs, [0.001] * len(epochs), 'g-', label='Learning Rate', linewidth=2)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Validation accuracy improvement
        val_acc_diff = np.diff(self.history['val_acc'])
        ax4.plot(epochs[1:], val_acc_diff, 'purple', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('Validation Accuracy Improvement', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy Change (%)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('rtx4060_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self, detailed=True):
        """Evaluate model with detailed metrics"""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Evaluating'):
                data = data.to(self.device, non_blocking=True)

                with autocast():
                    output = self.model(data)

                _, predicted = torch.max(output, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.numpy())

        if detailed:
            # Classification report
            print("\n" + "="*60)
            print("CLASSIFICATION REPORT")
            print("="*60)
            print(classification_report(all_targets, all_preds, target_names=self.class_names))

            # Confusion matrix
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix - RTX 4060 Optimized Model', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            plt.savefig('rtx4060_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()

        # Calculate accuracy
        accuracy = 100. * sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
        return accuracy

def main():
    """Main training function optimized for RTX 4060"""
    # Configuration optimized for RTX 4060 8GB VRAM
    DATA_PATH = "/home/worm/A/ML/mlproj/Datasets/ASL_Alphabet_Dataset/asl_alphabet_train"
    IMG_SIZE = (224, 224)  # Optimal for RTX 4060
    BATCH_SIZE = None  # Auto-optimize
    EPOCHS = 50

    print("="*60)
    print("ASL ALPHABET TRAINING - RTX 4060 OPTIMIZED")
    print("="*60)

    # Initialize trainer
    trainer = OptimizedASLTrainer(DATA_PATH, IMG_SIZE, BATCH_SIZE)

    # Setup data loaders
    trainer.setup_data_loaders(val_split=0.2)

    # Build model - try efficient model first, fallback to transfer learning if needed
    try:
        print("Building efficient CNN model...")
        trainer.build_model(model_type='efficient', dropout_rate=0.3)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Switching to transfer learning model for better memory efficiency...")
            trainer.build_model(model_type='transfer', backbone='efficientnet_b2')
        else:
            raise e

    # Train model
    history = trainer.train_model(
        epochs=50,  # or your desired total epochs
        save_path='best_asl_model_rtx4060_optimized.pth',
        early_stopping_patience=15,
        start_epoch=27,
        resume_checkpoint='best_asl_model_rtx4060_optimized.pth'
    )

    # Plot training history
    trainer.plot_training_history()

    # Evaluate model
    final_accuracy = trainer.evaluate_model(detailed=True)

    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Validation Accuracy: {final_accuracy:.2f}%")
    print("RTX 4060 optimizations applied:")
    print("✅ Automatic Mixed Precision (AMP) - Tensor Cores utilized")
    print("✅ Memory optimization for 8GB VRAM")
    print("✅ Gradient accumulation for effective larger batch sizes")
    print("✅ Optimized data loading with prefetching")
    print("✅ CUDA memory management")
    print("="*60)

if __name__ == "__main__":
    main()
