"""
ASL Alphabet Model Evaluation Script - RTX 4060 Optimized

Evaluates the trained model on the test set and prints accuracy, classification report, and confusion matrix.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# Import model and dataset classes from training script
from asl_training_pytorch import EfficientASLNet, OptimizedASLTransferLearningNet, ASLDataset, optimize_cuda_settings

def main():
    import argparse
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    # Argument parser for test dataset selection
    parser = argparse.ArgumentParser(description="ASL Model Evaluation")
    parser.add_argument('--testset', type=str, default="MergedDataset/Test", help="Test dataset directory (relative to workspace root)")
    args = parser.parse_args()

    MODEL_PATH = "/home/worm/A/ML/mlproj/best_asl_model_rtx4060_optimized.pth"
    TEST_DATA_PATH = "/home/worm/A/ML/mlproj/Datasets/TestNew"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 24
    RESULTS_DIR = "/home/worm/A/ML/mlproj/ModelResults"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        optimize_cuda_settings()
    print(f"Using device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint.get('class_names')
    num_classes = len(class_names)

    try:
        model = EfficientASLNet(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception:
        model = OptimizedASLTransferLearningNet(num_classes=num_classes, backbone='efficientnet_b2')
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ASLDataset(TEST_DATA_PATH, transform=test_transform, classes=class_names)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True
    )

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())

    labels = list(range(len(class_names)))
    # Metrics
    accuracy = 100. * sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    precision = precision_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    f1 = f1_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, labels=labels, average='macro', zero_division=0)
    micro_f1 = f1_score(all_targets, all_preds, labels=labels, average='micro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, labels=labels, average='weighted', zero_division=0)

    # Save metrics to file
    with open(os.path.join(RESULTS_DIR, f"metrics_{os.path.basename(args.testset)}.txt"), "w") as f:
        f.write(f"Test set: {TEST_DATA_PATH}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Micro F1: {micro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")
        f.write("\nPer-class metrics:\n")
        for idx, cname in enumerate(class_names):
            f.write(f"{cname}: Precision={precision[idx]:.4f}, Recall={recall[idx]:.4f}, F1={f1[idx]:.4f}\n")

    # Save classification report
    report = classification_report(all_targets, all_preds, target_names=class_names, labels=labels, zero_division=0)
    with open(os.path.join(RESULTS_DIR, f"classification_report_{os.path.basename(args.testset)}.txt"), "w") as f:
        f.write(report)

    # Save confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {os.path.basename(args.testset)}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{os.path.basename(args.testset)}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Detailed metrics and plots saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
