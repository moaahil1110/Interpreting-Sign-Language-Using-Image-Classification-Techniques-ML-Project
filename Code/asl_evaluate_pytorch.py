"""
ASL Alphabet Model Evaluation Script - RTX 4060 Optimized (Extended)
Generates metrics, bar graphs, metric heatmap, and confusion matrix.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from asl_training_pytorch import EfficientASLNet, ASLDataset, optimize_cuda_settings


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ASL Model Evaluation with Visualizations")
    parser.add_argument('--testset', type=str, default="MergedDataset/Test", help="Test dataset directory")
    args = parser.parse_args()

    MODEL_PATH = "/home/worm/A/ML/mlproj/best_asl_model_rtx4060_optimized.pth"
    TEST_DATA_PATH = "/home/worm/A/ML/mlproj/Datasets/TestNew"
    RESULTS_DIR = "/home/worm/A/ML/mlproj/ModelResults"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 24
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        optimize_cuda_settings()
    print(f"Using device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint.get('class_names')
    num_classes = len(class_names)

    model = EfficientASLNet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ASLDataset(TEST_DATA_PATH, transform=test_transform, classes=class_names)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=min(8, os.cpu_count()), pin_memory=True)

    # Evaluate
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            output = model(data.to(device))
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())

    labels = list(range(len(class_names)))
    accuracy = 100. * sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    precision = precision_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    f1 = f1_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)

    # Save report
    report = classification_report(all_targets, all_preds, target_names=class_names, labels=labels, zero_division=0)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # --- Visualization Section ---
    import pandas as pd
    df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    })

    # 1️⃣ Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 2️⃣ Per-Class Metric Bar Chart
    df.set_index('Class')[['Precision', 'Recall', 'F1-score']].plot(kind='bar', figsize=(15,6))
    plt.title('Per-Class Precision, Recall, and F1-score')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_bar_chart.png"), dpi=300)
    plt.close()

    # 3️⃣ Metric Heatmap
    df_melted = df.melt(id_vars='Class', var_name='Metric', value_name='Score')
    pivot_df = df_melted.pivot(index='Class', columns='Metric', values='Score')
    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', linewidths=0.5)
    plt.title('Classification Metrics Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_heatmap.png"), dpi=300)
    plt.close()

    print(f"\n✅ Results saved in: {RESULTS_DIR}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
