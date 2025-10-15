"""
ASL Alphabet Model Evaluation Script - RTX 4060 Optimized (Extended)
Generates metrics, bar graphs, metric heatmap, and confusion matrix.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from asl_training_pytorch import EfficientASLNet, ASLDataset, optimize_cuda_settings


def evaluate(model_path: str, test_data_path: str, results_dir: str,
             img_size=(224, 224), batch_size: int = 24):
    """Run evaluation and save artifacts to results_dir.

    Returns a summary dict with metrics and file paths.
    """
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        optimize_cuda_settings()

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names')
    num_classes = len(class_names)

    model = EfficientASLNet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ASLDataset(test_data_path, transform=test_transform, classes=class_names)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
    )

    # Evaluate
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            output = model(data.to(device))
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())

    labels = list(range(len(class_names)))
    accuracy = 100.0 * sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    precision = precision_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    recall = recall_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    f1 = f1_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, labels=labels, average='macro', zero_division=0)
    micro_f1 = f1_score(all_targets, all_preds, labels=labels, average='micro', zero_division=0)

    # Save textual reports
    report = classification_report(all_targets, all_preds, target_names=class_names, labels=labels, zero_division=0)
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Also save compatibility filenames expected by the app
    report_path_test = os.path.join(results_dir, "classification_report_Test.txt")
    with open(report_path_test, "w") as f:
        f.write(report)

    metrics_txt = os.path.join(results_dir, "metrics_Test.txt")
    with open(metrics_txt, "w") as f:
        f.write("ASL Evaluation Metrics\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Micro F1: {micro_f1:.4f}\n")
        f.write("\nPer-class (Precision, Recall, F1):\n")
        for cls, p, r, f1c in zip(class_names, precision, recall, f1):
            f.write(f"{cls}: Precision={p:.4f}, Recall={r:.4f}, F1={f1c:.4f}\n")

    # --- Visualization Section ---
    import pandas as pd
    df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    })

    # 1) Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    # compatibility name
    cm_path_test = os.path.join(results_dir, "confusion_matrix_Test.png")
    try:
        import shutil
        shutil.copyfile(cm_path, cm_path_test)
    except Exception:
        pass

    # 2) Per-Class Metric Bar Chart
    df.set_index('Class')[['Precision', 'Recall', 'F1-score']].plot(kind='bar', figsize=(15, 6))
    plt.title('Per-Class Precision, Recall, and F1-score')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    bar_path = os.path.join(results_dir, "metrics_bar_chart.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # 3) Metric Heatmap
    df_melted = df.melt(id_vars='Class', var_name='Metric', value_name='Score')
    pivot_df = df_melted.pivot(index='Class', columns='Metric', values='Score')
    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', linewidths=0.5)
    plt.title('Classification Metrics Heatmap', fontsize=16)
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, "metrics_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    summary = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'results_dir': results_dir,
        'report_path': report_path,
        'report_path_test': report_path_test,
        'metrics_txt': metrics_txt,
        'confusion_matrix_png': cm_path,
        'confusion_matrix_png_test': cm_path_test,
        'metrics_bar_chart_png': bar_path,
        'metrics_heatmap_png': heatmap_path,
        'class_names': class_names,
    }
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ASL Model Evaluation with Visualizations")
    parser.add_argument('--testset', type=str, default="MergedDataset/Test", help="Test dataset directory")
    parser.add_argument('--model', type=str, default="best_asl_model_rtx4060_optimized.pth", help="Model .pth path")
    parser.add_argument('--results', type=str, default="validationResults", help="Results output directory")
    args = parser.parse_args()

    # Resolve defaults to absolute/working paths as needed
    model_path = args.model
    test_data_path = args.testset
    results_dir = args.results

    summary = evaluate(model_path, test_data_path, results_dir)
    print(f"\n✅ Results saved in: {summary['results_dir']}")
    print(f"Accuracy: {summary['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
