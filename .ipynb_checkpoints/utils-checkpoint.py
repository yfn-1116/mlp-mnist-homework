import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR, LOG_DIR


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    """Create required directories if they do not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def calculate_accuracy(outputs, labels):
    """Calculate classification accuracy."""
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def plot_loss_curve(train_losses, save_path):
    """Plot and save training loss curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_acc_curve(test_accuracies, save_path):
    """Plot and save test accuracy curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(test_accuracies, marker='o')
    plt.title("Test Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_predictions(images, labels, preds, save_path):
    """Save prediction samples as a figure."""
    plt.figure(figsize=(10, 10))

    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"True: {labels[i]} | Pred: {preds[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
