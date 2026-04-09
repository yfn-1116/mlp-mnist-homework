import os
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR, LOG_DIR


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def plot_loss_curve(train_losses, save_path):
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
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"True: {labels[i]} | Pred: {preds[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_hidden1_weights(model, save_path):
    w1 = model.fc1.weight.detach().cpu().numpy()
    num_neurons = w1.shape[0]

    cols = min(8, num_neurons)
    rows = math.ceil(num_neurons / cols)

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(num_neurons):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(w1[i].reshape(28, 28), cmap="seismic")
        plt.title(f"H1-{i}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_hidden2_effective_weights(model, save_path):
    w1 = model.fc1.weight.detach().cpu().numpy()
    w2 = model.fc2.weight.detach().cpu().numpy()

    effective = np.matmul(w2, w1)
    num_neurons = effective.shape[0]

    cols = min(8, num_neurons)
    rows = math.ceil(num_neurons / cols)

    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(num_neurons):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(effective[i].reshape(28, 28), cmap="seismic")
        plt.title(f"H2-{i}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_inference_activation_demo(model, dataset, device, save_path, activation_type="relu", sample_idx=0):
    model.eval()

    image, label = dataset[sample_idx]
    x = image.view(1, -1).to(device)

    with torch.no_grad():
        logits, z1, h1, z2, h2 = model.forward_with_activations(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

    h1 = h1.cpu().numpy().squeeze()
    h2 = h2.cpu().numpy().squeeze()

    if activation_type.lower() == "relu":
        active1 = h1 > 0
        active2 = h2 > 0
    else:
        active1 = h1 >= h1.mean()
        active2 = h2 >= h2.mean()

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Input Image | True Label: {label}")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.bar(np.arange(len(probs)), probs)
    plt.title("Output Probabilities")
    plt.xlabel("Class")
    plt.ylabel("Probability")

    plt.subplot(2, 2, 3)
    colors1 = ["tab:orange" if a else "lightgray" for a in active1]
    plt.bar(np.arange(len(h1)), h1, color=colors1)
    plt.title(f"Hidden Layer 1 Activations | Active: {active1.sum()}/{len(h1)}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation")

    plt.subplot(2, 2, 4)
    colors2 = ["tab:orange" if a else "lightgray" for a in active2]
    plt.bar(np.arange(len(h2)), h2, color=colors2)
    plt.title(f"Hidden Layer 2 Activations | Active: {active2.sum()}/{len(h2)}")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
