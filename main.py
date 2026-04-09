import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src import model, utils, config

from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    FIGURE_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    DEVICE,
    SEED,
    EXP_NAME,
    ACTIVATION,
)
from model import MLP
from utils import (
    set_seed,
    ensure_dirs,
    plot_loss_curve,
    plot_acc_curve,
    save_predictions,
    save_hidden1_weights,
    save_hidden2_effective_weights,
    save_inference_activation_demo,
)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return running_loss / len(test_loader), total_correct / total_samples


def save_sample_predictions(model, test_loader, device, save_path):
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images_device = images.view(images.size(0), -1).to(device)
            outputs = model(images_device)
            _, preds = torch.max(outputs, dim=1)

            save_predictions(
                images[:9].cpu(),
                labels[:9].cpu().numpy(),
                preds[:9].cpu().numpy(),
                save_path,
            )
            break


def main():
    print(f"Experiment: {EXP_NAME}")
    print(f"Using device: {DEVICE}")

    set_seed(SEED)
    ensure_dirs()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    test_accuracies = []

    best_acc = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, f"{EXP_NAME}_best_model.pth")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        _, test_acc = evaluate(model, test_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        test_accuracies.append(test_acc)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"- Train Loss: {train_loss:.4f} "
            f"- Test Accuracy: {test_acc * 100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"Best Test Accuracy: {best_acc * 100:.2f}%")
    print(f"Best model saved to: {best_model_path}")

    loss_curve_path = os.path.join(FIGURE_DIR, f"{EXP_NAME}_loss_curve.png")
    acc_curve_path = os.path.join(FIGURE_DIR, f"{EXP_NAME}_acc_curve.png")
    pred_path = os.path.join(FIGURE_DIR, f"{EXP_NAME}_predictions.png")
    h1_path = os.path.join(FIGURE_DIR, f"{EXP_NAME}_hidden1_weights.png")
    h2_path = os.path.join(FIGURE_DIR, f"{EXP_NAME}_hidden2_effective_weights.png")
    activation_demo_path = os.path.join(FIGURE_DIR, f"{EXP_NAME}_inference_activation_demo.png")

    plot_loss_curve(train_losses, loss_curve_path)
    plot_acc_curve(test_accuracies, acc_curve_path)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    save_sample_predictions(model, test_loader, DEVICE, pred_path)
    save_hidden1_weights(model, h1_path)
    save_hidden2_effective_weights(model, h2_path)
    save_inference_activation_demo(
        model=model,
        dataset=test_dataset,
        device=DEVICE,
        save_path=activation_demo_path,
        activation_type=ACTIVATION,
        sample_idx=0
    )

    print("Training finished.")
    print(f"Loss curve saved to: {loss_curve_path}")
    print(f"Accuracy curve saved to: {acc_curve_path}")
    print(f"Prediction samples saved to: {pred_path}")
    print(f"Hidden layer 1 weights saved to: {h1_path}")
    print(f"Hidden layer 2 effective weights saved to: {h2_path}")
    print(f"Inference activation demo saved to: {activation_demo_path}")


if __name__ == "__main__":
    main()
