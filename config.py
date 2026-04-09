import os
import torch

# =========================
# Experiment Configuration
# =========================
EXP_NAME = "limit_relu_64_64"

# =========================
# Path Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# =========================
# Training Configuration
# =========================
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEBUG_EPOCHS = 3
EPOCHS = 10

# =========================
# Model Configuration
# =========================
INPUT_SIZE = 28 * 28
HIDDEN1 = 64
HIDDEN2 = 64
NUM_CLASSES = 10
ACTIVATION = "relu"

# =========================
# Device Configuration
# =========================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# =========================
# Random Seed
# =========================
SEED = 42
