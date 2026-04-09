import torch
import torch.nn as nn

from config import INPUT_SIZE, HIDDEN1, HIDDEN2, NUM_CLASSES


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, NUM_CLASSES)
        )

    def forward(self, x):
        return self.model(x)
