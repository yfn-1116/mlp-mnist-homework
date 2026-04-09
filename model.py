import torch
import torch.nn as nn

from config import INPUT_SIZE, HIDDEN1, HIDDEN2, NUM_CLASSES, ACTIVATION


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        if ACTIVATION.lower() == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif ACTIVATION.lower() == "sigmoid":
            self.act1 = nn.Sigmoid()
            self.act2 = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {ACTIVATION}")

        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, NUM_CLASSES)

    def forward(self, x):
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        out = self.fc3(h2)
        return out

    def forward_with_activations(self, x):
        z1 = self.fc1(x)
        h1 = self.act1(z1)
        z2 = self.fc2(h1)
        h2 = self.act2(z2)
        out = self.fc3(h2)
        return out, z1, h1, z2, h2
