import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(784, 2000, device="cuda")
        self.second_layer = nn.Linear(2000, 10, device="cuda")
    def forward(self, x):
        first_layer_out = self.first_layer(x)
        return softmax(self.second_layer(relu(first_layer_out)), dim=-1)
