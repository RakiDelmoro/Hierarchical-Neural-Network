import torch
import torch.nn as nn
from torch import Tensor

class CapsulePerceptron(nn.Module):
    def __init__(self, feature_sizes: list, capsule_tall: int, capsule_wide: int):
        super().__init__()
        self.capsule_tall = capsule_tall
        self.capsule_wide = capsule_wide
        self.layers = []
        for i in range(len(feature_sizes)-1):
            assert feature_sizes[i] % capsule_tall == 0, f"{capsule_tall} should divisible by {feature_sizes[i]}"
            input_feature = feature_sizes[i] // capsule_tall
            output_feature = feature_sizes[i+1] // capsule_tall
            layer = nn.Sequential(nn.Linear(input_feature, output_feature, device="cuda"), nn.ReLU())
            self.layers.append(layer)
        self.output_layer = nn.Linear(feature_sizes[-1], 10, device="cuda")

    def feature_view(self, x: Tensor):
        assert x.shape[-1] % self.capsule_tall == 0, f"Input feature {x.shape[-1]} should divisible by {self.capsule_tall}"
        feature_view = x.shape[-1] // self.capsule_tall

        # batch | capsule tall | feature view
        new_input_shape = x.shape[:-1] + (self.capsule_tall, feature_view)
        return x.view(new_input_shape)

    def view_layer(self, x, feature_layer, layer_idx):
        outputs = []
        for i in range(x.shape[1]):
            output = feature_layer(x[:, i, :])
            outputs.append(output)
        if layer_idx % 2 == 0:
            outputs.append(outputs[0][:, :1])
            return torch.concat(outputs, dim=1)[:, 1:]
        else:
            outputs.insert(0, outputs[-1][:, -1:])
            return torch.concat(outputs, dim=1)[:, :-1]
        
    def forward(self, x):
        previous_layer_output = x 
        for _ in range(self.capsule_wide):
            for layer_index, layer in enumerate(self.layers):
                input_view = self.feature_view(previous_layer_output)
                previous_layer_output = self.view_layer(input_view, layer, layer_index)

        return self.output_layer(previous_layer_output)
