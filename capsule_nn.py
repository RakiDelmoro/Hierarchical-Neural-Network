import torch
import torch.nn as nn
from torch import Tensor

class CapsuleNeuralNetwork(nn.Module):
    def __init__(self, num_capsule_wide: int, feature_sizes: list, input_feature_size: int,
                 roll_each_layer: int):
        super().__init__()
        assert input_feature_size % feature_sizes[0] == 0, f"Input tensor expected to have a size ({input_feature_size}) that divides evenly by the first feature size ({feature_sizes[0]})."
        assert feature_sizes[0] == feature_sizes[-1], f"First ({feature_sizes[0]}) and last ({feature_sizes[-1]}) feature sizes need to be the same."

        self.layers = []
        for i in range(len(feature_sizes)-1):
            layer = nn.Sequential(nn.Linear(feature_sizes[i], feature_sizes[i+1], device="cuda"), nn.ReLU())
            self.layers.append(layer)
    
        self.num_capsule_tall = input_feature_size // feature_sizes[0]
        self.num_capsule_wide = num_capsule_wide
        self.roll_each_layer = roll_each_layer
        self.output_probability = nn.Sequential(nn.Linear(15, 10, device="cuda"), nn.Softmax(-1))

    def forward(self, x: Tensor):
        previous_layer_output = x
        for _ in range(self.num_capsule_wide):
            for j, layer in enumerate(self.layers):
                outputs = []
                for i in range(self.num_capsule_tall):
                    input_size = previous_layer_output.shape[-1] // self.num_capsule_tall
                    feature_view_start = i * input_size
                    feature_view_end = feature_view_start + input_size
                    input_feature_view = previous_layer_output[:, feature_view_start:feature_view_end]
                    output = layer(input_feature_view)
                    outputs.append(output)
                if j % 2 == 0:
                    outputs.append(outputs[0][:, :self.roll_each_layer])
                    previous_layer_output = torch.concat(outputs, dim=1)[:, self.roll_each_layer:]
                else:
                    outputs.insert(0, outputs[-1][:, -self.roll_each_layer:])
                    previous_layer_output = torch.concat(outputs, dim=1)[:, :-self.roll_each_layer]
        
        return self.output_probability(previous_layer_output)

input = torch.randn(2, 15, device="cuda")
input_feature = input.shape[-1]
m = CapsuleNeuralNetwork(num_capsule_wide=4, feature_sizes=[5, 7, 13, 9, 5], input_feature_size=input_feature,
                         roll_each_layer=1)
out = m(input)
print(out)
