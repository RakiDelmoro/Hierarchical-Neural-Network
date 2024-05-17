import torch
import torch.nn as nn
from torch import Tensor

class CapsuleNeuralNetwork(nn.Module):
    def __init__(self, num_capsule_wide: int, num_capsule_tall: int, feature_sizes: list, input_data_size: int, roll_each_layer: int):
        super().__init__()
        # assert input_data_size % feature_sizes[0] == 0, f"Input tensor expected to have a size ({input_data_size}) that divides evenly by the first feature size ({feature_sizes[0]})."
        assert feature_sizes[0] == feature_sizes[-1], f"First ({feature_sizes[0]}) and last ({feature_sizes[-1]}) feature sizes need to be the same."

        self.layers = []
        for i in range(len(feature_sizes)-1):
            num_input_features = feature_sizes[i] + 1 if i != 0 else input_data_size // num_capsule_tall + feature_sizes[0] + 1
            num_output_features = feature_sizes[i+1]
            layer = nn.Sequential(nn.Linear(num_input_features, num_output_features, device="cuda"), nn.ReLU())
            self.layers.append(layer)

        self.input_data_size = input_data_size
        self.feature_sizes = feature_sizes
        self.num_capsule_tall = num_capsule_tall
        self.num_capsule_wide = num_capsule_wide
        self.roll_each_layer = roll_each_layer

        self.output_probability = nn.Linear(self.feature_sizes[-1], 10, device="cuda")

    def forward(self, primary_input: Tensor):
        previous_layer_output = torch.zeros(size=(primary_input.shape[0], self.feature_sizes[0]*self.num_capsule_tall), device="cuda")
        for capsule_column_index in range(self.num_capsule_wide):
            for layer_index, layer in enumerate(self.layers):
                outputs = []
                for capsule_row_index in range(self.num_capsule_tall):
                    input_size = self.feature_sizes[layer_index]
                    feature_view_start = capsule_row_index * input_size
                    feature_view_end = feature_view_start + input_size
                    input_feature_view = previous_layer_output[:, feature_view_start:feature_view_end]
                    input_to_next_layer = torch.concat([input_feature_view, torch.full([input_feature_view.shape[0], 1], capsule_column_index, device="cuda")], dim=1)
                    if layer_index == 0:
                        primary_input_size = self.input_data_size // self.num_capsule_tall
                        primary_input_view_start = capsule_row_index * primary_input_size
                        primary_input_view_end = primary_input_view_start + primary_input_size
                        primary_input_view = primary_input[:, primary_input_view_start:primary_input_view_end]
                        input_to_next_layer = torch.concat([primary_input_view, input_to_next_layer], dim=1)
                    output = layer(input_to_next_layer)
                    outputs.append(output)
                if layer_index % 2 == 0:
                    outputs.append(outputs[0][:, :self.roll_each_layer])
                    previous_layer_output = torch.concat(outputs, dim=1)[:, self.roll_each_layer:]
                else:
                    outputs.insert(0, outputs[-1][:, -self.roll_each_layer:])
                    previous_layer_output = torch.concat(outputs, dim=1)[:, :-self.roll_each_layer]

        return self.output_probability(previous_layer_output)

# x = torch.randn(1, 784, device="cuda")
# m = CapsuleNeuralNetwork(num_capsule_wide=4, num_capsule_tall=4, feature_sizes=[784, 2000, 784], input_data_size=784, roll_each_layer=1)
# print(m(x))