import torch
import torch.nn as nn
from torch import Tensor

class CapsuleNeuralNetworkV2(nn.Module):
    def __init__(self, feature_sizes, capsule_wide, capsule_tall):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(feature_sizes)-1):
            assert feature_sizes[i] % capsule_tall == 0, f"{capsule_tall} should divisible by {feature_sizes[i]}"
            input_feature_view = feature_sizes[i] // capsule_tall
            output_feature_view = feature_sizes[i+1] // capsule_tall
            capsule_tall_layer = []
            for _ in range(capsule_tall):
                input_view_layer = nn.Linear(input_feature_view, output_feature_view, device="cuda")
                capsule_tall_layer.append(input_view_layer)
            self.layers.append(nn.Sequential(*capsule_tall_layer, nn.ReLU()))

        self.capsule_tall = capsule_tall
        self.capsule_wide = capsule_wide
        self.output_decoder = nn.Linear(feature_sizes[0], 10, device="cuda")

    def apply_capsule_tall_dim(self, x: Tensor):
        assert x.shape[-1] % self.capsule_tall == 0, f"Input feature {x.shape[-1]} should divisible by {self.capsule_tall}"
        feature_view = x.shape[-1] // self.capsule_tall

        # batch | capsule tall | feature view
        new_input_shape = x.shape[:-1] + (self.capsule_tall, feature_view)
        return x.view(new_input_shape)
 
    def forward(self, x: Tensor):
        previous_output = x
        for _ in range(self.capsule_wide):
            for layer in self.layers:
                input_feature_view = self.apply_capsule_tall_dim(previous_output)
                view_outputs = []
                for view_idx in range(input_feature_view.shape[1]):
                    output = layer[view_idx](input_feature_view[:, view_idx, :])
                    view_outputs.append(output)
                previous_output = torch.concat(view_outputs, dim=1)

        return self.output_decoder(previous_output)
    