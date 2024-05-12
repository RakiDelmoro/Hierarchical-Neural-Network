import torch
import torch.nn as nn
from torch import Tensor

class CapsuleNeuralNetworkV2(nn.Module):
    def __init__(self, feature_sizes, capsule_wide, capsule_tall):
        super().__init__()
        self.layers = []
        for i in range(len(feature_sizes)-1):
            layers = nn.Sequential(nn.Linear(feature_sizes[i], feature_sizes[i+1], device="cuda"))
            self.layers.append(layers)

        self.capsule_tall = capsule_tall
        self.capsule_wide = capsule_wide
        self.output_probability = nn.Sequential(nn.Linear(feature_sizes[0], 10, device="cuda"), nn.Softmax(-1))

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
                # batch | input feature -> batch | capsule tall | feature view
                previous_output_new_shape = self.apply_capsule_tall_dim(previous_output)
                # TODO: after getting a new shape apply a mechanism of each input feature view communicate each other
                # shape after each input view communicate -> batch | capsule tall*feature_view 

        return self.output_probability(previous_output)


