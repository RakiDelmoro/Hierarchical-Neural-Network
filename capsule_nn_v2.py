import torch
import torch.nn as nn
from torch import Tensor

class CapsuleNeuralNetworkV2(nn.Module):
    def __init__(self, capsule_wide, capsule_tall):
        super().__init__()
        # self.layers = []
        # for i in range(len(feature_sizes)-1):
        #     layers = nn.Sequential(nn.Linear(feature_sizes[i], feature_sizes[i+1], device="cuda"))
        #     self.layers.append(layers)

        self.first_copy = nn.Linear(784//capsule_tall, 784//capsule_tall, device="cuda")
        self.second_copy = nn.Linear(784//capsule_tall, 784//capsule_tall, device="cuda")
        self.third_copy = nn.Linear(784//capsule_tall, 784//capsule_tall, device="cuda")

        self.num_capsule = 4
        self.capsule_tall = capsule_tall
        self.capsule_wide = capsule_wide

        self.output_decoder = nn.Sequential(nn.Linear(784, 784, device="cuda"), nn.ReLU(), nn.Linear(784, 784, device="cuda"))
        self.output_probability = nn.Sequential(nn.Linear(784, 10, device="cuda"), nn.Softmax(-1))

    def apply_capsule_tall_dim(self, x: Tensor):
        assert x.shape[-1] % self.capsule_tall == 0, f"Input feature {x.shape[-1]} should divisible by {self.capsule_tall}"
        feature_view = x.shape[-1] // self.capsule_tall

        # batch | capsule tall | feature view
        new_input_shape = x.shape[:-1] + (self.capsule_tall, feature_view)
        # batch | capsule tall | feature view
        return x.view(new_input_shape)

    def interact_feature_view(self, x: Tensor):
        first_copy_out = self.first_copy(x)
        second_copy_out = self.second_copy(x)
        third_copy_out = self.third_copy(x)

        feature_interaction_score = torch.matmul(first_copy_out, second_copy_out.transpose(-2, -1))
        feature_interaction_probability = nn.functional.softmax(feature_interaction_score, dim=-1)
        feature_score_for_all_capsule = torch.matmul(feature_interaction_probability, third_copy_out)

        # batch | capsule tall | feature view -> batch | capsule tall*feature view
        return feature_score_for_all_capsule.view(x.shape[0], -1)
    
    def forward(self, x: Tensor):
        previous_output = x
        for _ in range(self.capsule_wide):
            # batch | input feature -> batch | capsule tall | feature view
            previous_output_new_shape = self.apply_capsule_tall_dim(previous_output)
            previous_output = self.interact_feature_view(previous_output_new_shape)
        
        output = self.output_decoder(previous_output)
        return self.output_probability(output)
