import torch
from torch import empty
from torch.nn import Parameter
from functools import reduce
from torch.nn.functional import relu

def capsule_neural_network(feature_sizes: list, input_feature: int, capsule_tall: int, capsule_wide: int):
    parameters = []
    def linear_layer(in_features: int, out_features: int, bias: bool=True, device: str="cuda"):
        nonlocal parameters
        weight = Parameter(empty((out_features, in_features), device=device))
        bias = Parameter(empty(out_features, device=device))
        parameters.extend([weight, bias])
        return lambda x: relu(torch.matmul(x, weight.t()) + bias)

    layers = []
    for i in range(len(feature_sizes)-1):
        input_feature = feature_sizes[i] + 1 if i != 0 else input_feature // capsule_tall + feature_sizes[0] + 1
        output_feature = feature_sizes[i+1]
        layer = linear_layer(input_feature, output_feature)
        layers.extend(layer)

    def feature_view(x: torch.Tensor):
        input_view = x.shape[-1] // capsule_tall
        new_input_shape = x.shape[:-1] + (capsule_tall, input_view)
        return x.view(new_input_shape)
    
    def viewed_layer(x, layer, layer_idx):
        outputs = []
        for i in range(x.shape[1]):
            output = layer(x[:, i, :])
            outputs.append(output)
        # return torch.concat(outputs, dim=1)
        if layer_idx % 2 == 0:
            outputs.append(outputs[0][:, :1])
            return torch.concat(outputs, dim=1)[:, 1:]
        else:
            outputs.insert(0, outputs[-1][:, -1:])
            return torch.concat(outputs, dim=1)[:, :-1]

    def forward_pass(input_tensor, layer):
        previous_layer_output = torch.zeros(size=(input_tensor.shape[0], feature_sizes[0]*capsule_tall), device="cuda")
        for capsule_wide_idx in range(capsule_wide):
            for idx in range(len(layers)-1):
                # TODO: concatenate capsule_wide ix to the input
                capsule_idx_as_tensor = torch.full([previous_layer_output.shape[0], 1], capsule_wide_idx, device="cuda")
                if idx == 0:
                    input_for_layer = torch.concat([previous_layer_output, capsule_idx_as_tensor, input_tensor], dim=1)
                else:
                    input_for_layer = torch.concat([previous_layer_output, capsule_idx_as_tensor], dim=1)
                input_feature_view = feature_view(input_for_layer)
                previous_layer_output = viewed_layer(input_feature_view, layer, idx)
        return previous_layer_output

    def model(input_batch):
        return reduce(forward_pass, layers, input_batch)

    return model, parameters

# x = torch.randn(1, 10, device="cuda")
# m, param = capsule_neural_network([10, 10, 10], 10, 1, 1)
# print(m(x))
