import torch
from torch import empty
from torch.nn import Parameter
from torch.nn.functional import relu

def capsule_neural_network(feature_sizes: list, input_feature: int, capsule_tall: int, capsule_wide: int):
    def linear_layer(in_features: int, out_features: int, device: str="cuda"):
        weight = Parameter(empty((out_features, in_features), device=device))
        bias = Parameter(empty(out_features, device=device))
        return lambda x: relu(torch.matmul(x, weight.t()) + bias), weight, bias

    layers = []
    parameters = []
    for i in range(len(feature_sizes)-1):
        input_feature = feature_sizes[i] + 1 if i != 0 else input_feature // capsule_tall + feature_sizes[0] + 1
        output_feature = feature_sizes[i+1] // capsule_tall
        layer, w, b = linear_layer(input_feature, output_feature)
        layers.append(layer)
        parameters.extend([w, b])

    def feature_view(x: torch.Tensor):
        input_view = x.shape[-1] // capsule_tall
        new_input_shape = x.shape[:-1] + (capsule_tall, input_view)
        return x.view(new_input_shape)
    
    def viewed_layer(x: torch.Tensor, layer, layer_idx: int):
        outputs = []
        for i in range(x.shape[1]):
            output = layer(x[:, i, :])
            outputs.append(output)
        if layer_idx % 2 == 0:
            outputs.append(outputs[0][:, :1])
            return torch.concat(outputs, dim=1)[:, 1:]
        else:
            outputs.insert(0, outputs[-1][:, -1:])
            return torch.concat(outputs, dim=1)[:, :-1]

    def forward_pass(input_batch: torch.Tensor):
        previous_layer_output = torch.zeros(size=(input_batch.shape[0], feature_sizes[0]*capsule_tall), device="cuda")
        for capsule_wide_idx in range(capsule_wide):
            for layer_idx, layer in enumerate(layers):
                capsule_idx_as_tensor = torch.full([input_batch.shape[0], 1], capsule_wide_idx, device="cuda")
                if layer_idx == 0:
                    input_for_layer = torch.concat([previous_layer_output, capsule_idx_as_tensor, input_batch], dim=1)
                else:
                    input_for_layer = torch.concat([previous_layer_output, capsule_idx_as_tensor], dim=1)
                input_feature_view = feature_view(input_for_layer)
                previous_layer_output = viewed_layer(input_feature_view, layer, layer_idx)
        return previous_layer_output

    return forward_pass, parameters

x = torch.randn(1, 784, device="cuda")
m, param = capsule_neural_network([10, 2000, 10], 784, 1, 1)
print(m(x))
print(len(param))
