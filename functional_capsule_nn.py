import torch
from layer import linear_layer

def capsule_neural_network(feature_sizes: list, input_feature: int, capsule_tall: int, capsule_wide: int):
    layers = []
    parameters: list[torch.nn.Parameter] = []
    for i in range(len(feature_sizes)-1):
        input_feature = feature_sizes[i] + 1 if i != 0 else (input_feature + input_feature % capsule_tall) // capsule_tall + feature_sizes[0] + 1
        output_feature = feature_sizes[i+1]
        layer, w, b = linear_layer(input_feature, output_feature)
        layers.append(layer)
        parameters.extend([w, b])

    output_layer, w, b = linear_layer(feature_sizes[-1]*capsule_tall, 10)
    parameters.extend([w, b])
    
    def feature_view(x: torch.Tensor):
        assert x.shape[-1] % capsule_tall == 0, f'{capsule_tall} should divisible by {x.shape[-1]}'
        input_view = x.shape[-1] // capsule_tall
        new_input_shape = x.shape[:-1] + (capsule_tall, input_view)
        return x.view(new_input_shape)

    def viewed_layer(x: torch.Tensor, concatenated_capsule_column_and_zeros, layer, layer_idx: int):
        outputs = []
        for i in range(capsule_tall):
            input_for_layer = torch.concat([x[:, i, :], concatenated_capsule_column_and_zeros[:, i, :]], dim=-1)
            output = layer(input_for_layer)
            outputs.append(output)
        if layer_idx % 2 == 0:
            outputs.append(outputs[0][:, :1])
            return torch.concat(outputs, dim=1)[:, 1:]
        else:
            outputs.insert(0, outputs[-1][:, -1:])
            return torch.concat(outputs, dim=1)[:, :-1]
        
    def apply_capsule_tall_dim(x: torch.Tensor):
        # batch | capsule tall | feature_sizes[0] + 1
        return x.unsqueeze(1).repeat(1, capsule_tall, 1)

    def forward_pass(input_batch: torch.Tensor):
        previous_layer_output = input_batch
        tensor_zeros_to_add = torch.zeros(size=(input_batch.shape[0], feature_sizes[0]), device="cuda")
        for capsule_wide_idx in range(capsule_wide):
            for layer_idx, layer in enumerate(layers):
                capsule_column_idx = torch.full([input_batch.shape[0], 1], capsule_wide_idx, device="cuda")
                if layer_idx == 0:
                    capsule_column_and_zeros = apply_capsule_tall_dim(torch.concat([tensor_zeros_to_add, capsule_column_idx], dim=1))
                    input_feature_view = feature_view(previous_layer_output)
                else:
                    capsule_column_and_zeros = apply_capsule_tall_dim(capsule_column_idx)
                    input_feature_view = feature_view(previous_layer_output)
                previous_layer_output = viewed_layer(input_feature_view, capsule_column_and_zeros, layer, layer_idx)
        return output_layer(previous_layer_output)

    return forward_pass, parameters

# x = torch.randn(1, 784, device="cuda")
# m, param = capsule_neural_network([10, 2000, 10], 784, 4, 1)
# print(m(x))
