import torch
from layer import linear_layer
from torch.nn.functional import softmax, relu
def capsule_neural_network(feature_sizes: list, input_feature: int, capsule_tall: int, capsule_wide: int, rotation_amount: int):
    parameters: list[torch.nn.Parameter] = []

    first_layer_feature_size = (input_feature + input_feature % capsule_tall) // capsule_tall + feature_sizes[-1] + 1
    first_layer, w, b = linear_layer(first_layer_feature_size, feature_sizes[0])
    parameters.extend([w, b])

    layers = []
    for i in range(len(feature_sizes)-1):
        input_feature = feature_sizes[i] + 1 
        output_feature = feature_sizes[i+1]
        layer, w, b = linear_layer(input_feature, output_feature)
        layers.append(layer)
        parameters.extend([w, b])
    layers.insert(0, first_layer)

    output_layer, w, b = linear_layer(feature_sizes[-1]*capsule_tall, 10)
    parameters.extend([w, b])

    def input_feature_view(x: torch.Tensor):
        assert x.shape[-1] % capsule_tall == 0, f'{capsule_tall} should divisible by {x.shape[-1]}'
        input_view = x.shape[-1] // capsule_tall
        new_input_shape = x.shape[:-1] + (capsule_tall, input_view)
        return x.view(new_input_shape)

    def rotate_features(outputs: list, rotation_amount: int, layer_idx: int):
        if layer_idx % 2 == 0:
            outputs.append(outputs[0][:, :rotation_amount])
            return torch.concat(outputs, dim=1)[:, rotation_amount:]
        else:
            outputs.insert(0, outputs[-rotation_amount][:, -rotation_amount:])
            return torch.concat(outputs, dim=1)[:, :-rotation_amount]

    def capsule_layer(input_for_layer: torch.Tensor, layer, layer_idx: int):
        capsule_outputs = []
        for vertical_capsule_index in range(capsule_tall):
            view_features_for_capsule = input_for_layer[:, vertical_capsule_index, :]
            capsule_output = layer(view_features_for_capsule)
            capsule_outputs.append(capsule_output)
        return rotate_features(capsule_outputs, rotation_amount, layer_idx)

    def forward_pass(input_batch: torch.Tensor):
        input_feature_viewed = input_feature_view(input_batch)
        previous_layer_output = torch.zeros(size=(input_batch.shape[0], capsule_tall, feature_sizes[-1]), device="cuda")
        for capsule_wide_idx in range(capsule_wide):
            for layer_idx, layer in enumerate(layers):
                capsule_column_idx = torch.full([input_batch.shape[0], capsule_tall, 1], capsule_wide_idx, device="cuda")
                if layer_idx == 0:
                    input_for_layer = torch.concat([input_feature_viewed, previous_layer_output, capsule_column_idx], dim=-1)
                else:
                    previous_layer_output_viewed_feature = input_feature_view(previous_layer_output)
                    input_for_layer = torch.concat([previous_layer_output_viewed_feature, capsule_column_idx], dim=-1)
                previous_layer_output = capsule_layer(input_for_layer, layer, layer_idx)
        return previous_layer_output
    return forward_pass, parameters

# x = torch.randn(1, 784, device="cuda")
# m, param = capsule_neural_network([2000, 10], 784, 1, 1, 1)
# print(m(x))
