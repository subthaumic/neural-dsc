# Model generation submodule for experiments
from neural_dsc.mlp import MLP
import torch


def create_quadrants_mono():
    """
    Create a handmade MLP for a monosemantic quadrants classifier.

    weights and biases are chosen such that:
        (+1,+1) → (1,0,0,0) -> (1,0,0,0),
        (-1,+1) → (0,1,0,0) -> (0,1,0,0),
        (-1,-1) → (0,0,1,0) -> (0,0,1,0),
        (+1,-1) → (0,0,0,1) -> (0,0,0,0).
    """
    weight_ratio = 1.25
    layer_sizes = [2, 4, 4, 4]

    net = MLP(layer_sizes=layer_sizes)
    with torch.no_grad():
        for layer in net.hidden_layers:
            layer.bias.zero_()
        net.output_layer.bias.zero_()
        W_in = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
        net.hidden_layers[0].weight.copy_(W_in)
        W_out = torch.tensor(
            [
                [1.0, -weight_ratio, -weight_ratio, 1.0],
                [-weight_ratio, 1.0, -weight_ratio, 1.0],
                [-weight_ratio, 1.0, 1.0, -weight_ratio],
                [1.0, -weight_ratio, 1.0, -weight_ratio],
            ]
        )
        net.hidden_layers[1].weight.copy_(W_out)
        W_identity = torch.eye(4)
        net.output_layer.weight.copy_(W_identity)
    return net


def create_quadrants_poly():
    """
    Create a handmade MLP for a polysemantic quadrants classifier.

    weights and biases are chosen such that:
        (+1,+1) → (1,0,0) -> (1,0,0,0),
        (-1,+1) → (0,1,0) -> (0,1,0,0),
        (-1,-1) → (0,0,1) -> (0,0,1,0),
        (+1,-1) → (0,0,0) -> (0,0,0,1).

    """
    layer_sizes = [2, 3, 4, 4]  # Added identity layer size
    net = MLP(layer_sizes=layer_sizes)
    with torch.no_grad():
        for layer in net.hidden_layers:
            layer.bias.zero_()
        net.output_layer.bias.zero_()

        # First layer
        W_in = torch.tensor([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
        net.hidden_layers[0].weight.copy_(W_in)

        # Second layer
        W_out = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]]
        )
        b_out = torch.tensor([0.0, 0.0, 0.0, 2.0])
        net.hidden_layers[1].weight.copy_(W_out)
        net.hidden_layers[1].bias.copy_(b_out)

        # Identity matrix for the final output layer remains unchanged
        W_identity = torch.eye(4)
        net.output_layer.weight.copy_(W_identity)
    return net
