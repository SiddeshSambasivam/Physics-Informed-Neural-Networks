import torch
import torch.nn as nn
from collections import OrderedDict

class PhysicsINN(nn.Module):
    def __init__(
        self, num_layers: int = 2, num_neurons: int = 20, activation_fn: str = "tanh"
    ) -> None:

        """
        Hyper Parameters for the Model:
        1. Number of layers in the network
        2. Number of neurons in the network
        3. Activation function for the input and hidden layers

        """

        super(PhysicsINN, self).__init__()

        self.activation_fns = {
            "Tanh": torch.nn.Tanh,
            "Sigmoid": torch.nn.Sigmoid,
            "ReLU": torch.nn.ReLU,
        }

        if self.activation_fns.get(activation_fn) == None:
            raise ValueError(
                "Invalid Activation function. Use Sigmoid, Tanh or ReLU as activation function"
            )
        else:
            # Each hidden layer contained 20 neurons and a hyperbolic tangent activation function by default.
            self.activation_func = torch.nn.Tanh

        self.num_layers = num_layers
        self.num_neurons = num_neurons

        ordered_layers = list()

        ordered_layers.append(("input_layer", nn.Linear(2, self.num_neurons)))
        ordered_layers.append(("input_activation", self.activation_func()))

        # Create num_layers-2 linear layers with num_neuron neurons and tanh activation function
        self.num_hidden_layers = self.num_layers - 2
        for i in range(self.num_hidden_layers):

            ordered_layers.append(
                ("layer_%d" % (i + 1), nn.Linear(self.num_neurons, self.num_neurons))
            )
            ordered_layers.append(
                ("layer_%d_activation" % (i + 1), self.activation_func())
            )

        ordered_layers.append(("output_layer", nn.Linear(self.num_neurons, 1)))

        self.net = nn.Sequential(OrderedDict(ordered_layers))

        self.init_weights()

    def init_weights(
        self,
    ) -> None:
        """
        Initializes the weights and biases of all the layers in the model

        NOTE: According to the paper, the model's weights are initialized by xaviers' distribution
        and biases are initialized as zeros

        """
        for param in self.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.xavier_normal_(param)
            elif len(param.shape) == 1:
                torch.nn.init.zeros_(param)

    def forward(self, inputs) -> torch.Tensor:
        """returns the output from the model"""

        out = self.net(inputs)

        return out
        