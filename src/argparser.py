import argparse

def get_parser():
    """
    Returns a parser with all the required arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="../Data/burgers_shock.mat",
        type=str,
        help="Path to the dataset",
    )

    parser.add_argument(
        "--n_u",
        default=100,
        type=int,
        help="Number of initial and boudary points",
    )
    parser.add_argument(
        "--n_f",
        default=10000,
        type=int,
        help="Number of collocation points",
    )
    parser.add_argument(
        "--num_neurons",
        default=20,
        type=int,
        help="Number of neurons in hidden layers",
    )
    parser.add_argument(
        "--num_layers",
        default=2,
        type=int,
        help="Total number of layers in network",
    )
    parser.add_argument(
        "--activation_fn",
        default="Tanh",
        type=str,
        help="Activation function",
    )
    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
        help="Number of epochs to train the model",
    )

    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        help="Config file can be used to run multiple variations of HPs for experimentation. This overrides all the above arguments",
    )

    return parser
