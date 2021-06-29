import os
from datetime import datetime
from itertools import product

import torch
import scipy.io
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange

from pylab import meshgrid
from matplotlib import pyplot as plt

from model import PhysicsINN
from argparser import get_parser
from utils import read_config


def prepare_data(data_path: str, n_u: int, n_f: int) -> list:
    """
    Creates a list of training data and labels.
    Converts them to tensors and returns them as a list.
    """

    data_dict = scipy.io.loadmat(data_path)
    x_data, t_data, u_data = data_dict["x"], data_dict["t"], data_dict["usol"]

    u_t, u_x = meshgrid(t_data, x_data)

    u_data_transformed = u_data.flatten()[:, None]  # (25600,1)
    training_points = np.hstack(
        (u_t.flatten()[:, None], u_x.flatten()[:, None])
    )  # (25600, 2)

    IC_X, IC_Y = list(), list()  # Initial and boundary points
    CC_X, CC_Y = list(), list()  # Collocation points

    for idx, sample in enumerate(training_points):

        t, x = sample
        if t in [0, 1] or x in [-1, 1]:
            IC_X.append(sample)
            IC_Y.append(u_data_transformed[idx])
        else:
            CC_X.append(sample)
            CC_Y.append(u_data_transformed[idx])

    IC_X = np.array(IC_X)
    IC_Y = np.array(IC_Y)

    CC_X = np.array(CC_X)
    CC_Y = np.array(CC_Y)

    n_u_idx = list(np.random.choice(len(IC_X), n_u))
    n_f_idx = list(np.random.choice(len(CC_X), n_f))

    u_x = torch.tensor(IC_X[n_u_idx, 1:2], requires_grad=True).float()
    u_t = torch.tensor(IC_X[n_u_idx, 0:1], requires_grad=True).float()
    u_u = torch.tensor(IC_Y[n_u_idx, :], requires_grad=True).float()

    f_x = torch.tensor(CC_X[n_f_idx, 1:2], requires_grad=True).float()
    f_t = torch.tensor(CC_X[n_f_idx, 0:1], requires_grad=True).float()
    f_u = torch.tensor(CC_Y[n_f_idx, :], requires_grad=True).float()

    train_x = torch.cat((u_x, f_x), dim=0)
    train_t = torch.cat((u_t, f_t), dim=0)
    train_u = torch.cat((u_u, f_u), dim=0)

    return [train_x, train_t, train_u]


def trainer(
    train_x, train_t, train_u, epochs, num_neurons, num_layers, activation_fn
) -> list:

    model = PhysicsINN(
        num_layers=num_layers, num_neurons=num_neurons, activation_fn=activation_fn
    )

    optimizer = torch.optim.LBFGS(model.parameters())

    t_bar = trange(epochs)

    losses = list()

    for epoch in t_bar:

        def closure():
            optimizer.zero_grad()

            output = model(torch.cat((train_t, train_x), dim=1))

            u_grad_x = torch.autograd.grad(
                output,
                train_x,
                retain_graph=True,
                create_graph=True,
                grad_outputs=torch.ones_like(output),
                allow_unused=True,
            )[0]
            u_grad_xx = torch.autograd.grad(
                u_grad_x,
                train_x,
                retain_graph=True,
                create_graph=True,
                grad_outputs=torch.ones_like(output),
                allow_unused=True,
            )[0]

            u_grad_t = torch.autograd.grad(
                output,
                train_t,
                retain_graph=True,
                create_graph=True,
                grad_outputs=torch.ones_like(output),
                allow_unused=True,
            )[0]

            f = u_grad_t + output * u_grad_x - (0.01 / np.pi) * u_grad_xx

            mse_f = torch.mean(torch.square(f))
            mse_u = torch.mean(torch.square(output - train_u))

            loss = mse_f + mse_u

            loss.backward()

            t_bar.set_description("loss: %.20f" % loss.item())
            losses.append(loss.item())
            t_bar.refresh()  # to show immediately the update

            return loss

        optimizer.step(closure)

        if len(losses) > 1 and losses[-1] > 1:
            return []

    return losses


def run_from_args(
    data_path: str,
    n_u: int,
    n_f: int,
    num_neurons: int,
    num_layers: int,
    activation_fn: str,
    epochs: int,
) -> None:
    """
    Loads all the parameters from the CLI arguments.
    Prepares the data, runs the trainer loop and logs the experiment results.
    """
    train_x, train_t, train_u = prepare_data(data_path, n_u, n_f)
    losses = trainer(
        train_x, train_t, train_u, epochs, num_neurons, num_layers, activation_fn
    )
    print(losses[-1])


def run_from_config_file(config_path: str) -> None:

    """
    Loads all the parameters from the config file.
    1. Run all combinations of n_u and n_f for num_layers=9 and num_neurons=20
    2.Run all combinations of num_layers and num_neurons
    3. Try Different optimizer
    """

    args = read_config(config_path)
    print(
        "running from config: ",
    )
    exp_points = list(product(args["n_u"], args["n_f"]))
    experiment_1 = {"n_u": [], "n_f": [], "mse_loss": []}
    # for i, (n_u_local, n_f_local) in enumerate(exp_points):
    i = 0
    while i < len(exp_points):

        (n_u_local, n_f_local) = exp_points[i]

        train_x, train_t, train_u = prepare_data(
            args["data_path"], n_u_local, n_f_local
        )
        print(f"Batch {i}: n_u={n_u_local}\tn_f={n_f_local}")
        try:
            loss = trainer(
                train_x,
                train_t,
                train_u,
                100,
                20,
                9,
                "Tanh",
            )[-1]
        except:
            continue

        experiment_1["n_u"].append(n_u_local)
        experiment_1["n_f"].append(n_f_local)
        experiment_1["mse_loss"].append(loss)
        i += 1

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y__%H:%M:%S")

    pd.DataFrame(experiment_1).to_csv(f"./logs/{dt_string}_experiment_1.csv")


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    if args.config_path != None:
        run_from_config_file(config_path=args.config_path)
    else:
        # parse all the arguments and pass it to run_from_args
        data_path = args.data_path
        n_u = args.n_u
        n_f = args.n_f
        num_neurons = args.num_neurons
        num_layers = args.num_layers
        activation_fn = args.activation_fn
        num_epochs = args.num_epochs

        run_from_args(
            data_path, n_u, n_f, num_neurons, num_layers, activation_fn, num_epochs
        )
