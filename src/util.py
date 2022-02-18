import argparse

import yaml

import torch
import wandb


def log_conf_matrix(model, dataloader):
    """Logs a wandb conf matrix, from a model and dataloader"""
    preds = []
    y = []
    
    for x in dataloader:
        x_hat = model(x)
        probs_ = x_hat.softmax(dim=1)
        y.append(x)
        preds.append(probs_.argmax(dim=1))

    preds = torch.cat(preds, dim=0)
    y = torch.cat(y, dim=0)

    preds = preds.view((-1))
    y = y.view((-1))
    not_nans = ~y.isnan()
    wandb.log(
        {
            "conf": wandb.plot.confusion_matrix(
                preds=preds[not_nans].cpu().numpy(), y_true=y[not_nans].cpu().numpy()
            )
        }
    )


def create_argparser(default_yml_path: str) -> argparse.ArgumentParser:
    """Creates an argparser from a yaml file. Where each component is a flag to set and
    each subcomponent is an argument to be passed to the argparse

    .. code::

        epochs:
            help: Number of epochs to train over
            default: 100

    to make the code more flexible, help also be called desc default also called value.
    to indicate that this value both can be use to set a value and used to set a
    default. This also makes them compatible with the wieght and biases format:
    https://docs.wandb.ai/guides/track/config#file-based-configs

    Args:
        default_yml_path (str): Path to yaml file containing default arguments to parse.

    Returns:
        argparse.ArgumentParser: The ArgumentParser for parsing all the arguments in the
            config.
    """

    with open(default_yml_path, "r") as f:
        default_dict = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    for k, args in default_dict.items():
        if isinstance(args, dict):
            if ("value" in args) and ("default" in args):
                raise ValueError(
                    f"{k} contains both a 'value' and a 'default'," + " Remove one."
                )
            if ("desc" in args) and ("help" in args):
                raise ValueError(
                    f"{k} contains both a 'desc' and a 'help'." + " Remove one."
                )
            if "value" in args:
                args["default"] = args["value"].pop()
            if "desc" in args:
                args["default"] = args["value"].pop()
            parser.add_argument(f"--{k}", **args)

        # if not dict assume args is the default values
        else:
            parser.add_argument(f"--{k}", default=args)

    return parser


def config_yaml_to_dict(path: str) -> dict:
    """Convert a config yaml into a dict of hyperparameters.

    for instance:

    .. code::

        epochs:
            desc: Number of epochs to train over
            value: 100

    becomes:
    .. code::

        {epochs: 100}

    Args:
        path (str): Path to yaml file containing arguments to parse.
    """
    with open(path, "r") as f:
        default_dict = yaml.safe_load(f)
    hyperparameter_dict = {}
    for k, args in default_dict.items():
        if isinstance(args, dict):
            if ("value" in args) and ("default" in args):
                raise ValueError(
                    f"{k} contains both a 'value' and a 'default'," + " Remove one."
                )
            if "value" in args:
                args["default"] = args["value"].pop()
            hyperparameter_dict[k] = args["default"]
        else:
            hyperparameter_dict[k] = args
    return hyperparameter_dict
