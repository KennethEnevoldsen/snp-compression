from typing import Union
import argparse
from argparse import Namespace

import yaml


def create_config_namespace(
    default_yml_path: str, config: Union[dict, str] = {}
) -> Namespace:
    """Creates a config NameSpace from a default yaml dict which can
    for instance be used when replacing the wandb config. E.g. used
    when loading in the model without wanting to use wandb.

    Args:
        default_yml_path (str): The default yml path
        config (Union[dict, str], optional): An additional config which overwrites
        arguments in the default yaml path.

    Returns:
        Namespace: A namespace config
    """
    config_ = load_yaml_config(default_yml_path)
    if config:
        if isinstance(config, str):
            config = load_yaml_config(config)
        else:
            config = __clean_yaml_config(config)

        for k, item in config.items():
            config_[k] = item
    # collapse arguments
    config_ = {k: args["default"] for k, args in config_.items()}
    return Namespace(**config_)


def load_yaml_config(yaml_config_path: str) -> dict:
    with open(yaml_config_path, "r") as f:
        yaml_config_dict = yaml.safe_load(f)
    return __clean_yaml_config(yaml_config_dict)


def __clean_yaml_config(yaml_config_dict: dict) -> dict:
    for k, args in yaml_config_dict.items():
        if isinstance(args, dict):
            # normalize desc (to help) and value (to default)
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
            if "type" in args:
                args["type"] = eval(args["type"])
        else:
            # assume the value is the default
            yaml_config_dict[k] = {"default": args}

    return yaml_config_dict


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
    default_dict = load_yaml_config(default_yml_path)

    parser = argparse.ArgumentParser()
    for k, args in default_dict.items():
        parser.add_argument(f"--{k}", **args)

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
    default_dict = load_yaml_config(path)
    clean_dict = {k: args["default"] for k, args in default_dict.items()}
    return clean_dict
