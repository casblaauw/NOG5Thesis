import os
import random
from pathlib import Path
from types import ModuleType
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import yaml
from torch import nn as nn, optim as module_optimizer

from nog5.base import ModelBase
from nog5.base.base_parameterizedloss import ParameterizedLossBase
from nog5.utils.logger import setup_logger

log = setup_logger(__name__)


def setup_device(model: ModelBase, target_devices: List[int]) -> Tuple[ModelBase, torch.device]:
    """ Setup GPU device if available, move model into configured device
    Args:
        model: Module to move to GPU
        target_devices: list of target devices
    Returns:
        the model that now uses the gpu and the device
    """
    available_devices = list(range(torch.cuda.device_count()))

    if not available_devices:
        log.warning(
            "There's no GPU available on this machine. Training/prediction will be performed on CPU.")
        device = torch.device('cpu')
        model = model.to(device)
        return model, device

    if not target_devices:
        log.info("No GPU selected. Training/prediction will be performed on CPU.")
        device = torch.device('cpu')
        model = model.to(device)
        return model, device

    max_target_gpu = max(target_devices)
    max_available_gpu = max(available_devices)

    if max_target_gpu > max_available_gpu:
        msg = (f"Configuration requests GPU #{max_target_gpu} but only {max_available_gpu} "
                "available. Check the configuration and try again.")
        log.critical(msg)
        raise Exception(msg)

    log.info(f'Using devices {target_devices} of available GPUs {available_devices}')
    device = torch.device(f'cuda:{target_devices[0]}')

    if len(target_devices) > 1:
        model = nn.DataParallel(model, device_ids=target_devices).to(device)
    else:
        model = model.to(device)
    return model, device


def resume_checkpoint(resume_path: str, model: ModelBase, optimizer: module_optimizer, loss, config: Dict, device: torch.device):
    """ Resume from saved checkpoint. """
    if not resume_path:
        return model, optimizer, loss, 0

    log.info(f'Loading checkpoint: {resume_path}')
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    if isinstance(loss, ParameterizedLossBase) and 'loss' in checkpoint:
        # load parameterized loss state from checkpoint only when loss type is not changed.
        if checkpoint['config']['multitask_loss'] != config['multitask_loss']:
            log.warning("Warning: Parameterized loss type/args given in config file is different from "
                        "that of checkpoint. Loss parameters not being resumed.")
        else:
            loss.load_state_dict(checkpoint['loss'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint['config']['optimizer']['type'] != config['optimizer']['type']:
        log.warning("Warning: Optimizer type given in config file is different from "
                            "that of checkpoint. Optimizer parameters not being resumed.")
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])

    log.info(f'Checkpoint "{resume_path}" loaded')
    return model, optimizer, loss, checkpoint['epoch']+1


def get_instance(module: ModuleType, name: str, config: Dict, **kwargs: Any) -> Any:
    """ Helper to construct an instance of a class.
    Args
        module: Module containing the class to construct.
        name: Name of class, as would be returned by ``.__class__.__name__``.
        config: Dictionary containing an 'args' item, which will be used as ``kwargs`` to construct the class instance.
        kwargs : Keyword arguments which will be used as ``kwargs`` to construct the class instance.
    Returns:
        any instance of a class
    Errors:
        TypeError if any constructor arguments are specified in both config and kwargs
    """
    ctor_name = config[name]['type']

    if ctor_name is None:
        return None

    log.info(f'Building: {module.__name__}.{ctor_name}')
    return getattr(module, ctor_name)(**kwargs, **config[name].get('args', {}))


def seed_everything(seed: int):
    """ Sets a seed on python, numpy and pytorch
    Args:
        seed: number of the seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_config(filename: str) -> dict:
    """ Load a configuration file as YAML. """
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    return config


def load_model_data_loosely(model: ModelBase, model_data: str, device: torch.device):
    # strict=False is necessary to load models partially (e.g. just evaluator without embedding), but can be bad
    # if the model & checkpoint don't match at all, so we need to check for missing/unexpected keys
    data = torch.load(model_data, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(data['state_dict'], strict=False)
    if len(missing_keys) > 0:
        log.warning(
            f"""Missing key(s) in checkpoint (expected if using partial model): {', '.join(f'"{k}"' for k in missing_keys)}""")
    if len(unexpected_keys) > 0:
        log.error(
            f"""Unexpected key(s) in checkpoint (make sure that config model and checkpoint model match): {', '.join(f'"{k}"' for k in unexpected_keys)}""")


def load_best_model_data(model: ModelBase, checkpoint_dir: Path, device: torch.device):
    # try to load the best model checkpoint from the experiment
    model_best = checkpoint_dir / "model_best.pth"
    if model_best.exists():
        log.info("Loading best model from checkpoints")
        load_model_data_loosely(model, str(model_best), device)
    else:
        log.info("No best model in checkpoints, using input model as is")
