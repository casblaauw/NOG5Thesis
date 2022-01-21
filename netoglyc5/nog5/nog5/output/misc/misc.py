from typing import Dict, List

import numpy as np
import torch
from torch import Tensor


def get_mask(labels: Dict[str, Tensor], use_masks: List[str] = None) -> Tensor:
    """ Returns mask from labels
    Args:
        labels: tensor containing labels
    """
    mask = (labels['seq_mask'] > 0)
    if use_masks is not None:
        for mask_name in use_masks:
            if mask_name in labels:
                mask = mask * (labels[mask_name] > 0)
    return mask


def dihedral_to_radians(angle: Tensor) -> Tensor:
    """ Converts angles to radians
    Args:
        angle: tensor containing angle values
    """
    return angle * np.pi / 180


def arctan_dihedral(sin: Tensor, cos: Tensor) -> Tensor:
    """ Converts sin and cos back to diheral angles
    Args:
        sin: tensor with sin values
        cos: tensor with cos values
    """
    result = torch.where(cos >= 0, torch.arctan(sin / cos),
                         torch.arctan(sin / cos) + np.pi)
    result = torch.where((sin <= 0) & (cos <= 0), result - np.pi * 2, result)

    return result * 180 / np.pi
