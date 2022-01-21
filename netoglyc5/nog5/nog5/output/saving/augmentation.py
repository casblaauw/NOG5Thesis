from typing import Dict

import torch
from torch import Tensor

from nog5.output.misc import arctan_dihedral


def ss8_label(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats SS8 labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    return torch.argmax(outputs['ss8'], dim=2).unsqueeze(2)


def dis_label(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats mathews correlation coefficient disorder labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    return torch.round(torch.sigmoid(outputs['dis']))


def rsa_label(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats relative surface accesibility labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    return torch.sigmoid(outputs['rsa'])


def phi_label(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats phi angle labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    outputs = outputs['phi']
    outputs = arctan_dihedral(outputs[:, :, 0], outputs[:, :, 1]).unsqueeze(2)

    return outputs


def psi_label(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats psi angle labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    outputs = outputs['psi']
    outputs = arctan_dihedral(outputs[:, :, 0], outputs[:, :, 1]).unsqueeze(2)

    return outputs


def gly_label(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats glycosylation labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    return torch.round(torch.sigmoid(outputs['gly']))


def com_label(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats composition labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    return torch.round(torch.sigmoid(outputs['com']))


def multi_task_save_labels(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """ Formats labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    if 'ss8' in outputs:
        outputs['ss8'] = ss8_label(outputs)
    if 'dis' in outputs:
        outputs['dis'] = dis_label(outputs)
    if 'rsa' in outputs:
        outputs['rsa'] = rsa_label(outputs)
    if 'phi' in outputs:
        outputs['phi'] = phi_label(outputs)
    if 'psi' in outputs:
        outputs['psi'] = psi_label(outputs)
    if 'gly' in outputs:
        outputs['gly'] = gly_label(outputs)
    if 'com' in outputs:
        outputs['com'] = com_label(outputs)

    return outputs


def ss8_output(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats SS8 labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    return torch.softmax(outputs['ss8'], dim=2)


def dis_output(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats mathews correlation coefficient disorder labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    return torch.sigmoid(outputs['dis'])


def rsa_output(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats relative surface accesibility labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    return torch.sigmoid(outputs['rsa'])


def phi_output(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats phi angle labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    outputs = outputs['phi']
    outputs = arctan_dihedral(outputs[:, :, 0], outputs[:, :, 1]).unsqueeze(2)

    return outputs


def psi_output(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats psi angle labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    outputs = outputs['psi']
    outputs = arctan_dihedral(outputs[:, :, 0], outputs[:, :, 1]).unsqueeze(2)

    return outputs


def gly_output(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats glycosylation labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    return torch.sigmoid(outputs['gly'])


def com_output(outputs: Dict[str, Tensor]) -> Tensor:
    """ Formats composition labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    return torch.sigmoid(outputs['com'])


def multi_task_save_output(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """ Formats labels from outputs for saving
    Args:
        outputs: tensor with predicted values
    """
    if 'ss8' in outputs:
        outputs['ss8'] = ss8_output(outputs)
    if 'dis' in outputs:
        outputs['dis'] = dis_output(outputs)
    if 'rsa' in outputs:
        outputs['rsa'] = rsa_output(outputs)
    if 'phi' in outputs:
        outputs['phi'] = phi_output(outputs)
    if 'psi' in outputs:
        outputs['psi'] = psi_output(outputs)
    if 'gly' in outputs:
        outputs['gly'] = gly_output(outputs)
    if 'com' in outputs:
        outputs['com'] = com_output(outputs)

    return outputs
