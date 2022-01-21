from typing import Dict, List

import torch
from torch import Tensor

from nog5.output.loss.loss_functions import mse, ce, bce, nll, bce_logits
from nog5.output.misc import get_mask, dihedral_to_radians


def ss8_ce(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], class_weights: List[str] = None) -> Tensor:
    """ Returns SS8 loss
    Args:
        outputs: tensor with SS8 predictions
        labels: tensor with labels
    """
    mask = get_mask(labels).squeeze(2)

    outputs = outputs['ss8'].permute(0, 2, 1)
    labels = labels['ss8'].squeeze(2).long()

    #print('ss8_ce')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = ce(outputs, labels, mask, class_weights=class_weights)
    #print(loss)
    return loss


def ss8_bce(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], class_weights: List[str] = None) -> Tensor:
    """ Returns SS8 loss
    Args:
        outputs: tensor with SS8 predictions
        labels: tensor with labels
    """
    mask = get_mask(labels)

    outputs = outputs['ss8']
    labels = labels['ss8']

    #print('ss8_bce')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = bce_logits(outputs, labels, mask, class_weights=class_weights)
    #print(loss)
    return loss



def ss3_ce(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], class_weights: List[str] = None) -> Tensor:
    """ Returns SS3 loss from SS8 outputs and labels
    Args:
        outputs: tensor with SS3 predictions
        labels: tensor with labels
    Notes:
        This loss is more numerically unstable than SS8 due to doing softmax and log separately,
        but this is necessary since we need to sum over softmax output, and log(x+y) != log(x) + log(y)
    """
    mask = get_mask(labels).squeeze(2)

    ss3_groups = ((0, 1, 2), (3, 4), (5, 6, 7))
    ss3_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=mask.device)

    # softmax output, then sum together SS8 class probs into SS3 classes and stack them back together
    outputs = torch.softmax(outputs['ss8'].permute(0, 2, 1), dim=1)
    outputs = torch.log(torch.stack([outputs[:, group, :].sum(dim=1) for group in ss3_groups], dim=1))

    # convert SS8 label to SS3
    labels = labels['ss8'].squeeze(2)
    labels = ss3_mask[labels]

    #print('ss3_ce')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = nll(outputs, labels, mask, class_weights=class_weights)
    #print(loss)
    return loss


def ss3_bce(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], class_weights: List[str] = None) -> Tensor:
    """ Returns SS3 loss from SS8 outputs and labels
    Args:
        outputs: tensor with SS3 predictions
        labels: tensor with labels
    Notes:
        This loss is more numerically unstable than SS8 due to doing softmax and log separately,
        but this is necessary since we need to sum over softmax output, and log(x+y) != log(x) + log(y)
    """
    mask = get_mask(labels)

    ss3_groups = ((0, 1, 2), (3, 4), (5, 6, 7))

    # softmax output, then sum together SS8 class probs into SS3 classes and stack them back together
    outputs = torch.softmax(outputs['ss8'], dim=2)
    outputs = torch.log(torch.stack([outputs[:, :, group].sum(dim=2) for group in ss3_groups], dim=1))

    # sum together SS8 label class probs into SS3 classes and stack them back together
    labels = labels['ss8']
    labels = torch.log(torch.stack([labels[:, :, group].sum(dim=2) for group in ss3_groups], dim=1))

    print('ss3_bce')
    print(outputs.shape)
    print(labels.shape)
    print(mask.shape, mask.sum())
    loss = nll(outputs, labels, mask, class_weights=class_weights)
    print(loss)
    return loss


def dis_bce(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], positive_weight: float = None) -> Tensor:
    """ Returns disorder loss
    Args:
        outputs: tensor with disorder predictions
        labels: tensor with labels
    """
    mask = get_mask(labels).squeeze(2)

    outputs = outputs['dis'].squeeze(2)
    labels = labels['dis'].squeeze(2).float()
    
    #print('dis_bce')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = bce_logits(outputs, labels, mask, positive_weight=positive_weight)
    #print(loss)
    return loss


def dis_mse(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Tensor:
    """ Returns disorder loss
    Args:
        outputs: tensor with disorder predictions
        labels: tensor with labels
    """
    mask = get_mask(labels).squeeze(2)

    outputs = torch.sigmoid(outputs['dis']).squeeze(2)
    labels = labels['dis'].squeeze(2).float()

    #print('dis_mse')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = mse(outputs, labels, mask)
    #print(loss)
    return loss


def rsa_bce(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], positive_weight: float = None) -> Tensor:
    """ Returns relative surface accesibility loss
    Args:
        outputs: tensor with rsa predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, ['disorder_mask', 'unknown_mask']).squeeze(2)

    outputs = outputs['rsa'].squeeze(2)
    labels = labels['rsa'].squeeze(2).float()

    #print('rsa_bce')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = bce_logits(outputs, labels, mask, positive_weight=positive_weight)
    #print(loss)
    return loss


def rsa_mse(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Tensor:
    """ Returns relative surface accesibility loss
    Args:
        outputs: tensor with rsa predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, ['disorder_mask', 'unknown_mask']).squeeze(2)

    outputs = torch.sigmoid(outputs['rsa']).squeeze(2)
    labels = labels['rsa'].squeeze(2).float()

    #print('rsa_mse')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = mse(outputs, labels, mask)
    #print(loss)
    return loss


def phi_mse(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Tensor:
    """ Returns phi loss
    Args:
        outputs: tensor with phi predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, ['disorder_mask', 'unknown_mask'])
    mask = mask * (labels['phi'] != 360)

    outputs = outputs['phi']
    labels = labels['phi']
    labels = torch.cat((torch.sin(dihedral_to_radians(labels)), torch.cos(dihedral_to_radians(labels))), dim=2)

    #print('phi_mse')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = mse(outputs, labels, mask)
    #print(loss)
    return loss


def psi_mse(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Tensor:
    """ Returns psi loss
    Args:
        outputs: tensor with phi predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, ['disorder_mask', 'unknown_mask'])
    mask = mask * (labels['psi'] != 360)

    outputs = outputs['psi']
    labels = labels['psi']
    labels = torch.cat((torch.sin(dihedral_to_radians(labels)), torch.cos(dihedral_to_radians(labels))), dim=2)

    #print('psi_mse')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = mse(outputs, labels, mask)
    #print(loss)
    return loss


def gly_mse(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], positive_weight: float = None) -> Tensor:
    """ Returns glycosylation probability loss
    Args:
        outputs: tensor with glycosylation predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, ['glycosylation_mask', 'unknown_mask']).squeeze(2)

    outputs = outputs['gly'].squeeze(2)
    labels = labels['gly'].squeeze(2).float()

    #print('gly_mse')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = mse(outputs, labels, mask, positive_weight)
    #print(loss)
    return loss


def gly_unambiguous_mse(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], positive_weight: float = None) -> Tensor:
    """ Returns glycosylation probability loss
    Args:
        outputs: tensor with glycosylation predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, ['unambiguous_glycosylation_mask', 'unknown_mask']).squeeze(2)

    outputs = outputs['gly'].squeeze(2)
    labels = labels['gly'].squeeze(2).float()

    #print('gly_mse')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = mse(outputs, labels, mask, positive_weight)
    #print(loss)
    return loss


def gly_all_mse(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], positive_weight: float = None) -> Tensor:
    """ Returns glycosylation probability loss
    Args:
        outputs: tensor with glycosylation predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, ['seen']).squeeze(2)

    outputs = outputs['gly'].squeeze(2)
    labels = torch.clamp(labels['gly'], min=0, max=1).squeeze(2).float()

    #print('gly_mse')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = mse(outputs, labels, mask, positive_weight)
    #print(loss)
    return loss


def gly_hard_mse(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], positive_weight: float = None) -> Tensor:
    """ Returns glycosylation probability loss
    Args:
        outputs: tensor with glycosylation predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, ['glycosylation_mask', 'unknown_mask']).squeeze(2)

    outputs = torch.round(outputs['gly']).squeeze(2)
    labels = labels['gly'].squeeze(2).float()

    #print('gly_hard_mse')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = mse(outputs, labels, mask, positive_weight)
    #print(loss)
    return loss


def com_bce(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], class_weights: List[str] = None) -> Tensor:
    """ Returns glycosylation site composition multi-label probability loss
    Args:
        outputs: tensor with glycosylation predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, ['composition_mask', 'unknown_mask'])

    outputs = outputs['com']
    labels = labels['com'].float()

    #print('com_bce')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    loss = bce_logits(outputs, labels, mask, class_weights=class_weights)
    #print(loss)
    return loss
