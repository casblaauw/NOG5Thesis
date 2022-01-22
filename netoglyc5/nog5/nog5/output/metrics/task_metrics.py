from typing import Dict

import torch
from torch import Tensor
from nog5.output.metrics.metric_functions import fpr, mcc, pcc, mae_angle, accuracy, rmse, fnr

from nog5.output.misc import get_mask, arctan_dihedral


def ss8_accuracy(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns SS8 metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels).squeeze(2)
    outputs = torch.argmax(outputs['ss8'], dim=2)[mask == 1]
    labels = labels['ss8'].squeeze(2)[mask == 1]

    #print('ss8_accuracy')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    #print(mask.shape, mask.sum())
    metric = accuracy(outputs, labels)
    #print(metric)
    return metric


def ss8_pcc(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns SS8 metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels).squeeze(2)
    outputs = torch.softmax(outputs['ss8'], dim=2)[mask == 1]
    labels = labels['ss8'][mask == 1]

    #print('ss8_pcc')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    #print(mask.shape, mask.sum())
    metric = pcc(outputs, labels)
    #print(metric)
    return metric


def ss3_accuracy(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns SS3 metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels).squeeze(2)

    ss3_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=mask.device)

    outputs = ss3_mask[torch.argmax(outputs['ss8'], dim=2)][mask == 1]
    labels = ss3_mask[labels['ss8']].squeeze(2)[mask == 1]

    #print('ss3_accuracy')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    metric = accuracy(outputs, labels)
    #print(metric)
    return metric


def ss3_pcc(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns SS3 metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels).squeeze(2)

    ss3_groups = ((0, 1, 2), (3, 4), (5, 6, 7))

    # softmax output, then sum together SS8 class probs into SS3 classes and stack them back together
    outputs = torch.softmax(outputs['ss8'], dim=2)
    outputs = torch.log(torch.stack([outputs[:, :, group].sum(dim=2) for group in ss3_groups], dim=2))[mask == 1]

    # sum together SS8 label class probs into SS3 classes and stack them back together
    labels = labels['ss8']
    labels = torch.log(torch.stack([labels[:, :, group].sum(dim=2) for group in ss3_groups], dim=2))[mask == 1]

    #print('ss3_pcc')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    metric = pcc(outputs, labels)
    #print(metric)
    return metric


def dis_mcc(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns mathews correlation coefficient disorder metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels).squeeze(2)
    outputs = torch.round(torch.sigmoid(outputs['dis'])).squeeze(2).float()[mask == 1]
    labels = labels['dis'].squeeze(2)[mask == 1]

    #print('dis_mcc')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    metric = mcc(outputs, labels)
    #print(metric)
    return metric


def dis_fpr(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns false positive rate disorder metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels).squeeze(2)
    outputs = torch.round(torch.sigmoid(outputs['dis'])).squeeze(2).float()[mask == 1]
    labels = labels['dis'].squeeze(2)[mask == 1]

    #print('dis_fpr')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    metric = fpr(outputs, labels)
    #print(metric)
    return metric


def dis_pcc(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns mathews correlation coefficient disorder metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels).squeeze(2)
    outputs = torch.sigmoid(outputs['dis']).squeeze(2)[mask == 1]
    labels = labels['dis'].squeeze(2)[mask == 1]

    #print('dis_pcc')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    metric = pcc(outputs, labels)
    #print(metric)
    return metric


def rsa_pcc(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns relative surface accesibility metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, ['disorder_mask', 'unknown_mask']).squeeze(2)
    outputs = torch.sigmoid(outputs['rsa']).squeeze(2)[mask == 1]
    labels = labels['rsa'].squeeze(2)[mask == 1]

    #print('rsa_pcc')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    metric = pcc(outputs, labels)
    #print(metric)
    return metric


def phi_mae(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns phi angle metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, ['disorder_mask', 'unknown_mask'])
    mask = mask * (labels['phi'] != 360)
    mask = mask.squeeze(2)

    outputs = outputs['phi']
    outputs = arctan_dihedral(outputs[:, :, 0], outputs[:, :, 1])[mask == 1]
    labels = labels['phi'].squeeze(2)[mask == 1]

    #print('phi_mae')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    metric = mae_angle(outputs, labels)
    #print(metric)
    return metric


def psi_mae(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns psi angle metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, ['disorder_mask', 'unknown_mask'])
    mask = mask * (labels['psi'] != 360)
    mask = mask.squeeze(2)

    outputs = outputs['psi']
    outputs = arctan_dihedral(outputs[:, :, 0], outputs[:, :, 1])[mask == 1]
    labels = labels['psi'].squeeze(2)[mask == 1]

    #print('psi_mae')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    metric = mae_angle(outputs, labels)
    #print(metric)
    return metric


def gly_pcc(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns glycosylation metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, ['glycosylation_mask', 'unknown_mask']).squeeze(2)
    outputs = torch.sigmoid(outputs['gly']).squeeze(2)[mask == 1]
    labels = labels['gly'].squeeze(2)[mask == 1]

    #print('gly_pcc')
    #print(outputs.shape, outputs)
    #print(labels.shape, labels)
    #print(mask.shape, mask.sum())
    metric = pcc(outputs, labels)
    #print(metric)
    return metric

def gly_definite_mcc(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], threshold: float = 0.5) -> float:
    """ Returns glycosylation metric solely for definite sites (0 or 1)
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, ['definite_glycosylation_mask', 'unknown_mask']).squeeze(2)
    outputs = (torch.sigmoid(outputs['gly']) >= threshold).squeeze(2)[mask == 1]
    labels = labels['gly'].squeeze(2)[mask == 1]

    #print('gly_mcc')
    #print(outputs.shape, outputs)
    #print(labels.shape, labels)
    #print(mask.shape, mask.sum())
    metric = mcc(outputs, labels)
    #print(metric)
    return metric


def gly_ambiguous_mcc(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], threshold: float = 0.5) -> float:
    """ Returns glycosylation metric solely for ambiguous sites (>0 and <1)
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, ['ambiguous_glycosylation_mask', 'unknown_mask']).squeeze(2)
    outputs = (torch.sigmoid(outputs['gly']) >= threshold).squeeze(2)[mask == 1]
    labels = (labels['gly'] >= threshold).squeeze(2)[mask == 1]

    #print('gly_unambiguous_mcc')
    #print(outputs.shape, outputs)
    #print(labels.shape, labels)
    #print(mask.shape, mask.sum())
    metric = mcc(outputs, labels)
    #print(metric)
    return metric


def gly_fpr(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], threshold: float = 0.5) -> float:
    """ Returns glycosylation metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, ['glycosylation_mask', 'unknown_mask']).squeeze(2)
    outputs = (torch.sigmoid(outputs['gly']) >= threshold).squeeze(2)[mask == 1]
    labels = (labels['gly'] >= threshold).squeeze(2)[mask == 1]

    #print('gly_fpr')
    #print(outputs.shape, outputs)
    #print(labels.shape, labels)
    #print(mask.shape, mask.sum())
    metric = fpr(outputs, labels)
    #print(metric)
    return metric


def gly_fnr(outputs: Dict[str, Tensor], labels: Dict[str, Tensor], threshold: float = 0.5) -> float:
    """ Returns glycosylation metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, ['glycosylation_mask', 'unknown_mask']).squeeze(2)
    outputs = (torch.sigmoid(outputs['gly']) >= threshold).squeeze(2)[mask == 1]
    labels = (labels['gly'] >= threshold).squeeze(2)[mask == 1]

    #print('gly_fnr')
    #print(outputs.shape, outputs)
    #print(labels.shape, labels)
    #print(mask.shape, mask.sum())
    metric = fnr(outputs, labels)
    #print(metric)
    return metric


def com_accuracy(outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> float:
    """ Returns glycosylation site composition multi-label metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, ['composition_mask', 'unknown_mask']).squeeze(2)
    outputs = torch.round(torch.sigmoid(outputs['com'])).float()[mask == 1]
    labels = labels['com'][mask == 1]

    #print('com_accuracy')
    #print(outputs.shape)
    #print(labels.shape)
    #print(mask.shape, mask.sum())
    metric = accuracy(outputs, labels)
    #print(metric)
    return metric
