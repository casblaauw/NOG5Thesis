import torch
import numpy as np

from math import sqrt
from torch import Tensor

def fpr(pred: Tensor, labels: Tensor) -> float:
    """ Returns false positive rate
    Args:
        pred: tensor with binary values
        labels: tensor with binary values
    """
    if any(x not in (0,1) for x in pred):
        raise ValueError('Predictions must be rounded to 0 or 1 to calculate FPR.')
    if any(x not in (0,1) for x in labels):
        raise ValueError('Labels must be rounded to 0 or 1 to calculate FPR.')

    fp = int(torch.sum(torch.logical_and(torch.eq(pred, 1), torch.eq(labels, 0))))
    tn = int(torch.sum(torch.logical_and(torch.eq(pred, 0), torch.eq(labels, 0))))

    # All true positives means no FPR
    if (fp+tn) is 0:
        fpr = float(0)
    else:
        fpr = float(fp / (fp + tn))
    return fpr


def fnr(pred: Tensor, labels: Tensor) -> float:
    """ Returns false negative rate
    Args:
        pred: tensor with binary values
        labels: tensor with binary values
    """
    if any(x not in (0,1) for x in pred):
        raise ValueError('Predictions must be rounded to 0 or 1 to calculate FNR.')
    if any(x not in (0,1) for x in labels):
        raise ValueError('Labels must be rounded to 0 or 1 to calculate FNR.')
    
    fn = int(torch.sum(torch.logical_and(torch.eq(pred, 0), torch.eq(labels, 1))))
    tp = int(torch.sum(torch.logical_and(torch.eq(pred, 1), torch.eq(labels, 1))))

    # All true negatives means no FNR
    if (fn+tp) is 0:
        fnr = float(0)
    else:
        fnr = float(fn / (fn + tp))
    return fnr


def mcc(pred: Tensor, labels: Tensor) -> float:
    """ Returns Matthews correlation coefficient
    Args:
        pred: tensor with binary values
        labels: tensor with binary values
    """
    fp = int(torch.sum(torch.logical_and(torch.eq(pred, 1), torch.eq(labels, 0))))
    tp = int(torch.sum(torch.logical_and(torch.eq(pred, 1), torch.eq(labels, 1))))
    fn = int(torch.sum(torch.logical_and(torch.eq(pred, 0), torch.eq(labels, 1))))
    tn = int(torch.sum(torch.logical_and(torch.eq(pred, 0), torch.eq(labels, 0))))
    

    denom = sqrt(((tp + fp) * (fn + tn) * (tp + fn) * (fp + tn)))
    if denom == 0:
        denom = 1

    mcc = (tp * tn - fp * fn) / denom
    return mcc


def pcc(pred: Tensor, labels: Tensor) -> float:
    """ Returns pearson correlation coefficient
    Args:
        pred: tensor with predicted values
        labels: tensor with correct values
    """
    x = pred - torch.mean(pred)
    y = labels - torch.mean(labels)

    return (torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))).item()

def accuracy(pred: Tensor, labels: Tensor) -> float:
    """ Returns accuracy
    Args:
        pred: tensor with predicted values
        labels: tensor with correct values
    """

    return ((pred == labels).sum() / labels.numel()).item()


def rmse(pred: Tensor, labels: Tensor) -> float:
    """ Returns root mean squared error
    Args:
        pred: tensor with predicted values
        labels: tensor with correct values
    """

    return torch.sqrt(torch.mean((labels - pred) ** 2)).item()


def mae_angle(pred: Tensor, labels: Tensor) -> float:
    """ Returns mean absolute error for degree angles
    Args:
        pred: tensor with predicted values
        labels: tensor with correct values
    """
    err = torch.abs(labels - pred)
    return torch.mean(torch.fmin(err, 360 - err)).item()