import torch
from torch import Tensor


def fpr(pred: Tensor, labels: Tensor) -> float:
    """ Returns false positive rate
    Args:
        pred: tensor with binary values
        labels: tensor with binary values
    """
    fp = sum((pred == 1) & (labels == 0))
    tn = sum((pred == 0) & (labels == 0))

    return (fp / (fp + tn)).item()


def fnr(pred: Tensor, labels: Tensor) -> float:
    """ Returns false negative rate
    Args:
        pred: tensor with binary values
        labels: tensor with binary values
    """
    fn = sum((pred == 0) & (labels == 1))
    tp = sum((pred == 1) & (labels == 1))

    return (fn / (fn + tp)).item()


def mcc(pred: Tensor, labels: Tensor) -> float:
    """ Returns mathews correlation coefficient
    Args:
        pred: tensor with binary values
        labels: tensor with binary values
    """
    fp = sum((pred == 1) & (labels == 0))
    tp = sum((pred == 1) & (labels == 1))
    fn = sum((pred == 0) & (labels == 1))
    tn = sum((pred == 0) & (labels == 0))

    mcc = (tp * tn - fp * fn) / torch.sqrt(((tp + fp) * (fn + tn) * (tp + fn) * (fp + tn)).float())

    if torch.isnan(mcc):
        return 0

    return mcc.item()


def pcc(pred: Tensor, labels: Tensor) -> float:
    """ Returns pearson correlation coefficient
    Args:
        pred: tensor with predicted values
        labels: tensor with correct values
    """
    x = pred - torch.mean(pred)
    y = labels - torch.mean(labels)

    return (torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))).item()


def mae_angle(pred: Tensor, labels: Tensor) -> float:
    """ Returns mean absolute error for degree angles
    Args:
        pred: tensor with predicted values
        labels: tensor with correct values
    """
    err = torch.abs(labels - pred)
    return torch.mean(torch.fmin(err, 360 - err)).item()


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
