from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F


def mse(outputs: Tensor, labels: Tensor, mask: Tensor, positive_weight: float = None) -> Tensor:
    """ Returns mean squared loss using masking
    Args:
        outputs: tensor with predictions
        labels: tensor with labels
        mask: tensor with masking
    """
    weights = torch.ones(labels.shape, device=outputs.device)
    if positive_weight is not None:
        weights[labels > 0] = positive_weight

    loss = weights * torch.square(outputs - labels) * mask

    return torch.sum(loss) / torch.sum(mask)


def ce(outputs: Tensor, labels: Tensor, mask: Tensor, class_weights: List[str] = None) -> Tensor:
    """ Returns cross entropy loss using masking
    Args:
        outputs: tensor with predictions
        labels: tensor with labels
        mask: tensor with masking
    """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, device=outputs.device)

    loss = F.cross_entropy(outputs, labels, reduction='none', weight=class_weights) * mask

    return torch.sum(loss) / torch.sum(mask)


def bce(outputs: Tensor, labels: Tensor, mask: Tensor, class_weights: List[str] = None) -> Tensor:
    """ Returns binary cross entropy loss using masking
    Args:
        outputs: tensor with predictions
        labels: tensor with labels
        mask: tensor with masking
    """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, device=outputs.device)

    loss = F.binary_cross_entropy(outputs, labels, reduction='none', weight=class_weights) * mask

    return torch.sum(loss) / torch.sum(mask)


def nll(outputs: Tensor, labels: Tensor, mask: Tensor, class_weights: List[str] = None) -> Tensor:
    """ Returns negative log-likelihood loss using masking
        Args:
            outputs: tensor with predictions
            labels: tensor with labels
            mask: tensor with masking
        """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, device=outputs.device)

    loss = F.nll_loss(outputs, labels, reduction='none', weight=class_weights) * mask

    return torch.sum(loss) / torch.sum(mask)


def bce_logits(outputs: Tensor, labels: Tensor, mask: Tensor, class_weights: List[str] = None,
               positive_weight: float = None) -> Tensor:
    """ Returns binary cross entropy loss between logits using masking
    Args:
        outputs: tensor with predictions
        labels: tensor with labels
        mask: tensor with masking
    """
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, device=outputs.device)
    if positive_weight is not None:
        positive_weight = torch.tensor(positive_weight, device=outputs.device)

    loss = F.binary_cross_entropy_with_logits(outputs, labels, weight=class_weights, pos_weight=positive_weight,
                                              reduction='none') * mask

    return torch.sum(loss) / torch.sum(mask)
