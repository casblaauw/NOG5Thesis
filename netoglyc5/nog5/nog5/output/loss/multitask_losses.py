import inspect
from typing import Dict, List, Any

import torch
from torch import Tensor
from torch.optim import Optimizer

import nog5.output.loss.task_losses as module_taskloss
from nog5.base.base_parameterizedloss import ParameterizedLossBase


class WeightedLoss:
    """ Returns a weighted multi task loss """
    def __init__(self, loss_names: List[str], loss_weights: List[str], loss_args: List[Dict[str, Any]] = None):
        if len(loss_names) != len(loss_weights):
            raise ValueError("loss_names and loss_weights must be same length")
        if loss_args is not None:
            if len(loss_names) != len(loss_args):
                raise ValueError("loss_names and loss_args must be same length")
            self.loss_args = loss_args
        else:
            self.loss_args = [{}] * len(loss_names)
        self.loss_funcs = [getattr(module_taskloss, loss_name) for loss_name in loss_names]
        self.loss_weights = loss_weights

    def __call__(self, outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Tensor:
        loss_sum = 0
        for loss_func, loss_weight, loss_args in zip(self.loss_funcs, self.loss_weights, self.loss_args):
            loss_sum = loss_sum + (loss_func(outputs, labels, **loss_args).sum() * loss_weight)
        return loss_sum


class AutomaticWeightedLoss(ParameterizedLossBase):
    """Automatically weighted multi-task loss
    Based on https://github.com/Mikoto10032/AutomaticWeightedLoss
    Params:
        num: int，the number of loss
        x: multi-task loss
    Notes:
        # remember to add learnable parameters to optimizer (should a separate optimizer be used?)
        optimizer = optim.Adam([
                        {'params': model.parameters()},
                        {'params': awl.parameters(), 'weight_decay': 0}
                    ])
    """
    def __init__(self, loss_names: List[str], loss_args: List[Dict[str, Any]] = None, **param_group_args):
        super().__init__()
        if loss_args is not None:
            if len(loss_names) != len(loss_args):
                raise ValueError("loss_names and loss_args must be same length")
            self.loss_args = loss_args
        else:
            self.loss_args = [{}] * len(loss_names)
        self.loss_funcs = [getattr(module_taskloss, loss_name) for loss_name in loss_names]
        params = torch.ones(len(loss_names), requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.param_group_args = param_group_args

    def forward(self, outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Tensor:
        loss_sum = 0
        for idx, (loss_func, loss_args) in enumerate(zip(self.loss_funcs, self.loss_args)):
            loss_sum = loss_sum + (0.5 / (self.params[idx] ** 2) * loss_func(outputs, labels, **loss_args).sum() + torch.log(1 + self.params[idx] ** 2))
        return loss_sum

    def get_param_group(self, optimizer: Optimizer) -> Dict[str, Any]:
        param_group = {}
        # Set weight_decay to 0, but allow override or testing purposes
        if 'weight_decay' in inspect.signature(type(optimizer).__init__).parameters:
            param_group['weight_decay'] = 0
        param_group.update({'params': self.trainable_parameters(), **self.param_group_args})
        return param_group
