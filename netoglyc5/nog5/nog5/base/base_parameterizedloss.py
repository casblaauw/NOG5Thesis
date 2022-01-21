from typing import Dict, Any

from torch import nn, Tensor
from torch.optim import Optimizer

from nog5.utils.logger import setup_logger

log = setup_logger(__name__)


class ParameterizedLossBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs: Dict[str, Tensor], labels: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    def get_param_group(self, optimizer: Optimizer) -> Dict[str, Any]:
        raise NotImplementedError

    def trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                log.debug(f"\t{name}")
                yield param
