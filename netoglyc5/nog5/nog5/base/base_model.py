import torch.nn as nn

from nog5.utils.logger import setup_logger

log = setup_logger(__name__)


class ModelBase(nn.Module):
    """ Base class for all models """

    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        """ Forward pass logic """

        raise NotImplementedError

    def trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                log.debug(f"\t{name}")
                yield param
