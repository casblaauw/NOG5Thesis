from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from nog5.base import ModelBase
from nog5.embeddings import ESM1bEmbedding


class ESM1b_Linear(ModelBase):

    def __init__(self, in_features: int, embedding_pretrained: str, **kwargs):
        """ Constructor
        Args:
            in_features: size of the embedding features
            embedding_pretrained: path to the language model weights
        """
        super().__init__()

        self.embedding = ESM1bEmbedding(embedding_pretrained, **kwargs)

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=8),
        ])
        self.dis = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=1),
            nn.Sigmoid()
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=2),
            nn.Tanh()
        ])

    def forward(self, x: Tensor, seq_lengths: Tensor) -> Dict[str, Tensor]:
        """ Forwarding logic """

        x = self.embedding(x, seq_lengths)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        dis = self.dis(x)
        rsa = self.rsa(x)
        phi = self.phi(x)
        psi = self.psi(x)

        return {'ss8': ss8, 'dis': dis, 'rsa': rsa, 'phi': phi, 'psi': psi}
