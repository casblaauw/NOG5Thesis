from typing import Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nog5.base import ModelBase

from nog5.embeddings import ESM1bEmbedding


class Linear_NetOGlyc(ModelBase):

    def __init__(self, in_features: int):
        """ Baseline model for Linear_NetOGlyc
        Args:
            in_features: size of the embedding features
            embedding_pretrained: path to the language model weights
        """

        super().__init__()

         # Task block
        self.gly = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=1),
        ])

    def forward(self, x, seq_lengths, get_hidden_output=False) -> Dict[str, Tensor]:
        """ Forwarding logic """

        # hidden neurons to classes
        gly = self.gly(x)

        output = {'gly': gly}

        if get_hidden_output:
            output['hidden_output'] = x

        return output


class Linear_ESM1b_NetOGlyc(Linear_NetOGlyc):
    def __init__(self, embedding_pretrained: str = None, embedding_args: Dict[str, Any] = None, **kwargs):
        """ Adds embedding to superclass
        Args:
            embedding_pretrained: path to language model weights
            embedding_args: arguments for embedding
            kwargs: arguments for superclass
        """
        super().__init__(**kwargs)

        # ESM1b block
        self.embedding = ESM1bEmbedding(embedding_pretrained, **embedding_args)

    def forward(self, x: Tensor, seq_lengths: Tensor, get_hidden_output=False) -> Dict[str, Tensor]:
        """ Forwarding logic """
        x = self.embedding(x, seq_lengths)

        return super().forward(x, seq_lengths, get_hidden_output)


class Linear_NetOGlyc_NetSurfP(Linear_NetOGlyc):

    def __init__(self, **kwargs):
        """ Model with netsurfp multi-task """

        super().__init__(**kwargs)

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=kwargs['n_hidden'] * 2, out_features=8),
        ])
        self.dis = nn.Sequential(*[
            nn.Linear(in_features=kwargs['n_hidden'] * 2, out_features=1),
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features=kwargs['n_hidden'] * 2, out_features=1),
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=kwargs['n_hidden'] * 2, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=kwargs['n_hidden'] * 2, out_features=2),
            nn.Tanh()
        ])

    def forward(self, x, seq_lengths, get_hidden_output=False) -> Dict[str, Tensor]:
        """ Forwarding logic """

        output = super().forward(x, seq_lengths, True)
        x = output['hidden_output']

        # hidden neurons to classes
        ss8 = self.ss8(x)
        dis = self.dis(x)
        rsa = self.rsa(x)
        phi = self.phi(x)
        psi = self.psi(x)

        output.update({'ss8': ss8, 'dis': dis, 'rsa': rsa, 'phi': phi, 'psi': psi})

        if not get_hidden_output:
            del output['hidden_output']

        return output


class Linear_ESM1b_NetOGlyc_NetSurfP(Linear_NetOGlyc_NetSurfP):
    def __init__(self, embedding_pretrained: str = None, embedding_args: Dict[str, Any] = None, **kwargs):
        """ Adds embedding to superclass
        Args:
            embedding_pretrained: path to language model weights
            embedding_args: arguments for embedding
            kwargs: arguments for superclass
        """
        super().__init__(**kwargs)

        # ESM1b block
        self.embedding = ESM1bEmbedding(embedding_pretrained, **embedding_args)

    def forward(self, x: Tensor, seq_lengths: Tensor, get_hidden_output=False) -> Dict[str, Tensor]:
        """ Forwarding logic """
        x = self.embedding(x, seq_lengths)

        return super().forward(x, seq_lengths, get_hidden_output)
