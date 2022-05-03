# Placeholder

from typing import Dict, Any

import torch
import torch.nn as nn
from torch import Tensor, softmax

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nog5.base import ModelBase

from nog5.embeddings import ESM1bEmbedding


class CNNLocalGlobal_small(ModelBase):

    def __init__(self, embed_n_features: int, region_hidden_features: list, region_kernel_sizes: list, region_out_features: int,
                 localisation_out_features: int, full_hidden_features: int, full_kernel_size: int, dropout: float):
        """ Baseline model for CRF_TwoStep
        Args:
            embed_n_features: size of the incoming feature vector
            region_hidden_features:
            # region_out_features:
            region_kernel_sizes:
            localisation_out_features:
            full_kernel_size:

            # out_channels: amount of hidden neurons in the bidirectional lstm
            # cnn_layers: amount of cnn layers
            # kernel_size: kernel sizes of the cnn layers
            # padding: padding of the cnn layers
            # n_hidden: amount of hidden neurons
            # dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
        """

        super().__init__()

        # CNNs to predict regional glycosylatability
        self.region_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_n_features, out_channels=region_hidden_features[0],
                        kernel_size=region_kernel_sizes[0], padding=region_kernel_sizes[0]//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(region_hidden_features[0]),
            nn.Conv1d(in_channels=region_hidden_features[0], out_channels=region_hidden_features[1],
                        kernel_size=region_kernel_sizes[1], padding=region_kernel_sizes[1]//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(region_hidden_features[1])
        )
        self.region_linear = nn.Sequential(
            nn.Linear(in_features=region_hidden_features[1], out_features=region_out_features),
            nn.Sigmoid()
        )

        # Logistic regression on pooled vector to predict global state
        self.localisation = nn.Sequential(
            nn.Linear(in_features=embed_n_features, out_features=localisation_out_features),
            nn.Softmax(dim=-1)
        )

        self.global1 = nn.Sequential(
            nn.Linear(in_features=embed_n_features, out_features=1),
            nn.Sigmoid()
        )

        self.global2 = nn.Sequential(
            nn.Linear(in_features=embed_n_features, out_features=1),
            nn.Sigmoid()
        )

        # Overall CNN to predict sites
        full_dims = embed_n_features+1+localisation_out_features+2
        self.full_conv = nn.Sequential(
            nn.Conv1d(in_channels=full_dims, out_channels=full_hidden_features, kernel_size=full_kernel_size, padding=full_kernel_size//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(full_hidden_features)
        )
        self.full_linear = nn.Sequential(
            nn.Linear(in_features=full_hidden_features, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x, seq_lengths, get_hidden_output=False) -> Dict[str, Tensor]:
        """ Forwarding logic """

        # Starting shape: (batch, max_len, embed_dim)

        # Housekeeping: get max sequence length and reorder dimensions
        max_seq_length = int(max(seq_lengths))

        ## Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        ## Output shape: (batch, embed_dim, max_len) 
        x = x.permute(0, 2, 1)

        # Get regional glycosylatability score
        x1 = self.region_conv(x)
        x1 = x1.permute(0, 2, 1) # (batch, full_hidden_features, max_len) -> (batch, max_len, pioneer_full_features)
        x1 = self.region_linear(x1)
        x1 = x1.permute(0, 2, 1) # revert to (batch, 1, max_len)

        # Get global scores
        x_pooled = torch.mean(x, 2)
        x2 = self.localisation(x_pooled)
        x2 = torch.unsqueeze(x2, 2).repeat(1, 1, max_seq_length) # expand from (batch, localisation_out_features) to (batch, localisation_out_features, len_sequence)
        x3 = self.global1(x_pooled)
        x3 = x3.reshape((-1, 1, 1)).repeat(1, 1, max_seq_length) # expand from (batch) to (batch, 1, len_sequence) 
        x4 = self.global2(x_pooled)
        x4 = x4.reshape((-1, 1, 1)).repeat(1, 1, max_seq_length)

        # Predict glycosylation
        x = torch.cat((x, x1, x2, x3, x4), 1) # (batch, all_features, max_len)
        x = self.full_conv(x) # (batch, full_hidden_features, max_len)
        x = x.permute(0, 2, 1) # (batch, max_len, full_hidden_features)
        x = self.full_linear(x) # (batch, max_len, 1)
        x = x.permute(0, 2, 1) # (batch, 1, max_len)

        # Return some hidden states/add extra scoring types

        # Return output
        output = {'gly': x.squeeze(1)}
        return output

        # return output


class CNNLocalGlobal_ESM1b_small(CNNLocalGlobal_small):
    def __init__(self, embedding_pretrained: str = None, embedding_args: Dict[str, Any] = None, **kwargs):
        """ Adds embedding to superclass
        Args:
            embedding_pretrained: path to language model weights
            embedding_args: arguments for embedding
            kwargs: arguments for superclass
        """
        super().__init__(**kwargs)

        if embedding_args is None:
            embedding_args = {}

        # ESM1b block
        self.embedding = ESM1bEmbedding(embedding_pretrained, **embedding_args)

    def forward(self, x: Tensor, seq_lengths: Tensor, get_hidden_output=False) -> Dict[str, Tensor]:
        """ Forwarding logic """
        x = self.embedding(x, seq_lengths)

        return super().forward(x, seq_lengths, get_hidden_output)


# class CNNbLSTM_NetOGlyc_NetSurfP(CNNbLSTM_NetOGlyc):

#     def __init__(self, **kwargs):
#         """ Model with netsurfp multi-task """

#         super().__init__(**kwargs)

#         # Task block
#         self.ss8 = nn.Sequential(*[
#             nn.Linear(in_features=kwargs['n_hidden'] * 2, out_features=8),
#         ])
#         self.dis = nn.Sequential(*[
#             nn.Linear(in_features=kwargs['n_hidden'] * 2, out_features=1),
#         ])
#         self.rsa = nn.Sequential(*[
#             nn.Linear(in_features=kwargs['n_hidden'] * 2, out_features=1),
#         ])
#         self.phi = nn.Sequential(*[
#             nn.Linear(in_features=kwargs['n_hidden'] * 2, out_features=2),
#             nn.Tanh()
#         ])
#         self.psi = nn.Sequential(*[
#             nn.Linear(in_features=kwargs['n_hidden'] * 2, out_features=2),
#             nn.Tanh()
#         ])

#     def forward(self, x, seq_lengths, get_hidden_output=False) -> Dict[str, Tensor]:
#         """ Forwarding logic """

#         output = super().forward(x, seq_lengths, True)
#         x = output['hidden_output']

#         # hidden neurons to classes
#         ss8 = self.ss8(x)
#         dis = self.dis(x)
#         rsa = self.rsa(x)
#         phi = self.phi(x)
#         psi = self.psi(x)

#         output.update({'ss8': ss8, 'dis': dis, 'rsa': rsa, 'phi': phi, 'psi': psi})

#         if not get_hidden_output:
#             del output['hidden_output']

#         return output


# class CNNbLSTM_ESM1b_NetOGlyc_NetSurfP(CNNbLSTM_NetOGlyc_NetSurfP):
#     def __init__(self, embedding_pretrained: str = None, embedding_args: Dict[str, Any] = None, **kwargs):
#         """ Adds embedding to superclass
#         Args:
#             embedding_pretrained: path to language model weights
#             embedding_args: arguments for embedding
#             kwargs: arguments for superclass
#         """
#         super().__init__(**kwargs)

#         if embedding_args is None:
#             embedding_args = {}

#         # ESM1b block
#         self.embedding = ESM1bEmbedding(embedding_pretrained, **embedding_args)

#     def forward(self, x: Tensor, seq_lengths: Tensor, get_hidden_output=False) -> Dict[str, Tensor]:
#         """ Forwarding logic """
#         x = self.embedding(x, seq_lengths)

#         return super().forward(x, seq_lengths, get_hidden_output)
