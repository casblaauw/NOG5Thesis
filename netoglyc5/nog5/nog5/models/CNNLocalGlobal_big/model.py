# Placeholder

from typing import Dict, Any

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F
from torch import Tensor

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nog5.base import ModelBase

from nog5.embeddings import ESM1bEmbedding


class CNNLocalGlobal_big(ModelBase):

    def __init__(self, embed_n_features: int, region_hidden_features: int, region_lstm_layers: int, 
                 global_number_filters: list, global_kernel_sizes: list,
                 localisation_out_features: int, pioneer_hidden_features: int, pioneer_kernel_size: int,
                 full_hidden_features: int, full_kernel_size: int, dropout: float):
        """ Baseline model for CRF_TwoStep
        Args:
            embed_n_features: size of the incoming feature vector
            region_hidden_features:
            region_out_features:
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

        # LSTM to predict regional glycosylatability
        self.region_lstm = nn.LSTM(input_size=embed_n_features, hidden_size=region_hidden_features, batch_first=True,
                            num_layers=region_lstm_layers, bidirectional=True, dropout=dropout)
        self.region_lstm_dropout = nn.Dropout(p=dropout)
        self.region_linear = nn.Linear(in_features=2*region_hidden_features, out_features=1)


        # Pooled CNN to detect far-away globally important features like signal peptides
        self.global_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_n_features, out_channels=global_number_filters[i], 
                          kernel_size=global_kernel_sizes[i], padding=global_kernel_sizes[i]//2),
                nn.ReLU(),
                # nn.MaxPool1d(kernel_size=global_kernel_sizes[i]))
                nn.AdaptiveMaxPool1d(output_size = 1))
            for i in range(len(global_number_filters))
        ])
        # self.global_linear = nn.Linear(in_features=np.sum(global_number_filters), out_features=)

        # Logistic regression on pooled vector to predict global state
        self.localisation = nn.Linear(in_features=embed_n_features, out_features=localisation_out_features)

        self.global1 = nn.Sequential(
            nn.Linear(in_features=embed_n_features, out_features=1),
            nn.Sigmoid()
        )

        self.global2 = nn.Sequential(
            nn.Linear(in_features=embed_n_features, out_features=1),
            nn.Sigmoid()
        )

        # Overall CNN to predict sites
        # Combine embeddings, LSTM regional prediction, global convolution features, all global linear features
        pioneer_dims = embed_n_features+1+np.sum(global_number_filters)+(localisation_out_features+2)
        self.pioneer_conv = nn.Sequential(
            nn.Conv1d(in_channels=pioneer_dims, out_channels=pioneer_hidden_features, kernel_size=pioneer_kernel_size, padding=pioneer_kernel_size//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(pioneer_hidden_features)
        )
        self.pioneer_linear = nn.Sequential(
            nn.Linear(in_features=pioneer_hidden_features, out_features=1),
            nn.Sigmoid()
        )

        # Combine all information with pioneer sites
        full_dims = pioneer_dims + 1
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
        

    def forward(self, x, mask, get_hidden_output=False) -> Dict[str, Tensor]:
        """ Forwarding logic """

        # Housekeeping: get individual seq lengths and max seq length (batch shape)
        # Starting shape: (batch, max_len, embed_dim)
        seq_lengths = torch.sum(mask, dim=1).cpu()
        max_seq_length = x.shape[1]

        # Get regional glycosylatability score
        # Expects (batch, max_len, embed_dim), makes (batch, max_len, region_hidden_features) -> (batch, max_len, 1) -> (batch, 1, max_len)
        x1 = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        x1, _ = self.region_lstm(x1)
        x1, _ = pad_packed_sequence(x1, total_length=max_seq_length, batch_first=True)
        x1 = self.region_lstm_dropout(x1)
        x1 = self.region_linear(x1)
        x1 = x1.permute(0, 2, 1)

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (batch, embed_dim, max_len)
        x = x.permute(0, 2, 1)

        # Get global conv score
        # Expects (batch, embed_dim, max_len), returns (batch, sum(n_filters), max_len)) (repeating over max_len) 
        x5 = [conv(x) for conv in self.global_conv]
        x5 = torch.cat([x_pool for x_pool in x5], dim=1) # concatenate max-pooled results across sequence from diff kernel sizes into (batch, sum(n_filters), 1)
        x5 = x5.repeat(1, 1, max_seq_length) # repeat global score to add max score across seq to every element of the sequence (batch, sum(n_filters), max_len)

        # Get global linear scores
        x_pooled = torch.mean(x, 2)
        x2 = self.localisation(x_pooled)
        x2 = torch.unsqueeze(x2, 2).repeat(1, 1, max_seq_length) # expand from (batch, localisation_out_features) to (batch, localisation_out_features, max_len)
        x3 = self.global1(x_pooled)
        x3 = x3.reshape((-1, 1, 1)).repeat(1, 1, max_seq_length) # expand from (batch) to (batch, 1, max_len) 
        x4 = self.global2(x_pooled)
        x4 = x4.reshape((-1, 1, 1)).repeat(1, 1, max_seq_length)

        # Predict pioneer glycosylation
        x = torch.cat((x, x1, x2, x3, x4, x5), 1) # Combine embeddings, glycosylatability score, and global protein scores into (batch, all_features, max_len)
        x6 = self.pioneer_conv(x) 
        x6 = x6.permute(0, 2, 1) # (batch, pioneer_hidden_features, max_len) -> (batch, max_len, pioneer_hidden_features) because linear needs in_features at end
        x6 = self.pioneer_linear(x6)
        x6 = x6.permute(0, 2, 1) # revert to (batch, 1, max_len)

        # Predict full glycosylation
        x = torch.cat((x, x6), 1) # Append first-round glycosylation to combined features
        x = self.full_conv(x)
        x = x.permute(0, 2, 1) # (batch, full_hidden_features, max_len) -> (batch, max_len, full_hidden_features)
        x = self.full_linear(x)
        x = x.permute(0, 2, 1) # revert to (batch, 1, max_len)

        # Return some hidden states/add extra scoring types

        # Return output
        output = {'gly': x.squeeze(1)}
        return output

        # return output


class CNNLocalGlobal_ESM1b_big(CNNLocalGlobal_big):
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
