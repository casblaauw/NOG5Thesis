# Placeholder

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchcrf import CRF

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from nog5.base import ModelBase

from nog5.embeddings import ESM1bEmbedding


class LSTMCRF(ModelBase):

    def __init__(self, embed_n_features: int, lstm_hidden_features: list, lstm_layers: list, 
                 num_tags: int, dropout: float):
        """ Baseline model for CRF_TwoStep
        Args:
            embed_n_features: size of the incoming feature vector


            dropout: amount of dropout
        """

        super().__init__()

        # 'Encoder': biLSTM-dropout-dense on representations 
        self.lstm = nn.LSTM(input_size=embed_n_features, hidden_size=lstm_hidden_features, batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_linear = nn.Linear(in_features=2*lstm_hidden_features, out_features=num_tags)
        self.lstm_dropout = nn.Dropout(p=dropout)

        # CRF
        self.crf = CRF(num_tags = num_tags, batch_first = True)


    def forward(self, x, mask, target = None, get_hidden_output=False) -> Dict[str, Tensor]:
        """ Forwarding logic """

        # Housekeeping: get individual seq lengths and max seq length (batch shape)
        # Starting shape: (batch, max_len, embed_dim)
        seq_lengths = torch.sum(mask, dim=1).cpu().int()
        max_seq_length = x.shape[1]
        mask_bool = torch.eq(mask, 1)
        start_device = x.device

        # Get LSTM score
        # Expects (batch, max_len, embed_dim), makes (batch, max_len, lstm_hidden_features) -> (batch, max_len, num_tags)
        x = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length=max_seq_length, batch_first=True)
        x = self.lstm_linear(x)
        x = self.lstm_dropout(x)

        # Get CRF loss (if training) or prediction (if evalulating after model.eval())
        if self.training:
            if target is None:
                raise ValueError("The CRF module requires true labels as argument 'target' of model.forward() when training.")
            # Expects (batch_size, seq_length, num_tags), returns a single float (sum of loglikelihoods of all sequences)
            x = self.crf.forward(emissions = x, tags = target['region'], mask = mask_bool)
            return -x # Return negative log likelihood (to minimise)
        else:
            results = {}
            results['region_lstm'] = x
            results['region_lstm_softmax'] = F.softmax(x, dim = 2)
            # Expects (batch_size, seq_length, num_tags), returns (batch_size, seq_length)
            x = self.crf.decode(emissions = x, mask = mask_bool)
            x = [torch.tensor(elem, device = start_device) for elem in x]
            x = pad_sequence(x, batch_first = True, padding_value = 0)
            results['region'] = x
            return results



class LSTMCRF_ESM1b(LSTMCRF):
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
