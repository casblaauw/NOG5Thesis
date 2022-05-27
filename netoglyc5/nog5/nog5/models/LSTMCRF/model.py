# Placeholder

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from torchcrf import CRF

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from nog5.base import ModelBase
from nog5.utils.crf import CRF
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
        # mask_bool = torch.eq(mask, 1)
        start_device = x.device
        if target is not None and target.get('info_mask') is not None:
            loss_mask = target['info_mask']
            # loss_mask_bool = torch.eq(loss_mask, 1)
        else:
            loss_mask = mask
            # loss_mask_bool = mask_bool

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
            x = self.crf.forward(emissions = x, tags = target['region'], mask = mask, loss_mask = loss_mask)
            return -x # Return negative log likelihood (to minimise)
        else:
            results = {}
            results['region_lstm'] = x
            results['region_lstm_softmax'] = F.softmax(x, dim = 2)
            # Decode (i.e. get predicted sequence labels)
            # Expects (batch_size, seq_length, num_tags), returns (batch_size, seq_length)
            preds = self.crf.decode(emissions = x, mask = mask)
            preds = [torch.tensor(elem, device = start_device) for elem in preds]
            preds = pad_sequence(preds, batch_first = True, padding_value = 0)
            results['region'] = preds
            # Get marginal probabilities (i.e. get probabilities of each label)
            probs = self.crf.compute_marginal_probabilities(emissions = x, mask = mask)
            results['region_probs'] = probs.to(start_device)
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


class LSTMCRFCNN(ModelBase):

    def __init__(self, embed_n_features: int, lstm_hidden_features: list, lstm_layers: list, 
                 num_tags: int, cnn_kernel: int, cnn_heads: int, dropout: float):
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

        # CNN
        self.cnn = nn.Conv1d(in_channels=embed_n_features+1, out_channels=cnn_heads, kernel_size=cnn_kernel, padding=cnn_kernel//2)
        self.cnn_linear = nn.Linear(in_features=cnn_heads, out_features=1)
        self.cnn_dropout = nn.Dropout(p=dropout)
 

    def forward(self, x, mask, target = None, get_hidden_output=False) -> Dict[str, Tensor]:
        """ Forwarding logic """

        # Housekeeping: get individual seq lengths and max seq length (batch shape)
        # Starting shape: (batch, max_len, embed_dim)
        seq_lengths = torch.sum(mask, dim=1).cpu().int()
        max_seq_length = x.shape[1]
        # mask_bool = torch.eq(mask, 1)
        start_device = x.device
        # if target is not None and target.get('info_mask') is not None:
        #     loss_mask = target['info_mask']
        # else:
        #     loss_mask = mask

        # Get LSTM score
        # Expects (batch, max_len, embed_dim), makes (batch, max_len, lstm_hidden_features) -> (batch, max_len, num_tags)
        lstm_score = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        lstm_score, _ = self.lstm(lstm_score)
        lstm_score, _ = pad_packed_sequence(lstm_score, total_length=max_seq_length, batch_first=True)
        lstm_score = self.lstm_linear(lstm_score)
        lstm_score = self.lstm_dropout(lstm_score)

        # Get marginal probabilities (i.e. get probabilities of each label)
        region_probs = self.crf.compute_marginal_probabilities(emissions = lstm_score, mask = mask)

        # Decode (i.e. get predicted sequence labels)
        # Expects (batch_size, seq_length, num_tags), returns (batch_size, seq_length)
        region_preds = self.crf.decode(emissions = lstm_score, mask = mask)
        region_preds = [torch.tensor(elem, device = start_device) for elem in region_preds]
        region_preds = pad_sequence(region_preds, batch_first = True, padding_value = 0)

        # Get CNN score
        # Expects (batch, embed_dim+1, len), returns (batch, len, 1)
        site_preds = x.transpose(1, 2)
        site_preds = torch.cat([site_preds, region_preds.unsqueeze(1)], dim = 1)
        site_preds = self.cnn(site_preds) # Returns (batch, cnn_heads, len)
        site_preds = self.cnn_dropout(site_preds)
        site_preds = site_preds.transpose(1, 2) # Returns (batch, len, cnn_heads)
        site_preds = self.cnn_linear(site_preds) # Needs (batch, len, cnn_heads) (or just *, *, features really)
        site_preds = site_preds.squeeze()
        site_preds = torch.sigmoid(site_preds)
        # site_preds = F.softmax(site_preds, dim = 1)


        # Get CRF loss (if training) or prediction (if evalulating after model.eval())
        if self.training:
            if target is None:
                raise ValueError("The CRF module requires true labels as argument 'target' of model.forward() when training.")
            # Expects (batch_size, seq_length, num_tags), returns a single float (sum of loglikelihoods of all sequences)
            crf_loss = -self.crf.forward(emissions = lstm_score, tags = target['region'], mask = mask, reduction = 'mean') * 0.01
            # cnn_loss = torch.sum(F.binary_cross_entropy(site_preds, target['gly'], reduction='none') * mask)/torch.sum(mask)
            cnn_loss = torch.sum(F.binary_cross_entropy(site_preds, target['gly'], weight = torch.tensor([0.01, 1]), reduction='none')*target['glycosylation_mask']) / torch.sum(target['glycosylation_mask'])
            return crf_loss + cnn_loss

            
        else:
            results = {}
            results['region_lstm'] = lstm_score
            results['region_lstm_softmax'] = F.softmax(lstm_score, dim = 2)
            results['region'] = region_preds
            results['region_probs'] = region_probs.to(start_device)
            results['gly'] = site_preds
            return results