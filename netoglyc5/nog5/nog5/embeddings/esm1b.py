import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.pretrained import load_model_and_alphabet_core, load_hub_workaround
from torch import Tensor

from nog5.utils.logger import setup_logger

log = setup_logger(__name__)


class ESM1bEmbedding(nn.Module):
    """ ESM1b embedding layer module """

    MODEL_MAX_EMBEDDING_SIZE = 1024
    MODEL_EMBEDDING_FEATURES = 1280

    def __init__(self, embedding_pretrained=None, ft_embed_tokens: bool = False, ft_transformer: bool = False,
                 ft_contact_head: bool = False, ft_embed_positions: bool = False,
                 ft_emb_layer_norm_before: bool = False, ft_emb_layer_norm_after: bool = False,
                 ft_lm_head: bool = False, max_embedding: int = MODEL_MAX_EMBEDDING_SIZE, concat_overlap: int = 200,
                 keep_padding: bool = True):
        """ Constructor
        Args:
            embedding_pretrained: patht to pretrained model
            ft_embed_tokens: finetune embed tokens layer
            ft_transformer: finetune transformer layer
            ft_contact_head: finetune contact head
            ft_embed_positions: finetune embedding positions
            ft_emb_layer_norm_before: finetune embedding layer norm before
            ft_emb_layer_norm_after: finetune embedding layer norm after
            ft_lm_head: finetune lm head layer
            max_embedding: maximum sequence length for language model
            concat_overlap: overlap offset when concatenating sequences above max embedding
        """
        super().__init__()

        self.max_embedding = max_embedding
        self.concat_overlap = concat_overlap
        self.keep_padding = keep_padding

        assert 0 < self.concat_overlap < self.max_embedding <= self.MODEL_MAX_EMBEDDING_SIZE

        # if given model path then use pretrained
        # Workaround for load_model_and_alphabet_local bug in v0.4.0 https://github.com/facebookresearch/esm/pull/102
        # Also avoids loading unused ContactPredictionHead regression weights
        if embedding_pretrained:
            model_data = torch.load(embedding_pretrained, map_location="cpu")
        else:
            model_data = load_hub_workaround("https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt")

        # Ignores warning: Regression weights not found, predicting contacts will not produce correct results.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model, _ = load_model_and_alphabet_core(model_data, None)

        # finetuning, freezes all layers by default
        self.finetune = [ft_embed_tokens, ft_transformer, ft_contact_head,
                         ft_embed_positions, ft_emb_layer_norm_before, ft_emb_layer_norm_after, ft_lm_head]

        # finetune by freezing unchoosen layers
        for i, child in enumerate(self.model.children()):
            if not self.finetune[i]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, batch_tokens: Tensor, seq_lengths: Tensor = None) -> Tensor:
        """ Convert tokens to embeddings
        Args:
            batch_tokens: tensor with sequence tokens
            seq_lengths: length of each sequence in batch_tokens without padding
        """
        batch_original_length = batch_tokens.shape[1]

        # remove padding
        if seq_lengths is not None:
            max_unpadded_seq_length = int(max(seq_lengths))
            batch_tokens = batch_tokens[:, :max_unpadded_seq_length + 1]

        batch_length = batch_tokens.shape[1]

        embeddings = self.model(batch_tokens[:, :self.max_embedding], repr_layers=[33])["representations"][33]

        # if length above max embedding, then concatenate multiple embeddings with averaged overlaps
        if batch_length > self.max_embedding:
            n_extra_embeddings = math.ceil((self.max_embedding - batch_length) / (self.concat_overlap - self.max_embedding))
            for i in range(1, n_extra_embeddings + 1):
                o1 = (self.max_embedding - self.concat_overlap) * i
                o2 = o1 + self.max_embedding
                next_embeddings = self.model(batch_tokens[:, o1:o2], repr_layers=[33])["representations"][33]
                concat_1 = embeddings[:, :-self.concat_overlap, :]

                concat_2_overlap_1 = embeddings[:, -self.concat_overlap:, :]
                concat_2_overlap_2 = next_embeddings[:, :self.concat_overlap, :]

                # We use a uniform sampling from sigmoid as weights for a smooth transition in overlap
                concat_2_overlap_weights = torch.sigmoid(torch.linspace(-1, 1, concat_2_overlap_1.shape[1], device=concat_2_overlap_1.device)).unsqueeze(0).unsqueeze(2)
                concat_2 = (concat_2_overlap_1 * concat_2_overlap_weights) + (concat_2_overlap_2 * (1 - concat_2_overlap_weights))

                concat_3 = next_embeddings[:, self.concat_overlap:, :]
                embeddings = torch.cat([concat_1, concat_2, concat_3], dim=1)

        # add back padding
        if self.keep_padding:
            embeddings = F.pad(embeddings, (0, 0, 0, batch_original_length - embeddings.shape[1]))

        return embeddings[:, 1:-1, :]
