from typing import Union, List, Sequence
from os import PathLike
from glyc_processing.annotation import ProteinSet
import zipfile
import pickle

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from nog5.base import DatasetBase

class ZipDataset(DatasetBase):
    """Dataset representing indexed and compressed ESM embeddings.
    Path is expected to be a path to an zip file containing the .pt tensor arrays for each protein.
    Then indexes into that zip file to find the f"esm_embeddings_{prot_id} file."
    Info is expected to be a ProteinSet object or a path to a pickled file containing a ProteinSet."""
    def __init__(self, dataset_path: str, info: Union[ProteinSet, str, PathLike], truncate_seq_length: int = None):
        # Process info
        if isinstance(info, ProteinSet):
            self.info = info
        else:
            with open(info, 'rb') as f:
               self.info = pickle.load(f)

        # Set base properties
        self.dataset_path = dataset_path
        self.truncate_seq_length = truncate_seq_length if truncate_seq_length is not None else 2147483647 # We just set a very large number if no truncation length is provided


    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, idx):
        protname = list(self.info.keys())[idx]

        # Read embeddings
        with zipfile.ZipFile(self.dataset_path, 'r') as zip:
            with zip.open(f"esm_embeddings_{protname}.pt") as myfile:
                embed = torch.load(myfile).squeeze() # Return as (prot_len, embedding_dims) for padding

        # Get labels
        label = {}
        label['gly'] = torch.tensor(self.info[protname].get_glycosylation_labels())
        label['seq_mask'] = torch.ones_like(label['gly'])
        ## Mask in the loss context indicates detected glycosylation sites
        label['glycosylation_mask'] = torch.where(label['gly'] >= 0, 1, 0) 
        label['definite_glycosylation_mask'] = torch.where(torch.eq(label['gly'], 0) | torch.eq(label['gly'], 1), 1, 0) 
        label['ambiguous_glycosylation_mask'] = torch.where(torch.gt(label['gly'], 0) & torch.lt(label['gly'], 1), 1, 0)

        ## Mask in the training/model context used to indicate sequence length integer (to counteract batch padding)
        # mask = torch.tensor(len(self.info[protname].protein_seq))
        return embed, label, torch.ones_like(label['gly'])

    def custom_collate_fn(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # print(x[0])
        # # x: list of (embed, label, mask) tuples of len batch_size
        # if isinstance(x[0][0], Sequence):
        #     for output in x:
        #         embed_tensor = output[0]
        #     all_sizes = [elem.shape[-1] for elem in x[0]]
        #     max_size = max(all_sizes)
        #     if not all(len_elem == all_sizes[0] for len_elem in all_sizes):
        #         x[0] = [F.pad(elem, (0, max_size - len_elem)) for elem, len_elem in zip(x[0], all_sizes)]

        # return default_collate(x)

        inputs = [elem[0] for elem in x]                # List of 2D tensors
        labels = [elem[1] for elem in x]                # List of dictionaries containing 1D tensors
        masks = [elem[2] for elem in x]                 # List of integer tensors

        # Convert list of individual dicts to dict of lists with entire batch
        ## [{a: tensor, b: tensor}, ...] -> {a: [tensor, ...], b: [tensor, ...]}
        ## Requires all entries to have same label sets as first
        labels_dict = {k: [dic[k] for dic in labels] for k in labels[0]} 

        inputs = pad_sequence(inputs, batch_first=True) 
        labels = {label_type: pad_sequence(label_values, batch_first=True) for label_type, label_values in labels_dict.items()}
        masks = torch.stack(masks)
        return (inputs, labels, masks)



class ZipPandasDataset(DatasetBase):
    """Dataset representing indexed and compressed ESM embeddings.
    Path is expected to be a path to an zip file containing the .pt tensor arrays for each protein.
    Then indexes into that zip file to find the f"esm_embeddings_{prot_id} file."
    Info is expected to be a pandas dataframe containing columns 'prot_id' and 'label' (-1 or (0,1] )."""
    def __init__(self, dataset_path: str, info_dataframe: pd.DataFrame, truncate_seq_length: int = None):
        self.dataset_path = dataset_path
        # We just set a very large number if no truncation length is provided
        self.truncate_seq_length = truncate_seq_length if truncate_seq_length is not None else 2147483647
        self.info = info_dataframe.reset_index() 

    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, idx):
        with zipfile.ZipFile(self.dataset_path, 'r') as zip:
            with zip.open(f"esm_embeddings_{self.info['prot_id'][idx]}.pt") as myfile:
                embed = torch.load(myfile).T

        label = {}
        label['gly'] = torch.tensor(self.info['label'][idx])

        # Mask in the loss context indicates detected glycosylation sites
        label['glycosylation_mask'] = torch.where(label['gly'] >= 0, 1, 0) 
        label['definite_glycosylation_mask'] = torch.where(label['gly'] == 1, 1, 0)
        # Mask in the training/model context indicates sequence length (to remove padding)
        mask = torch.ones_like(label['gly'])
        return embed, label, mask