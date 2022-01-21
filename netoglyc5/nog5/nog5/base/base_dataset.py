import math
from typing import Union, TypeVar, List, Tuple, Any

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

T = TypeVar("T")


class DatasetBase(Dataset):
    """ Base class for NetSurfP datasets """

    def __init__(self, dataset_path: str, truncate_seq_length: int = None):
        self.dataset_path = dataset_path
        # We just set a very large number if no truncation length is provided
        self.truncate_seq_length = truncate_seq_length if truncate_seq_length is not None else 2147483647

    def __getitem__(self, index: int):
        """ Returns data, target and mask data (+ identifier & sequence if PredictionDataset)
        Args:
            index: Index at the array
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """ Returns the length of the data """
        raise NotImplementedError

    def custom_collate_fn(self, x: Union[T, List[T]]) -> T:
        """ Collates dataset samples into a batch
        This function should be overridden by subclass if:
            arrays in samples can't simply be stacked together (e.g. variable lengths, non-numpy/torch arrays)
            there is some information that should not just be concatenated (e.g. a max of sequence lengths)
        Args:
            x: either a single sample or a list of samples
        Returns:
            If single sample: adds batch dimension to any arrays and converts them to tensors
            If list of samples: stacks together arrays to form single tensors in batch
        """
        return default_collate(x)


class TrainEvalDatasetBase(DatasetBase):

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """ Returns data, target and mask data
        Args:
            index: Index at the array
        """
        raise NotImplementedError


class PredictionDatasetBase(DatasetBase):

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any, Any]:
        """ Returns data, target and mask data
        Args:
            index: Index at the array
        """
        raise NotImplementedError
