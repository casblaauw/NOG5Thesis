from typing import Optional, Union, List

import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler

from nog5.base import DataLoaderBase, DatasetBase


class BasicDataLoader(DataLoaderBase):
    """ DataLoader to load NetSurfP & H5 data """

    def __init__(self, dataset: Union[DatasetBase, ConcatDataset[DatasetBase]], batch_size: int, num_workers: int,
                 shuffle: bool, validation_split: float = None, training_indices: List[int] = None,
                 validation_indices: List[int] = None):
        """ Constructor
        Args:
            train_path: path to the training dataset
            dataset_loader: dataset loader class
            batch_size: size of the batch
            shuffle: shuffles the training data
            validation_split: decimal for the split of the validation
            nworkers: workers for the dataloader class
            test_path: path to the test dataset(s)
        """
        self.dataset = dataset
        if isinstance(self.dataset, ConcatDataset):
            self.custom_collate_fn = self.dataset.datasets[0].custom_collate_fn
        else:
            self.custom_collate_fn = self.dataset.custom_collate_fn

        self.training_sampler = None
        self.validation_sampler = None

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
        }

        if training_indices is None:
            training_indices = list(range(len(self.dataset)))  # no SubsetSequentialSampler exists, so just pass indices

        if validation_indices is not None:
            # remove any training indices that are also found in validation indices
            for idx in validation_indices:
                try:
                    del training_indices[training_indices.index(idx)]
                except ValueError:
                    pass
        elif validation_split is not None:
            # pick random samples from training indices based off the validation split
            num_samples = len(training_indices)
            sample_indices = np.array(training_indices)
            np.random.shuffle(sample_indices)
            num_validation_samples = int(num_samples * validation_split)
            training_indices = sorted(sample_indices[num_validation_samples:].tolist())
            validation_indices = sorted(sample_indices[:num_validation_samples].tolist())

        if len(training_indices) == len(self.dataset):
            self.training_sampler = RandomSampler(self.dataset) if shuffle else SequentialSampler(self.dataset)
        else:
            self.training_sampler = SubsetRandomSampler(training_indices) if shuffle else training_indices

        if validation_indices is not None:
            self.validation_sampler = validation_indices  # no SubsetSequentialSampler exists, so just pass indices

        super().__init__(self.dataset, sampler=self.training_sampler, collate_fn=self.custom_collate_fn, **self.init_kwargs)

    def get_validation_dataloader(self) -> Optional[DataLoader]:
        """ Returns the validation data """
        if self.validation_sampler is None:
            raise ValueError("Dataset has not been split")
        else:
            return DataLoader(self.dataset, sampler=self.validation_sampler, collate_fn=self.custom_collate_fn, **self.init_kwargs)
