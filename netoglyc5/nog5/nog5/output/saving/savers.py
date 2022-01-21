from typing import List, Tuple

import h5py
import torch

from nog5.base import SaverBase
from nog5.utils.dataset_io import validate_h5_file
from nog5.utils.logger import setup_logger

log = setup_logger(__name__)


class H5Saver(SaverBase):

    def __init__(self, output_path: str, labels_transform: callable = None, embedding_features: int = None,
                 label_names: List[str] = None, label_sizes: List[int] = None, target_is_output_labels: bool = False,
                 data_is_output_embeddings: bool = False):
        super().__init__(output_path)
        # https://stackoverflow.com/questions/46045512/h5py-hdf5-database-randomly-returning-nans-and-near-very-small-data-with-multi
        self.file = None
        self.labels_transform = labels_transform
        self.embedding_features = embedding_features
        self.target_is_output_labels = target_is_output_labels
        self.data_is_output_embeddings = data_is_output_embeddings

        # Save either embeddings or output labels depending on arguments
        if label_names is not None and label_sizes is not None:
            if len(label_names) == len(label_sizes):
                if len(label_names) == 0:
                    raise ValueError("Failed to create H5Saver, either specify at least one label in 'label_names' "
                                     "and 'label_sizes' or remove them")
                self.labels = list(zip(label_names, label_sizes))
            else:
                raise ValueError("Failed to create H5Saver, 'label_names' and 'label_sizes' are not the same length")
        else:
            self.labels = None

        if self.embedding_features is not None and self.labels is not None and not (self.target_is_output_labels or self.data_is_output_embeddings):
            raise ValueError("Failed to create H5Saver, 'model_embedding_features' and 'label_names/sizes' cannot be "
                             "combined unless 'target_is_output_labels' is True")
        if self.embedding_features is None and self.labels is None:
            raise ValueError("Failed to create H5Saver, you must provide 'model_embedding_features' and/or "
                             "'label_names/sizes'")

    def write(self, identifiers, sequences, data, target, mask, output):
        n_seqs = len(identifiers)
        max_seq_length = max(len(seq) for seq in sequences)
        output_labels = target if self.target_is_output_labels else output
        output_embeddings = data if self.data_is_output_embeddings else output

        if self.labels_transform is not None:
            output_labels = self.labels_transform(output_labels)

        if self.mode == 'w':
            self.file = h5py.File(self.output_path, 'w')

            identifiers_dataset = self.file.create_dataset("identifiers", (n_seqs,), h5py.string_dtype(), maxshape=(None,))
            sequences_dataset = self.file.create_dataset("sequences", (n_seqs,), h5py.string_dtype(), maxshape=(None,))

            if self.embedding_features is not None:
                embeddings_dataset = self.file.create_dataset("embeddings",
                                                              (n_seqs, max_seq_length, self.embedding_features), 'f4',
                                                              maxshape=(None, None, self.embedding_features))
            else:
                embeddings_dataset = None

            if self.labels is not None:
                labels_group = self.file.create_group("labels")
                for label, size in self.labels:
                    label_dataset = labels_group.create_dataset(label, (n_seqs, max_seq_length, size), dtype='f4',
                                                                maxshape=(None, None, size))
                    label_dataset.attrs['cast_type'] = str(output_labels[label].numpy().dtype)
            else:
                labels_group = None

            start_idx = 0
            self.mode = 'a'

        elif self.mode == 'a':
            if not self.file:
                self.file = h5py.File(self.output_path, 'a')
                validate_h5_file(self.file, self.output_path, labels=self.labels,
                                 embedding_features=self.embedding_features)

            identifiers_dataset = self.file["identifiers"]
            sequences_dataset = self.file["sequences"]
            embeddings_dataset = self.file["embeddings"] if self.embedding_features is not None else None
            labels_group = self.file["labels"] if self.labels is not None else None

            if embeddings_dataset is not None:
                old_max_seq_length = embeddings_dataset.shape[1]
            elif labels_group is not None:
                old_max_seq_length = labels_group[self.labels[0][0]].shape[1]
            else:
                raise ValueError("Failed to append in H5Saver, could not determine maximum sequence length of dataset")

            start_idx = len(identifiers_dataset)
            expanded_n_seqs = n_seqs + start_idx
            expanded_max_seq_length = max(max_seq_length, old_max_seq_length)

            identifiers_dataset.resize((expanded_n_seqs,))
            sequences_dataset.resize((expanded_n_seqs,))

            if embeddings_dataset is not None:
                embeddings_dataset.resize((expanded_n_seqs, expanded_max_seq_length, self.embedding_features))
            if labels_group is not None:
                for label, size in self.labels:
                    labels_group[label].resize((expanded_n_seqs, expanded_max_seq_length, size))
        else:
            raise ValueError("Failed to write to H5Saver, write mode is unknown")

        identifiers_dataset[start_idx:] = identifiers
        sequences_dataset[start_idx:] = sequences

        if embeddings_dataset is not None:
            embeddings_dataset[start_idx:, :max_seq_length, :] = output_embeddings.numpy()
        if labels_group is not None:
            for label, size in self.labels:
                try:
                    labels_group[label][start_idx:, :max_seq_length, :] = output_labels[label].numpy()
                except TypeError as e:
                    log.error(f"TypeError raised while saving label '{label}'")
                    raise e

    def close(self):
        if self.file is not None:
            self.file.close()
