from typing import List, Tuple

# import h5py

from nog5.utils.logger import setup_logger

log = setup_logger(__name__)


# def validate_h5_file(file: h5py.File, file_path, labels: List[Tuple[str, int]] = None, embedding_features: int = None):
#     error_string = f"Existing H5 file {file_path} did not pass format validation for section '{{}}'"

#     if (
#         "identifiers" not in file or
#         not isinstance(file["identifiers"], h5py.Dataset) or
#         len(file["identifiers"].shape) != 1 or
#         file["identifiers"].dtype != 'O'
#     ):
#         raise IOError(error_string.format("identifiers"))

#     n_seqs = file["identifiers"].shape[0]

#     if (
#         "sequences" not in file or
#         not isinstance(file["sequences"], h5py.Dataset) or
#         file["sequences"].shape != (n_seqs,) or
#         file["sequences"].dtype != 'O'
#     ):
#         raise IOError(error_string.format("sequences"))

#     max_seq_length = max(len(seq) for seq in file["sequences"].asstr())

#     if labels is not None:
#         if (
#             "labels" not in file or
#             not isinstance(file["labels"], h5py.Group)
#         ):
#             raise IOError(error_string.format("labels"))

#         for label, size in labels:
#             if (
#                 label not in file["labels"] or
#                 file["labels"][label].shape != (n_seqs, max_seq_length, size) or
#                 file["labels"][label].dtype != 'f4'
#             ):
#                 raise IOError(error_string.format(f"labels/{label}"))

#     if embedding_features is not None:
#         if (
#             "embeddings" not in file or
#             not isinstance(file["embeddings"], h5py.Dataset) or
#             file["embeddings"].shape != (n_seqs, max_seq_length, embedding_features) or
#             file["embeddings"].dtype != 'f4'
#         ):
#             raise IOError(error_string.format("embeddings"))

#     expected_sections = 2 + (1 if labels is not None else 0) + (1 if embedding_features is not None else 0)
#     if len(file) != expected_sections or (labels is not None and len(file["labels"]) != len(labels)):
#         log.warning(f"Existing H5 file {file_path} has unknown sections,"
#                     "make sure that you know what these are and keep them synchronized with known sections")


# def copy_h5_indices(old_f: h5py.File, new_f: h5py.File, indices: List[int], embeddings_batch_size: int):
#     n_seqs = len(indices)
#     max_seq_length = max(map(lambda seq: len(seq), old_f['sequences'].asstr()[indices]))

#     new_f.create_dataset('identifiers', (n_seqs,), h5py.string_dtype(), maxshape=(None,))
#     new_f['identifiers'][:] = old_f['identifiers'].asstr()[indices]

#     new_f.create_dataset("sequences", (n_seqs,), h5py.string_dtype(), maxshape=(None,))
#     new_f['sequences'][:] = old_f['sequences'].asstr()[indices]

#     if 'labels' in old_f:
#         old_labels_group = old_f["labels"]
#         new_labels_group = new_f.create_group("labels")
#         for label in old_labels_group:
#             size = old_labels_group[label].shape[2]
#             new_labels_group.create_dataset(label, (n_seqs, max_seq_length, size), dtype='f4', maxshape=(None, None, size))
#             new_labels_group[label].attrs['cast_type'] = old_labels_group[label].attrs['cast_type']
#             new_labels_group[label][:] = old_labels_group[label][indices, :max_seq_length]

#     if 'embeddings' in old_f:
#         embedding_features = old_f['embeddings'].shape[2]
#         new_embeddings_dataset = new_f.create_dataset("embeddings", (n_seqs, max_seq_length, embedding_features), 'f4', maxshape=(None, None, embedding_features))
#         for idx in range(0, n_seqs, embeddings_batch_size):
#             batch_indices = indices[idx:idx+embeddings_batch_size]
#             new_embeddings_dataset[idx:idx+embeddings_batch_size] = old_f['embeddings'][batch_indices, :max_seq_length]
