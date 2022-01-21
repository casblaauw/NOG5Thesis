import re
from typing import Mapping, Sequence, Union

import torch
from torch import Tensor
from torch.nn import functional as F

np_str_obj_array_pattern = re.compile(r'[SaUO]')
concat_output_list_err_msg_format = (
    "concat_output_list: output must contain tensors, numpy arrays, numbers, strings, dicts, lists or tuples; found {}"
)
tensors_to_device_err_msg_format = (
    "tensors_to_device: output must contain tensors, numpy arrays, numbers, strings, dicts, lists or tuples; found {}"
)


def concat_list(output):
    """
    Tries to concatenate the list of arbitrary outputs/targets from predictor,
        e.g. tensors/arrays are padded and concatenated and dicts/lists are gathered together.
    This might fail for more exotic model outputs (weird types, inconsistent output format etc.),
        in which case a custom function should be used.
    Code modified from:
    https://github.com/pytorch/pytorch/blob/b67eaec8535b8d2a1fa1ddddfdbbf54b4f624840/torch/utils/data/_utils/collate.py
    """
    elem = output[0]
    elem_type = type(elem)
    if isinstance(elem, Tensor):
        elem_shape = elem.shape
        if not all(elem.shape[-1] == elem_shape[-1] for elem in output):
            raise RuntimeError('each tensor element in list should be of equal size in last dimension')
        if len(elem_shape) > 1:
            max_elem_length = max(elem.shape[-2] for elem in output)
            return torch.cat(
                [F.pad(elem, (0, 0, 0, max_elem_length - elem.shape[-2])) for elem in output]
            )
        else:
            return torch.stack(output, dim=0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(concat_output_list_err_msg_format.format(elem.dtype))
            return concat_list([torch.as_tensor(b) for b in output])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(output)
    elif isinstance(elem, (bool, float, int, str, bytes)) or elem is None:
        return output
    elif isinstance(elem, Mapping):
        try:
            return elem_type({key: concat_list([d[key] for d in output]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: concat_list([d[key] for d in output]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(concat_list(samples) for samples in zip(*output)))
    elif isinstance(elem, Sequence):
        # check to make sure that the elements in output have consistent size
        it = iter(output)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of output should be of equal size')
        transposed = list(zip(*output))  # It may be accessed twice, so we use a list.
        try:
            return elem_type([concat_list(samples) for samples in transposed])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [concat_list(samples) for samples in transposed]
    else:
        raise TypeError(concat_output_list_err_msg_format.format(elem_type))


def _tensor_to_device(tensor: Tensor, device: torch.device):
    if device.type == 'cpu':
        return tensor.detach().to(device)
    elif device.type == 'cuda':
        return tensor.to(device)


def tensors_to_device(output, device: torch.device):
    """
    Tries to make sure that all tensors in arbitrary output from predictor are detached from graph and put on the cpu
    This might fail for more exotic model outputs (weird types etc.), in which case a custom function should be used.
    Code modified from:
    https://github.com/pytorch/pytorch/blob/b67eaec8535b8d2a1fa1ddddfdbbf54b4f624840/torch/utils/data/_utils/collate.py
    """
    output_type = type(output)
    if isinstance(output, Tensor):
        return _tensor_to_device(output, device)
    elif isinstance(output, (bool, float, int, str, bytes)) or output is None:
        return output
    elif isinstance(output, Mapping):
        try:
            return output_type(((key, tensors_to_device(output[key], device)) for key in output))
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: tensors_to_device(output[key], device) for key in output}
    elif isinstance(output, tuple) and hasattr(output, '_fields'):  # namedtuple
        return output_type(tensors_to_device(field, device) for field in output)
    elif isinstance(output, Sequence):
        try:
            return output_type(tensors_to_device(elem, device) for elem in output)
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [tensors_to_device(elem, device) for elem in output]
    else:
        raise TypeError(tensors_to_device_err_msg_format.format(output_type))
