from torch.utils.data import DataLoader


class DataLoaderBase(DataLoader):
    """ Base class for all data loader """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split_validation(self) -> DataLoader:
        """ Return a `torch.utils.data.DataLoader` for validation, or None if not available. """

        raise NotImplementedError
