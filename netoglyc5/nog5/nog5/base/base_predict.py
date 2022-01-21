import torch
from torch.types import Device

from nog5.base.base_dataloader import DataLoaderBase
from nog5.base.base_model import ModelBase
from nog5.base.base_saver import SaverBase
from nog5.utils.logger import setup_logger

log = setup_logger(__name__)


class PredictorBase:
    """ Base class for all trainers """

    def __init__(self, model: ModelBase, config: dict, device: Device,
                 prediction_dataloader: DataLoaderBase, saver: SaverBase = None):
        """ Constructor
        Args:
            model: model to use for the training
            device: device for the tensors
        """

        # Load model data
        self.model = model
        self.device = device
        self.saver = saver

        self.prediction_dataloader = prediction_dataloader
        self.n_batches = len(self.prediction_dataloader)
        self.batch_size = self.prediction_dataloader.batch_size

        self._setup_monitoring(config)

        self.model.eval()

    def predict(self):
        """ Run prediction """
        raise NotImplementedError

    def _setup_monitoring(self, config: dict) -> None:
        self.log_step = config.get('log_step', 1)

    def log_progress(self, batches_complete):
        percent = batches_complete / self.n_batches
        log.info(f'Prediction progress: [{batches_complete:0{len(str(self.n_batches))}}/{self.n_batches} batches ({percent:06.2%})]')

