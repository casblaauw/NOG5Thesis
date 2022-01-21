import torch
from torch import Tensor

from nog5.base import ModelBase, SaverBase
from nog5.base.base_predict import PredictorBase
from nog5.utils.logger import setup_logger
from nog5.utils.conversion import concat_list, tensors_to_device

log = setup_logger(__name__)


class Predictor(PredictorBase):
    """ Responsible for training loop and validation. """

    def __init__(self, model: ModelBase, prediction_dataloader, config: dict, device: torch.device,
                 saver: SaverBase = None, data_transform: callable = None, mask_transform: callable = None,
                 target_transform: callable = None, concat_transform: callable = None,
                 tensors_to_device_transform: callable = None):
        super().__init__(model, config, device, prediction_dataloader, saver)

        self.data_transform = data_transform
        self.mask_transform = mask_transform
        self.target_transform = target_transform
        self.output_list_transform = concat_transform
        if self.output_list_transform is None:
            self.concat_transform = concat_list
        self.tensors_to_device_transform = tensors_to_device_transform
        if self.tensors_to_device_transform is None:
            self.tensors_to_device_transform = tensors_to_device

    def predict(self):
        all_identifiers, all_sequences, all_targets, all_outputs = [], [], [], []

        with torch.no_grad():
            for batch_idx, (identifiers, sequences, data, target, mask) in enumerate(self.prediction_dataloader):
                if self.data_transform:
                    data = self.data_transform(data)
                if self.mask_transform:
                    mask = self.mask_transform(mask)
                if self.target_transform:
                    target = self.target_transform(target)

                data_ondevice = tensors_to_device(data, self.device)
                output = self.model(data_ondevice, mask)
                output = self.tensors_to_device_transform(output, torch.device('cpu'))

                if self.saver is not None:
                    self.saver.write(identifiers, sequences, data, target, mask, output)
                else:
                    all_identifiers.extend(identifiers)
                    all_sequences.extend(sequences)
                    all_targets.extend(target)
                    all_outputs.append(output)

                if batch_idx % self.log_step == 0 or batch_idx == self.n_batches - 1:
                    self.log_progress((batch_idx + 1))

        if self.saver is not None:
            self.saver.close()
        else:
            all_targets = self.concat_transform(all_targets)
            all_outputs = self.concat_transform(all_outputs)
            return all_identifiers, all_sequences, all_targets, all_outputs
