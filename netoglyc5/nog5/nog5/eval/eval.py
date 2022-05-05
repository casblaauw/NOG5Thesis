from pathlib import Path
from typing import Optional
import gc

import torch
from torch import Tensor

from nog5.base import EvaluatorBase, AverageMeter, ModelBase, DataLoaderBase
from nog5.utils.conversion import tensors_to_device


class Evaluator(EvaluatorBase):
    """ Responsible for test evaluation and the metrics. """

    def __init__(self, model: ModelBase, metrics: list, config: dict, device: torch.device,
                 testing_dataloader: DataLoaderBase, dataset_path: str, writer_dir: Path, checkpoint_dir: Path,
                 data_transform: callable = None, mask_transform: callable = None, target_transform: callable = None):
        super().__init__(model, metrics, checkpoint_dir, writer_dir, config, device, testing_dataloader, dataset_path)
        """ Constructor
        Args:
            model: model to use for the evaluation
            metrics: list of the metrics
            checkpoint_dir: directory of the checkpoints to save config and get best model if available
            writer_dir: directory to write evaluation results
            device: device for the tensors
            test_data_loader: list Dataloader containing the test data
        """

        self.data_transform = data_transform
        self.mask_transform = mask_transform
        self.target_transform = target_transform
    
    def _evaluate_epoch(self) -> dict:
        """ Evaluation of test """

        self.model.eval()

        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]
        # get test evaluation from metrics
        with torch.no_grad():
            for batch_idx, (data, target, mask) in enumerate(self.testing_dataloader):
                if self.data_transform:
                    data = self.data_transform(data)
                if self.mask_transform:
                    mask = self.mask_transform(mask)
                if self.target_transform:
                    target = self.target_transform(target)

                data, target, mask = tensors_to_device(data, self.device), tensors_to_device(target, self.device), tensors_to_device(mask, self.device)

                output = self.model(data, mask)
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))

        # cleanup
        del data
        del target
        del output
        torch.cuda.empty_cache()
        gc.collect()

        # return results
        results = {}
        for mtr in metric_mtrs:
            results[mtr.name] = mtr.avg

        return results

    def _write_test(self):
        """ Write test results """

        with open(self.writer_dir / "evaluation_results.txt", "a") as evalf:
            evalf.write(self.dataset_path + "\n")
            for metric, value in self.evaluations.items():
                evalf.write("{}: {}\n".format(metric, value))
