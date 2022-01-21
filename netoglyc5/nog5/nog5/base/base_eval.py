from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import Tensor

from nog5.base.base_model import ModelBase
from nog5.base.base_dataloader import DataLoaderBase
from nog5.utils.logger import setup_logger
from nog5.utils.paths import trainer_paths
from nog5.utils.setup import load_model_data_loosely, load_best_model_data

log = setup_logger(__name__)


class EvaluatorBase:
    """ Base class for all evaluators """

    def __init__(self, model: ModelBase, metrics: list, checkpoint_dir: Path, writer_dir: Path, config: dict,
                 device: torch.device, testing_dataloader: DataLoaderBase, dataset_path: str):
        """ Constructor
        Args:
            model: model to use for the evaluation
            metrics: list with the metrics
            checkpoint_dir: directory of the checkpoints
            writer_dir: directory to write results
            config: loaded configuration file
            device: device for the tensors
        """

        self.model = model
        self.metrics = metrics
        self.device = device
        self.testing_dataloader = testing_dataloader
        self.dataset_path = dataset_path
        self.config = config
        self.writer_dir = writer_dir
        self.checkpoint_dir = checkpoint_dir
        self.evaluations = {}

        # Save configuration file into checkpoint directory:
        with open(self.checkpoint_dir / 'config.yml', 'w') as handle:
            yaml.dump(config, handle, default_flow_style=False)

    def evaluate(self):
        """ Full evaluation logic """

        log.info("Starting evaluating...")
        for _ in range(1):
            result = self._evaluate_epoch()

            # save logged informations into log dict
            for key, value in result.items():
                if key == self.evaluations.keys():
                    self.evaluations[key].update(value.avg)
                else:
                    self.evaluations[key] = value

        # write metrics to log
        log.info(f"Evaluation result - {self.dataset_path}:")
        for metric, value in self.evaluations.items():
            log.info("{}: {}".format(metric, float(value)))

        self._write_test()

    def _eval_metrics(self, output: Tensor, target: Tensor) -> float:
        """ Evaluation of metrics
        Args:
            output: tensor with output values from the model
            target: tensor with target values
        """

        with torch.no_grad():
            for metric in self.metrics:
                yield metric(output, target)

    def _evaluate_epoch(self) -> dict:
        """ Evaluation logic for the single epoch. """
        raise NotImplementedError

    def _write_test(self) -> dict:
        """ Write finished evaluation """
        raise NotImplementedError
