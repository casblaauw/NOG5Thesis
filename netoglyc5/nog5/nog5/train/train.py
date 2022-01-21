import torch
import optuna
from torch import Tensor

from nog5.base import TrainerBase, AverageMeter
from nog5.utils.conversion import tensors_to_device
from nog5.utils.logger import setup_logger

log = setup_logger(__name__)


class Trainer(TrainerBase):
    """ Responsible for training loop and validation. """

    def __init__(self, model, loss, metrics, optimizer, start_epoch, config, device, training_dataloader,
                 validation_dataloader=None, data_transform: callable = None, mask_transform: callable = None,
                 target_transform: callable = None, lr_scheduler=None, trial=None):
        super().__init__(model, loss, metrics, optimizer, start_epoch, config, device)
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.do_validation = self.validation_dataloader is not None
        self.lr_scheduler = lr_scheduler
        self.data_transform = data_transform
        self.mask_transform = mask_transform
        self.target_transform = target_transform
        self.trial = trial

    def _train_epoch(self, epoch: int) -> dict:
        """ Training logic for an epoch
        Args:
            epoch: current epoch
        Returns:
            dictionary containing results for the epoch.
        """
        
        self.model.train()

        loss_mtr = AverageMeter('loss')
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]

        n_batches = len(self.training_dataloader)
        for batch_idx, (data, target, mask) in enumerate(self.training_dataloader):
            if self.data_transform:
                data = self.data_transform(data)
            if self.mask_transform:
                mask = self.mask_transform(mask)

            data, target = tensors_to_device(data, self.device), tensors_to_device(target, self.device)

            # Compute prediction and loss
            output = self.model(data, mask)
            loss = self.loss(output, target)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # write results and metrics
            loss_mtr.update(loss.item(), data.size(0))

            if batch_idx % self.log_step == 0 or batch_idx == n_batches - 1:
                self.writer.set_step(epoch * n_batches + batch_idx)
                self.writer.add_scalar('batch/loss', loss.item())
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))
                    self.writer.add_scalar(f'batch/{mtr.name}', value)
                self._log_batch(
                    epoch, batch_idx, self.training_dataloader.batch_size,
                    n_batches, loss.item()
                )

        # cleanup
        del data
        del target
        del output
        torch.cuda.empty_cache()

        # write results
        self.writer.add_scalar('epoch/loss', loss_mtr.avg)
        for mtr in metric_mtrs:
            self.writer.add_scalar(f'epoch/{mtr.name}', mtr.avg)

        results = {
            'loss': loss_mtr.avg,
            'metrics': [mtr.avg for mtr in metric_mtrs]
        }

        if self.do_validation:
            val_results = self._valid_epoch(epoch)
            results = {**results, **val_results}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return results


    def _log_batch(self, epoch: int, batch_idx: int, batch_size: int, n_batches: int, loss: float):
        """ Logging of the batches
        Args:
            epoch: current epoch
            batch_idx: indexes of the batch
            batch_size: size of the batch
            n_batches: length of the data
            loss: training loss of the batch
        """

        n_samples = batch_size * n_batches
        n_complete = batch_idx * batch_size
        fraction = batch_idx / n_batches
        msg = f'Train Epoch: {epoch} [{n_complete:0{len(str(n_samples))}}/{n_samples} ({fraction:06.2%})] Loss: {loss:.6f}'
        log.info(msg)


    def _eval_metrics(self, output: Tensor, target: Tensor) -> float:
        """ Evaluate metrics
        Args:
            output: output from model
            target: labels matching the output
        Returns:
            values from each metric
        """

        with torch.no_grad():
            for metric in self.metrics:
                yield metric(output, target)


    def _valid_epoch(self, epoch: int) -> dict:
        """ Validate after training an epoch
        Args:
            epoch: current epoch
        Returns:
            contains keys 'val_loss' and 'val_metrics'.
        """

        self.model.eval()

        loss_mtr = AverageMeter('loss')
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]

        # loss and metrics of validation data 
        with torch.no_grad():
            for batch_idx, (data, target, mask) in enumerate(self.validation_dataloader):
                if self.data_transform:
                    data = self.data_transform(data)
                if self.mask_transform:
                    mask = self.mask_transform(mask)
                if self.target_transform:
                    target = self.target_transform(target)

                data, target = tensors_to_device(data, self.device), tensors_to_device(target, self.device)

                output = self.model(data, mask)
                loss = self.loss(output, target)

                # update loss
                loss_mtr.update(loss.item(), data.size(0))
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))

        # cleanup
        del data
        del target
        del output
        torch.cuda.empty_cache()

        # write results
        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar('loss', loss_mtr.avg)
        for mtr in metric_mtrs:
            self.writer.add_scalar(mtr.name, mtr.avg)

        if self.trial:
            self.trial.report(loss_mtr.avg, epoch)

            # Handle pruning based on the intermediate value.
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return {
            'val_loss': loss_mtr.avg,
            'val_metrics': [mtr.avg for mtr in metric_mtrs]
        }
