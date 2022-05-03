from typing import List

import torch
# from torch import nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler
import optuna
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

import nog5.dataloaders.augmentation as module_loadaug
# import nog5.datasets.traineval as module_traineval
# import nog5.datasets.prediction as module_pred
import nog5.dataloaders.datasets as module_datasets
import nog5.dataloaders as module_load
import nog5.output.loss.multitask_losses as module_multiloss
import nog5.output.metrics as module_metric
import nog5.output.saving.augmentation as module_saveaug
import nog5.output.saving.savers as module_save
import nog5.models as module_arch
import nog5.embeddings as module_embed
from nog5.base.base_parameterizedloss import ParameterizedLossBase
from nog5.train import Trainer
from nog5.eval import Evaluator
from nog5.predict import Predictor
from nog5.utils.logger import setup_logger
from nog5.utils.paths import optim_results_path, trainer_paths
from nog5.utils.setup import setup_device, resume_checkpoint, get_instance, seed_everything, \
    load_model_data_loosely, load_best_model_data

log = setup_logger(__name__)


def train(cfg: dict, resume: str = None):
    """ Loads configuration and trains and evaluates a model
    args:
        cfg: dictionary containing the configuration of the experiment
        resume: path to previous resumed model
    """
    log.debug(f'Training: {cfg}')
    seed_everything(cfg['seed'])
    torch.backends.cudnn.benchmark = False  # disable if not consistent input sizes

    log.info('Loading model, optimizer, loss and metrics')
    model = get_instance(module_arch, 'arch', cfg)
    model, device = setup_device(model, cfg['target_devices'])
    optimizer = get_instance(module_optimizer, 'optimizer', cfg, params=[{'params': model.trainable_parameters()}])
    loss = get_instance(module_multiloss, 'multitask_loss', cfg)
    metrics = [getattr(module_metric, met) for met in cfg['metrics']]

    param_groups = []
    if isinstance(loss, ParameterizedLossBase):
        param_groups.append(loss.get_param_group(optimizer))
    if len(param_groups) > 0:
        log.info('Loading additional optimizer parameter groups')
        for param_group in param_groups:
            optimizer.add_param_group(param_group)

    model, optimizer, loss, start_epoch = resume_checkpoint(resume, model, optimizer, loss, cfg, device)

    log.info('Loading scheduler if specified')
    lr_scheduler = get_instance(module_scheduler, 'lr_scheduler', cfg, optimizer=optimizer)

    log.info("Loading training dataset(s)")
    training_dataloader = _get_dataloader(module_datasets, 'training', cfg)

    if 'validation' in cfg['dataloaders']:
        log.info("Loading validation dataset(s)")
        validation_dataloader = _get_dataloader(module_datasets, 'validation', cfg)
    else:
        log.info(f"Split training dataset for training and validation")
        validation_dataloader = training_dataloader.get_validation_dataloader()

    if 'testing' in cfg['dataloaders']:
        log.info("Loading testing dataset(s)")
        testing_dataloaders = _get_dataloaders(module_datasets, 'testing', cfg)
    else:
        log.warning("No testing datasets were included, skipping testing")
        testing_dataloaders = []

    log.info("Loading transforms if specified")
    training_data_transform = get_transform('data_transform', cfg['training'])
    training_mask_transform = get_transform('mask_transform', cfg['training'])
    training_target_transform = get_transform('target_transform', cfg['training'])

    if 'testing' in cfg:
        testing_data_transform = get_transform('data_transform', cfg['testing'])
        testing_mask_transform = get_transform('mask_transform', cfg['testing'])
        testing_target_transform = get_transform('target_transform', cfg['testing'])
    else:
        log.warning("No testing section in config, which is fine if no transforms are needed for test data")
        testing_data_transform, testing_mask_transform, testing_target_transform = None, None, None

    log.info('Initialising trainer')
    trainer = Trainer(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        start_epoch=start_epoch,
        config=cfg,
        device=device,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        data_transform=training_data_transform,
        mask_transform=training_mask_transform,
        target_transform=training_target_transform,
        lr_scheduler=lr_scheduler
    )

    trainer.train()

    log.info('Initialising evaluation')
    # try to load the best model checkpoint from the experiment if available
    load_best_model_data(model, trainer.checkpoint_dir, device)

    for dataset_path, testing_dataloader in testing_dataloaders:
        evaluator = Evaluator(
            model=model,
            metrics=metrics,
            config=cfg,
            device=device,
            dataset_path=dataset_path,
            testing_dataloader=testing_dataloader,
            checkpoint_dir=trainer.checkpoint_dir,
            data_transform=testing_data_transform,
            mask_transform=testing_mask_transform,
            target_transform=testing_target_transform,
            writer_dir=trainer.writer_dir
        )

        evaluator.evaluate()

    log.info('Finished!')


def evaluate(cfg: dict, model_data: str):
    """ Loads configuration and trains and evaluates a model
    args:
        cfg: dictionary containing the configuration of the experiment
        model_data: path to trained model
    """
    log.debug(f'Evaluating: {cfg}')
    seed_everything(cfg['seed'])
    torch.backends.cudnn.benchmark = False  # disable if not consistent input sizes

    log.info('Loading model and metrics')
    model = get_instance(module_arch, 'arch', cfg)
    model, device = setup_device(model, cfg['target_devices'])
    metrics = [getattr(module_metric, met) for met in cfg['metrics']]
    load_model_data_loosely(model, model_data, device)

    log.info("Loading testing dataset(s)")
    testing_dataloaders = _get_dataloaders(module_datasets, 'testing', cfg)

    if 'testing' in cfg:
        testing_data_transform = get_transform('data_transform', cfg['testing'])
        testing_mask_transform = get_transform('mask_transform', cfg['testing'])
        testing_target_transform = get_transform('target_transform', cfg['testing'])
    else:
        log.warning("No testing section in config, which is fine if no transforms are needed for test data")
        testing_data_transform, testing_mask_transform, testing_target_transform = None, None, None

    log.info('Initialising evaluation')

    checkpoint_dir, writer_dir = trainer_paths(cfg)

    for dataset_path, testing_dataloader in testing_dataloaders:
        evaluator = Evaluator(
            model=model,
            metrics=metrics,
            config=cfg,
            device=device,
            dataset_path=dataset_path,
            testing_dataloader=testing_dataloader,
            checkpoint_dir=checkpoint_dir,
            writer_dir=writer_dir,
            data_transform=testing_data_transform,
            mask_transform=testing_mask_transform,
            target_transform=testing_target_transform,
        )

        evaluator.evaluate()

    log.info('Finished!')


def predict(cfg: dict, model_data: str, input_paths: List[str], output_path: str = None):
    """ Predict using trained model and file or string input
    Args:
        cfg: configuration of model
        model_data: path to trained model
        input_paths: file paths of prediction input, if multiple they are concatenated
        output_path: file path of prediction output, output is returned if not specified
    """
    log.debug(f'Predicting: {cfg}')
    torch.backends.cudnn.benchmark = False  # disable if not consistent input sizes

    # Only allow GPU inference if specified
    if cfg['prediction'].get('allow_cuda', False) is False:
        cfg['target_devices'] = []

    # override any pretrained model from config
    if "embedding_pretrained" in cfg['arch']['args']:
        cfg['arch']['args'].pop("embedding_pretrained")

    log.info("Loading model")
    # allow loading both models and embeddings for prediction
    if cfg['arch'].get('embedding', False) is True:
        model = get_instance(module_embed, 'arch', cfg)
    else:
        model = get_instance(module_arch, 'arch', cfg)
    model, device = setup_device(model, cfg['target_devices'])
    if model_data is not None:
        load_model_data_loosely(model, model_data, device)

    log.info("Loading prediction dataset(s)")
    # override any input files in config for prediction
    cfg['dataloaders']['prediction']['paths'] = input_paths
    prediction_dataloader = _get_dataloader(module_pred, 'prediction', cfg)

    log.info("Loading transforms if specified")
    if output_path is not None:
        if 'labels_transform' in cfg['prediction']['saver']:
            saver_labels_transform = getattr(module_saveaug, cfg['prediction']['saver']['labels_transform'])
        else:
            saver_labels_transform = None
        saver = get_instance(module_save, 'saver', cfg['prediction'], output_path=output_path, labels_transform=saver_labels_transform)
    else:
        saver = None

    predictor_data_transform = get_transform('data_transform', cfg['prediction'])
    predictor_mask_transform = get_transform('mask_transform', cfg['prediction'])
    predictor_target_transform = get_transform('target_transform', cfg['prediction'])
    predictor_concat_transform = get_transform('concat_transform', cfg['prediction'])
    predictor_tensors_to_device_transform = get_transform('tensors_to_device_transform', cfg['prediction'])

    log.info('Initialising predictor')
    predictor = Predictor(
        model=model,
        prediction_dataloader=prediction_dataloader,
        config=cfg['prediction'],
        device=device,
        saver=saver,
        data_transform=predictor_data_transform,
        mask_transform=predictor_mask_transform,
        target_transform=predictor_target_transform,
        concat_transform=predictor_concat_transform,
        tensors_to_device_transform=predictor_tensors_to_device_transform,
    )

    result = predictor.predict()
    log.info('Finished!')
    return result


def hyperparameter_optim(cfg: dict, resume: str, results_path: str):
    """ Loads configuration and hyperparameter optimizes the learning rate
    args:
        cfg: dictionary containing the configuration of the experiment
        resume: path to a previous resumed model
    """
    # create results directory and study
    results_path = optim_results_path(results_path, cfg)
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        cfg['training']['optimizer']['args']['lr'] = lr

        log.debug(f'Training w. Hyperparameters: {cfg}')
        seed_everything(cfg['seed'])
        torch.backends.cudnn.benchmark = False  # disable if not consistent input sizes

        log.info('Loading model, optimizer, loss and metrics')
        model = get_instance(module_arch, 'arch', cfg)
        model, device = setup_device(model, cfg['target_devices'])
        optimizer = get_instance(module_optimizer, 'optimizer', cfg)
        loss = get_instance(module_multiloss, 'multitask_loss', cfg)
        metrics = [getattr(module_metric, met) for met in cfg['metrics']]

        log.info('Loading optimizer parameter groups')
        param_groups = setup_param_groups(model, loss, optimizer)
        for param_group in param_groups:
            optimizer.add_param_group(param_group)

        model, optimizer, loss, start_epoch = resume_checkpoint(resume, model, optimizer, loss, cfg, device)

        log.info('Loading scheduler if specified')
        lr_scheduler = get_instance(module_scheduler, 'lr_scheduler', cfg, optimizer=optimizer)

        log.info("Loading training dataset(s)")
        training_dataloader = _get_dataloader(module_traineval, 'training', cfg)

        if 'validation' in cfg['dataloaders']:
            log.info("Loading validation dataset(s)")
            validation_dataloader = _get_dataloader(module_traineval, 'validation', cfg)
        else:
            log.info(f"Split training dataset for training and validation")
            validation_dataloader = training_dataloader.get_validation_dataloader()

        log.info("Loading transforms if specified")
        training_data_transform = get_transform('data_transform', cfg['training'])
        training_mask_transform = get_transform('mask_transform', cfg['training'])
        training_target_transform = get_transform('target_transform', cfg['training'])

        log.info('Initialising trainer')
        trainer = Trainer(
            model=model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            start_epoch=start_epoch,
            config=cfg,
            device=device,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            data_transform=training_data_transform,
            mask_transform=training_mask_transform,
            target_transform=training_target_transform,
            lr_scheduler=lr_scheduler
        )

        results = trainer.train()
        return results['val_loss']

    study.optimize(objective, n_trials=10, timeout=None)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    log.info("Study statistics:")
    log.info("Number of finished trials: {}".format(len(study.trials)))
    log.info("Number of pruned trials: {}".format(len(pruned_trials)))
    log.info("Number of complete trials: {}".format(len(complete_trials)))

    log.info("Best trial:")
    trial = study.best_trial

    log.info("Value: {}".format(trial.value))

    log.info("Params:")
    for key, value in trial.params.items():
        log.info("{}: {}".format(key, value))

    # Plot optimization history
    plt.style.use(['science', 'ieee'])
    plt.rcParams["figure.figsize"] = (3, 2)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("")
    plt.xlabel("$trials$")
    plt.ylabel("$Objective$ $value$")
    plt.savefig(str(results_path / "optimization_history.png"))


def _get_dataloader(module_data, dataloader_name: str, cfg: dict):
    embedding_path = cfg['dataloaders'][dataloader_name]['embedding_path']
    annotation_path = cfg['dataloaders'][dataloader_name]['annotation_path']

    if len(embedding_path) > 1 and not isinstance(embedding_path, str):
        datasets = []
        for epath, apath in zip(embedding_path, annotation_path):
            datasets.append(get_instance(module_data, 'dataset', cfg['dataloaders'][dataloader_name], dataset_path = epath, info = apath))
        dataset = ConcatDataset(datasets)
    elif len(embedding_path) == 1 and not isinstance(embedding_path, str):
        dataset = get_instance(module_data, 'dataset', cfg['dataloaders'][dataloader_name], dataset_path=embedding_path[0], info=annotation_path[0])
    else:
        dataset = get_instance(module_data, 'dataset', cfg['dataloaders'][dataloader_name], dataset_path=embedding_path, info=annotation_path)
        
    return get_instance(module_load, dataloader_name, cfg['dataloaders'], dataset=dataset)


def _get_dataloaders(module_data, dataloader_name: str, cfg: dict):
    embedding_path = cfg['dataloaders'][dataloader_name]['embedding_path']
    annotation_path = cfg['dataloaders'][dataloader_name]['annotation_path']

    dataloaders = []
    for epath, apath in zip(embedding_path, annotation_path):
        dataset = get_instance(module_data, 'dataset', cfg['dataloaders'][dataloader_name], dataset_path=epath, info=apath)
        dataloaders.append((epath, get_instance(module_load, dataloader_name, cfg['dataloaders'], dataset=dataset)))
    return dataloaders


def get_transform(transform_name: str, cfg_section: dict):
    if transform_name in cfg_section:
        return get_instance(module_loadaug, transform_name, cfg_section)
    return None

# def setup_param_groups(model: nn.Module, config: dict) -> list:
#     """ Setup model parameters
#     Args:
#         model: pytorch model
#         config: configuration containing params
#     Returns:
#         list with model parameters
#     """
#     return [{'params': model.parameters(), **config}]