from typing import List

import click

from nog5 import main
from nog5.utils.setup import load_config
from nog5.utils.logger import setup_logging

config_paths_option = click.option('-c', '--config_paths', required=True, multiple=True,
                                   help=(
                                       'Path to training configuration file. If multiple are provided, runs will be '
                                       'executed in order'
                                   ))
config_path_option = click.option('-c', '--config_path', type=str, required=True, help='Path to model configuration file')
resume_option = click.option('-r', '--resume', type=str, help='path to checkpoint')
log_config_path_option = click.option('-l', '--log_config_path', type=str, help='Path to logging configuration file')
verbose_option = click.option('-v', '--verbose', count=True, help="Stdout logging level. -v for INFO, -vv for DEBUG")


@click.group()
def cli():
    """ CLI for nog5 """
    pass


@cli.command()
@config_paths_option
@resume_option
@log_config_path_option
@verbose_option
def train(config_paths: List[str], resume: str = None, log_config_path: str = None, verbose: int = 0):
    """ Entry point to start training run(s). """
    configs = [load_config(f) for f in config_paths]
    for config in configs:
        setup_logging(config, verbose, log_config_path)
        main.train(config, resume)


@cli.command()
@config_path_option
@click.option('-d', '--model_data', type=str, required=True, help='Path to model data checkpoint')
@log_config_path_option
@verbose_option
def evaluate(config_path: str, model_data: str, log_config_path: str = None, verbose: int = 0):
    """ Entry point to start evaluation run(s). """
    config = load_config(config_path)
    setup_logging(config, verbose, log_config_path)
    main.evaluate(config, model_data)


@cli.command()
@config_path_option
@click.option('-d', '--model_data', type=str, help='Path to model data checkpoint')
@click.option('-i', '--input_paths', multiple=True, required=True, type=str,
              help='Path to input data. If multiple are provided, they will be concatenated in the output')
@click.option('-o', '--output_path', required=True, type=str, help='Path to output data file')
@log_config_path_option
@verbose_option
def predict(config_path: str, model_data: str, input_paths: List[str], output_path: str, log_config_path: str = None,
            verbose: int = 0):
    config = load_config(config_path)
    setup_logging(config, verbose, log_config_path)
    main.predict(config, model_data, input_paths, output_path)


@cli.command()
@config_paths_option
@resume_option
@click.option('-o', '--results_path', required=True, type=str, help='Path to directory where optimization results are saved')
@log_config_path_option
@verbose_option
def optimize(config_paths: List[str], resume: str, results_path: str, log_config_path: str = None, verbose: int = 0):
    """ Entry point to start training run(s). """
    configs = [load_config(f) for f in config_paths]
    for config in configs:
        setup_logging(config, verbose, log_config_path)
        main.hyperparameter_optim(config, resume, results_path)
