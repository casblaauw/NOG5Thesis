====
**Thesis**: Prediction of protein O-glycosylation using language models
====

The repository contains the source code for the updated version of NetOGlyc, which replaces external tools with embeddings from the pretrained model ESM-1b.


.. contents:: Table of Contents
   :depth: 2

Folder Structure
================

::

  NOG5Thesis/
  │
  ├── netoglyc5/ - model training/prediction library based on NetSurfP-3.0
  │    │
  │    ├── nog5/nog5/
  │    │    │
  │    │    ├── logging.yml - logging configuration
  │    │    ├── cli.py - command line interface
  │    │    ├── main.py - main script to start train/test
  │    │    │
  │    │    ├── base/ - abstract base classes
  │    │    │
  │    │    ├── dataloaders/ - anything about data loading goes here
  │    │    │
  │    │    ├── datasets/ - anything about dataset formats goes here
  │    │    │
  │    │    ├── embeddings/ - folder containing the ESM1b model
  │    │    │
  │    │    ├── models/ - models
  │    │    │
  │    │    ├── output/
  │    │    │    │
  │    │    │    ├── loss/ - loss functions
  │    │    │    │
  │    │    │    ├── metrics/ - performance metric functions
  │    │    │    │
  │    │    │    ├── misc/ - masking etc.
  │    │    │    │
  │    │    │    └── saving/ - prediction output savers
  │    │    │
  │    │    ├── predict/ - predictors
  │    │    │
  │    │    ├── train/ - trainers
  │    │    │
  │    │    ├── eval/ - evaluators
  │    │    │
  │    │    └── utils/ - utilities for logging and tensorboard visualization
  │    │
  │    ├── data/ - directory for storing input data
  │    │
  │    ├── study/ - directory for storing optuna studies
  │    │
  │    ├── models/ - directory for storing pre-trained models
  │    │
  │    └── saved/ - directory for checkpoints, logs and evaluation results
  │
  ├── notebooks/ - directory of notebooks used for data pipeline
  │
  └── glyc_processing - MS-peptide glycosylation data cleaning library


Usage
=====
Start by creating an environment to install the project requirements
.. code-block::

  $ conda env create --file environment_loose.yml
  $ conda activate nog5

Now you can either use the package out of the box

.. code-block::

  $ cd netoglyc5/nog5
  $ pip install .

Or develop further the project. This will create a symbolic link to the package. Changes to the source code will be automatically applied.

.. code-block::

  $ pip install -e .

Training a model based on a experiment configuration (includes evaluating in the end with best model)

.. code-block::

  $ nog5 train -c experiments/config.yml

Predicting, which uses a model and its prediction configuration
.. code-block::

  $ nog5 predict -c config.yml -d model.pth -i example_input.txt -o example_output.txt


Config file format
------------------
Config files are in `.yml` format:

.. code-block:: HTML

	name: CNNTrans_NetOGlyc_NetSurfP
	save_dir: saved/nog5
	seed: 1234
	target_devices: [0]

	arch:
	  type: CNNTrans_NetOGlyc_NetSurfP
	  args:
	    init_n_channels: 1280
	    out_channels: 32
	    cnn_layers: 2
	    kernel_size: [129, 257]
	    padding: [64, 128]
	    n_head: 21
	    dropout: 0.5
	    encoder_layers: 2
	    #embedding_pretrained: "models/esm1b_t33_650M_UR50S.pt"

	dataloaders:
	  training:
	    paths: ["protein_embeddings_netsurfp_output_glyc_labels_max.h5"]
	    type: BasicDataLoader
	    args:
	      batch_size: 16
	      num_workers: 2
	      shuffle: true
	      validation_split: 0.05
	      #training_indices: [0, 1, 2, 3]
	      #validation_indices: [4, 44, 53, 71, 83]
	    dataset:
	      type: H5TrainEvalDataset
	      args:
		truncate_seq_length: 1022
		embedding_features: 1280
		label_names: [ss8, dis, rsa, phi, psi, gly]
		label_sizes: [8, 1, 1, 1, 1, 1]
	  testing:
	    paths: ["protein_embeddings_netsurfp_output_glyc_labels_max.h5"]
	    type: BasicDataLoader
	    args:
	      batch_size: 16
	      num_workers: 2
	      shuffle: false
	      training_indices: [15, 25, 50, 66, 78, 87]
	    dataset:
	      type: H5TrainEvalDataset
	      args:
		truncate_seq_length: 1022
		embedding_features: 1280
		label_names: [ss8, dis, rsa, phi, psi, gly]
		label_sizes: [8, 1, 1, 1, 1, 1]
	  prediction:
	    paths: ["protein_embeddings_netsurfp_output_glyc_labels_max.h5"]
	    type: BasicDataLoader
	    args:
	      batch_size: 8
	      num_workers: 2
	      shuffle: false
	      training_indices: [15, 25, 50, 66, 78, 87, 102]
	    dataset:
	      type: H5PredictionDataset
	      args:
		embedding_features: 1280

	prediction:
	  allow_cuda: True
	  log_step: 50
	  #data_transform:
	    #type: ESM1bTokenize
	  saver:
	    type: H5Saver
	    args:
	      #embedding_features: 1280
	      label_names: [ss8, dis, rsa, phi, psi, gly]
	      label_sizes: [8, 1, 1, 1, 1, 1]
	      #target_is_output_labels: True
	      #data_is_output_embeddings: True
	    labels_transform: multi_task_save_output

	training:
	  early_stop: 3
	  epochs: 100
	  monitor: max val_gly_unambiguous_mcc
	  save_period: 1
	  log_step: 1
	  tensorboard: true

	optimizer:
	  type: AdamW
	  args:
	    lr: 5.0e-05
	    #weight_decay: 1.0e-3
	    #momentum: 0.9

	lr_scheduler:
	  type: null

	multitask_loss:
	#  type: AutomaticWeightedLoss
	  type: WeightedLoss
	  args:
	    loss_names: [ss8_bce, dis_mse, rsa_mse, phi_mse, psi_mse, gly_definite_mse]
	    loss_weights: [1, 1, 1, 1, 1, 2000]
	    loss_args: [{}, {}, {}, {}, {}, {positive_weight: 0.01}]
	
	metrics: [ss8_pcc, ss3_pcc, dis_pcc, rsa_pcc, phi_mae, psi_mae, gly_pcc, gly_definite_mcc, gly_ambiguous_mcc, gly_fpr, gly_fnr]


Add additional configurations if you need.

Using config files
------------------
Modify the configurations in `.yml` config files, then run:

.. code-block::

  $ nsp3 train -c experiments/<config>.yml

Resuming from checkpoints
-------------------------
You can resume from a previously saved checkpoint by:

.. code-block::

  nsp3 train -c experiments/<config>.yml -r path/to/checkpoint.pth

Checkpoints
-----------
You can specify the name of the training session in config files:

.. code-block:: HTML

  "name": "CNNTrans_NetOGlyc_NetSurfP"

The checkpoints will be saved in `save_dir/name/timestamp/checkpoints/`, with timestamp in
YYYY-mmdd-HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:

.. code-block:: python

  checkpoint = {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config,
    'loss': self.loss.state_dict(), # Only if using AutomaticWeightedLoss
  }

Tensorboard Visualization
--------------------------
This template supports `<https://pytorch.org/docs/stable/tensorboard.html>`_ visualization.

1. Run training

    Set `tensorboard` option in config file true.

2. Open tensorboard server

    Type `tensorboard --logdir saved/experiment_name_here/` at the project root, then server will open at
    `http://localhost:6006`

By default, values of loss and metrics specified in config file will be logged.
If you need more visualizations, use `add_scalar('tag', data)`,
`add_image('tag', image)`, etc in the `trainer._train_epoch` method. `add_something()` methods in
this template are basically wrappers for those of `tensorboard.SummaryWriter` module.

**Note**: You don't have to specify current steps, since `TensorboardWriter` class defined at
`utils/visualization.py` will track current steps.
