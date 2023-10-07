import os
import copy
import pytorch_lightning as pl
import datetime
import wandb

from pytorch_lightning.loggers import WandbLogger
from model.model_interface import ModelInterface
from dataset.data_interface import DataInterface
from pytorch_lightning.strategies import DDPStrategy


def load_wandb(config):
    # initialize wandb
    wandb_config = config.setting.wandb_config
    wandb_logger = WandbLogger(project=wandb_config.project, config=config,
                               name=wandb_config.name,
                               settings=wandb.Settings(start_method='fork'))
    
    return wandb_logger


def load_model(config):
    # initialize model
    model_config = copy.deepcopy(config)
    kwargs = model_config.pop('kwargs')
    model_config.update(kwargs)
    return ModelInterface.init_model(**model_config)


def load_dataset(config):
    # initialize dataset
    dataset_config = copy.deepcopy(config)
    kwargs = dataset_config.pop('kwargs')
    dataset_config.update(kwargs)
    return DataInterface.init_dataset(**dataset_config)


# def load_plugins():
#     config = get_config()
#     # initialize plugins
#     plugins = []
#
#     if "Trainer_plugin" not in config.keys():
#         return plugins
#
#     if not config.Trainer.logger:
#         if hasattr(config.Trainer_plugin, "LearningRateMonitor"):
#             config.Trainer_plugin.pop("LearningRateMonitor", None)
#
#     if not config.Trainer.enable_checkpointing:
#         if hasattr(config.Trainer_plugin, "ModelCheckpoint"):
#             config.Trainer_plugin.pop("ModelCheckpoint", None)
#
#     for plugin, kwargs in config.Trainer_plugin.items():
#         plugins.append(eval(plugin)(**kwargs))
#
#     return plugins


# Initialize strategy
def load_strategy(config):
    config = copy.deepcopy(config)
    if "timeout" in config.keys():
        timeout = int(config.pop('timeout'))
        config["timeout"] = datetime.timedelta(seconds=timeout)

    return DDPStrategy(**config)


# Initialize a pytorch lightning trainer
def load_trainer(config):
    trainer_config = copy.deepcopy(config.Trainer)
    
    # Initialize wandb
    if trainer_config.logger:
        trainer_config.logger = load_wandb(config)
    else:
        trainer_config.logger = False

    # Initialize plugins
    # plugins = load_plugins()
    
    # Initialize strategy
    strategy = load_strategy(trainer_config.pop('strategy'))
    return pl.Trainer(**trainer_config, strategy=strategy, callbacks=[])
