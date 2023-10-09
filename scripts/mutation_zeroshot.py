import sys
sys.path.append('.')
import yaml
import argparse

from tqdm import tqdm
from easydict import EasyDict
from utils.others import setup_seed
from utils.module_loader import *


def run(config):
    # Initialize a model
    model = load_model(config.model)

    # Initialize a dataset
    data_module = load_dataset(config.dataset)

    # Initialize a trainer
    trainer = load_trainer(config)

    # Record results
    if config.setting.os_environ.NODE_RANK == 0 and config.setting.out_path is not None:
        out_path = config.setting.out_path
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        w = open(config.setting.out_path, 'w')
        w.write("dataset\tspearman\n")

    # Save logs for ClinVar benchmark
    if config.model.kwargs.get("log_dir", None) is not None:
        os.makedirs(config.model.kwargs.log_dir, exist_ok=True)

    for name in tqdm(os.listdir(config.setting.dataset_dir)):
        print(name)
        path = os.path.join(config.setting.dataset_dir, name)
        data_module.test_lmdb = path
        result = trainer.test(model=model, datamodule=data_module)
        spearman = result[0]['spearman']

        if config.setting.os_environ.NODE_RANK == 0 and config.setting.out_path is not None:
            w.write(f"{name}\t{spearman:.4f}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="Running configurations", type=str, required=True)
    return parser.parse_args()


def main():
    with open(args.config, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))

    if config.setting.seed:
        setup_seed(config.setting.seed)

    # set os environment variables
    for k, v in config.setting.os_environ.items():
        if v is not None and k not in os.environ:
            os.environ[k] = str(v)

    run(config)


if __name__ == '__main__':
    args = get_args()
    main()