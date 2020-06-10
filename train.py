import fire

from configurations.configuration import config
from tasks.fault_diagnosis import fault_diagnosis_train
from data_util.dataset import Example
from data_util.dataset import Topology


def train(**kwargs):
    config.parse_args(kwargs)
    fault_diagnosis_train(config=config,
                          save_dir='experiment_3/train_data_g',
                          filename='train_data_g_filenames.txt')


if __name__ == '__main__':
    fire.Fire(train)
