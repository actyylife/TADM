import fire

from configurations.configuration import config
from tasks.fault_diagnosis import fault_diagnosis_infer
from data_util.dataset import Example
from data_util.dataset import Topology


def infer(**kwargs):
    config.parse_args(kwargs)
    fault_diagnosis_infer(config=config,
                          save_dir='experiment_3/test_data_g',
                          filename='test_data_g_filenames.txt')


if __name__ == '__main__':
    fire.Fire(infer)
