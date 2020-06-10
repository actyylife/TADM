from typing import Optional, Any

from torch import nn
import torch
import torch.nn.functional as F

import pickle

# from models.tsia import TSIA
from models.deep_iterative_network import DeepIterativeNetwork
from models.diagnosis_sl import DiagnosisSL

from data_util.dataset import Example
from data_util.dataset import Topology


class AFDSL(nn.Module):
    def __init__(self,
                 emb_size, T,
                 compute_device=torch.device('cpu')):
        super(AFDSL, self).__init__()

        # self.tsia = TSIA(emb_size, T, compute_device)
        self.din = DeepIterativeNetwork(emb_size, T, compute_device)
        self.diagnosis_sl = DiagnosisSL(emb_size, compute_device)
        self.compute_device = compute_device

        self.initialize_parameters()

    def initialize_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                param.data.normal_(mean=0, std=0.1)

    def forward(self, example):
        """

        :param example:
        :return:
        """
        V, grid_emb = self.din(example)
        predict = self.diagnosis_sl(V, grid_emb)    # (num_dev,)

        return predict


if __name__ == '__main__':
    afd_sl = AFDSL(emb_size=128, T=4)
    
    print('============================================================')
    example_00 = pickle.load(
        open('/data/lf/graduate_projects/TSIA/dataset/data/57_1_10_10_9.example', 'rb')
    )  # type: Example
    res_00 = afd_sl(example_00)
    print(res_00.size())
    label_00 = example_00.device_state  # numpy (num_dev,)
    label_00 = torch.from_numpy(label_00).long()
    loss_00 = F.cross_entropy(res_00, label_00)
    print(loss_00)

    print('============================================================')
    example_01 = pickle.load(
        open('/data/lf/graduate_projects/TSIA/dataset/data/39_1_10_10_67.example', 'rb')
    )
    res_01 = afd_sl(example_01)
    print(res_01.size())
    label_01 = example_01.device_state  # numpy (num_dev,)
    label_01 = torch.from_numpy(label_01).long()
    loss_01 = F.cross_entropy(res_01, label_01)
    print(loss_01)

    print('============================================================')
    total_loss = loss_00 + loss_01
    print(total_loss)
    total_loss.backward()
    print('unit test')
