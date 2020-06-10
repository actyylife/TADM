import pickle

import torch
from torch import nn
from torch.nn import Parameter

from data_util.dataset import Topology
from data_util.dataset import Example

__all__ = ['DeepIterativeNetwork']


class UpdateLayer(nn.Module):
    def __init__(self, emb_size):
        super(UpdateLayer, self).__init__()

        self.weights = nn.Linear(2 * emb_size, 3 * emb_size)

    def forward(self, V_pre, embedded_info):
        """

        :param V_pre:   Tensor (bz, emb_size)
        :param embedded_info: (bz, emb_size)
        :return:
        """
        v_and_emb_info = torch.cat((V_pre, embedded_info), dim=1)  # Tensor (bz, 2*emb_size)
        gates = self.weights(v_and_emb_info)  # type: torch.Tensor

        keep_gate, update_gate, new_info = gates.chunk(chunks=3, dim=1)  # Tensor (bz, emb_size)
        keep_gate = torch.sigmoid(keep_gate)
        update_gate = torch.sigmoid(update_gate)
        new_info = torch.tanh(new_info)
        V = torch.tanh(V_pre * keep_gate + update_gate * new_info)  # Tensor (bz, emb_size)

        return V


class EmbeddingLayer(nn.Module):
    def __init__(self, emb_size, compute_device=torch.device('cpu')):
        super(EmbeddingLayer, self).__init__()

        self.compute_device = compute_device

        # embedding protectors
        self.omega_0 = nn.Sequential(nn.Linear(emb_size, emb_size),
                                     nn.Tanh())
        self.omega_1 = nn.Sequential(nn.Linear(1, emb_size),
                                     nn.Tanh())
        self.omega_2 = nn.Sequential(nn.Linear(1, emb_size),
                                     nn.Tanh())

        # embedding breakers
        self.omega_3 = nn.Sequential(nn.Linear(emb_size, emb_size),
                                     nn.Tanh())
        self.omega_4 = nn.Sequential(nn.Linear(1, emb_size),
                                     nn.Tanh())

        # embedding neighbors
        self.omega_5 = nn.Sequential(nn.Linear(emb_size, emb_size),
                                     nn.Tanh())

        # embedding info
        self.embedding = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=1,
                                                 kernel_size=1, stride=1, bias=True),
                                       nn.Tanh())

    def forward(self, V_pre, devices, breakers, protector_sate, breaker_state):
        """

        :param V_pre: 特征向量; (num_dev, emb_size)
        :param devices: list
        :param breakers: list
        :param protector_sate: numpy; (num_dev, 3)
        :param breaker_state: numpy; (num_bre,)
        :return:
        """
        embedded_info = list()
        for dev_id, con_breaker_ids in enumerate(devices):
            # embedding breakers
            con_breaker_state = breaker_state[con_breaker_ids]  # numpy (num_nei,)
            breaker_emb = torch.from_numpy(con_breaker_state).unsqueeze(1).float().to(
                self.compute_device)  # Tensor (num_nei, 1)
            breaker_emb = self.omega_4(breaker_emb)  # Tensor (num_nei, emb_size)
            breaker_emb = torch.sum(breaker_emb, dim=0).unsqueeze(0)  # Tensor (1, emb_size,)
            breaker_emb = self.omega_3(breaker_emb)  # Tensor (1, emb_size,)
            breaker_emb = breaker_emb.squeeze(0)  # Tensor (emb_size,)

            # embedding protectors
            protectors = protector_sate[dev_id]  # numpy (3,)
            protector_emb = torch.from_numpy(protectors).unsqueeze(1).float().to(self.compute_device)  # Tensor (3,1)
            protector_emb = self.omega_1(protector_emb)  # Tensor (3, emb_size)

            tmp_breaker_emb = torch.from_numpy(con_breaker_state).unsqueeze(1).float().to(
                self.compute_device)  # Tensor (num_nei, 1)
            tmp_breaker_emb = torch.sum(tmp_breaker_emb, dim=0).unsqueeze(0)  # Tensor (1, 1)
            tmp_breaker_emb = self.omega_2(tmp_breaker_emb)  # Tensor (1, emb_size)
            tmp_breaker_emb = tmp_breaker_emb.repeat(3, 1)  # Tensor (3, emb_size)

            protector_emb = protector_emb + tmp_breaker_emb  # Tensor (3, emb_size)
            protector_emb = torch.sum(protector_emb, dim=0).unsqueeze(0)  # Tensor (1, emb_size)
            protector_emb = self.omega_0(protector_emb)  # Tensor (1, emb_size)
            protector_emb = protector_emb.squeeze(0)  # Tensor (emb_size,)

            # embedding neighbors
            neighbor_ids = []
            for conbreaker_id in con_breaker_ids:
                breaker = breakers[conbreaker_id]
                neighbor_id = breaker[0] if dev_id != breaker[0] else breaker[1]
                neighbor_ids.append(neighbor_id)
            neighbor_emb = V_pre[neighbor_ids]  # Tensor (num_neighbors, emb_size)
            # neighbor_emb = self.omega_7(neighbor_emb)  # Tensor (num_neighbors, emb_size)
            neighbor_emb = torch.sum(neighbor_emb, dim=0).unsqueeze(0)  # Tensor (1,emb_size,)
            neighbor_emb = self.omega_5(neighbor_emb)  # Tensor (1,emb_size)
            neighbor_emb = neighbor_emb.squeeze(0)  # Tensor (emb_size,)

            # embedding info
            emb_info_dev = torch.stack([protector_emb, breaker_emb, neighbor_emb])  # Tensor (3, emb_size)
            emb_info_dev = emb_info_dev.unsqueeze(1).unsqueeze(0)  # Tensor (1,3, 1, emb_size)
            emb_info_dev = self.embedding(emb_info_dev)  # Tensor (1, 1, 1, emb_size)
            emb_info_dev = emb_info_dev.view(-1)  # Tensor (emb_size,)

            embedded_info.append(emb_info_dev)
        embedded_info = torch.stack(embedded_info)

        return embedded_info  # Tensor (num_dev, emb_size)


class EmbeddingGrid(nn.Module):
    def __init__(self, emb_size):
        super(EmbeddingGrid, self).__init__()

        self.embedding = nn.Linear(emb_size, emb_size)

    def forward(self, inputs):
        """

        :param inputs: Tensor (num_dev, emb_size)
        :return:
        """
        grid_emb = torch.sum(inputs, dim=0).unsqueeze(0)  # Tensor (1,emb_size,)
        grid_emb = self.embedding(grid_emb).squeeze(0)  # (emb_size,)

        return grid_emb


class DeepIterativeNetwork(nn.Module):
    def __init__(self,
                 emb_size, T,
                 compute_device=torch.device('cpu')):
        """
        :param emb_size:
        :param T:
        """
        super(DeepIterativeNetwork, self).__init__()

        self.emb_size = emb_size
        self.T = T
        self.compute_device = compute_device

        # embedding layer
        self.embedding_info = EmbeddingLayer(emb_size, compute_device)

        # update layer
        self.update = UpdateLayer(emb_size)

        # embed grid
        self.embedding_grid = EmbeddingGrid(emb_size)

        # initialized V
        self.v0 = Parameter(data=torch.zeros(self.emb_size), requires_grad=False).float()  # Tensor (emb_size,)

    def initialize_V(self, num_dev):
        """

        :param num_dev: (num_device, emb_size)
        :return:
        """
        V_0 = self.v0.unsqueeze(0).repeat(num_dev, 1)  # Tensor (num_dev, emb_size)

        return V_0

    def forward(self, example):
        """
        only support one batch size
        :type example: Example
        :param example:
        :return:
        """
        self.to(self.compute_device)

        # parse the data
        topology = example.topology  # type: Topology
        devices = topology.devices  # list
        breakers = topology.breakers  # list

        # 分离故障信息
        protector_state = example.protector_state  # numpy (num_dev,3)
        breaker_state = example.breaker_state  # numpy (num_bre,)

        # 初始化特征向量
        V = self.initialize_V(len(devices))  # Tensor (num_dev, emb_size)

        # 执行T步迭代更新算法
        for t in range(self.T):
            V = self.forward_step(V, devices, breakers, protector_state, breaker_state)

        grid_emb = self.embedding_grid(V)

        return V, grid_emb  # Tensor (num_dev, emb_size), (emb_size,)

    def forward_step(self, V_pre, devices, breakers, protector_sate, breaker_state):
        """

        :param V_pre: 特征向量; (num_dev, emb_size)
        :param devices: list
        :param breakers: list
        :param protector_sate: numpy; (num_dev, 3)
        :param breaker_state: numpy; (num_bre,)
        :return:
        """
        embedded_info = self.embedding_info(V_pre, devices, breakers, protector_sate, breaker_state)
        V = self.update(V_pre, embedded_info)

        return V  # Tensor (num_dev, emb_size)
