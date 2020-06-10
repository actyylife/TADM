import os
import shutil

from torch import nn
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F

from data_util.dataset import Example
from data_util.dataset import Topology
from models.afd_sl import AFDSL

import pickle


class Criterion(nn.Module):
    def __init__(self, beta=1.):
        super(Criterion, self).__init__()
        self.beta = beta

    def forward(self, predict, label):
        """

        :param predict: Tensor (num_dev,)
        :param label: Tensor (num_dev,)
        :return:
        """
        label = label.float()
        loss = -self.beta*label * torch.log(predict) - (1 - label) * torch.log(1 - predict)   # Tensor (num_dev,)

        return torch.sum(loss), predict.size(0)


class SLTrainer(nn.Module):
    def __init__(self,
                 afd_sl,
                 criterion,
                 learning_rate,
                 clip_norm,
                 ckpt_dir,
                 compute_device=torch.device('cpu')):
        """

        :type afd_sl: AFDSL
        """
        super(SLTrainer, self).__init__()

        self.afd_sl = afd_sl
        self.criterion = criterion
        self.lr = learning_rate
        self.clip_norm = clip_norm
        self.ckpt_dir = ckpt_dir

        self.compute_device = compute_device

        self.optimizer = self.initialize_optimizer()

    def forward(self, examples):
        """

        :param examples: list of example, 容量为一个batch_size
        :return:
        """
        dev_count = 0
        total_loss = list()
        for example in examples:  # type: Example
            label = example.device_state  # numpy (num_dev,)
            label = torch.from_numpy(label).long().to(self.compute_device)  # Tensor (num_dev,)

            res = self.afd_sl(example)  # Tensor (num_dev,)

            loss, num_dev = self.criterion(res, label)
            dev_count += num_dev
            total_loss.append(loss)

        total_loss = torch.stack(total_loss)
        total_loss = torch.sum(total_loss)
        avg_loss = total_loss / dev_count  # Tensor ()

        return avg_loss

    def initialize_optimizer(self):
        optimizer = optim.Adam(
            filter(lambda param: param.requires_grad, self.afd_sl.parameters()),
            lr=self.lr
        )

        return optimizer

    def train_step(self, examples):
        """

        :param examples:
        :return:
        """
        self.optimizer.zero_grad()
        loss = self(examples)
        loss.backward()
        clip_grad_norm_(self.afd_sl.parameters(), self.clip_norm)

        self.optimizer.step()

        return loss

    def save(self, filename='afd_sl.pth', save_optimizer=True):
        """

        :param filename:
        :param save_optimizer:
        :return:
        """
        # print('\nSaving parameters...')

        save_dict = dict()

        save_dict['model'] = self.afd_sl.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if not filename:
            filename = 'afd_sl.pth'
        filename = os.path.join(self.ckpt_dir, filename)

        if not os.path.exists(self.ckpt_dir):
            try:
                os.mkdir(self.ckpt_dir)
            except FileNotFoundError:
                os.mkdir(os.path.dirname(self.ckpt_dir))
                os.mkdir(self.ckpt_dir)

        path_dir = os.path.dirname(filename)
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)

        torch.save(save_dict, filename)

        return filename

    def load(self, filename='afd_sl.pth', load_optimizer=True):
        filename = os.path.join(self.ckpt_dir, filename)
        if os.path.exists(filename):
            print('\nLoading parameters from checkpoint')
            state_dict = torch.load(filename)
            if 'model' in state_dict:
                self.afd_sl.load_state_dict(state_dict['model'])
            else:
                self.afd_sl.load_state_dict(state_dict)

            if 'optimizer' in state_dict and load_optimizer:
                self.optimizer.load_state_dict(state_dict['optimizer'])

        return self


if __name__ == '__main__':
    example_00 = pickle.load(
        open('/data/lf/graduate_projects/TSIA/dataset/data/10_1_10_10_23.example', 'rb')
    )  # type: Example
    examples = [example_00]

    afd_sl = AFDSL(emb_size=128, T=4)
    criterion = nn.CrossEntropyLoss()
    sl_trainer = SLTrainer(afd_sl=afd_sl,
                           criterion=criterion,
                           learning_rate=0.001,
                           clip_norm=5.0,
                           ckpt_dir=''
                           )

    loss = sl_trainer.train_step(examples)
    print(loss.item())
