import torch
from torch import nn
from torch.nn import functional as F


class DiagnosisSL(nn.Module):
    def __init__(self, emb_size, compute_device=torch.device('cpu')):
        super(DiagnosisSL, self).__init__()

        self.compute_device = compute_device

        # classifier
        self.omega_10 = nn.Linear(2 * emb_size, 1)

        # process grid emb
        self.omega_11 = nn.Sequential(nn.Linear(emb_size, emb_size),
                                      nn.Tanh())

        # process vector
        self.omega_12 = nn.Sequential(nn.Linear(emb_size, emb_size),
                                      nn.Tanh())

    def forward(self, V, grid_emb):
        """

        :type V: torch.Tensor
        :param V:   (num_dev, emb_size)
        :param grid_emb: (emb_size,)
        :return:
        """
        self.to(self.compute_device)

        num_dev = V.size(0)
        grid_emb = grid_emb.unsqueeze(0).repeat(num_dev, 1)  # Tensor (num_dev, emb_size)
        grid_emb = self.omega_11(grid_emb)  # Tensor (num_dev, emb_size)
        V = self.omega_12(V)    # Tensor (num_dev, emb_size)
        res = self.omega_10(torch.cat((grid_emb, V), 1)).view(-1)    # Tensor (num_dev, 2*emb_size) -> (num_dev,)
        res = torch.sigmoid(res)

        return res
