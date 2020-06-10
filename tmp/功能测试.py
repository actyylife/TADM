import numpy as np
import torch

a = np.array([[1, 2],
              [3, 4]])

tmp = torch.from_numpy(a)
print(torch.sum(tmp, dim=0).size())

b = np.array([1, 2])
b = torch.from_numpy(b)
b = b.unsqueeze(0)
print(b.size())
b = b.repeat(2, 1)

print(b.view(-1).size())

tmp = torch.zeros(2)
print(tmp)

tmp = []
print(not tmp)

a = np.array([1, 2, 3])
b = torch.from_numpy(a)
print((a > 2).astype(np.int))
c = (b > 2).long()
print(c)
print(b * c)

b = np.array([1, 2])
b = torch.from_numpy(b)
c = b.size(0)
print(type(c))

a = [1, 2, 3, 4, 5]
print(a[2:20])
