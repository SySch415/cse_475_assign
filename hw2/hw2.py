import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from torch import optim

plt.show()

sys.path.append("../..")
sys.path.append("..")
import torch
from torch import nn
from torch.utils import data

from my_utils import *


class Model(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.linear_layer = nn.Linear(2, 1)

    def forward(self, x):
        # [TODO] your code
        lin_score = self.linear_layer(x)
        probabilty = torch.sigmoid(lin_score)
        return probabilty


trainloader = data.DataLoader(
    data.TensorDataset(torch.Tensor(1), torch.Tensor(1)), batch_size=1, shuffle=True
)
device = torch.device("cpu")

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)
random.seed(seed)

X, y = sample_gaussian()
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plot_scatter(ax, X, y)

model = Model().to(device)
optimzer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-6)

"""
outp = model(X)
print(outp.shape)
print(outp.min(), outp.max())
print()

print(model.linear_layer.weight)
print(model.linear_layer.bias)
"""
