import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data

sys.path.append("../..")
sys.path.append("..")
sys.path.append("../hw2")
from my_utils import *


class Model(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.linear_layer = nn.Linear(2, 1)

    def forward(self, x):
        y = self.linear_layer(x)
        y = torch.sigmoid(y)
        return y


# dummy trainloader
trainloader = data.DataLoader(
    data.TensorDataset(torch.Tensor(1), torch.Tensor(1)), batch_size=1, shuffle=True
)
device = torch.device("cpu")


seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)
random.seed(seed)

X, y = sample_annuli()
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plot_scatter(ax, X, y)

model = Model().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-6)

for itr in range(1, 1001):
    optimizer.zero_grad()
    yh = model(X)
    loss = nn.BCELoss()(yh.squeeze(), y.float())
    print(itr, loss.item())
    loss.backward()
    optimizer.step()
