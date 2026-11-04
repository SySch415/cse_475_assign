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

with torch.no_grad():
    fig = plt.figure(figsize=(4, 4))
    axes = []
    axes.append(fig.add_subplot(1, 1, 1))
    xs, ys = X, y
    y_pred = model(xs)
    print(y_pred.shape)
    label = (y_pred[:, 0] >= 0.5).long()
    print(label)
    colors = ["lime", "tomato"]
    for i in range(512):
        axes[0].scatter(xs[i, 0], xs[i, 1], c=colors[label[i]], edgecolor="none", s=30)
        axes[0].scatter(
            xs[i + 512 :, 0],
            xs[i + 512 :, 1],
            c=colors[label[i + 512]],
            edgecolor="none",
            s=30,
        )
    axes[0].set_xlim(xs[:, 0].min(), xs[:, 0].max())
    axes[0].set_ylim(xs[:, 1].min(), xs[:, 1].max())
    plt.show()

    print(xs.shape)
    err = torch.sum(torch.abs(label - y))
    print(err)


class Model(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.linear_layer = nn.Linear(2, 1)  # [TODO] define a linear layer

    def forward(self, x):
        # [TODO] your code
        y = self.linear_layer(x**2)
        y = torch.sigmoid(y)
        return y


model = Model().to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-6)

for itr in range(1, 1001):
    optimizer.zero_grad()
    yh = model(X)
    loss = nn.BCELoss()(yh.squeeze(), y.float())
    print(itr, loss.item())
    loss.backward()
    optimizer.step()

with torch.no_grad():
    fig = plt.figure(figsize=(4, 4))
    axes = []
    axes.append(fig.add_subplot(1, 1, 1))
    xs, ys = X, y
    y_pred = model(xs)
    print(y_pred.shape)
    label = (y_pred[:, 0] >= 0.5).long()
    print(label)
    colors = ["lime", "tomato"]
    for i in range(512):
        axes[0].scatter(xs[i, 0], xs[i, 1], c=colors[label[i]], edgecolor="none", s=30)
        axes[0].scatter(
            xs[i + 512 :, 0],
            xs[i + 512 :, 1],
            c=colors[label[i + 512]],
            edgecolor="none",
            s=30,
        )
    axes[0].set_xlim(xs[:, 0].min(), xs[:, 0].max())
    axes[0].set_ylim(xs[:, 1].min(), xs[:, 1].max())
    plt.show()

    print(xs.shape)
    err = torch.sum(torch.abs(label - y))
    print(err)

# spiral

X, y = sample_spiral()
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plot_scatter(ax, X, y)


class Model(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()  # my ruff linter prefers super() over super(Model, self)

        # [TODO] your code
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        y = self.net(x)
        y = torch.sigmoid(y)
        return y


model = Model().to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-6)

for itr in range(1, 1001):
    optimizer.zero_grad()
    yh = model(X)
    loss = nn.BCELoss()(yh.squeeze(), y.float())
    print(itr, loss.item())
    loss.backward()
    optimizer.step()

with torch.no_grad():
    fig = plt.figure(figsize=(4, 4))
    axes = []
    axes.append(fig.add_subplot(1, 1, 1))  # ,sharex=True,sharey=True))
    xs, ys = X, y  # sample_gaussian(n_samples=200) ; s = torch.linspace(0, 1, 10)
    y_pred = model(xs)
    print(y_pred.shape)
    label = (y_pred[:, 0] >= 0.5).long()
    print(label)
    colors = ["lime", "tomato"]
    for i in range(1024):
        axes[0].scatter(xs[i, 0], xs[i, 1], c=colors[label[i]], edgecolor="none", s=30)
        axes[0].scatter(
            xs[i + 1024 :, 0],
            xs[i + 1024 :, 1],
            c=colors[label[i + 1024]],
            edgecolor="none",
            s=30,
        )
    axes[0].set_xlim(xs[:, 0].min(), xs[:, 0].max())
    axes[0].set_ylim(xs[:, 1].min(), xs[:, 1].max())
    plt.show()

    print(xs.shape)
    err = torch.sum(torch.abs(label - y))
    print(err)

# sigmoid

with torch.no_grad():
    out = model.net(X).squeeze()
    pred = torch.sigmoid(out)

    colors = ["green", "red"]
    plot_colors = []
    for label in y:
        plot_colors.append(colors[int(label)])

    plt.figure(figsize=(6, 4))
    plt.scatter(out, pred, c=plot_colors, s=20)
    s = torch.linspace(out.min() - 2, out.max() + 2, 200)
    plt.plot(s, torch.sigmoid(s), "k-")

    plt.show()
