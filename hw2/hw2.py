import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from torch import optim

# plt.show()

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
optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-6)

"""
outp = model(X)
print(outp.shape)
print(outp.min(), outp.max())
print()

print(model.linear_layer.weight)
print(model.linear_layer.bias)
"""

criterion = nn.BCELoss()
for itr in range(1, 2001):
    optimizer.zero_grad()
    yh = model(X)  # forward pass
    # compute loss # [TODO] your code
    loss = criterion(yh.squeeze(-1), y.float())
    print(itr, loss.item())
    loss.backward()  # backward pass
    optimizer.step()

with torch.no_grad():
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()

    probability = model(X).squeeze(-1)
    predicted = (probability >= 0.5).long()
    plot_scatter(ax, X, predicted)

    theta0 = model.linear_layer.bias[0].item()
    thetal = model.linear_layer.weight[0, 0].item()
    theta2 = model.linear_layer.weight[0, 1].item()

    x1_min = X[:, 0].min().item()
    x1_max = X[:, 0].max().item()
    x2_min = X[:, 1].min().item()
    x2_max = X[:, 1].max().item()

    if abs(theta2) > 1e-8:
        x1_val = np.linspace(x1_min, x1_max, 100)
        x2_val = -(theta0 + thetal * x1_val) / theta2
        ax.plot(x1_val, x2_val, "m")

    else:
        ax.axvline(-theta0 / thetal, color="m")

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    plt.show()


model2 = Model().to(device)

# print(model2.linear_layer.weight)
# print(model2.linear_layer.bias)


def binary_cross_entropy(y_true, y_pred, epsilon=1e-12):
    """
    Compute the binary cross-entropy loss.

    Parameters:
    y_true (torch array): Ground truth labels (0 or 1).
    y_pred (torch array): Predicted probabilities (between 0 and 1).
    epsilon (float): Small value to prevent log(0). [NOTE] do not change this

    Returns:
    float: Binary cross-entropy loss.
    """
    # Clip predictions to avoid log(0) issues
    y_pred = torch.clip(y_pred, epsilon, 1 - epsilon)

    # Compute BCE loss
    loss = -torch.mean(
        y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
    )
    return loss


learning_rate = 1e-2
for t in range(2000):
    # Forward pass: compute predicted y
    yh = model2(X)

    # Compute and print loss
    loss = binary_cross_entropy(
        y.float(), yh.squeeze(-1)
    )  # np.square(y_pred - y).sum()
    print(loss)

    # [TODO] your code: iteratively update the model parameters
    # The model parameters theta_0, theta_1, theta_2 can be updated individually
    # by (1) computing the gradient using the analytical expression you obtained from the part 1 of the homework
    # by (2) updating the model parameters into the "DESCENDING" direction by the step size (that is, learning rate)
