import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Q-Network Model with 17 inputs, two 64-unit hidden layers, and 8 outputs."""

    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(17, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, state):
        """Forward pass through the network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN_QNetwork(nn.Module):
    """Unused CNN-style Q-Network (left untouched)"""

    def __init__(self):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Net(nn.Module):
    def __init__(self, net_structure):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(int(net_structure[layer]), int(net_structure[layer+1]))
            for layer in range(len(net_structure) - 1)
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out

    def save_checkpoint(self, path, index=None):
        torch.save(self.state_dict(), os.path.join(path, f'qnetwork_torch_madqn_{index}.pth' if index is not None else 'qnetwork_torch_dqn.pth'))

    def load_checkpoint(self, path, index=None):
        self.load_state_dict(torch.load(os.path.join(path, f'qnetwork_torch_madqn_{index}.pth' if index is not None else 'qnetwork_torch_dqn.pth'), map_location=device))
