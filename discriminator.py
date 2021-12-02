import os
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class discriminator(nn.Module):
    def __init__(self, input_size = 32, n_class = 10):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.f2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    def forward(self, input):
        x = F.relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.relu(self.fc4(x))

        return x

