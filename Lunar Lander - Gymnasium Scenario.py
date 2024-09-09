##!pip install gymnasium
##!pip install "gymnasium[atari, accept-rom-license]"
##!apt-get install -y swig
##!pip install gymnasium[box2d]

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple

class Cerebro(nn.Module):
    def __init__(self, state_size, action_size, seed = 42):
        super(Cerebro, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcl = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self,state):
        x = self.fcl(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
    
    import gymnasium as gym
    
    cenario = gym.make("LunarLander-v2")
    state_size = cenario.observation_space.shape
    state_shape = cenario.observation_space.shape[0]
    acoes = cenario.action_space.n
    print('State_Shape: ', state_shape)
    print('State_Size: ', state_size)
    print('Ações: ', acoes)