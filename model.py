import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """        
        super(QNetwork, self).__init__()        
        FC1_SIZE = state_size
        FC2_SIZE = state_size
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, FC1_SIZE)
        self.fc2 = nn.Linear(FC1_SIZE, FC2_SIZE)
        self.outp = nn.Linear(FC2_SIZE, action_size)
        
    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.outp(x)
        return x
