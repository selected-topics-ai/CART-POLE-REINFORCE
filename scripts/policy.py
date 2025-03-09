import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self,state_dim:int=4, hidden_state_dim:int=64, action_space:int=2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_state_dim)
        self.fc2 = nn.Linear(hidden_state_dim, action_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)