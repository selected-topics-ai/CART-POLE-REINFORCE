import torch

from torch.utils.data import Dataset

class StateActionDataset(Dataset):

    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'states': torch.FloatTensor(self.states[idx]),
            'action': torch.FloatTensor([self.actions[idx]]),
        }
