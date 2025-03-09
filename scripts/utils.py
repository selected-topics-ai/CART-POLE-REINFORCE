import os
import torch
import random
import numpy as np

from policy import PolicyNetwork
from torch.distributions.categorical import Categorical

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.mps.is_available():
        device = "mps"

    return device


def select_action(state, policy: PolicyNetwork, device=get_device()):
    state_tensor = torch.tensor(state, dtype=torch.float).to(device)
    probs = policy(state_tensor)
    dist = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action), dist.entropy()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)