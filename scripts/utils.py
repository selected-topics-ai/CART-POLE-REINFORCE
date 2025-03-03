import os
import torch
import random
import numpy as np
import torch.nn as nn

from torch.distributions.categorical import Categorical
from device import get_device

def select_action(state: np.ndarray, policy: nn.Module, sample=True, device=get_device()) -> (int, torch.Tensor, torch.Tensor):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs, _ = policy(state)
    probs = probs.to(device)
    m = Categorical(probs=probs)
    if sample:
        action = m.sample()
    else:
        action = probs.argmax(dim=1)
    return action.item(), m.log_prob(action), probs.squeeze(0)


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