import os
import torch
import random
import numpy as np

from tqdm import tqdm
from gymnasium import Env
from policy import PolicyNetwork
from torch.distributions.categorical import Categorical

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.mps.is_available():
        device = "mps"

    return device


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


def select_action(state, policy: PolicyNetwork, device=get_device(), sample:bool=True):
    state_tensor = torch.tensor(state.tolist(), dtype=torch.float).to(device)
    probs = policy(state_tensor)
    dist = Categorical(probs)
    if sample:
        action = dist.sample()
    else:
        action = torch.argmax(dist.probs, dim=-1)
    return action.item(), dist.log_prob(action), dist.entropy()


def _generate_state_actions(env: Env, policy: PolicyNetwork, seed: int, steps_until_truncate: int, device=get_device()) -> tuple[np.ndarray, np.ndarray]:

    policy = policy.to(device)

    states = []
    actions = []
    state, _ = env.reset(seed=seed)
    for step in range(steps_until_truncate):
        states.append(state)
        action, _, _ = select_action(state, policy, device=get_device(), sample=True)
        actions.append(action)
        new_state, reward, terminate, _, _ = env.step(action)
        if terminate:
            break
        state = new_state

    return np.array(states), np.array(actions)


def generate_state_actions(env: Env, policy: PolicyNetwork, seeds: list[int], steps_until_truncate: int) -> tuple[np.ndarray, np.ndarray]:

    states = []
    actions = []

    for seed in tqdm(seeds, desc="Generating state-actions"):
        s, a = _generate_state_actions(env, policy, seed, steps_until_truncate)

        states.append(s)
        actions.append(a)

    return np.concat(states), np.concat(actions)
