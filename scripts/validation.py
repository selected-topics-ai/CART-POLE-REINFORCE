import torch
import numpy as np

from typing import List
from gymnasium import Env
from policy import PolicyNetwork
from utils import get_device, select_action


def validate(environment: Env, policy: PolicyNetwork, eval_seeds: List[int], episode_steps_until_truncate: int = 700, device=get_device()):
    policy.eval()
    policy.to(device)
    with torch.inference_mode():
        total_episode_rewards = []
        average_episodes_entropy = []
        for seed in eval_seeds:
            episode_rewards = []
            episode_entropy = []
            state, _ = environment.reset(seed=seed)
            for step in range(episode_steps_until_truncate):
                action, _, entropy = select_action(state, policy, device=device)
                next_state, reward, terminate, _, _ = environment.step(action)
                if terminate:
                    break
                episode_rewards.append(float(reward))
                episode_entropy.append(entropy.cpu().detach().item())
                state = next_state
            total_episode_rewards.append(sum(episode_rewards))
            average_episodes_entropy.append(np.mean(episode_entropy))
    return total_episode_rewards, average_episodes_entropy