import os
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from gymnasium import Env
from validation import validate
from policy import PolicyNetwork
from utils import select_action, get_device
from torch.optim.lr_scheduler import LRScheduler


class ValueFunction(nn.Module):
    def __init__(self, input_size: int=4, hidden_size: int=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(policy: PolicyNetwork,
          value_function: ValueFunction,
          environment: Env,
          total_episodes: int,
          train_seeds: List[int],
          eval_every_th_episode: int,
          eval_seeds: List[int],
          total_episode_steps: int,
          gamma: float,
          beta: float,
          optimizer: torch.optim.Optimizer,
          value_function_optimizer: torch.optim.Optimizer,
          scheduler: LRScheduler=None,
          value_function_scheduler: LRScheduler=None,
          log_to_wandb: bool = True,
          device = get_device(),
          trainer_name="cart-pole-value_function-reward-baseline",
          ouput_dir="./models"):

    policy.to(device)
    value_function.to(device)

    if log_to_wandb:
        run = wandb.init(project=trainer_name,
                         config={
                             "lr": float(optimizer.param_groups[0]["lr"]),
                             "value_function_lr": float(value_function_optimizer.param_groups[0]["lr"]),
                             "total_episodes": total_episodes,
                             "episode_steps_until_truncate": total_episode_steps,
                             "beta": beta,
                             "gamma": gamma,
                         })

    episodes = [i for i in range(total_episodes)]

    if len(train_seeds) <= total_episodes:
        train_seeds_ = train_seeds * (total_episodes // len(train_seeds)) + train_seeds[0:total_episodes % len(train_seeds)]
    else:
        train_seeds_ = train_seeds[0:total_episodes]

    current_min_validation_reward = 600

    for episode, seed in zip(episodes, train_seeds_):

        policy.train()
        value_function.train()

        episode_rewards = []
        value_function_rewards = []
        log_probs = []
        entropies = []

        state, _ = environment.reset(seed=seed)

        for step in range(total_episode_steps):

            action, log_prob, entropy = select_action(state, policy)
            next_state, reward, terminate, _, _ = environment.step(action)

            if terminate:
                break

            log_probs.append(log_prob)
            episode_rewards.append(float(reward))
            entropies.append(entropy.cpu().detach().item())

            state_tensor = torch.tensor(state, dtype=torch.float).to(device)
            value_fuction_reward = value_function(state_tensor)
            value_function_rewards.append(value_fuction_reward)

            state = next_state

        G = 0
        returns = []
        for r in reversed(episode_rewards):
            G = float(r) + gamma * G
            returns.insert(0, G)

        episode_losses = []
        loss = 0
        for log_prob, entropy, G, vf_reward in zip(log_probs, entropies, returns, value_function_rewards):

            advantage = G - vf_reward.cpu().detach().item()

            loss -= log_prob * advantage
            loss -= beta * entropy

            episode_losses.append(loss.item())

        # Optimize policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(episode)

        # Optimize value function
        value_function_rewards_tensor = torch.tensor(value_function_rewards, dtype=torch.float).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float).to(device)

        value_function_optimizer.zero_grad()
        vf_loss = F.mse_loss(returns_tensor, value_function_rewards_tensor)
        vf_loss.backward()
        value_function_optimizer.step()

        if value_function_scheduler is not None:
            value_function_scheduler.step()

        if episode % eval_every_th_episode == 0:
            val_total_episode_rewards, val_average_episodes_entropy = validate(policy=policy,
                                                                       environment=environment,
                                                                       eval_seeds=eval_seeds)

            if log_to_wandb:
                run.log({
                    "val/avg_episode_reward": np.mean(val_total_episode_rewards),
                    "val/min_episode_reward": np.min(val_total_episode_rewards),
                    "val/avg_episode_entropy": np.mean(val_average_episodes_entropy),
                }, step=episode)
            else:
                print(
                    f'"val/avg_episode_reward": {np.mean(val_total_episode_rewards)}',
                    f'val/min_episode_reward": {np.min(val_total_episode_rewards)}',
                    f'"val/avg_episode_entropy": {np.mean(val_average_episodes_entropy)}'
                )

            # Save best checkpoint
            min_val_reward = np.min(val_total_episode_rewards)
            if min_val_reward > current_min_validation_reward:
                checkpoint = {
                    'episode': episode,
                    'seed': seed,
                    'policy': policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(checkpoint, os.path.join(ouput_dir, f'{trainer_name}-checkpoint-{episode}-val_reward-{min_val_reward}.pth'))
                current_min_validation_reward = min_val_reward

        if log_to_wandb:
            run.log({
                "train/avg_episode_loss": np.mean(episode_losses),
                "train/total_episode_reward": np.sum(episode_rewards),
                "train/avg_train_entropy": np.mean(entropies),
                "train/avg_value_function_rewards": np.mean(value_function_rewards),
            },  step=episode)
        else:
            print(
                f'"train/avg_episode_loss": {np.mean(episode_losses)}',
                f'"train/total_episode_reward": {np.sum(episode_rewards)}',
                f'"train/avg_train_entropy": {np.mean(entropies)}',
                f'train/avg_value_function_rewards": {np.mean(value_function_rewards)}',
            )