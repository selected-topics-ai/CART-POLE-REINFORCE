import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy import stats
from gymnasium import Env
from config import ReinforceConfig
from utils import select_action
from typing import List


class ValueFunction(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_vpg_with_bvf(environment: Env, policy: nn.Module, value_function: nn.Module, config: ReinforceConfig):
    policy = policy.to(config.device)
    value_function = value_function.to(config.device)

    episode_rewards = []
    validation_rewards: List[List[float]] = [] # Списки наград за каждый валидационный эпизод для отслеживания дисперсии между эпизодами
    validation_mean_rewards: List[float] = [] # Средняя награда за все валидационные эпизоды в рамках одного тренировочного эпизода

    best_validation_rewards = [0.0] * config.validation_episodes

    train_seed = 1
    for episode in range(config.episodes):
        policy.train()
        state, _ = environment.reset(seed=train_seed)

        log_probs  = []
        probs      = []
        rewards    = []
        vf_rewards = []

        step = 0
        while True:

            action, log_prob, prob = select_action(state, policy, device=config.device)
            log_probs.append(log_prob)
            probs.append(prob)

            state, reward, terminate, truncated, _ = environment.step(action)
            rewards.append(reward)

            vf_reward = value_function(torch.tensor(state, dtype=torch.float32, device=config.device))
            vf_rewards.append(vf_reward)

            if terminate:
                break
            if step >= config.episode_steps_truncating:
                break
            step += 1

        # Discounted rewards
        discounted_rewards = []
        R_t = 0
        for r in reversed(rewards):
            R_t = float(r) + config.discount_factor * R_t
            discounted_rewards.insert(0, R_t)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(config.device)

        # Value Function reward estimation
        vf_rewards = torch.stack(vf_rewards, dim=-1).squeeze(0)
        advantage = discounted_rewards - vf_rewards.detach()

        log_probs = torch.stack(log_probs, dim=-1).squeeze(0)
        policy_loss = (-log_probs * advantage).mean()

        if config.entropy_regularization != 0.0:
            probs = torch.stack(probs, dim=0)
            entropy = -torch.sum(probs * torch.log(probs), dim=1)
            policy_loss += config.entropy_regularization * entropy.mean()

        # Optimization block
        config.optimizer.zero_grad()
        policy_loss.backward()
        config.optimizer.step()

        config.vf_optimizer.zero_grad()
        vf_loss = F.mse_loss(vf_rewards, discounted_rewards)
        vf_loss.backward()
        config.vf_optimizer.step()

        # Валидация
        if episode % config.validate_every_th_episode == 0:
            policy.eval()
            with torch.no_grad():
                validation_seed = config.episodes + 100
                v = []
                for val_episode in range(config.validation_episodes):
                    state, _ = environment.reset(seed=validation_seed)
                    val_episode_reward = []
                    step = 0
                    while True:
                        action, log_prob, probs = select_action(state, policy, sample=False, device=config.device)
                        state, reward, terminate, truncated, _ = environment.step(action)
                        if terminate:
                            break
                        if step >= config.episode_steps_truncating:
                            break
                        step += 1
                        val_episode_reward.append(reward)
                    v.append(sum(val_episode_reward))
                    validation_seed += 1

                validation_rewards.append(v)
                validation_mean = np.array(v).mean()
                validation_std = np.array(v).std()

                if len(best_validation_rewards) == 0 or validation_mean > np.mean(best_validation_rewards):
                    t_statistic, p_value = stats.ttest_ind(v, best_validation_rewards)
                    if p_value < 0.05 or (validation_std < np.std(best_validation_rewards)):
                        torch.save(policy.state_dict(), os.path.join(config.save_path, f"{config.model_name}_entropy_{config.entropy_regularization}_best.pth"))
                        print(f"Saved best model. Old best validation reward: {np.mean(best_validation_rewards)}, new best {validation_mean}")
                        best_validation_rewards = v
                    validation_mean_rewards.append(validation_mean - validation_std)

                print(f"Episode: {episode}, reward: {sum(rewards)}, validation mean reward: {validation_mean}, validation std reward: {validation_std}")

        episode_rewards.append(sum(rewards))
        train_seed += 1

    model_metadata_save_path = f'{config.save_path}/{config.model_name}_entropy_{config.entropy_regularization}'
    np.savetxt(f'{model_metadata_save_path}_episodes_reward.txt', np.array(episode_rewards), fmt='%d')
    np.savetxt(f'{model_metadata_save_path}_validation_rewards.txt', np.array(validation_rewards), fmt='%d')
    return episode_rewards, validation_rewards