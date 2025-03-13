import os
import torch
import wandb
import numpy as np

from typing import List
from gymnasium import Env
from datetime import datetime
from validation import validate
from policy import PolicyNetwork
from utils import get_device, select_action
from torch.optim.lr_scheduler import LRScheduler


def train(policy: PolicyNetwork,
          environment: Env,
          total_episodes: int,
          train_seeds: List[int],
          eval_every_th_episode: int,
          eval_seeds: List[int],
          total_episode_steps: int,
          gamma: float,
          beta: float,
          optimizer: torch.optim.Optimizer,
          scheduler: LRScheduler=None,
          log_to_wandb: bool = True,
          device = get_device(),
          trainer_name="cart-pole-rloo-baseline",
          ouput_dir="./models"):

    policy.to(device)

    if log_to_wandb:
        run = wandb.init(project="cart-pole-rloo-baseline",
                         name=f"{trainer_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
                         config={
                             "lr": float(optimizer.param_groups[0]["lr"]),
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

        episode_rewards = []
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

            state = next_state

        # returns
        G = 0
        returns = []
        for r in reversed(episode_rewards):
            G = float(r) + gamma * G
            returns.insert(0, G)

        # rloo returns
        T = len(episode_rewards)
        llo_returns = []
        for t in range(T):
            G_t = returns[t]
            G_loo = (np.sum(returns) - G_t) / (T - 1)
            advantage = G_t - G_loo
            llo_returns.append(advantage)

        episode_losses = []
        loss = 0
        for log_prob, entropy, advantage in zip(log_probs, entropies, llo_returns):

            loss -= log_prob * advantage
            loss -= beta * entropy

            episode_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step(episode)

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
                format_entropy = str(beta).replace('.', '_')
                checkpoint_name = f'{trainer_name}-checkpoint-{episode}-val_reward-{min_val_reward}-entropy-{format_entropy}.pth'
                torch.save(checkpoint, os.path.join(ouput_dir, checkpoint_name))
                current_min_validation_reward = min_val_reward

        if log_to_wandb:
            run.log({
                "train/avg_episode_loss": np.mean(episode_losses),
                "train/total_episode_reward": np.sum(episode_rewards),
                "train/avg_train_entropy": np.mean(entropies),
            },  step=episode)
        else:
            print(
                f'"train/avg_episode_loss": {np.mean(episode_losses)}',
                f'"train/total_episode_reward": {np.sum(episode_rewards)}',
                f'"train/avg_train_entropy": {np.mean(entropies)}'
            )

    if log_to_wandb:
        wandb.finish()