import os
import torch
import wandb
import numpy as np

from typing import List
from gymnasium import Env
from utils import get_device
from datetime import datetime
from validation import validate
from policy import PolicyNetwork
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler


def train(policy: PolicyNetwork,
          epochs: int,
          environment: Env,
          optimizer: Optimizer,
          criterion,
          train_loader: DataLoader,
          source_seeds: List[int],
          eval_seeds: List[int],
          sample_size: int,
          device=get_device(),
          trainer_name: str = 'behavioral-cloning',
          log_to_wandb: bool = True,
          lr_scheduler: LRScheduler = None,
          validate_every: int = 10,):

    if log_to_wandb:
        run = wandb.init(project=trainer_name,
                         name=f"{trainer_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
                         config={
                             "lr": float(optimizer.param_groups[0]["lr"]),
                             "epochs": epochs,
                             "device": device,
                             "eval_seeds": eval_seeds,
                             "source_seeds": source_seeds,
                             "sample_size": sample_size,
                         })

    current_min_validation_reward = 0

    for epoch in range(epochs):
        policy.to(device=device)

        train_loss = []

        for batch in train_loader:

            state, actions = batch['states'].to(device), batch['action'].to(device)
            predicted_actions = policy(state)
            loss = criterion(predicted_actions, actions.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if lr_scheduler is not None:
                lr_scheduler.step()

        run.log({"loss": np.mean(train_loss)}, step=epoch)

        if epoch % validate_every == 0:

            (source_total_episode_rewards,
             source_average_episodes_entropy) = validate(environment=environment,
                                                      policy=policy,
                                                      eval_seeds=source_seeds)

            (eval_total_episode_rewards,
             eval_average_episodes_entropy) = validate(environment=environment,
                                                      policy=policy,
                                                      eval_seeds=eval_seeds)

            # Save best checkpoint
            min_val_reward = np.min(eval_total_episode_rewards)
            if min_val_reward > current_min_validation_reward:
                checkpoint = {
                    'epoch': epoch,
                    'policy': policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                checkpoint_name = f'{trainer_name}-checkpoint-{epoch}-val_reward-{min_val_reward}.pth'
                torch.save(checkpoint, os.path.join('./models', checkpoint_name))
                current_min_validation_reward = min_val_reward

            run.log({
                "val/source/avg_episode_reward": np.mean(source_total_episode_rewards),
                "val/source/min_episode_reward": np.min(source_total_episode_rewards),
                "val/source/avg_episode_entropy": np.mean(source_average_episodes_entropy),
                "val/avg_episode_reward": np.mean(eval_total_episode_rewards),
                "val/min_episode_reward": np.min(eval_total_episode_rewards),
                "val/avg_episode_entropy": np.mean(eval_average_episodes_entropy),
            }, step=epoch)

    if log_to_wandb:
        wandb.finish()
