import gymnasium as gym
import torch.optim as optim

from utils import get_device
from policy import PolicyNetwork
from torch.optim.lr_scheduler import CosineAnnealingLR
from scripts.reinforce_value_function import ValueFunction, train


if __name__ == '__main__':

    total_episodes = 2_000
    train_seeds = [i for i in range(total_episodes // 10)]
    eval_every_th_episode = 50
    eval_seeds = [i for i in range(10_000, 10_100)]
    total_episode_steps_until_truncation = 1_000

    gamma = 0.99
    betas = [0.0, 0.001, 0.01, 0.1]

    env = gym.make("CartPole-v1")

    policy = PolicyNetwork().to(get_device())
    value_function = ValueFunction().to(get_device())

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=total_episodes)

    value_function_optimizer = optim.Adam(value_function.parameters(), lr=1e-3)
    value_function_scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=total_episodes)

    for beta in betas:

        train(policy=policy,
              value_function=value_function,
              environment=env,
              total_episodes=total_episodes,
              train_seeds=train_seeds,
              eval_every_th_episode=eval_every_th_episode,
              eval_seeds=eval_seeds,
              total_episode_steps=total_episode_steps_until_truncation,
              gamma=gamma,
              beta=beta,
              optimizer=optimizer,
              value_function_optimizer=value_function_optimizer,
              scheduler=scheduler,
              value_function_scheduler=value_function_scheduler,
              log_to_wandb=True)
