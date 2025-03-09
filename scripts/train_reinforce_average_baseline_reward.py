import gymnasium as gym
import torch.optim as optim

from utils import get_device
from policy import PolicyNetwork
from scripts.reinforce_average_reward import train
from torch.optim.lr_scheduler import CosineAnnealingLR

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

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=total_episodes)

    for beta in betas:

        train(policy=policy,
              environment=env,
              total_episodes=total_episodes,
              train_seeds=train_seeds,
              eval_every_th_episode=eval_every_th_episode,
              eval_seeds=eval_seeds,
              total_episode_steps=total_episode_steps_until_truncation,
              gamma=gamma,
              beta=beta,
              optimizer=optimizer,
              scheduler=scheduler,
              log_to_wandb=True)