import torch
import numpy as np
import gymnasium as gym

from policy import PolicyNetwork
from behavioral_cloning import train
from torch.utils.data import DataLoader
from utils import get_device, seed_everything
from scripts.bc_dataset import StateActionDataset


if __name__ == "__main__":

    seed_everything(42)

    path = '../models/cart-pole-value_function-reward-baseline-checkpoint-150-val_reward-700.0-entropy-1_0.pth'

    best_checkpoint = torch.load(path, map_location=get_device())['policy']

    policy = PolicyNetwork()
    policy.load_state_dict(best_checkpoint)

    source_seeds = [i for i in range(10_000, 10_100)]
    useen_seeds = [i for i in range(100_000, 100_100)]

    env = gym.make('CartPole-v1')

    # states, actions = generate_state_actions(env=gym.make('CartPole-v1'),
    #                                          seeds=eval_seeds,
    #                                          policy=policy,
    #                                          steps_until_truncate=600)

    states = np.loadtxt('../data/states.txt')
    actions = np.loadtxt('../data/actions.txt')

    N = [1000]
    epochs = 100

    for n in N:

        random_indexes = np.random.randint(0, states.shape[0], size=n)

        states = states[random_indexes]
        actions = actions[random_indexes]

        dataset = StateActionDataset(states, actions)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        pupil_policy = PolicyNetwork()

        optimizer = torch.optim.Adam(pupil_policy.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        train(environment=env,
              epochs=epochs,
              policy=pupil_policy,
              criterion=criterion,
              optimizer=optimizer,
              train_loader=dataloader,
              eval_seeds=useen_seeds,
              source_seeds=source_seeds,
              sample_size=n,)
