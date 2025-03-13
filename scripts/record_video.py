import torch
import argparse
import gymnasium as gym

from policy import PolicyNetwork
from utils import get_device, select_action

if __name__ == '__main__':

    seed = 100_001
    steps = 500
    video_folder = '../data/video'
    policy_checkpoint_path = 'models/behavioral-cloning-without-position-steps-checkpoint-30-val_reward-203.0.pth'

    policy_checkpoint = torch.load(policy_checkpoint_path, map_location=get_device())['policy']
    policy = PolicyNetwork()
    policy.load_state_dict(policy_checkpoint)
    policy.to(get_device())
    policy.eval()

    cart_pole_env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env=cart_pole_env,
                                   video_folder=video_folder,
                                   name_prefix='cart_pole',
                                   episode_trigger=lambda x: x % 2 == 0)

    state, _ = env.reset(seed=seed)
    with torch.inference_mode():
        for step in range(steps):
            action, _, _ = select_action(state, policy, device=get_device())
            next_state, _, terminate, _, _ = env.step(action)
            if terminate:
                break
            state = next_state

    env.close()
