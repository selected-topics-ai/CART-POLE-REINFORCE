import gymnasium
import torch.optim as optim

from utils import seed_everything
from policy import PolicyNetwork
from config import ReinforceConfig
from reinforce_average_reward import train_reinforce_with_ar_baseline

def train():
    pass

if __name__ == '__main__':
    seed_everything(42)

    entropy_weights = [0.0, 0.001, 0.01, 0.1]
    av_environment = gymnasium.make("CartPole-v1")

    for entropy_weight in entropy_weights:
        av_policy = PolicyNetwork(state_dim=4, hidden_state_dim=64, action_space=2)
        optimizer = optim.Adam(av_policy.parameters(), lr=1e-3)
        config = ReinforceConfig(episodes=2_000,
                                 discount_factor=0.99,
                                 optimizer=optimizer,
                                 model_name="average_reward_new",
                                 entropy_regularization=entropy_weight,
                                 validation_episodes=50,
                                 validate_every_th_episode=10)

        train_reinforce_with_ar_baseline(environment=av_environment, policy=av_policy, config=config)