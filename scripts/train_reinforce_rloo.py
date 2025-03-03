import gymnasium
import torch.optim as optim

from utils import seed_everything
from policy import PolicyNetwork
from config import ReinforceConfig
from reinforce_rloo import train_reinforce_with_baseline_rloo


if __name__ == '__main__':
    seed_everything(42)

    entropy_weights = [0.0, 0.001, 0.01, 0.1]
    rloo_environment = gymnasium.make("CartPole-v1")

    for entropy_weight in entropy_weights:
        rloo_policy = PolicyNetwork(state_dim=4, hidden_state_dim=64, action_space=2)
        optimizer = optim.Adam(rloo_policy.parameters(), lr=1e-3)
        rloo_config = ReinforceConfig(episodes=1_000,
                                      discount_factor=0.99,
                                      optimizer=optimizer,
                                      model_name="rloo_model_new",
                                      entropy_regularization=entropy_weight,
                                      validation_episodes=50,
                                      validate_every_th_episode=10)

        train_reinforce_with_baseline_rloo(rloo_environment, rloo_policy, rloo_config)