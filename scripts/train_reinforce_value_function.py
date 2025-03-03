import gymnasium
import torch.optim as optim

from utils import seed_everything
from policy import PolicyNetwork
from config import ReinforceConfig
from reinforce_value_function import ValueFunction, train_vpg_with_bvf

if __name__ == '__main__':
    seed_everything(42)

    entropy_weights = [0.0, 0.001, 0.01, 0.1]
    ac_environment = gymnasium.make("CartPole-v1")

    for entropy_weight in entropy_weights:
        actor = PolicyNetwork(state_dim=4, hidden_state_dim=64, action_space=2)
        critic = ValueFunction(input_size=4, hidden_size=64)

        actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
        critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

        ac_config = ReinforceConfig(episodes=2_000,
                                    discount_factor=0.99,
                                    optimizer=actor_optimizer,
                                    vf_optimizer=critic_optimizer,
                                    entropy_regularization=entropy_weight,
                                    validation_episodes=50,
                                    validate_every_th_episode=10,
                                    model_name=f"ac_model_new")

        episode_rewards, validation_rewards = train_vpg_with_bvf(ac_environment, actor, critic, ac_config)