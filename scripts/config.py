import torch.optim as optim

from dataclasses import dataclass
from device import get_device

@dataclass
class ReinforceConfig:
    episodes: int
    discount_factor: float
    optimizer: optim.Optimizer
    model_name: str
    vf_optimizer: optim.Optimizer = None
    entropy_regularization: float = 0.0
    save_path: str = './models'
    validation_episodes: int = 100
    validate_every_th_episode: int = 20
    episode_steps_truncating: int = 600
    device: str = get_device()