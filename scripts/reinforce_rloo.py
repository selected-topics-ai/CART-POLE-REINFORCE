import numpy as np
import torch
import torch.nn as nn

from scipy import stats
from tqdm import tqdm
from gymnasium import Env
from gymnasium import Env
from utils import select_action
from config import ReinforceConfig
from typing import List, Tuple

def train_reinforce_with_baseline_rloo(environment: Env, policy: nn.Module, config: ReinforceConfig):

    policy = policy.to(config.device)

    episode_rewards = []
    validation_rewards: List[List[float]] = [] # Списки наград за каждый валидационный эпизод для отслеживания дисперсии между эпизодами
    validation_mean_rewards: List[float] = [] # Средняя награда за все валидационные эпизоды в рамках одного тренировочного эпизода

    best_validation_rewards = [0.0] * config.validation_episodes

    train_seed = 1
    for episode in range(config.episodes):

        state, info = environment.reset(seed=train_seed)

        log_probs = []
        probs     = []
        rewards   = []
        states    = []

        step = 0
        while True:

            action, log_prob, prob = select_action(state, policy, device=config.device)
            log_probs.append(log_prob)
            probs.append(prob)

            state, reward, terminate, truncated, _ = environment.step(action)
            rewards.append(reward)

            if terminate:
                break
            if step >= config.episode_steps_truncating:
                break
            step += 1

        # Вычисляем дисконтированные награды для всех временных шагов
        discounted_rewards = []
        R_t = 0
        for r in reversed(rewards):
            R_t = float(r) + config.discount_factor * R_t
            discounted_rewards.insert(0, R_t)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(config.device)

        # Вычисляем LLO поправку
        T = len(rewards)
        policy_loss = []
        for t in range(T):
            G_t = discounted_rewards[t]
            G_loo = (discounted_rewards.sum() - G_t) / (T - 1)
            advantage = G_t - G_loo
            policy_loss.append(-log_probs[t] * advantage)


        policy_loss = torch.stack(policy_loss).mean()

        if config.entropy_regularization != 0.0:
            probs = torch.stack(probs, dim=0)
            entropy = -torch.sum(probs * torch.log(probs), dim=1)
            policy_loss += config.entropy_regularization * entropy.mean()

        config.optimizer.zero_grad()
        policy_loss.backward()
        config.optimizer.step()

        # Валидация
        if episode % config.validate_every_th_episode == 0:
            policy.eval()
            with torch.no_grad():
                validation_seed = config.episodes + 100
                v = []
                for val_episode in range(config.validation_episodes):
                    state, _ = environment.reset(seed=validation_seed)
                    val_episode_reward = []
                    step = 0
                    while True:
                        action, log_prob, probs = select_action(state, policy, sample=False, device=config.device)
                        state, reward, terminate, truncated, _ = environment.step(action)
                        if terminate:
                            break
                        if step >= config.episode_steps_truncating:
                            break
                        step += 1
                        val_episode_reward.append(reward)
                    v.append(sum(val_episode_reward))
                    validation_seed += 1

                validation_rewards.append(v)
                validation_mean = np.array(v).mean()
                validation_std = np.array(v).std()

                if len(best_validation_rewards) == 0 or validation_mean > np.mean(best_validation_rewards):
                    t_statistic, p_value = stats.ttest_ind(v, best_validation_rewards)
                    if p_value < 0.05 or (validation_std < np.std(best_validation_rewards)):
                        torch.save(policy.state_dict(), os.path.join(config.save_path, f"{config.model_name}_entropy_{config.entropy_regularization}_best.pth"))
                        print(f"Saved best model. Old best validation reward: {np.mean(best_validation_rewards)}, new best {validation_mean}")
                        best_validation_rewards = v
                    validation_mean_rewards.append(validation_mean - validation_std)

                print(f"Episode: {episode}, reward: {sum(rewards)}, validation mean reward: {validation_mean}, validation std reward: {validation_std}")

        episode_rewards.append(sum(rewards))
        train_seed += 1

    model_metadata_save_path = f'{config.save_path}/{config.model_name}_entropy_{config.entropy_regularization}'
    np.savetxt(f'{model_metadata_save_path}_episodes_reward.txt', np.array(episode_rewards), fmt='%d')
    np.savetxt(f'{model_metadata_save_path}_validation_rewards.txt', np.array(validation_rewards), fmt='%d')
    return episode_rewards, validation_rewards