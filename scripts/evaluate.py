from typing import List

import random
import numpy as np
from utils import select_action

def evaluate_policy(policy, env, n_episodes=10, seeds: List[int]=None, render=True):

    if seeds is None:
        seeds = []
        for episode in range(n_episodes):
            seeds.append(random.randint(10_000, 100_000))

    episode_rewards = []  # Список для хранения вознаграждений за каждый эпизод

    for seed in range(len(seeds)):

        obs, _ = env.reset(seed=seed)  # Сбрасываем окружение и получаем начальное наблюдение
        done = False
        total_reward = 0  # Суммарное вознаграждение за эпизод

        while not done:
            if render:
                env.render()  # Визуализация окружения (если включено)

            action, log_prob = select_action(obs, policy, sample=True, device="mps")  # Получаем действие от политики
            obs, reward, done, truncated, info = env.step(action)  # Шаг в окружении
            total_reward += reward  # Добавляем вознаграждение

        episode_rewards.append(total_reward)  # Сохраняем вознаграждение за эпизод
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Вычисляем среднее и стандартное отклонение вознаграждений
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"Evaluation over {n_episodes} episodes: Mean Reward = {mean_reward}, Std Reward = {std_reward}")
    return mean_reward, std_reward