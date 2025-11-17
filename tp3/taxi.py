"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import itertools
import typing as t
import time
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent


env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore

rewards_history = {"Q-Learning": [], "Q-Learning Epsilon Scheduling": [], "SARSA": []}
param_grid = {
    "learning_rate": [0.1, 0.3, 0.5, 0.7],
    "epsilon": [0.01, 0.05, 0.1, 0.25],
    "gamma": [0.9, 0.95, 0.99],
}


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        agent.update(s, a, r, next_s)
        s = next_s
        total_reward += r
        if done:
            break
        # END SOLUTION

    return total_reward


def train_agent(env, agent, label, num_episodes=1000, verbose=False):
    start_time = time.time()
    rewards = []
    for i in range(num_episodes):
        rewards.append(play_and_train(env, agent))
        if i % 100 == 0 and label != "SARSA" and not verbose:
            print("mean reward", np.mean(rewards[-100:]))
    end_time = time.time()
    if not verbose:
        print(f"Training {label} took {end_time - start_time:.2f} seconds")
    rewards_history[label] = rewards
    if label != "SARSA" and not verbose:
        print("Final mean reward", np.mean(rewards[-100:]))

    return rewards


def grid_search(env, agent_class, label, param_grid, num_episodes=1000):
    best_mean_reward = -float("inf")
    best_params = {}
    best_rewards = []

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for params in param_combinations:
        agent = agent_class(
            learning_rate=params["learning_rate"],
            epsilon=params["epsilon"],
            gamma=params["gamma"],
            legal_actions=list(range(n_actions)),
        )
        rewards = train_agent(env, agent, label, num_episodes, verbose=True)
        mean_reward = np.mean(rewards[-100:])

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_params = params
            best_rewards = rewards

    print(
        f"Best params for {label}: {best_params} with mean reward: {best_mean_reward}"
    )
    return best_params, best_rewards


def record_agent_video(env_id: str, agent, video_path: str, max_steps: int = 200):
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda e: True)

    s, _ = env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        a = agent.get_action(s)
        next_s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        s = next_s
        step += 1

    env.close()


print("Grid Search for Q-Learning...")
best_ql_params, ql_rewards = grid_search(env, QLearningAgent, "Q-Learning", param_grid)

print("Grid Search for Q-Learning Epsilon Scheduling...")
best_ql_eps_params, ql_eps_rewards = grid_search(
    env, QLearningAgentEpsScheduling, "Q-Learning Epsilon Scheduling", param_grid
)

print("Grid Search for SARSA...")
best_sarsa_params, sarsa_rewards = grid_search(env, SarsaAgent, "SARSA", param_grid)


agent_ql = QLearningAgent(**best_ql_params, legal_actions=list(range(n_actions)))
print("Training Q-Learning...")
ql_rewards = train_agent(
    env,
    agent_ql,
    label="Q-Learning",
)
record_agent_video("Taxi-v3", agent_ql, "./videos/qlearning_agent")

agent_ql_eps = QLearningAgentEpsScheduling(
    **best_ql_eps_params, legal_actions=list(range(n_actions))
)
print("Training Q-Learning Epsilon Scheduling...")
ql_eps_rewards = train_agent(
    env,
    agent_ql_eps,
    label="Q-Learning Epsilon Scheduling",
)
record_agent_video("Taxi-v3", agent_ql_eps, "./videos/qlearning_eps_scheduling_agent")

agent_sarsa = SarsaAgent(**best_sarsa_params, legal_actions=list(range(n_actions)))
print("Training SARSA...")
ql_sarsa_rewards = train_agent(
    env,
    agent_sarsa,
    label="SARSA",
)
record_agent_video("Taxi-v3", agent_sarsa, "./videos/sarsa_agent")

# Plotting learning curves
plt.figure(figsize=(10, 6))
for label, rewards in rewards_history.items():
    plt.plot(np.convolve(rewards, np.ones(100) / 100, mode="valid"), label=label)
plt.xlabel("Episodes")
plt.ylabel("Mean Reward (smoothed)")
plt.title("Learning Curves for Q-Learning vs Q-Learning Epsilon Scheduling vs SARSA")
plt.legend()
plt.grid()
plt.savefig("learning_curves.png")
plt.show()
