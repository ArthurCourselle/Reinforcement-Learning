import os
import ale_py

import gymnasium as gym

import torch

from dqn import DQNAgent

# from utils import preprocess, stack_frames
from utils import to_torch_order
from config import Config

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


def train(config: Config):
    env = gym.make(config.env_name, render_mode="rgb_array", frameskip=1)

    env = AtariPreprocessing(
        env,
        noop_max=30,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True,
        frame_skip=4,
        terminal_on_life_loss=True,
    )
    env = FrameStackObservation(env, config.num_frames)

    agent = DQNAgent(env, config)
    writer = SummaryWriter(log_dir=f"runs/{config.env_name}")
    episode = 0

    if os.path.exists(f"checkpoints/{config.env_name}_dqn_latest.pth"):
        episode = agent.load(f"checkpoints/{config.env_name}_dqn_latest.pth")

    obs, _ = env.reset()
    state_stack = to_torch_order(obs)

    print("Filling replay memory...")
    while len(agent.memory) < config.min_replay_size:
        action = env.action_space.sample()
        next_obs, reward, done, truncated, _ = env.step(action)
        next_state_stack = to_torch_order(next_obs)

        agent.memory.push(state_stack, action, reward, next_state_stack, done)
        state_stack = next_state_stack
        if done:
            obs, _ = env.reset()
            state_stack = to_torch_order(obs)

    print("Starting training...")

    rewards_history = []
    loss_history = []
    q_value_history = []

    while True:
        obs, _ = env.reset()
        state_stack = to_torch_order(obs)
        # state = preprocess(obs)
        # state_stack, frames = stack_frames(None, state, True, config.num_frames)
        total_reward = 0
        total_loss = 0
        total_q_value = 0
        done = False

        # rewards_history_ep = []
        loss_history_ep = []
        q_value_history_ep = []

        # cpt_loss = 0

        while not done:
            print("Step:", agent.steps_done, end="\r")
            action = agent.select_action(state_stack)
            next_obs, reward, done, truncated, _ = env.step(action)
            # next_frame = preprocess(next_obs)
            # next_state_stack, frames = stack_frames(
            #     frames, next_frame, False, config.num_frames
            # )
            next_state_stack = to_torch_order(next_obs)
            agent.memory.push(state_stack, action, reward, next_state_stack, done)
            state_stack = next_state_stack
            total_reward += reward

            loss, q_value = agent.optimize()
            loss_history_ep.append(loss)
            q_value_history_ep.append(q_value)

            total_loss += loss
            total_q_value += q_value

            # cpt_loss += 1

            if agent.steps_done % config.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

                # writer.add_scalar("Q_Value/Train", q_value, agent.steps_done)

        episode += 1
        rewards_history.append(total_reward)
        loss_history.append(total_loss)
        q_value_history.append(total_q_value)

        if episode % config.log_every == 0:
            mean_reward = np.mean(rewards_history[-config.log_every :])
            writer.add_scalar("Reward/Mean_50ep", mean_reward, agent.steps_done)
            mean_loss = np.mean(loss_history[-config.log_every :])
            writer.add_scalar("Loss/Mean_50ep", mean_loss, agent.steps_done)
            mean_q_value = np.mean(q_value_history[-config.log_every :])
            writer.add_scalar("Q_Value/Mean_50ep", mean_q_value, agent.steps_done)

            print(
                f"Step {agent.steps_done}: Mean reward (last {config.log_every} steps) = {mean_reward:.2f}, Mean loss = {mean_loss:.4f}, Mean Q value = {mean_q_value:.2f}"
            )

            # writer.add_scalar("Reward/Train", total_reward, agent.steps_done)
            # writer.add_scalar("Loss/Mean_50ep", total_loss / cpt_loss, agent.steps_done)

            # print(
            #     f"Step {agent.steps_done}: Reward = {total_reward:.2f}, Loss = {total_loss / cpt_loss:.4f}"
            # )

        if episode % config.save_every == 0:
            os.makedirs("checkpoints/ALE", exist_ok=True)
            agent.save(f"checkpoints/{config.env_name}_dqn_{episode}.pth", episode)
            agent.save(f"checkpoints/{config.env_name}_dqn_latest.pth", episode)

        print(
            f"Episode {episode}, Total Reward: {total_reward}, Total Loss: {loss_history[-1]}, Total Q: {q_value_history[-1]}, Epsilon: {agent.epsilon:.3f}"
        )


if __name__ == "__main__":
    train(Config())
