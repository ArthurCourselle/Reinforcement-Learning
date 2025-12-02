import ale_py

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

import torch
import numpy as np
from collections import deque
import random
from config import Config

def init_env(config: Config, render_mode: str = "rgb_array") -> gym.Env:

    env = gym.make(config.env_name, render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)

    env = AtariPreprocessing(
        env,
        noop_max=30,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True,
        frame_skip=4,
        terminal_on_life_loss=False,
    )
    env = FrameStackObservation(env, config.num_frames)

    return env

def to_torch_order(obs):
    obs = np.array(obs, dtype=np.float32)
    return obs

class ReplayMemory:
    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.device = device

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(dones).to(self.device),
        )

    def __len__(self) -> int:
        return len(self.memory)
