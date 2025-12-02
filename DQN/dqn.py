import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import ReplayMemory
from config import Config


class DQN(nn.Module):
    def __init__(self, input_shape: tuple, num_actions: int) -> None:
        super(DQN, self).__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 9 * 9, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, env: gym.Env, config: Config) -> None:
        self.env = env

        obs_shape = (config.num_frames, 84, 84)
        self.num_actions = env.action_space.n
        self.policy_net = DQN(obs_shape, self.num_actions).to(config.device)
        self.target_net = DQN(obs_shape, self.num_actions).to(config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(config.memory_size, config.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=config.lr)

        self.steps_done = 0
        self.epsilon = config.eps_start
        self.config = config
        self.loss = F.smooth_l1_loss

    def select_action(self, state: np.ndarray) -> int:
        self.epsilon = max(
            self.config.eps_end,
            self.config.eps_start - (self.config.eps_start - self.config.eps_end) * (self.steps_done / self.config.eps_decay)
        )
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state = torch.tensor(
                state, dtype=torch.float32, device=self.config.device
            ).unsqueeze(0)
            return self.policy_net(state).argmax(1).item()

    def optimize(self) -> None:
        if len(self.memory) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )
        dones = dones.float()
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target = rewards + self.config.gamma * next_q_values * (1 - dones)

        loss = self.loss(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        return loss.item(), q_values.mean().item()

    def save(self, path: str, episode: int) -> None:
        torch.save(
            {
                "episode": episode,
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.config.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.steps_done = checkpoint["steps_done"]
        self.epsilon = checkpoint["epsilon"]

        return checkpoint["episode"]
