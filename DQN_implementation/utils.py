import torch
import numpy as np
from collections import deque
import cv2
import random


def to_torch_order(obs):
    # obs = np.array(obs, dtype=np.float32)  # (84,84,4)
    # obs /= 255.0
    return obs


# def preprocess(frame: np.ndarray) -> np.ndarray:
#     # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     # frame = cv2.resize(frame, (110, 84), interpolation=cv2.INTER_AREA)
#     # # Random crop to 84x84
#     # frame =
#     # return frame.astype(np.float32) / 255.0


# def stack_frames(
#     frames: deque, new_frame: np.ndarray, is_new_episode: bool, num_frames: int
# ) -> tuple[np.ndarray, deque]:
#     if is_new_episode:
#         frames = deque([new_frame] * num_frames, maxlen=num_frames)
#     else:
#         frames.append(new_frame)
#     return np.stack(frames, axis=0), frames


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
            torch.tensor(np.array(states), device=self.device),
            torch.tensor(actions, device=self.device),
            torch.tensor(rewards, device=self.device),
            torch.tensor(np.array(next_states), device=self.device),
            torch.tensor(dones, device=self.device),
        )

    def __len__(self) -> int:
        return len(self.memory)
