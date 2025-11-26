from dataclasses import dataclass
import torch


@dataclass
class Config:
    env_name: str = "ALE/Pong-v5"
    # ENV_NAME = "ALE/Breakout-v5"
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 32
    # memory_size: int = 100000
    memory_size: int = 100000
    min_replay_size: int = 10000
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay: int = 1000000
    target_update: int = 1000
    num_frames: int = 4
    device: torch.device = torch.device("mps")
    # device = (
    #     torch.device("mps") if torch.backends.mps.is_available()
    #     else torch.device("cuda") if torch.cuda.is_available()
    #     else torch.device("cpu")
    # )

    save_every: int = 50
    log_every: int = 50
